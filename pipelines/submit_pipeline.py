import argparse
import atexit
import json
import logging
import os
import signal
import traceback
import uuid
from datetime import datetime
from pathlib import Path
import sys

from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import PipelineJob, Environment
from azure.identity import (
    ChainedTokenCredential,
    ManagedIdentityCredential,
    AzureCliCredential,
)
import yaml

# Module logger — used for non-fatal warnings instead of bare except: pass
logger = logging.getLogger("submit_pipeline")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# K2: schema validation gate — HARD FAIL if the validator cannot be imported.
# The K2 schema check is a security gate (catches missing target_column,
# unknown task_type, etc.) and MUST run before any Azure work.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.orchestration.config_schema import validate_config as _validate_config  # type: ignore
except Exception as _e:  # pragma: no cover - validator must be present in repo
    print(f"❌ K2: config validator unavailable ({_e}). Refusing to submit without schema gate.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Duplicate-submission prevention helpers
# ---------------------------------------------------------------------------
# Operator state (lock file, audit log, last-job pointer) lives under
# $MLOPS_STATE_DIR (default ~/.mlops). NEVER inside the repo: ``git clean -fdx``
# would otherwise wipe audit history and the lock file would land in commits.
_USER_STATE_DIR = Path(os.environ.get("MLOPS_STATE_DIR", Path.home() / ".mlops"))
_LOCK_DIR = _USER_STATE_DIR / "locks"
_LOCK_DIR.mkdir(parents=True, exist_ok=True)
_LOCK_FILE = _LOCK_DIR / ".submit.lock"
_LAST_JOB_FILE = _USER_STATE_DIR / "last_submitted_job.json"

_LOCK_MAX_AGE_SEC = 4 * 60 * 60                      # 4 hours hard ceiling — protects
                                                     # against PID-recycling false hits.

# Audit trail for --force submissions (security-relevant; keep alongside lock file)
_FORCE_AUDIT_FILE = _LOCK_DIR / ".force_submit_audit.jsonl"


def _acquire_lock() -> bool:
    """Try to create a lock file.  Return True if acquired.

    Hardening notes:
    * ``os.kill(pid, 0)`` returning ``PermissionError`` (EPERM) means the PID
      exists but is owned by a different user — the lock is GENUINE, never stale.
    * Only ``ProcessLookupError`` (ESRCH) means the process is truly gone.
    * A hard ``_LOCK_MAX_AGE_SEC`` ceiling protects against PID recycling on
      long-lived shared machines.
    """
    if _LOCK_FILE.exists():
        try:
            lock_data = json.loads(_LOCK_FILE.read_text())
            lock_pid = lock_data.get("pid")
            lock_ts  = lock_data.get("ts", 0)
            lock_expires = lock_data.get("expires", 0)
            now_ts = datetime.now().timestamp()
            age = now_ts - lock_ts

            # TTL: if the lock is past its declared expiry, treat as stale
            past_ttl = lock_expires and now_ts > lock_expires
            past_age = age >= _LOCK_MAX_AGE_SEC

            if lock_pid and not past_ttl and not past_age:
                try:
                    os.kill(lock_pid, 0)          # 0-signal existence check
                    return False                  # process alive (same uid) → genuine lock
                except PermissionError:
                    # Process exists but is owned by another user — lock is REAL.
                    return False
                except ProcessLookupError:
                    pass                          # process truly gone → stale lock
                except OSError as _exc:
                    # Any other OS error: be conservative — treat as alive.
                    logger.warning("os.kill probe failed (%s); treating lock as live", _exc)
                    return False
            # Lock is stale (expired by TTL/age, or process gone) – remove it
            _LOCK_FILE.unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError) as _exc:
            logger.warning("Could not parse existing lock file (%s); reclaiming", _exc)
            _LOCK_FILE.unlink(missing_ok=True)

    now = datetime.now()
    lock_payload = json.dumps({
        "pid": os.getpid(),
        "ts": now.timestamp(),
        "started": now.isoformat(),
        "expires": now.timestamp() + _LOCK_MAX_AGE_SEC,
        "user": os.getenv("USER", "unknown"),
    })
    try:
        fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as lock_file:
        lock_file.write(lock_payload)
        lock_file.flush()
        os.fsync(lock_file.fileno())
    return True


def _release_lock():
    """Remove the lock file (safe to call multiple times)."""
    _LOCK_FILE.unlink(missing_ok=True)


def _handle_submit_signal(signum, _frame):
    """Release the submit lock before process interruption exits."""
    _release_lock()
    raise SystemExit(128 + signum)


def _check_active_jobs(ml_client: MLClient, experiment_name: str) -> list:
    """Return list of active (non-terminal) jobs in the experiment."""
    active_statuses = {"Running", "Queued", "Preparing", "Starting", "NotStarted",
                       "Provisioning", "CancelRequested"}
    active = []
    try:
        for j in ml_client.jobs.list():
            if getattr(j, "experiment_name", None) != experiment_name:
                continue
            if j.status in active_statuses:
                active.append({
                    "name": j.name,
                    "status": j.status,
                    "display_name": getattr(j, "display_name", ""),
                })
    except Exception as exc:
        logger.warning("Could not query active jobs: %s", exc)
    return active


# ---------------------------------------------------------------------------
# Repository root + dataset traversal guard
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (REPO_ROOT / "data").resolve()

# Bounded reads — Phase-1 profiling never needs more than statistics, and the
# submit host is shared / NFS-mounted so we MUST cap memory and IO.
MAX_LOCAL_CSV_BYTES = 500 * 1024 * 1024              # 500 MB hard cap
PROFILE_NROWS = 50_000                               # rows used for profiling only

# Variant safety caps — Azure ML pipeline parameter has a ~2 KB string limit
MAX_VARIANTS_PER_RUN = 50
MAX_VARIANT_LIST_CHARS = 1800


def _safe_join_data_path(blob_path: str) -> Path:
    """Resolve ``blob_path`` under ``DATA_ROOT`` and refuse traversal.

    Refuses absolute paths, paths containing ``..`` segments, and any resolved
    location that escapes ``DATA_ROOT``. Raises ``ValueError`` on any violation.
    """
    if not blob_path or not isinstance(blob_path, str):
        raise ValueError("blob_path must be a non-empty string")
    candidate = Path(blob_path)
    if candidate.is_absolute():
        raise ValueError(f"blob_path must be relative, got absolute path: {blob_path!r}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"blob_path traversal blocked (contains '..'): {blob_path!r}")
    resolved = (DATA_ROOT / candidate).resolve()
    if not str(resolved).startswith(str(DATA_ROOT) + os.sep) and resolved != DATA_ROOT:
        raise ValueError(f"blob_path traversal blocked (escapes DATA_ROOT): {blob_path!r}")
    return resolved


def _check_csv_size_within_cap(local_path: Path, max_bytes: int = MAX_LOCAL_CSV_BYTES) -> None:
    """Refuse to read CSVs larger than ``max_bytes`` on the submit host."""
    size = local_path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Local dataset {local_path.name} is {size / 1024 / 1024:.1f} MB, "
            f"exceeds {max_bytes / 1024 / 1024:.0f} MB cap. "
            "Profiling/gating must run as a remote step instead."
        )
    logger.info("Local dataset %s = %.1f MB (within %.0f MB cap)",
                local_path.name, size / 1024 / 1024, max_bytes / 1024 / 1024)


def _record_force_audit(args, user: str) -> None:
    """Append a tamper-evident audit record when --force is used."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "user": user,
        "pid": os.getpid(),
        "config": getattr(args, "config", None),
        "experiment_name": getattr(args, "experiment_name", None),
        "display_name": getattr(args, "display_name", None),
        "compute": getattr(args, "compute", None),
    }
    try:
        with open(_FORCE_AUDIT_FILE, "a") as af:
            af.write(json.dumps(record) + "\n")
    except OSError as _exc:
        logger.warning("Could not write force-submit audit log %s: %s", _FORCE_AUDIT_FILE, _exc)


# Recipe selector — sys.path was already prepared at module top for the K2 import.
from src.utils.recipe_selector import select_recipes_for_tier

# Import variant selection components (Phase 1)
try:
    from src.utils.dataset_profiler import DatasetProfiler
    from src.utils.variant_recommender import VariantRecommender
    from src.utils.variant_selector import select_variants, load_and_validate_variants
    from src.utils.variant_schema import load_variant
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False
    print("⚠️ Phase 1 components not found, falling back to legacy architecture")

# Import bundle gating (AIM-Tournament)
try:
    from src.utils.bundle_gating import (
        compute_data_signals, load_bundle_catalog,
        select_enabled_bundles, resolve_variant_paths,
        write_gating_artifacts,
    )
    BUNDLES_AVAILABLE = True
except ImportError:
    BUNDLES_AVAILABLE = False

# Pipeline builder import — component YAMLs are loaded once at import time.
# To pick up component-YAML edits, restart the process (do NOT importlib.reload
# inside a long-lived submitter — it has historically masked stale-component bugs).
from pipelines.pipeline_builder import full_pipeline, full_pipeline_v2


def _azure_from_local_config(cfg):
    """Extract azureml connection defaults from an already-loaded config dict."""
    if not cfg or not isinstance(cfg, dict):
        return None, None, None
    azure_cfg = cfg.get("azureml") or cfg.get("azure_ml") or {}
    return (
        azure_cfg.get("subscription_id"),
        azure_cfg.get("resource_group"),
        azure_cfg.get("workspace_name"),
    )


def derive_experiment_name(config_path: str) -> str:
    """Derive generic reusable experiment name from config filename.
    
    Example: config_classification_telecom_churn_azureml.yml → classification_telecom_churn_v3
    """
    config_stem = Path(config_path).stem
    normalized = config_stem.replace("config_", "").replace("_azureml", "").replace("_local", "")
    return f"{normalized}_v3"


def derive_display_name(experiment_name: str) -> str:
    """Generate unique display name for this job submission.
    
    Format: {experiment_name}_{timestamp}_{random_id}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{experiment_name}_{timestamp}_{unique_id}"


# ---------------------------------------------------------------------------
# Imputation preset filter — filters selected variants by imputation family
# ---------------------------------------------------------------------------
IMPUTATION_PRESET_MAP = {
    "auto": None,  # No filter — pass all variants through
    "statistical": ["mean", "median", "mode", "trimmed_mean", "winsorized_mean"],
    "ml_based": ["knn", "iterative"],
    "removal": ["drop"],
    "pandas_native": ["forward_fill", "backward_fill", "interpolate_linear", "constant", "zero_fill"],
    "composite": ["numeric_mean_cat_mode", "numeric_median_cat_mode"],
    "sampling": ["random_sample"],
    "advanced": None,  # All Tier 1 methods — same as "auto" until Tier 2+ is added
}


def filter_variants_by_imputation_preset(
    variant_paths: list,
    preset: str,
) -> list:
    """Filter variant paths to only those whose imputation method matches the preset.

    Args:
        variant_paths: List of absolute or relative paths to variant YAML files.
        preset: One of the keys in IMPUTATION_PRESET_MAP.

    Returns:
        Filtered list of paths. If preset is "auto" or "advanced", all paths are kept.
    """
    allowed = IMPUTATION_PRESET_MAP.get(preset)
    if allowed is None:
        # "auto" / "advanced" — no filtering
        return variant_paths

    filtered = []
    for p in variant_paths:
        try:
            variant = load_variant(str(p))
            method = variant.stage3_preprocessing.imputation.method.lower().strip()
            if method in allowed:
                filtered.append(p)
        except Exception as exc:
            # If a variant YAML is malformed, skip it with a warning
            logger.warning("Could not read variant %s for imputation filter: %s", p, exc)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Submit V3 pipeline with proper experiment/display naming")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--subscription_id", required=False, help="Azure subscription ID")
    parser.add_argument("--resource_group", required=False, help="Azure resource group")
    parser.add_argument("--workspace_name", required=False, help="Azure ML workspace name")
    parser.add_argument("--compute", required=False, default=None,
                        help="Compute target (default: $AZURE_COMPUTE env var; required if unset)")
    parser.add_argument(
        "--experiment_name",
        required=False,
        help="Reusable experiment name (auto-derived from config if not provided)",
    )
    parser.add_argument(
        "--display_name",
        required=False,
        help="Unique job display name (auto-generated with timestamp if not provided)",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for job to complete")
    parser.add_argument("--stop_compute", action="store_true",
                        help="Stop compute cluster after job completes (requires --wait)")
    parser.add_argument("--use_phase1", action="store_true", help="Use Phase 1 intelligent variant runner (NEW)")
    # V3-Proposed Planner flags
    parser.add_argument("--enable_planner", action="store_true", help="Enable V3-Proposed adaptive planner mode")
    parser.add_argument("--round1_max_variants", type=int, default=40, help="Max variants for Round 1 proxy training")
    parser.add_argument("--round2_max_variants", type=int, default=10, help="Max variants for Round 2 full training")
    parser.add_argument("--proxy_prune_threshold", type=float, default=0.50, help="Proxy metric threshold for pruning")
    parser.add_argument("--disable_cache", action="store_true", help="Disable preprocessing cache")
    parser.add_argument("--bundles_dir", required=False, default=None,
                        help="Path to variant_bundles/<task> directory for AIM-Tournament bundle gating")
    parser.add_argument("--drift_baseline_in", required=False, default=None,
                        help="Optional previous s13 drift_baseline uri_folder for baseline comparison")
    parser.add_argument("--imputation_preset", required=False, default=None,
                        choices=["auto", "statistical", "ml_based", "removal",
                                 "pandas_native", "composite", "sampling", "advanced"],
                        help="Filter variants by imputation family (overrides config value)")
    parser.add_argument("--force", action="store_true",
                        help="Skip duplicate-submission guards (lock file + active-job check). "
                             "AUDITED: appends to ~/.mlops/locks/.force_submit_audit.jsonl")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose tracebacks and debug-only diagnostics (URIs, etc.)")
    parser.add_argument("--env_version", default=None,
                        help="Azure ML environment tag (default: read from environments/azureml_unified_env.yml)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build the pipeline job and print its YAML — do NOT submit to Azure ML")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # ----- Load and validate config ONCE up front (H1) ---------------------
    config_path = args.config
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(config_path, "r") as _f:
            cfg = yaml.safe_load(_f) or {}
    except yaml.YAMLError as _ye:
        logger.error("Config %s is not valid YAML: %s", config_path, _ye)
        raise SystemExit(2) from _ye
    if not isinstance(cfg, dict):
        logger.error("Config %s did not parse to a mapping (got %s)", config_path, type(cfg).__name__)
        raise SystemExit(2)

    # K2 schema gate — fail fast BEFORE any Azure work
    if _validate_config is not None:
        try:
            _validate_config(cfg)
            logger.info("K2: config schema validation passed for %s", config_path)
            print(f"✅ K2: config schema validation passed for {config_path}")
        except Exception as _ve:
            print(f"❌ K2: config schema validation FAILED: {_ve}")
            raise SystemExit(2) from _ve

    # If CLI context missing, fall back to azureml block in cfg
    if not args.subscription_id or not args.resource_group or not args.workspace_name:
        sub, rg, ws = _azure_from_local_config(cfg)
        args.subscription_id = args.subscription_id or sub
        args.resource_group = args.resource_group or rg
        args.workspace_name = args.workspace_name or ws

    # Then env-var fallback (CLI > config > env). NEVER hardcode.
    args.subscription_id = args.subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
    args.resource_group  = args.resource_group  or os.environ.get("AZURE_RESOURCE_GROUP")
    args.workspace_name  = args.workspace_name  or os.environ.get("AZURE_WORKSPACE_NAME")
    args.compute         = args.compute         or os.environ.get("AZURE_COMPUTE")

    _missing_ctx = [n for n, v in (
        ("subscription_id (--subscription_id / AZURE_SUBSCRIPTION_ID)", args.subscription_id),
        ("resource_group (--resource_group / AZURE_RESOURCE_GROUP)",   args.resource_group),
        ("workspace_name (--workspace_name / AZURE_WORKSPACE_NAME)",   args.workspace_name),
        ("compute (--compute / AZURE_COMPUTE)",                         args.compute),
    ) if not v]
    if _missing_ctx:
        print("❌ Missing Azure context: " + ", ".join(_missing_ctx), file=sys.stderr)
        print("   See .env.example for the full list of required variables.", file=sys.stderr)
        sys.exit(2)

    # FIXED: Use config filename only (from uploaded code directory)
    # Avoid workspaceblobstore upload by passing filename as string parameter
    config_name = Path(config_path).name
    print(f"✅ Using config filename: {config_name} (from uploaded code/configs directory)")

    # Resolve env_version (CLI > config > default)
    env_version = (
        args.env_version
        or (cfg.get("azureml") or cfg.get("azure_ml") or {}).get("environment")
        or "mlops-v3-unified:20"
    )

    # Derive experiment name (reusable, generic)
    if not args.experiment_name:
        args.experiment_name = derive_experiment_name(config_path)
    
    # Derive display name (unique per submission, with timestamp)
    if not args.display_name:
        args.display_name = derive_display_name(args.experiment_name)
    
    print("\n" + "="*80)
    print("NAMING CONFIGURATION")
    print("="*80)
    print(f"📊 Experiment name (reusable):  {args.experiment_name}")
    print(f"🎯 Display name (unique):       {args.display_name}")
    print("="*80 + "\n")
    datastore_name = (cfg.get("dataset") or {}).get("datastore_name", "mlops_blob")

    # Dataset folder URI (Azure ML will mount it). The full URI exposes the
    # subscription ID; only print it when --debug is set.
    dataset_folder_uri = (
        f"azureml://subscriptions/{args.subscription_id}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace_name}"
        f"/datastores/{datastore_name}/paths/"
    )
    print(f"Using datastore: {datastore_name}")
    if args.debug:
        print(f"Dataset folder URI: {dataset_folder_uri}")
    else:
        logger.debug("Dataset folder URI: %s", dataset_folder_uri)

    # Determine task-specific recipes based on task_type from config
    # Dynamic recipe selection — ALL selected recipes will be passed to the variant runner
    all_selected_recipes = []
    task_type = cfg.get("task_type", "classification")
    try:
        # Check for phase_b_recipes config. If omitted, use committed variant_search
        # recipes through the selector rather than embedding recipe file names here.
        phase_b_config = cfg.get("phases", {}).get("phase_b_recipes", {})
        if not phase_b_config:
            phase_b_config = {
                "tier": "progressive",
                "library": "variant_search",
                "max_recipes": 2,
                "runtime_budget_sec": 300,
            }
            print("⚠️ No phase_b_recipes config found; using variant_search selector defaults")
        
        # Use dynamic tier-based selection
        tier = phase_b_config.get("tier", "balanced_performance")
        library = phase_b_config.get("library", "variant_search")
        max_recipes = phase_b_config.get("max_recipes", 8)
        runtime_budget = phase_b_config.get("runtime_budget_sec", None)
        
        print(f"🎯 Task type: {task_type}")
        print(f"📚 Using {library} recipe library, tier: {tier}, max_recipes: {max_recipes}")
        
        recipes_base_dir = Path(__file__).resolve().parents[1] / "configs" / "recipes"
        all_selected_recipes = select_recipes_for_tier(
            task_type=task_type,
            tier=tier,
            count=max_recipes,
            library=library,
            max_runtime_sec=runtime_budget,
            recipes_base_dir=recipes_base_dir
        )
        if not all_selected_recipes:
            raise ValueError(f"No Phase B recipes selected for task_type={task_type}")
        
        print(f"✅ Selected {len(all_selected_recipes)} Phase B recipes:")
        for i, r in enumerate(all_selected_recipes, 1):
            print(f"   [{i}] {r}")
        print()
    
    except Exception as e:
        print(f"❌ Could not determine task_type/recipes from config: {e}")
        raise SystemExit(2) from e
    
    # Build comma-separated variants list for the variant runner
    variants_list_str = ",".join(all_selected_recipes)
    if len(all_selected_recipes) > MAX_VARIANTS_PER_RUN:
        raise SystemExit(
            f"Refusing to submit: {len(all_selected_recipes)} variants exceed cap of "
            f"{MAX_VARIANTS_PER_RUN}. Reduce phase_b_recipes.max_recipes in config."
        )
    if len(variants_list_str) >= MAX_VARIANT_LIST_CHARS:
        raise SystemExit(
            f"Refusing to submit: variants_list string is {len(variants_list_str)} chars, "
            f"exceeds Azure ML pipeline-parameter cap of {MAX_VARIANT_LIST_CHARS}."
        )

    # ============================================================================
    # AIM-TOURNAMENT: BUNDLE GATING (data-driven variant selection)
    # ============================================================================
    bundle_gated_variants = None  # Will be set if bundle gating succeeds

    if args.bundles_dir and BUNDLES_AVAILABLE:
        print("\n" + "="*80)
        print("AIM-TOURNAMENT: BUNDLE GATING")
        print("="*80)
        try:
            import pandas as _bg_pd
            _bg_task = cfg.get("task_type", "classification")
            _bg_target = cfg.get("dataset", {}).get("target_column")
            _bg_blob = cfg.get("dataset", {}).get("blob_path", "")
            try:
                _bg_local = _safe_join_data_path(_bg_blob)
            except ValueError as _path_err:
                logger.error("Bundle gating refused dataset path %r: %s", _bg_blob, _path_err)
                raise

            if _bg_local.exists():
                _check_csv_size_within_cap(_bg_local)
                _bg_df = _bg_pd.read_csv(_bg_local, nrows=PROFILE_NROWS)
                signals = compute_data_signals(_bg_df, _bg_target, _bg_task)
                print(f"\n📡 Data signals computed ({len(signals)} signals)")
                for _sk, _sv in sorted(signals.items()):
                    print(f"   {_sk}: {_sv}")

                catalog = load_bundle_catalog(args.bundles_dir)
                enabled, decisions = select_enabled_bundles(signals, catalog)

                bundle_gated_variants = resolve_variant_paths(enabled, str(REPO_ROOT))
                write_gating_artifacts(signals, decisions, str(REPO_ROOT / "outputs" / "signals"))

                print(f"\n✅ Bundle gating: {len(enabled)}/{len(catalog)} bundles enabled → {len(bundle_gated_variants)} variants")
            else:
                print(f"⚠️ Local dataset not found at {_bg_local} — skipping bundle gating")
        except Exception as _bg_err:
            logger.warning("Bundle gating failed (non-fatal): %s", _bg_err)
            if args.debug:
                import traceback
                traceback.print_exc()

    # ============================================================================
    # PHASE 1: INTELLIGENT VARIANT SELECTION (NEW ARCHITECTURE)
    # ============================================================================
    variants_json_path = None
    use_phase1_pipeline = args.use_phase1 and PHASE1_AVAILABLE
    
    if use_phase1_pipeline:
        print("\n" + "="*80)
        print("PHASE 1: INTELLIGENT VARIANT RECOMMENDATION SYSTEM")
        print("="*80)
        
        try:
            phase_b_config = cfg.get("phases", {}).get("phase_b", {})
            if not phase_b_config:
                print("⚠️ No phase_b config found, using defaults")
                phase_b_config = {
                    "enable_profiling": True,
                    "library_dir": f"configs/recipes/{task_type}/variant_search",
                    "max_variants": 20,
                    "selection_strategy": "scored",
                    "min_relevance_score": 30.0,
                    "diversity_boost": True,
                    "runtime_budget_sec": 180,
                    "time_budget_per_variant": 600,
                    "engines": ["pycaret", "flaml"]
                }
            
            # 1. Load and profile dataset
            print("\n📊 STEP 1: Dataset Profiling")
            print("-" * 80)
            
            # Construct local dataset path for profiling
            # In Azure ML job, this would be mounted, but for submission we need local path
            dataset_blob_path = cfg.get("dataset", {}).get("blob_path", "")
            try:
                local_dataset_path = _safe_join_data_path(dataset_blob_path)
            except ValueError as _path_err:
                logger.error("Phase 1 refused dataset path %r: %s", dataset_blob_path, _path_err)
                raise
            
            if not local_dataset_path.exists():
                print(f"⚠️ Local dataset not found at {local_dataset_path}")
                print("⚠️ Falling back to alphabetical variant selection (no profiling)")
                phase_b_config["selection_strategy"] = "alphabetical"
                phase_b_config["enable_profiling"] = False
            
            profile = None
            if phase_b_config.get("enable_profiling", True) and local_dataset_path.exists():
                import pandas as pd
                _check_csv_size_within_cap(local_dataset_path)
                df = pd.read_csv(local_dataset_path, nrows=PROFILE_NROWS)
                target_column = cfg.get("dataset", {}).get("target_column")
                
                profiler = DatasetProfiler(task_type=task_type)
                profile = profiler.profile_dataset(df, target_column)
                
                print(profile.generate_profile_summary())
                recommendations = profile.recommend_preprocessing_strategies()
                
                print("\n🎯 PREPROCESSING RECOMMENDATIONS:")
                for dim, strategies in recommendations.items():
                    if dim != "priority_scores" and dim != "reasoning":
                        print(f"  • {dim}: {', '.join(strategies)}")
                
                print("\n💡 REASONING:")
                for reason in recommendations.get("reasoning", []):
                    print(f"  • {reason}")
            
            # 2. Load all variants and score
            print("\n\n🔍 STEP 2: Variant Scoring and Selection")
            print("-" * 80)
            
            library_dir = Path(__file__).resolve().parents[1] / phase_b_config.get("library_dir", f"configs/recipes/{task_type}/variant_search")
            all_variant_paths = sorted(library_dir.glob("variant_*.yml"))
            
            print(f"Found {len(all_variant_paths)} variants in library: {library_dir.name}")
            
            # Select variants based on strategy
            selection_strategy = phase_b_config.get("selection_strategy", "scored")
            max_variants = phase_b_config.get("max_variants", 20)
            runtime_budget = phase_b_config.get("runtime_budget_sec")
            
            if selection_strategy == "scored" and profile:
                # Intelligent selection using dataset profiling
                all_variants = [load_variant(str(p)) for p in all_variant_paths]
                recommender = VariantRecommender(profile, all_variants)
                selected = recommender.select_top_variants(
                    max_variants=max_variants,
                    min_relevance_score=phase_b_config.get("min_relevance_score", 30.0),
                    diversity_boost=phase_b_config.get("diversity_boost", True)
                )
                
                # Generate selection report
                report = recommender.generate_selection_report(selected)
                print(report)
                
                selected_paths = [str(v.metadata.source_path) for v, _, _ in selected]
            else:
                # Fallback: alphabetical or random selection
                selected_paths = select_variants(
                    task_type=task_type,
                    library_dir=str(library_dir),
                    max_variants=max_variants,
                    selection_strategy=selection_strategy,
                    runtime_budget_sec=runtime_budget
                )
                print(f"✅ Selected {len(selected_paths)} variants using '{selection_strategy}' strategy")
            
            print(f"\n✅ Selected {len(selected_paths)} variants for Phase B pipeline")
            
            # ── Imputation preset filter ──────────────────────────────────
            # CLI flag overrides config; default is "auto" (no filter).
            imputation_preset = (
                args.imputation_preset
                or phase_b_config.get("imputation_preset", "auto")
            )
            if imputation_preset and imputation_preset != "auto":
                pre_filter_count = len(selected_paths)
                selected_paths = filter_variants_by_imputation_preset(
                    selected_paths, imputation_preset
                )
                print(
                    f"🔬 Imputation preset '{imputation_preset}': "
                    f"{pre_filter_count} → {len(selected_paths)} variants "
                    f"(allowed methods: {IMPUTATION_PRESET_MAP.get(imputation_preset)})"
                )
                if not selected_paths:
                    print("⚠️ No variants matched the imputation preset — falling back to 'auto' (all methods)")
                    selected_paths = filter_variants_by_imputation_preset(
                        [str(v.metadata.source_path) for v, _, _ in selected]
                        if selection_strategy == "scored" and profile
                        else select_variants(
                            task_type=task_type,
                            library_dir=str(library_dir),
                            max_variants=max_variants,
                            selection_strategy=selection_strategy,
                            runtime_budget_sec=runtime_budget,
                        ),
                        "auto",
                    )
            else:
                print(f"🔬 Imputation preset: 'auto' (no filter applied)")

            # CRITICAL FIX: Convert absolute paths to relative paths from repo root
            repo_root = Path(__file__).resolve().parents[1]
            relative_paths = []
            for p in selected_paths:
                path_obj = Path(p)
                if path_obj.is_absolute():
                    try:
                        # Convert to path relative to repo root
                        rel_path = path_obj.relative_to(repo_root)
                        relative_paths.append(str(rel_path))
                    except ValueError:
                        # If path is not under repo_root, try to extract just the filename pattern
                        # This handles cases where path structure differs between environments
                        print(f"⚠️ Warning: Could not make path relative: {path_obj}")
                        relative_paths.append(str(path_obj.name))
                else:
                    # Already relative
                    relative_paths.append(str(path_obj))
            
            print(f"📝 Converted to relative paths (sample): {relative_paths[0] if relative_paths else 'N/A'}")
            print("="*80 + "\n")
            
            # Store as comma-separated string
            variants_list_str = ",".join(relative_paths)
            if len(relative_paths) > MAX_VARIANTS_PER_RUN:
                raise SystemExit(
                    f"Refusing to submit: Phase 1 selected {len(relative_paths)} variants "
                    f"(cap {MAX_VARIANTS_PER_RUN}). Tighten max_variants/min_relevance_score."
                )
            if len(variants_list_str) >= MAX_VARIANT_LIST_CHARS:
                raise SystemExit(
                    f"Refusing to submit: Phase 1 variants_list is {len(variants_list_str)} chars "
                    f"(cap {MAX_VARIANT_LIST_CHARS})."
                )
            
        except Exception as e:
            logger.error("Phase 1 variant selection failed: %s", e)
            print(f"\n❌ Phase 1 variant selection failed: {e}")
            print("⚠️ Falling back to legacy pipeline\n")
            use_phase1_pipeline = False
            if args.debug:
                import traceback
                traceback.print_exc()

    # Build pipeline job (config filename passed as string, no upload needed)
    if use_phase1_pipeline and 'variants_list_str' in locals():
        # ADVANCED: Use Phase 1 intelligent variant runner with planner mode
        print("🚀 Using Phase 1 intelligent variant runner pipeline (advanced planner)\n")
        
        engine_list = ",".join(phase_b_config.get("engines", ["pycaret", "flaml"]))
        time_budget_per_variant = phase_b_config.get("time_budget_per_variant", 600)
        
        # V3-Proposed Planner settings (from CLI or config)
        planner_config = phase_b_config.get("planner", {})
        planner_enabled = args.enable_planner or planner_config.get("enabled", False)
        round1_max = args.round1_max_variants or planner_config.get("round1_max_variants", 40)
        round2_max = args.round2_max_variants or planner_config.get("round2_max_variants", 10)
        proxy_threshold = args.proxy_prune_threshold or planner_config.get("proxy_prune_threshold", 0.50)
        cache_enabled = not args.disable_cache and planner_config.get("cache_enabled", True)
        
        if planner_enabled:
            print("="*80)
            print("V3-PROPOSED PLANNER MODE ENABLED")
            print("="*80)
            print(f"  Round 1 max variants: {round1_max}")
            print(f"  Round 2 max variants: {round2_max}")
            print(f"  Proxy prune threshold: {proxy_threshold}")
            print(f"  Preprocessing cache: {'ENABLED' if cache_enabled else 'DISABLED'}")
            print("="*80 + "\n")
        
        drift_baseline_input = Input(path=args.drift_baseline_in, type="uri_folder") if args.drift_baseline_in else None
        job = full_pipeline_v2(
            config_name=config_name,
            dataset_folder=Input(path=dataset_folder_uri, type="uri_folder"),
            variants_list=variants_list_str,
            engine_list=engine_list,
            time_budget_per_variant=time_budget_per_variant,
            drift_baseline_in=drift_baseline_input,
            # V3-Proposed Planner parameters
            planner_enabled=planner_enabled,
            round1_max_variants=round1_max,
            round2_max_variants=round2_max,
            proxy_prune_threshold=proxy_threshold,
            cache_enabled=cache_enabled
        )
    else:
        # DEFAULT: Use production pipeline with ALL selected variants
        # The variant runner processes every recipe in a single step
        engine_list_str = "pycaret,flaml"
        if task_type == "clustering":
            engine_list_str = "pycaret"  # FLAML doesn't support clustering
        
        # Read time_budget_per_variant from config (fallback 600s)
        _pb_cfg = cfg.get("phases", {}).get("phase_b", {}) if 'cfg' in dir() else {}
        _time_budget = _pb_cfg.get("time_budget_per_variant", 600)
        print(f"🚀 Using production pipeline with {len(all_selected_recipes)} variants × engines={engine_list_str}, time_budget={_time_budget}s\n")
        drift_baseline_input = Input(path=args.drift_baseline_in, type="uri_folder") if args.drift_baseline_in else None
        job = full_pipeline(
            config_name=config_name,
            dataset_folder=Input(path=dataset_folder_uri, type="uri_folder"),
            variants_list=variants_list_str,
            engine_list=engine_list_str,
            time_budget_per_variant=_time_budget,
            drift_baseline_in=drift_baseline_input,
        )
    
    # 🚀 Set display names for Phase B step (variant runner)
    if not use_phase1_pipeline:
        # Default pipeline: Set display name for variant runner step
        try:
            job.settings.default_compute = args.compute
            if hasattr(job, 'jobs') and 's06' in job.jobs:
                variant_count = len(all_selected_recipes)
                job.jobs['s06'].display_name = f"s06_phaseb_variant_runner__{variant_count}_variants"
                print(f"✅ Set display name: s06_phaseb_variant_runner__{variant_count}_variants")
        except Exception as e:
            logger.warning("Could not set display name (non-critical): %s", e)
    else:
        # Phase 1 pipeline: Set display name for variant runner step
        try:
            job.settings.default_compute = args.compute
            if hasattr(job, 'jobs') and 's06' in job.jobs:
                job.jobs['s06'].display_name = f"s06_phaseb_variant_runner__intelligent"
                print(f"✅ Set display name for intelligent variant runner")
        except Exception as e:
            logger.warning("Could not set display name (non-critical): %s", e)
    
    job.experiment_name = args.experiment_name
    job.display_name = args.display_name

    # Add job-level tags for dataset/task/preset and pipeline version
    dataset_tag = (cfg.get('dataset') or {}).get('name') or 'unknown'
    task_tag = cfg.get('task_type') or 'unknown'
    preset_tag = cfg.get('preset') or 'unknown'
    job.tags = {
        'dataset': dataset_tag,
        'task': task_tag,
        'preset': preset_tag,
        'pipeline_version': 'v3',
        'environment': env_version,
    }
    if args.force:
        job.tags['force_submit'] = 'true'
        job.tags['force_submitted_by'] = os.getenv('USER', 'unknown')

    # If Azure ML context provided, submit; else print YAML
    if args.dry_run:
        print("\n🔍 --dry_run: emitting pipeline job (NOT submitting)\n")
        print(job)
        return

    if args.subscription_id and args.resource_group and args.workspace_name:
        # ---------- Duplicate-submission guard: lock file ----------
        if not args.force:
            if not _acquire_lock():
                try:
                    lock_info = json.loads(_LOCK_FILE.read_text())
                except Exception as _le:
                    logger.warning("Could not read lock file %s: %s", _LOCK_FILE, _le)
                    lock_info = {}
                print("\n" + "="*80)
                print("🚫  DUPLICATE SUBMISSION BLOCKED")
                print("="*80)
                print(f"Another submit_pipeline.py is already running (PID {lock_info.get('pid')}, "
                      f"started {lock_info.get('started', '?')}, user {lock_info.get('user', '?')}).")
                print(f"If that process is dead, delete the lock file:")
                print(f"   rm {_LOCK_FILE}")
                print(f"Or use --force to submit anyway (audited).")
                print("="*80 + "\n")
                sys.exit(1)
            # Ensure lock is released on exit / signals
            atexit.register(_release_lock)
            signal.signal(signal.SIGTERM, _handle_submit_signal)
            signal.signal(signal.SIGINT, _handle_submit_signal)
        else:
            _force_user = os.getenv('USER', 'unknown')
            print("\n" + "="*80)
            print(f"⚠️  SECURITY NOTICE: --force bypassed all submission guards")
            print(f"   user={_force_user}  pid={os.getpid()}  time={datetime.now().isoformat()}")
            print("="*80 + "\n")
            _record_force_audit(args, _force_user)

        ml_client = MLClient(
            ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential()),
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name,
        )

        # ---------- Duplicate-submission guard: active-job check ----------
        if not args.force:
            active_jobs = _check_active_jobs(ml_client, args.experiment_name)
            if active_jobs:
                print("\n" + "="*80)
                print("⚠️   ACTIVE JOBS DETECTED in experiment: " + args.experiment_name)
                print("="*80)
                for aj in active_jobs:
                    print(f"   • {aj['name']}  [{aj['status']}]  {aj['display_name']}")
                print("\nA pipeline is already running. Submitting again will create a duplicate.")
                print("Use --force to submit anyway.")
                print("="*80 + "\n")
                _release_lock()
                sys.exit(1)

        # Environment version from component YAMLs
        print(f"Note: Using environment {env_version} from component YAMLs (includes azureml-core, sweetviz, etc.)\n")

        print("🚀 Submitting pipeline to Azure ML (this may take several minutes on NFS)...")
        submitted = ml_client.jobs.create_or_update(job)
        print(f"✅ Submitted job: {submitted.name}")
        # H2: do NOT leak subscription/rg/workspace IDs in the URL by default.
        if args.debug:
            print(f"🌐 Web View: https://ml.azure.com/runs/{submitted.name}?wsid=/subscriptions/{args.subscription_id}/resourcegroups/{args.resource_group}/workspaces/{args.workspace_name}")
        else:
            print(f"🌐 Web View: https://ml.azure.com/runs/{submitted.name}")

        # Write marker file for easy status checks later
        try:
            _USER_STATE_DIR.mkdir(parents=True, exist_ok=True)
            _LAST_JOB_FILE.write_text(json.dumps({
                "name": submitted.name,
                "display_name": args.display_name,
                "experiment": args.experiment_name,
                "submitted_at": datetime.now().isoformat(),
                "config": config_name,
            }, indent=2))
            print(f"📝 Job name saved to {_LAST_JOB_FILE}")
        except OSError as _me:
            logger.warning("Could not write last-job marker %s: %s", _LAST_JOB_FILE, _me)

        # Release lock after successful submission
        _release_lock()

        if args.wait:
            print("\n⏳ Waiting for pipeline to complete...")
            try:
                ml_client.jobs.stream(submitted.name)
            finally:
                if args.stop_compute:
                    compute_name = args.compute
                    print(f"\n🛑 Stopping compute cluster '{compute_name}'...")
                    try:
                        ml_client.compute.begin_stop(compute_name).wait()
                        print(f"✅ Compute cluster '{compute_name}' stopped successfully.")
                    except Exception as stop_err:
                        print(f"⚠️ Could not stop compute '{compute_name}': {stop_err}")
        elif args.stop_compute:
            print("⚠️ --stop_compute requires --wait. Compute will not be stopped.")
    else:
        # Local dry-run: emit job yaml for inspection
        print(job)


if __name__ == "__main__":
    main()
