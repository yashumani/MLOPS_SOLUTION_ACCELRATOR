import argparse
import atexit
import json
import os
import signal
import uuid
from datetime import datetime
from pathlib import Path
import importlib
import sys

from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import PipelineJob, Environment
from azure.identity import DefaultAzureCredential
import yaml

# K2: schema validation gate — refuse to submit if the config does not pass
# the JSON-schema + cross-field checks (e.g. missing target_column).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.orchestration.config_schema import validate_config as _validate_config  # type: ignore
except Exception as _e:  # pragma: no cover - validator must be present in repo
    _validate_config = None
    print(f"⚠️  K2: config validator unavailable ({_e}); proceeding without schema check")

# ---------------------------------------------------------------------------
# Duplicate-submission prevention helpers
# ---------------------------------------------------------------------------
_LOCK_DIR = Path(__file__).resolve().parent          # pipelines/
_LOCK_FILE = _LOCK_DIR / ".submit.lock"
_LAST_JOB_FILE = _LOCK_DIR / ".last_submitted_job"
_LOCK_MAX_AGE_SEC = 1800                             # 30 min stale threshold


def _acquire_lock() -> bool:
    """Try to create a lock file.  Return True if acquired."""
    if _LOCK_FILE.exists():
        try:
            lock_data = json.loads(_LOCK_FILE.read_text())
            lock_pid = lock_data.get("pid")
            lock_ts  = lock_data.get("ts", 0)
            age = datetime.now().timestamp() - lock_ts

            # If the locking process is still alive AND lock is fresh → blocked
            if lock_pid and age < _LOCK_MAX_AGE_SEC:
                try:
                    os.kill(lock_pid, 0)          # 0-signal existence check
                    return False                  # process alive → genuine lock
                except OSError:
                    pass                          # process gone → stale lock
            # Lock is stale – remove it
            _LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            _LOCK_FILE.unlink(missing_ok=True)

    lock_payload = json.dumps({
        "pid": os.getpid(),
        "ts": datetime.now().timestamp(),
        "started": datetime.now().isoformat(),
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
        print(f"⚠️  Could not query active jobs: {exc}")
    return active

# Add src to path for recipe selector and variant selection
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

# Force reload of pipeline_builder module to pick up latest component YAML changes
# This ensures component version increments are respected
if 'pipeline_builder' in sys.modules:
    import pipeline_builder
    importlib.reload(pipeline_builder)
    from pipeline_builder import full_pipeline, full_pipeline_v2
else:
    from pipeline_builder import full_pipeline, full_pipeline_v2


def _azure_from_local_config(config_path: str):
    """Load azureml connection defaults from a local YAML config if available."""
    p = Path(config_path)
    if not p.exists():
        return None, None, None
    try:
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        azure_cfg = cfg.get("azureml") or cfg.get("azure_ml") or {}
        return (
            azure_cfg.get("subscription_id"),
            azure_cfg.get("resource_group"),
            azure_cfg.get("workspace_name"),
        )
    except Exception:
        return None, None, None


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
            print(f"⚠️ Could not read variant {p} for imputation filter: {exc}")
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Submit V3 pipeline with proper experiment/display naming")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--subscription_id", required=False, help="Azure subscription ID")
    parser.add_argument("--resource_group", required=False, help="Azure resource group")
    parser.add_argument("--workspace_name", required=False, help="Azure ML workspace name")
    parser.add_argument("--compute", required=False, default="mlopsv2computecluster", help="Compute target")
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
                        help="Skip duplicate-submission guards (lock file + active-job check)")
    args = parser.parse_args()

    # If CLI context missing, try to read from local config YAML (when path is local)
    config_path = args.config
    if (not args.subscription_id or not args.resource_group or not args.workspace_name) and Path(args.config).exists():
        sub, rg, ws = _azure_from_local_config(args.config)
        args.subscription_id = args.subscription_id or sub
        args.resource_group = args.resource_group or rg
        args.workspace_name = args.workspace_name or ws
    
    # FIXED: Use config filename only (from uploaded code directory)
    # Avoid workspaceblobstore upload by passing filename as string parameter
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_name = Path(config_path).name
    print(f"✅ Using config filename: {config_name} (from uploaded code/configs directory)")

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
    datastore_name = "mlops_blob"
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            datastore_name = cfg.get('dataset', {}).get('datastore_name', 'mlops_blob')
    except Exception:
        pass

    # K2: validate the loaded config BEFORE doing any Azure work. Fail fast on
    # schema violations (missing target_column, bad task_type, etc.).
    if _validate_config is not None:
        try:
            with open(config_path, 'r') as _f:
                _cfg_for_validate = yaml.safe_load(_f) or {}
            _validate_config(_cfg_for_validate)
            print(f"✅ K2: config schema validation passed for {config_path}")
        except Exception as _ve:
            print(f"❌ K2: config schema validation FAILED: {_ve}")
            raise SystemExit(2) from _ve
    
    # Dataset folder URI (Azure ML will mount it)
    dataset_folder_uri = (
        f"azureml://subscriptions/{args.subscription_id}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace_name}"
        f"/datastores/{datastore_name}/paths/"
    )
    print(f"Using datastore: {datastore_name}")
    print(f"Dataset folder URI: {dataset_folder_uri}")

    # Determine task-specific recipes based on task_type from config
    # Dynamic recipe selection — ALL selected recipes will be passed to the variant runner
    all_selected_recipes = []
    task_type = "classification"
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        task_type = cfg.get("task_type", "classification")
        
        # Check for phase_b_recipes config
        phase_b_config = cfg.get("phases", {}).get("phase_b_recipes", {})
        
        if phase_b_config:
            # Use dynamic tier-based selection
            tier = phase_b_config.get("tier", "balanced_performance")
            library = phase_b_config.get("library", "v1_generated")
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
            
            print(f"✅ Selected {len(all_selected_recipes)} Phase B recipes:")
            for i, r in enumerate(all_selected_recipes, 1):
                print(f"   [{i}] {r}")
            print()
        else:
            # Fall back to hardcoded task-specific recipes (legacy behavior)
            if task_type == "classification":
                all_selected_recipes = [
                    "classification/recipe_smote_target_standard.yml",
                    "classification/recipe_knn_onehot_minmax.yml",
                ]
            elif task_type == "regression":
                all_selected_recipes = [
                    "regression/recipe_outlier_iqr_standard.yml",
                    "regression/recipe_winsorize_robust.yml",
                ]
            elif task_type == "clustering":
                all_selected_recipes = [
                    "clustering/recipe_knn_onehot_minmax.yml",
                ]
            
            print(f"Task type: {task_type}")
            print(f"⚠️ No phase_b_recipes config found, using hardcoded recipes:")
            for i, r in enumerate(all_selected_recipes, 1):
                print(f"   [{i}] {r}")
            print()
    
    except Exception as e:
        print(f"⚠️ Could not determine task_type/recipes from config: {e}")
        all_selected_recipes = ["recipe_knn_onehot_minmax.yml"]
        print(f"Using default recipes: {all_selected_recipes}\n")
    
    # Build comma-separated variants list for the variant runner
    variants_list_str = ",".join(all_selected_recipes)

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
            with open(config_path, 'r') as f:
                _bg_cfg = yaml.safe_load(f)
            _bg_task = _bg_cfg.get("task_type", "classification")
            _bg_target = _bg_cfg.get("dataset", {}).get("target_column")
            _bg_blob = _bg_cfg.get("dataset", {}).get("blob_path", "")
            _bg_local = Path(__file__).resolve().parents[1] / "data" / _bg_blob

            if _bg_local.exists():
                _bg_df = _bg_pd.read_csv(_bg_local)
                signals = compute_data_signals(_bg_df, _bg_target, _bg_task)
                print(f"\n📡 Data signals computed ({len(signals)} signals)")
                for _sk, _sv in sorted(signals.items()):
                    print(f"   {_sk}: {_sv}")

                catalog = load_bundle_catalog(args.bundles_dir)
                enabled, decisions = select_enabled_bundles(signals, catalog)

                repo_root = str(Path(__file__).resolve().parents[1])
                bundle_gated_variants = resolve_variant_paths(enabled, repo_root)
                write_gating_artifacts(signals, decisions, "outputs/signals")

                print(f"\n✅ Bundle gating: {len(enabled)}/{len(catalog)} bundles enabled → {len(bundle_gated_variants)} variants")
            else:
                print(f"⚠️ Local dataset not found at {_bg_local} — skipping bundle gating")
        except Exception as _bg_err:
            print(f"⚠️ Bundle gating failed (non-fatal): {_bg_err}")
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
            # Load config for phase_b settings
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            
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
                    "time_budget_per_variant": 300,
                    "engines": ["pycaret", "flaml"]
                }
            
            # 1. Load and profile dataset
            print("\n📊 STEP 1: Dataset Profiling")
            print("-" * 80)
            
            # Construct local dataset path for profiling
            # In Azure ML job, this would be mounted, but for submission we need local path
            dataset_blob_path = cfg.get("dataset", {}).get("blob_path", "")
            local_dataset_path = Path(__file__).resolve().parents[1] / "data" / dataset_blob_path
            
            if not local_dataset_path.exists():
                print(f"⚠️ Local dataset not found at {local_dataset_path}")
                print("⚠️ Falling back to alphabetical variant selection (no profiling)")
                phase_b_config["selection_strategy"] = "alphabetical"
                phase_b_config["enable_profiling"] = False
            
            profile = None
            if phase_b_config.get("enable_profiling", True) and local_dataset_path.exists():
                import pandas as pd
                df = pd.read_csv(local_dataset_path)
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
            
        except Exception as e:
            print(f"\n❌ Phase 1 variant selection failed: {e}")
            print("⚠️ Falling back to legacy pipeline\n")
            use_phase1_pipeline = False
            import traceback
            traceback.print_exc()

    # Build pipeline job (config filename passed as string, no upload needed)
    if use_phase1_pipeline and 'variants_list_str' in locals():
        # ADVANCED: Use Phase 1 intelligent variant runner with planner mode
        print("🚀 Using Phase 1 intelligent variant runner pipeline (advanced planner)\n")
        
        engine_list = ",".join(phase_b_config.get("engines", ["pycaret", "flaml"]))
        time_budget_per_variant = phase_b_config.get("time_budget_per_variant", 300)
        
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
        
        # Read time_budget_per_variant from config (fallback 300s)
        _pb_cfg = cfg.get("phases", {}).get("phase_b", {}) if 'cfg' in dir() else {}
        _time_budget = _pb_cfg.get("time_budget_per_variant", 300)
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
            print(f"⚠️ Could not set display name (non-critical): {e}")
    else:
        # Phase 1 pipeline: Set display name for variant runner step
        try:
            job.settings.default_compute = args.compute
            if hasattr(job, 'jobs') and 's06' in job.jobs:
                job.jobs['s06'].display_name = f"s06_phaseb_variant_runner__intelligent"
                print(f"✅ Set display name for intelligent variant runner")
        except Exception as e:
            print(f"⚠️ Could not set display name (non-critical): {e}")
    
    job.experiment_name = args.experiment_name
    job.display_name = args.display_name

    # Add job-level tags for dataset/task/preset and pipeline version
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        dataset = (cfg.get('dataset') or {}).get('name') or 'unknown'
        task = cfg.get('task_type') or 'unknown'
        preset = cfg.get('preset') or 'unknown'
        job.tags = {
            'dataset': dataset,
            'task': task,
            'preset': preset,
            'pipeline_version': 'v3',
            'environment': 'mlops-v3-unified:20',
        }
    except Exception:
        job.tags = {'pipeline_version': 'v3', 'environment': 'mlops-v3-unified:20'}

    # If Azure ML context provided, submit; else print YAML
    if args.subscription_id and args.resource_group and args.workspace_name:
        # ---------- Duplicate-submission guard: lock file ----------
        if not args.force:
            if not _acquire_lock():
                try:
                    lock_info = json.loads(_LOCK_FILE.read_text())
                except Exception:
                    lock_info = {}
                print("\n" + "="*80)
                print("🚫  DUPLICATE SUBMISSION BLOCKED")
                print("="*80)
                print(f"Another submit_pipeline.py is already running (PID {lock_info.get('pid')}, "
                      f"started {lock_info.get('started', '?')}).")
                print(f"If that process is dead, delete the lock file:")
                print(f"   rm {_LOCK_FILE}")
                print(f"Or use --force to submit anyway.")
                print("="*80 + "\n")
                sys.exit(1)
            # Ensure lock is released on exit / signals
            atexit.register(_release_lock)
            signal.signal(signal.SIGTERM, _handle_submit_signal)
            signal.signal(signal.SIGINT, _handle_submit_signal)

        ml_client = MLClient(
            DefaultAzureCredential(),
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

        # Environment version from component YAMLs (mlops-v3-unified:20)
        print("Note: Using environment mlops-v3-unified:20 from component YAMLs (includes azureml-core, sweetviz, etc.)\n")

        print("🚀 Submitting pipeline to Azure ML (this may take several minutes on NFS)...")
        submitted = ml_client.jobs.create_or_update(job)
        print(f"✅ Submitted job: {submitted.name}")
        print(f"🌐 Web View: https://ml.azure.com/runs/{submitted.name}?wsid=/subscriptions/{args.subscription_id}/resourcegroups/{args.resource_group}/workspaces/{args.workspace_name}")

        # Write marker file for easy status checks later
        try:
            _LAST_JOB_FILE.write_text(json.dumps({
                "name": submitted.name,
                "display_name": args.display_name,
                "experiment": args.experiment_name,
                "submitted_at": datetime.now().isoformat(),
                "config": config_name,
            }, indent=2))
            print(f"📝 Job name saved to {_LAST_JOB_FILE}")
        except Exception:
            pass

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
