"""
Phase B Variant Runner - Industrial-Grade Variant Execution Engine

Runs multiple preprocessing variants in one Azure ML step, using nested
MLflow runs for each variant×engine combination.

HARDENED FEATURES (Industrial Accelerator):
- Variant-level checkpointing (resume capability)
- Per-variant time budget enforcement
- Stable champion manifest contract
- Normalized variant results schema
- Defensive guards (failure tolerance)

Author: MLOps Solution Accelerator V3
Date: 2026-01-26 (Hardened)
"""

import argparse
import json
import logging
import sys
import time
import hashlib
import subprocess
import signal
import secrets
import math
import random
import multiprocessing
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import mlflow
import os
import yaml

# Ensure src/ on path (single canonical insertion at front of sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Module-level logger for diagnostic/debug messages.
logger = logging.getLogger(__name__)

from utils.variant_schema import load_variant, validate_variant_for_task, validate_variant_yaml, VariantConfig
from utils.dataset_profiler import DatasetProfiler
from utils.variant_recommender import VariantRecommender
from utils.variant_planner import (
    EdaPriors, VariantPlan, build_variant_plan, score_variant_relevance,
    diverse_sample, compute_preprocessing_hash, get_default_planner_config, VariantScore
)
from utils.preprocessing_cache import PreprocessingCache, create_preprocessing_cache
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_stage_table, sha256_file,
)
from utils.model_universe import get_model_list
from orchestration.config_compiler import (
    CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS,
    PHASE_B_TIMEOUT_CAP_SECONDS,
    ROUND1_MAX_VARIANTS_CAP,
    ROUND2_MAX_VARIANTS_CAP,
    compile_config,
)
from orchestration.contracts import (
    CandidateRecord,
    ExecutionManifest,
    QualityDecision,
    SplitManifest,
    canonical_hash,
    dataset_version_identity,
)
from utils.recipe_catalog import normalize_recipe, semantic_recipe_hash
from utils.fitted_variant_preprocessor import FittedVariantPreprocessor
from utils.model_bundle import (
    ModelBundle,
    capture_input_schema,
    save_model_bundle,
)
from utils.common_evaluator import (
    EvaluationSpec,
    build_fold_local_pipeline,
    evaluate_candidate,
)


LEGACY_VARIANTS_LIST_MAX_CHARS = 1800
_CANDIDATE_CATALOG_IDENTITY_FIELDS = (
    "execution_id",
    "recipe_catalog_hash",
    "recipe_paths",
    "recipe_ids",
    "candidate_ids",
    "candidate_records",
)


def _parse_bool(value) -> bool:
    """Parse booleans from Azure ML component string values."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# ============================================================================
# INDUSTRIAL CONTRACTS - DO NOT MODIFY WITHOUT COORDINATION
# ============================================================================

@dataclass
class VariantResult:
    """Normalized result structure - STABLE CONTRACT for downstream phases."""
    variant_id: str
    engine: str
    algorithm: str
    metrics: Dict[str, float]
    runtime_sec: float
    timed_out: bool
    failed: bool
    failure_reason: Optional[str] = None
    leakage_risk: str = "none"
    n_features: int = 0
    mlflow_run_id: Optional[str] = None
    candidate_id: Optional[str] = None
    candidate_record: Optional[Dict[str, Any]] = None


@dataclass
class ChampionManifest:
    """Champion pipeline configuration - LOCKED SCHEMA for Phase C/Registry."""
    variant_id: str
    variant_path: str
    engine: str
    algorithm: str
    primary_metric_name: str  # e.g., "accuracy", "r2"
    primary_metric_value: float  # actual metric value
    metrics: Dict[str, float]
    preprocessing_config: Dict[str, Any]
    feature_engineering_config: Dict[str, Any]
    data_fingerprint: Dict[str, Any]
    code_version: str
    timestamp: str
    leakage_risk: str
    task_type: str
    safety_net_review_required: bool = False
    review_status: str = "accepted"
    registration_eligible: bool = True
    review_reason: Optional[str] = None
    execution_id: str = ""
    candidate_id: str = ""
    mlflow_parent_run_id: Optional[str] = None
    mlflow_child_run_id: Optional[str] = None
    recipe: Dict[str, Any] = None
    model_bundle_id: Optional[str] = None


# ============================================================================
# CHECKPOINTING - Resume Capability
# ============================================================================

class CheckpointManager:
    """Manages resume state for long-running variant execution."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load existing checkpoint or create new."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {"completed": [], "last_updated": None}
    
    def is_completed(self, variant_id: str, engine: str) -> bool:
        """Check if variant×engine combination already completed."""
        key = f"{variant_id}::{engine}"
        return key in self.state["completed"]
    
    def mark_completed(self, variant_id: str, engine: str):
        """Mark variant×engine as completed and flush immediately."""
        key = f"{variant_id}::{engine}"
        if key not in self.state["completed"]:
            self.state["completed"].append(key)
            self.state["last_updated"] = datetime.utcnow().isoformat() + "Z"
            self._flush()
    
    def _flush(self):
        """Flush state to disk immediately."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_progress(self, total_expected: int = None) -> Dict[str, Any]:
        """Get current progress statistics."""
        completed = len(self.state["completed"])
        return {
            "completed": completed,
            "total": total_expected if total_expected is not None else completed,
            "last_updated": self.state["last_updated"]
        }


# ============================================================================
# SOFT TIMEOUT - No multiprocessing (Azure ML safe)
# ============================================================================

def set_deterministic_seed(seed: int = 42):
    """Set random seed across all libraries for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    # PyCaret and FLAML will use session_id/seed params separately


# ============================================================================
# DATA FINGERPRINTING
# ============================================================================

def compute_data_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute deterministic fingerprint of dataset for reproducibility."""
    # Sample-based hash for large datasets
    sample_size = min(1000, len(df))
    sample = df.head(sample_size).to_csv(index=False).encode('utf-8')
    data_hash = hashlib.sha256(sample).hexdigest()
    
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "hash": data_hash,
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def get_code_version() -> str:
    """Get git commit SHA if available, else timestamp."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return sha[:8]
    except (OSError, subprocess.CalledProcessError) as e:
        # OSError covers FileNotFoundError (git not installed); CalledProcessError
        # covers non-git directories. Fallback to timestamp-based version.
        logger.debug(f"git rev-parse unavailable, using timestamp version: {e}")
        return f"no_git_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def atomic_write(path: Path, content: str):
    """Atomic write: write to temp file, then replace target."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temp_path, 'w') as f:
            f.write(content)
        os.replace(str(temp_path), str(path))
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def publish_uri_file(source: Path, destination: str | Path) -> None:
    """Publish a required Azure ML ``uri_file`` output."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Azure ML file outputs expose the declared file path but may reject sibling
    # temporary files. The step's success boundary provides publication atomicity.
    shutil.copyfile(source, target)
    if target.stat().st_size != source.stat().st_size:
        raise IOError(f"Incomplete Azure ML output publication: {target}")


def write_variant_validation_report(output_path: Path, reports: list[dict]) -> None:
    if not reports:
        return
    json_path = output_path / "variant_validation_report.json"
    csv_path = output_path / "variant_validation_report.csv"
    atomic_write(json_path, json.dumps(reports, indent=2, default=str))
    pd.DataFrame(reports).to_csv(csv_path, index=False)
    print(f"📋 Variant YAML validation report: {json_path} / {csv_path}")


def build_variant_anomaly_report(
    variant: VariantConfig,
    transformed_df: pd.DataFrame,
    target_column: str | None,
) -> dict:
    feature_df = transformed_df.drop(columns=[target_column], errors="ignore") if target_column else transformed_df
    non_numeric = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_counts = transformed_df.isna().sum()
    missing_columns = {col: int(count) for col, count in missing_counts.items() if int(count) > 0}
    numeric_df = feature_df.select_dtypes(include=[np.number])
    infinite_columns: list[str] = []
    high_skew_columns: dict[str, float] = {}
    if not numeric_df.empty:
        infinite_counts = np.isinf(numeric_df.to_numpy()).sum(axis=0)
        infinite_columns = [
            col for col, count in zip(numeric_df.columns.tolist(), infinite_counts) if int(count) > 0
        ]
        skew_values = numeric_df.skew(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
        high_skew_columns = {
            col: round(float(value), 4)
            for col, value in skew_values.items()
            if abs(float(value)) > 2.0
        }
    status = "fail" if non_numeric or missing_columns or infinite_columns else "warn" if high_skew_columns else "pass"
    return {
        "variant_id": variant.variant_id,
        "status": status,
        "n_features": int(feature_df.shape[1]),
        "non_numeric_feature_count": len(non_numeric),
        "missing_columns_after_count": len(missing_columns),
        "infinite_columns_after_count": len(infinite_columns),
        "high_skew_feature_count": len(high_skew_columns),
        "non_numeric_features": non_numeric,
        "missing_columns": missing_columns,
        "infinite_columns": infinite_columns,
        "high_skew_columns": high_skew_columns,
    }


# ============================================================================
# DEFENSIVE GUARDS
# ============================================================================

def validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Guard against NaN/inf metrics - replace with sentinel values."""
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                print(f"⚠️ Invalid metric {key}={value}, replacing with 0.0")
                cleaned[key] = 0.0
            else:
                cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return cleaned


def check_leakage_risk(variant: VariantConfig) -> str:
    """Enhanced leakage detection - return risk level."""
    risk = variant.variant_metadata.leakage_risk if variant.variant_metadata else "unknown"
    
    # Additional heuristic checks
    if variant.stage3_preprocessing.encoding.categorical_method == "target":
        if risk == "none":
            risk = "medium"  # Target encoding has inherent risk
    
    return risk


def deadline_guard(deadline: float, label: str) -> bool:
    """Check if deadline exceeded. Returns True if exceeded."""
    if time.time() > deadline:
        print(f"   ⏰ DEADLINE EXCEEDED at {label}")
        return True
    return False


def require_phase_b_budget(deadline: float, label: str) -> float:
    """Return remaining seconds or fail the component before emitting outputs."""

    remaining = float(deadline) - time.time()
    if remaining <= 0:
        raise HardDeadlineExceeded(
            f"Phase B wall-clock budget exhausted at {label}"
        )
    return remaining


class HardDeadlineExceeded(TimeoutError):
    """Raised after a blocking child process has been terminated."""


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Force-stop one worker and its descendants without a grace overrun."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            [
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()


def run_subprocess_with_hard_deadline(
    command: list[str],
    *,
    timeout_seconds: float,
    env: Optional[dict[str, str]] = None,
) -> int:
    """Run an entire component process under one killable wall-clock ceiling."""
    if timeout_seconds <= 0:
        raise HardDeadlineExceeded("No Phase B component time remains")
    started_at = time.monotonic()
    process_options: dict[str, Any] = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True
    process = subprocess.Popen(command, env=env, **process_options)
    remaining = float(timeout_seconds) - (time.monotonic() - started_at)
    if remaining <= 0:
        _terminate_process_tree(process)
        raise HardDeadlineExceeded(
            "Phase B component hard end-to-end wall-clock budget expired "
            "during startup"
        )
    try:
        return int(process.wait(timeout=remaining))
    except subprocess.TimeoutExpired as error:
        _terminate_process_tree(process)
        raise HardDeadlineExceeded(
            "Phase B component exceeded its hard end-to-end wall-clock "
            f"budget of {float(timeout_seconds):.3f}s and was killed"
        ) from error


def _phase_b_budget_from_cli(argv: list[str]) -> float:
    """Read only the outer watchdog budget without duplicating the main parser."""
    option = "--phaseb_time_budget_sec"
    if option not in argv:
        return 10800.0
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise ValueError(f"{option} requires a value")
    budget = float(argv[index + 1])
    if not 1 <= budget <= PHASE_B_TIMEOUT_CAP_SECONDS:
        raise ValueError(
            "phaseb_time_budget_sec must be between 1 and "
            f"{PHASE_B_TIMEOUT_CAP_SECONDS}"
        )
    return budget


def run_phase_b_cli() -> int:
    """Wrap the complete S06 lifecycle, including I/O and artifact uploads."""
    token_option = "--_phaseb_watchdog_token"
    if token_option in sys.argv:
        index = sys.argv.index(token_option)
        supplied = sys.argv[index + 1] if index + 1 < len(sys.argv) else ""
        expected = os.getenv("MLOPS_S06_DEADLINE_TOKEN", "")
        if (
            not supplied
            or not expected
            or not secrets.compare_digest(supplied, expected)
        ):
            raise RuntimeError("Invalid Phase B watchdog worker token")
        del sys.argv[index : index + 2]
        main()
        return 0
    budget = _phase_b_budget_from_cli(sys.argv[1:])
    token = secrets.token_hex(32)
    worker_env = os.environ.copy()
    worker_env.pop("MLOPS_S06_DEADLINE_WORKER", None)
    worker_env["MLOPS_S06_DEADLINE_TOKEN"] = token
    return run_subprocess_with_hard_deadline(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            token_option,
            token,
            *sys.argv[1:],
        ],
        timeout_seconds=budget,
        env=worker_env,
    )


def _isolated_callable_worker(
    result_path: str,
    function,
    args: tuple,
    kwargs: dict,
) -> None:
    """Execute one pickleable callable and persist its result out of process."""

    import joblib

    try:
        value = function(*args, **kwargs)
        joblib.dump({"ok": True, "value": value}, result_path)
    except BaseException as exc:  # noqa: BLE001 - preserve child diagnostics
        joblib.dump(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            result_path,
        )


def run_with_hard_timeout(
    function,
    *args,
    timeout_seconds: float,
    **kwargs,
):
    """Run a blocking callable in a killable spawned process."""

    if timeout_seconds <= 0:
        raise HardDeadlineExceeded("No time remains before the hard deadline")
    with tempfile.TemporaryDirectory(prefix="mlops-s06-deadline-") as directory:
        result_path = str(Path(directory) / "result.joblib")
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=_isolated_callable_worker,
            args=(result_path, function, args, kwargs),
            daemon=True,
        )
        process.start()
        process.join(timeout=float(timeout_seconds))
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            raise HardDeadlineExceeded(
                f"{getattr(function, '__name__', 'operation')} exceeded "
                f"{float(timeout_seconds):.3f}s and was terminated"
            )
        if not Path(result_path).is_file():
            raise RuntimeError(
                "Isolated operation exited without a serialized result "
                f"(exit_code={process.exitcode})"
            )
        import joblib

        payload = joblib.load(result_path)
        if not payload.get("ok"):
            raise RuntimeError(
                f"{payload.get('error_type', 'ChildError')}: "
                f"{payload.get('error', 'unknown isolated failure')}\n"
                f"{payload.get('traceback', '')}".rstrip()
            )
        return payload["value"]


def get_primary_metric(task_type: str) -> str:
    """Return primary metric name for task type.

    For classification we optimise balanced accuracy so Phase B uses the
    same imbalanced-data production metric as Phase A aggregation.
    """
    if task_type == "classification":
        return "Balanced Accuracy"
    elif task_type == "regression":
        return "R2"
    elif task_type == "clustering":
        return "silhouette_score"
    return "AUC"


def get_metric_columns_for_task(task_type: str) -> list:
    """Return the appropriate metric columns for a given task type.
    
    This ensures leaderboards and output files contain the right metrics
    for each task type instead of hardcoding classification columns.
    """
    if task_type == "regression":
        return ["r2", "mae", "mse", "rmse", "rmsle", "mape"]
    elif task_type == "clustering":
        return ["silhouette", "calinski_harabasz", "davies_bouldin"]
    else:  # classification (default)
        return ["balanced_accuracy", "accuracy", "auc", "f1", "precision", "recall", "kappa", "mcc"]


def is_lower_better(metric_name: str) -> bool:
    """Return True if metric is lower-better (error metrics)."""
    metric_lower = metric_name.lower()
    lower_better_metrics = ["rmse", "mae", "mape", "logloss", "loss", "mse", "davies_bouldin"]
    return any(lb in metric_lower for lb in lower_better_metrics)


def safe_float(x) -> float:
    """Coerce to float, return -inf for None/NaN/inf."""
    if x is None:
        return float('-inf')
    try:
        val = float(x)
        if not math.isfinite(val):
            return float('-inf')
        return val
    except (TypeError, ValueError):
        return float('-inf')


def count_distinct_phaseb_candidates(valid_results) -> int:
    """Count distinct realized candidate identities in comparable results."""
    return len(
        {
            result.candidate_id
            or f"{result.variant_id}:{result.engine}:{result.algorithm}"
            for result in valid_results
        }
    )


def require_valid_phaseb_results(
    valid_results,
    minimum_candidates: int = 1,
) -> None:
    """Fail closed when too few candidates can produce selection evidence."""
    if minimum_candidates < 1:
        raise ValueError("minimum_candidates must be at least 1")
    candidate_count = count_distinct_phaseb_candidates(valid_results)
    if candidate_count >= minimum_candidates:
        return
    if minimum_candidates > 1:
        raise RuntimeError(
            f"Phase B produced {candidate_count} distinct comparable candidate(s); "
            f"at least {minimum_candidates} are required before champion selection"
        )
    raise RuntimeError(
        "Phase B produced no valid variants; cross-validation selection "
        "evidence is mandatory"
    )


def is_usable_phaseb_result(result: VariantResult) -> bool:
    """Accept every finite successful CV result; quality policy owns thresholds."""

    score = safe_float(result.metrics.get("primary_metric"))
    return (
        not result.failed
        and not result.timed_out
        and result.algorithm != "skipped"
        and math.isfinite(score)
    )


def get_result_score(result, primary_metric_name: str) -> float:
    """Unified scoring function for champion selection.
    
    Returns comparable score (higher is better) by negating lower-better metrics.
    """
    # Prefer 'primary_metric' key, fallback to metric_name
    # Use 'is not None' instead of truthiness to handle 0.0 correctly
    val = result.metrics.get("primary_metric")
    if val is None:
        val = result.metrics.get(primary_metric_name)
    val = safe_float(val)
    
    # Negate lower-better metrics for consistent comparison
    if is_lower_better(primary_metric_name):
        return -val
    return val


def apply_variant_preprocessing(
    df: pd.DataFrame,
    variant: VariantConfig,
    target_column: str = None,
    apply_smote: bool = True
) -> pd.DataFrame:
    """Apply variant preprocessing configuration to dataset.
    
    This is a simplified implementation for Phase 1.
    In Phase 2, this should reuse existing stage3/stage4 logic.
    """
    df_processed = df.copy()
    
    # Separate features and target
    if target_column and target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
    else:
        X = df_processed
        y = None
    
    # === IMPUTATION ===
    imputation_method = variant.stage3_preprocessing.imputation.method
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_imp = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if imputation_method == "mean":
        X = X.fillna(X.mean(numeric_only=True))
        # Fill remaining categorical NaNs with mode
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "median":
        X = X.fillna(X.median(numeric_only=True))
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "drop":
        X = X.dropna()
    elif imputation_method == "knn":
        from sklearn.impute import KNNImputer
        n_neighbors = getattr(variant.stage3_preprocessing.imputation, 'n_neighbors', 5)
        if num_cols and X[num_cols].isnull().any().any():
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            X[num_cols] = imputer.fit_transform(X[num_cols])
        # KNN is numeric-only; fill categorical with mode
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        max_iter = getattr(variant.stage3_preprocessing.imputation, 'max_iter', 10)
        if num_cols and X[num_cols].isnull().any().any():
            imputer = IterativeImputer(random_state=42, max_iter=max_iter)
            X[num_cols] = imputer.fit_transform(X[num_cols])
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method in ("mode", "most_frequent"):
        from sklearn.impute import SimpleImputer
        if cat_cols_imp and X[cat_cols_imp].isnull().any().any():
            imp_cat = SimpleImputer(strategy='most_frequent')
            X[cat_cols_imp] = imp_cat.fit_transform(X[cat_cols_imp])
        if num_cols and X[num_cols].isnull().any().any():
            imp_num = SimpleImputer(strategy='most_frequent')
            X[num_cols] = imp_num.fit_transform(X[num_cols])
    # --- Tier 1: pandas-native methods ---
    elif imputation_method == "forward_fill":
        X = X.ffill()
        # Any remaining NaNs at the start → backfill as safety net
        X = X.bfill()
    elif imputation_method == "backward_fill":
        X = X.bfill()
        X = X.ffill()
    elif imputation_method == "interpolate_linear":
        if num_cols:
            X[num_cols] = X[num_cols].interpolate(method='linear', limit_direction='both')
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "constant":
        fill_val = getattr(variant.stage3_preprocessing.imputation, 'fill_value', 0)
        X = X.fillna(fill_val)
    elif imputation_method == "zero_fill":
        if num_cols:
            X[num_cols] = X[num_cols].fillna(0)
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna("missing")
    # --- Tier 1: statistical variants ---
    elif imputation_method == "trimmed_mean":
        from scipy import stats
        trim_frac = getattr(variant.stage3_preprocessing.imputation, 'trim_fraction', 0.1)
        for col in num_cols:
            if X[col].isnull().any():
                t_mean = stats.trim_mean(X[col].dropna().values, proportiontocut=trim_frac)
                X[col] = X[col].fillna(t_mean)
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "winsorized_mean":
        from scipy.stats.mstats import winsorize
        trim_frac = getattr(variant.stage3_preprocessing.imputation, 'trim_fraction', 0.05)
        for col in num_cols:
            if X[col].isnull().any():
                valid = X[col].dropna().values
                wins_vals = winsorize(valid, limits=[trim_frac, trim_frac])
                X[col] = X[col].fillna(float(wins_vals.mean()))
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "random_sample":
        for col in num_cols:
            if X[col].isnull().any():
                non_null = X[col].dropna()
                n_missing = X[col].isnull().sum()
                X.loc[X[col].isnull(), col] = non_null.sample(n=n_missing, replace=True, random_state=42).values
        for col in cat_cols_imp:
            if X[col].isnull().any():
                non_null = X[col].dropna()
                n_missing = X[col].isnull().sum()
                X.loc[X[col].isnull(), col] = non_null.sample(n=n_missing, replace=True, random_state=42).values
    # --- Tier 1: column-aware composites ---
    elif imputation_method == "numeric_mean_cat_mode":
        if num_cols:
            X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "numeric_median_cat_mode":
        if num_cols:
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    else:
        import logging
        _imp_logger = logging.getLogger("s06.imputation")
        _imp_logger.warning(
            f"Unknown imputation method '{imputation_method}' for variant "
            f"'{variant.recipe_name}' — falling back to mean imputation"
        )
        X = X.fillna(X.mean(numeric_only=True))
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    
    # === OUTLIER HANDLING (H2) ===
    outlier_cfg = getattr(variant.stage3_preprocessing, 'outlier_handling', None)
    outlier_method = outlier_cfg.method if outlier_cfg else "none"
    if outlier_method and outlier_method != "none":
        _out_num = X.select_dtypes(include=[np.number]).columns.tolist()
        n_before_outlier = len(X)
        try:
            if outlier_method == "iqr_removal":
                Q1 = X[_out_num].quantile(0.25)
                Q3 = X[_out_num].quantile(0.75)
                IQR = Q3 - Q1
                mask = ~((X[_out_num] < (Q1 - 1.5 * IQR)) | (X[_out_num] > (Q3 + 1.5 * IQR))).any(axis=1)
                X = X[mask]
                if y is not None:
                    y = y[mask]
                print(f"    🔧 Outlier handling (IQR removal): {n_before_outlier} → {len(X)} rows ({n_before_outlier - len(X)} removed)")
            elif outlier_method == "iqr_capping":
                Q1 = X[_out_num].quantile(0.25)
                Q3 = X[_out_num].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                X[_out_num] = X[_out_num].clip(lower=lower, upper=upper, axis=1)
                print(f"    🔧 Outlier handling (IQR capping): values clipped to [Q1-1.5*IQR, Q3+1.5*IQR]")
            elif outlier_method == "zscore":
                from scipy import stats
                z_scores = np.abs(stats.zscore(X[_out_num], nan_policy='omit'))
                mask = (z_scores < 3).all(axis=1)
                X = X[mask]
                if y is not None:
                    y = y[mask]
                print(f"    🔧 Outlier handling (z-score): {n_before_outlier} → {len(X)} rows ({n_before_outlier - len(X)} removed)")
            elif outlier_method == "winsorize":
                from scipy.stats import mstats
                for col in _out_num:
                    X[col] = mstats.winsorize(X[col], limits=[0.05, 0.05])
                print(f"    🔧 Outlier handling (winsorize): 5th/95th percentile capping applied")
            elif outlier_method == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
                preds = iso.fit_predict(X[_out_num].fillna(0))
                mask = preds == 1
                X = X[mask]
                if y is not None:
                    y = y[mask]
                print(f"    🔧 Outlier handling (Isolation Forest): {n_before_outlier} → {len(X)} rows ({n_before_outlier - len(X)} removed)")
            else:
                print(f"    ⚠️ Outlier method '{outlier_method}' not implemented, skipping")
            # Reset index after row removal
            X = X.reset_index(drop=True)
            if y is not None:
                y = y.reset_index(drop=True)
        except Exception as e:
            print(f"    ⚠️ Outlier handling '{outlier_method}' failed: {e}, skipping")
    
    # === ENCODING ===
    encoding_method = variant.stage3_preprocessing.encoding.categorical_method
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if encoding_method == "label":
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
    elif encoding_method == "onehot":
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    elif encoding_method == "target" and len(cat_cols) > 0 and y is not None:
        from category_encoders import TargetEncoder
        te = TargetEncoder(cols=list(cat_cols), return_df=True)
        X = te.fit_transform(X, y)
        print(f"    🔧 Target encoding applied to {len(cat_cols)} columns")
    elif encoding_method and encoding_method != "none" and len(cat_cols) > 0:
        print(f"    ⚠️ Encoding method '{encoding_method}' not implemented, falling back to label encoding")
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
    
    # === SCALING ===
    scaling_method = variant.stage3_preprocessing.scaling.method
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if scaling_method == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method == "yeo_johnson" and len(numeric_cols) > 0:
        from sklearn.preprocessing import PowerTransformer
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method == "quantile" and len(numeric_cols) > 0:
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method and scaling_method != "none" and len(numeric_cols) > 0:
        print(f"    ⚠️ Scaling method '{scaling_method}' not implemented, falling back to standard scaling")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # === IMBALANCE HANDLING (C3) ===
    # Apply SMOTE/ADASYN/SMOTEENN etc. for classification tasks only.
    # Must run AFTER encoding + scaling (SMOTE requires all numeric, scaled data).
    # NOTE: apply_smote=False during training so PyCaret CV gets clean data
    # (prevents synthetic samples leaking into CV test folds).
    # SMOTE retraining happens AFTER model selection in train_pycaret_variant().
    imb_cfg = getattr(variant.stage3_preprocessing, 'imbalance_handling', None)
    imb_method = imb_cfg.method if imb_cfg else "none"
    is_classification = y is not None and y.nunique() <= 30  # heuristic
    if not apply_smote and imb_method and imb_method != "none" and is_classification:
        print(f"    ⏭️ Imbalance handling ({imb_method}) DEFERRED to post-model-selection retraining")
    elif apply_smote and imb_method and imb_method != "none" and is_classification and y is not None:
        try:
            n_before_imb = len(X)
            X_num = X.select_dtypes(include=[np.number])
            if len(X_num.columns) == len(X.columns):  # all numeric after encoding
                if imb_method == "smote":
                    from imblearn.over_sampling import SMOTE
                    sampler = SMOTE(random_state=42, n_jobs=-1)
                    X, y = sampler.fit_resample(X, y)
                    print(f"    ⚖️ Imbalance handling (SMOTE): {n_before_imb} → {len(X)} rows")
                elif imb_method == "adasyn":
                    from imblearn.over_sampling import ADASYN
                    sampler = ADASYN(random_state=42, n_jobs=-1)
                    X, y = sampler.fit_resample(X, y)
                    print(f"    ⚖️ Imbalance handling (ADASYN): {n_before_imb} → {len(X)} rows")
                elif imb_method == "smoteenn":
                    from imblearn.combine import SMOTEENN
                    sampler = SMOTEENN(random_state=42)
                    X, y = sampler.fit_resample(X, y)
                    print(f"    ⚖️ Imbalance handling (SMOTEENN): {n_before_imb} → {len(X)} rows")
                elif imb_method == "smotetomek":
                    from imblearn.combine import SMOTETomek
                    sampler = SMOTETomek(random_state=42)
                    X, y = sampler.fit_resample(X, y)
                    print(f"    ⚖️ Imbalance handling (SMOTETomek): {n_before_imb} → {len(X)} rows")
                else:
                    print(f"    ⚠️ Imbalance method '{imb_method}' not implemented, skipping")
            else:
                print(f"    ⚠️ Imbalance handling skipped: {len(X.columns) - len(X_num.columns)} non-numeric columns remain after encoding")
        except Exception as e:
            print(f"    ⚠️ Imbalance handling '{imb_method}' failed: {e}, continuing without resampling")
    elif imb_method and imb_method != "none" and not is_classification:
        print(f"    ℹ️ Imbalance handling '{imb_method}' skipped: not a classification task")
    
    # === FEATURE SELECTION ===
    fs_config = variant.stage4_feature_engineering.feature_selection
    fs_method = fs_config.method if fs_config else "none"
    fs_threshold = fs_config.threshold if fs_config and fs_config.threshold is not None else 0.01
    
    if fs_method and fs_method != "none" and y is not None:
        numeric_fs_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        n_before = len(numeric_fs_cols)
        
        if fs_method == "correlation":
            # Drop numeric features with low absolute correlation to target
            correlations = X[numeric_fs_cols].corrwith(y).abs()
            cols_to_keep = correlations[correlations >= fs_threshold].index.tolist()
            # Always keep non-numeric columns (already encoded but just in case)
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            cols_to_keep = list(set(cols_to_keep + non_numeric))
            cols_dropped = [c for c in X.columns if c not in cols_to_keep]
            if cols_dropped:
                X = X[cols_to_keep]
                print(f"    📉 Feature selection (correlation≥{fs_threshold}): {n_before} → {len(cols_to_keep)} features ({len(cols_dropped)} dropped)")
            else:
                print(f"    📉 Feature selection (correlation): all {n_before} features retained")
        
        elif fs_method == "variance":
            # Drop features with near-zero variance
            from sklearn.feature_selection import VarianceThreshold
            var_threshold = fs_threshold if fs_threshold > 0 else 0.01
            selector = VarianceThreshold(threshold=var_threshold)
            try:
                X_selected = selector.fit_transform(X[numeric_fs_cols])
                kept_mask = selector.get_support()
                kept_cols = [numeric_fs_cols[i] for i in range(len(numeric_fs_cols)) if kept_mask[i]]
                non_numeric = [c for c in X.columns if c not in numeric_fs_cols]
                X = X[kept_cols + non_numeric]
                print(f"    📉 Feature selection (variance≥{var_threshold}): {n_before} → {len(kept_cols)} features ({n_before - len(kept_cols)} dropped)")
            except Exception as e:
                print(f"    ⚠️ Variance feature selection failed: {e}, skipping")
        
        elif fs_method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            try:
                mi_func = mutual_info_classif if y.nunique() <= 20 else mutual_info_regression
                mi_scores = mi_func(X[numeric_fs_cols].fillna(0), y, random_state=42)
                mi_series = pd.Series(mi_scores, index=numeric_fs_cols)
                cols_to_keep = mi_series[mi_series >= fs_threshold].index.tolist()
                non_numeric = [c for c in X.columns if c not in numeric_fs_cols]
                if cols_to_keep:
                    X = X[cols_to_keep + non_numeric]
                    print(f"    📉 Feature selection (mutual_info≥{fs_threshold}): {n_before} → {len(cols_to_keep)} features")
                else:
                    print(f"    ⚠️ Mutual info would drop ALL features - skipping")
            except Exception as e:
                print(f"    ⚠️ Mutual info feature selection failed: {e}, skipping")
        
        else:
            print(f"    ℹ️ Feature selection method '{fs_method}' not implemented, skipping")
    elif fs_method and fs_method != "none" and y is None:
        print(f"    ⚠️ Feature selection '{fs_method}' requires target column, skipping")
    
    # Rejoin target if exists
    if y is not None:
        X[target_column] = y.values
    
    return X


def preprocess_training_data(
    df_train_raw: pd.DataFrame,
    variant: "VariantConfig",
    target_column: str,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, FittedVariantPreprocessor]:
    """Fit the complete recipe on training rows only.

    Phase B never receives or transforms the locked final-test partition.  The
    returned fitted object is the exact preprocessing graph later persisted in
    the immutable raw-input ModelBundle.
    """

    if target_column and target_column in df_train_raw.columns:
        raw_features = df_train_raw.drop(columns=[target_column])
        target = df_train_raw[target_column].copy()
    else:
        raw_features = df_train_raw.copy()
        target = None
    preprocessor = FittedVariantPreprocessor(
        variant.to_dict(),
        random_seed=random_seed,
    )
    transformed = preprocessor.fit_transform(raw_features, target)
    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(transformed)
    transformed = transformed.reset_index(drop=True)
    if target is not None:
        transformed[target_column] = target.reset_index(drop=True).values
    return transformed, preprocessor


# ============================================================================
# PHASE B SIGNAL HELPERS (Round 0 & Round 1)
# ============================================================================

def run_round0_feasibility(
    df: pd.DataFrame,
    variant: VariantConfig,
    target_column: str,
    df_transformed: pd.DataFrame | None = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Round 0: Feasibility check via transform-only (no model training).
    
    Returns:
        (df_transformed_or_none, report_dict)
        
    Report fields:
        - variant_id: str
        - status: "pass" | "fail" | "error"
        - reason: str (empty if pass)
        - transform_runtime_sec: float
        - n_features_before: int
        - n_features_after: int
        - feature_multiplier: float
        - feature_explosion: bool (>500 features or 5x multiplier)
    """
    start_time = time.time()
    n_features_before = df.shape[1] - (1 if target_column else 0)
    
    report = {
        "variant_id": variant.variant_id,
        "status": "pass",
        "reason": "",
        "transform_runtime_sec": 0.0,
        "n_features_before": n_features_before,
        "n_features_after": 0,
        "feature_multiplier": 0.0,
        "feature_explosion": False
    }
    
    try:
        # Apply preprocessing transform
        if df_transformed is None:
            df_transformed = apply_variant_preprocessing(
                df,
                variant,
                target_column,
                apply_smote=False,
            )
        n_features_after = df_transformed.shape[1] - (1 if target_column else 0)
        feature_multiplier = n_features_after / max(n_features_before, 1)
        
        report["transform_runtime_sec"] = time.time() - start_time
        report["n_features_after"] = n_features_after
        report["feature_multiplier"] = round(feature_multiplier, 2)
        
        # Zero-feature guard: preprocessing/feature selection dropped all features
        if n_features_after <= 0:
            report["status"] = "fail"
            report["reason"] = f"Zero features remaining after preprocessing ({n_features_before} → {n_features_after}). Check feature selection threshold."
            return None, report
        
        # Check feasibility gates with adaptive thresholds
        # Dynamic limit: min of 20k or max of (500, 50x original features)
        dyn_feature_limit = min(20000, max(500, 50 * n_features_before))
        report["dyn_feature_limit"] = dyn_feature_limit  # Log threshold used
        
        if n_features_after > dyn_feature_limit:
            report["status"] = "fail"
            report["reason"] = f"Feature explosion: {n_features_after} features exceeds adaptive threshold ({dyn_feature_limit})"
            report["feature_explosion"] = True
            return None, report
        
        if feature_multiplier > 10.0:  # More lenient multiplier (was 5x)
            report["status"] = "fail"
            report["reason"] = f"Feature multiplier {feature_multiplier:.1f}x exceeds threshold (10x)"
            report["feature_explosion"] = True
            return None, report
        
        # Passed all checks
        return df_transformed, report
        
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"Transform error: {str(e)}"
        report["transform_runtime_sec"] = time.time() - start_time
        return None, report


def fit_round1_proxy_preprocessor(
    X_train_raw: pd.DataFrame,
    X_validation_raw: pd.DataFrame,
    y_train: Optional[pd.Series],
    variant: VariantConfig,
    *,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    FittedVariantPreprocessor,
]:
    """Fit learned recipe transforms on proxy-training rows only."""

    preprocessor = FittedVariantPreprocessor(
        variant.to_dict(),
        random_seed=random_seed,
    )
    transformed_train = preprocessor.fit_transform(X_train_raw, y_train)
    transformed_validation = preprocessor.transform(X_validation_raw)
    return transformed_train, transformed_validation, preprocessor


def prepare_round1_proxy_partitions(
    df_raw: pd.DataFrame,
    variant: VariantConfig,
    target_column: str,
    task_type: str,
    max_samples: int = 5000,
    random_seed: int = 42,
) -> tuple[
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.Series],
    Optional[pd.Series],
    FittedVariantPreprocessor,
]:
    """Split raw proxy rows before fitting any learned recipe transform."""

    df_sample = df_raw.sample(
        n=min(max_samples, len(df_raw)),
        random_state=random_seed,
    )
    if target_column and target_column in df_sample.columns:
        raw_features = df_sample.drop(columns=[target_column])
    else:
        raw_features = df_sample.copy()

    if task_type == "clustering":
        preprocessor = FittedVariantPreprocessor(
            variant.to_dict(),
            random_seed=random_seed,
        )
        transformed = preprocessor.fit_transform(raw_features, None)
        return transformed, None, None, None, preprocessor

    if not target_column or target_column not in df_sample.columns:
        raise ValueError(f"Target column {target_column!r} missing from proxy data")
    target = df_sample[target_column]
    from sklearn.model_selection import train_test_split

    split_kwargs: dict[str, Any] = {
        "test_size": 0.3,
        "random_state": random_seed,
    }
    if task_type == "classification":
        split_kwargs["stratify"] = target
    X_train_raw, X_validation_raw, y_train, y_validation = train_test_split(
        raw_features,
        target,
        **split_kwargs,
    )
    transformed_train, transformed_validation, preprocessor = (
        fit_round1_proxy_preprocessor(
            X_train_raw,
            X_validation_raw,
            y_train,
            variant,
            random_seed=random_seed,
        )
    )
    return (
        transformed_train,
        transformed_validation,
        y_train,
        y_validation,
        preprocessor,
    )


def run_round1_proxy(
    df_raw: pd.DataFrame,
    variant: VariantConfig,
    target_column: str,
    task_type: str,
    max_samples: int = 5000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Round 1: Proxy leaderboard using cheap SGD model on sampled data.
    
    Returns report_dict with fields:
        - variant_id: str
        - status: "pass" | "fail" | "error"
        - reason: str
        - proxy_runtime_sec: float
        - proxy_metric: float (accuracy for classification, R2 for regression)
        - n_features: int
    """
    start_time = time.time()
    variant_id = variant.variant_id
    
    report = {
        "variant_id": variant_id,
        "status": "pass",
        "reason": "",
        "proxy_runtime_sec": 0.0,
        "proxy_metric": 0.0,
        "n_features": 0,
    }
    
    try:
        X_train, X_validation, y_train, y_validation, preprocessor = (
            prepare_round1_proxy_partitions(
                df_raw,
                variant,
                target_column,
                task_type,
                max_samples=max_samples,
                random_seed=random_seed,
            )
        )
        report["n_features"] = int(X_train.shape[1])
        report["preprocessor_fit_rows"] = int(len(X_train))
        report["proxy_validation_rows"] = int(
            0 if X_validation is None else len(X_validation)
        )
        report["preprocessor_task_type"] = str(
            preprocessor.recipe.get("task_type") or ""
        )

        # Train cheap proxy model with improved config
        if task_type == "classification":
            from sklearn.linear_model import SGDClassifier
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            
            # More iterations + early stopping for better convergence
            proxy_model = SGDClassifier(
                loss='log_loss',
                random_state=random_seed,
                max_iter=500,  # Increased from 100
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5
            )
            proxy_model.fit(X_train, y_train)
            y_pred = proxy_model.predict(X_validation)
            
            # Use balanced_accuracy for imbalanced datasets
            proxy_accuracy = accuracy_score(y_validation, y_pred)
            proxy_balanced_acc = balanced_accuracy_score(y_validation, y_pred)
            proxy_metric = proxy_balanced_acc  # Primary metric
            
            # Log both metrics
            report["proxy_accuracy"] = round(float(proxy_accuracy), 4)
            report["proxy_balanced_accuracy"] = round(float(proxy_balanced_acc), 4)
            
        elif task_type == "regression":
            from sklearn.linear_model import SGDRegressor
            from sklearn.metrics import r2_score
            
            proxy_model = SGDRegressor(
                random_state=random_seed,
                max_iter=500,  # Increased from 100
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5
            )
            proxy_model.fit(X_train, y_train)
            y_pred = proxy_model.predict(X_validation)
            
            # Use R2 as proxy metric
            proxy_metric = r2_score(y_validation, y_pred)
            
        elif task_type == "clustering":
            # Clustering: use KMeans + silhouette_score as proxy metric (no target column)
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            X_cluster = X_train.select_dtypes(include=['number']).copy()
            if target_column in X_cluster.columns:
                X_cluster = X_cluster.drop(columns=[target_column], errors='ignore')
            
            if X_cluster.shape[1] == 0:
                report["status"] = "error"
                report["reason"] = "No numeric features for clustering proxy"
                return report
            
            # Try k=3 as default proxy
            n_clusters = min(3, len(X_cluster) // 5)
            if n_clusters < 2:
                n_clusters = 2
            proxy_model = KMeans(
                n_clusters=n_clusters,
                random_state=random_seed,
                n_init=10,
            )
            labels = proxy_model.fit_predict(X_cluster)
            
            n_unique = len(set(labels))
            if n_unique > 1:
                proxy_metric = silhouette_score(X_cluster, labels)
            else:
                proxy_metric = -1.0
            
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        
        report["proxy_metric"] = round(float(proxy_metric), 4)
        report["proxy_runtime_sec"] = round(time.time() - start_time, 2)
        
        # Mark suspiciously low scores as warning (not fail)
        if task_type == "classification" and proxy_metric < 0.5:
            report["status"] = "warning"  # Not fail - may still train
            report["reason"] = f"Proxy metric below random baseline: {proxy_metric:.4f}"
        elif task_type == "regression" and proxy_metric < -0.5:
            report["status"] = "warning"
            report["reason"] = f"Proxy metric negative: {proxy_metric:.4f}"
        elif task_type == "clustering" and proxy_metric < 0.0:
            report["status"] = "warning"
            report["reason"] = f"Clustering silhouette negative: {proxy_metric:.4f}"
        
        return report
        
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"Proxy training error: {str(e)}"
        report["proxy_runtime_sec"] = round(time.time() - start_time, 2)
        return report


def rank_round2_proxy_survivors(
    reports: List[Dict[str, Any]],
    *,
    task_type: str,
    configured_threshold: float,
    max_variants: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Rank successful Round 1 evidence and return a bounded Round 2 list."""

    if not 1 <= max_variants <= ROUND2_MAX_VARIANTS_CAP:
        raise ValueError(
            f"Round 2 max_variants must be between 1 and {ROUND2_MAX_VARIANTS_CAP}"
        )
    effective_threshold = configured_threshold
    eligible: list[Dict[str, Any]] = []
    rejected: list[Dict[str, Any]] = []
    for report in reports:
        score = safe_float(report.get("proxy_metric"))
        status = str(report.get("status", "error"))
        if status != "pass" or score < effective_threshold:
            rejected.append(
                {
                    "variant_id": report.get("variant_id"),
                    "status": status,
                    "proxy_metric": score,
                    "reason": report.get("reason")
                    or f"proxy_metric below {effective_threshold}",
                }
            )
            continue
        eligible.append(report)
    eligible.sort(
        key=lambda report: (
            -safe_float(report.get("proxy_metric")),
            str(report.get("semantic_hash", "")),
            str(report.get("variant_id", "")),
            str(report.get("variant_path", "")),
        )
    )
    selected = eligible[:max_variants]
    for report in eligible[max_variants:]:
        rejected.append(
            {
                "variant_id": report.get("variant_id"),
                "status": "pruned",
                "proxy_metric": safe_float(report.get("proxy_metric")),
                "reason": f"outside top {max_variants} Round 2 budget",
            }
        )
    return [str(report["variant_path"]) for report in selected], rejected


def select_feasible_round1_candidates(
    catalog_scores: List[VariantScore],
    screening_reports: List[Dict[str, Any]],
    *,
    max_variants: int,
    diversity_min_hamming: int,
) -> List[VariantScore]:
    """Apply the Round 1 cap only after every catalog candidate is screened."""

    status_by_path = {
        str(report.get("variant_path")): str(report.get("status", "error"))
        for report in screening_reports
    }
    feasible = [
        candidate
        for candidate in catalog_scores
        if status_by_path.get(str(candidate.variant_path)) == "pass"
    ]
    return diverse_sample(feasible, max_variants, diversity_min_hamming)


def _canonical_recipe_path(value: str) -> str:
    """Normalize a recipe reference to its catalog-relative identity."""

    normalized = str(value).replace("\\", "/").strip()
    marker = "configs/recipes/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("./")


def resolve_recipe_paths(
    requested_paths: List[str],
    *,
    project_root: Path,
) -> List[str]:
    """Resolve recipes beneath ``configs/recipes`` and reject every escape."""

    root = project_root.resolve()
    recipes_root = (root / "configs" / "recipes").resolve()
    resolved: list[str] = []
    for requested in requested_paths:
        raw = Path(str(requested).strip())
        if not str(raw):
            raise ValueError("Recipe paths must be non-empty")
        candidates = (
            [raw]
            if raw.is_absolute()
            else [recipes_root / raw, root / raw]
        )
        existing = next(
            (candidate.resolve() for candidate in candidates if candidate.exists()),
            None,
        )
        if existing is None:
            raise FileNotFoundError(f"Cannot resolve recipe path: {requested!r}")
        try:
            existing.relative_to(recipes_root)
        except ValueError as exc:
            raise ValueError(
                f"Recipe path escapes allowed root {recipes_root}: {requested!r}"
            ) from exc
        if not existing.is_file():
            raise ValueError(f"Recipe path is not a file: {requested!r}")
        resolved.append(str(existing))
    return resolved


def load_candidate_catalog(path: str | Path) -> Tuple[Dict[str, Any], List[str]]:
    """Load the complete immutable catalog transported as an Azure ``uri_file``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate_catalog must contain a JSON object")
    missing = [
        field
        for field in _CANDIDATE_CATALOG_IDENTITY_FIELDS
        if field not in payload
    ]
    if missing:
        raise ValueError(
            "candidate_catalog is missing immutable identity fields: "
            + ", ".join(missing)
        )
    recipe_paths = payload["recipe_paths"]
    if (
        not isinstance(recipe_paths, list)
        or not recipe_paths
        or any(not isinstance(path, str) or not path.strip() for path in recipe_paths)
    ):
        raise ValueError(
            "candidate_catalog recipe_paths must be a non-empty string list"
        )
    if len(recipe_paths) != len(set(recipe_paths)):
        raise ValueError("candidate_catalog recipe_paths must be unique")
    if not isinstance(payload["candidate_records"], list) or not payload[
        "candidate_records"
    ]:
        raise ValueError(
            "candidate_catalog candidate_records must be a non-empty list"
        )
    return payload, list(recipe_paths)


def validate_candidate_catalog_binding(
    execution_payload: Dict[str, Any],
    candidate_catalog: Dict[str, Any],
) -> None:
    """Fail closed if either immutable submission artifact was substituted."""

    if not isinstance(execution_payload, dict):
        raise ValueError("execution_manifest must contain a JSON object")
    for field in _CANDIDATE_CATALOG_IDENTITY_FIELDS:
        if execution_payload.get(field) != candidate_catalog.get(field):
            raise ValueError(
                "candidate_catalog does not exactly match execution_manifest "
                f"field {field!r}"
            )


def bind_candidate_records_to_runtime_split(
    records: Tuple[CandidateRecord, ...],
    split_manifest: SplitManifest,
) -> Tuple[CandidateRecord, ...]:
    """Derive realized search identities from the actual Stage 2 row split."""

    return tuple(
        CandidateRecord(
            task_type=record.task_type,
            recipe_id=record.recipe_id,
            recipe_hash=record.recipe_hash,
            engine=record.engine,
            algorithm=record.algorithm,
            parameters=record.to_dict()["parameters"],
            split_id=split_manifest.split_id,
            data_version=record.data_version,
            code_sha=record.code_sha,
            environment_hash=record.environment_hash,
        )
        for record in records
    )


def validate_execution_manifest_for_run(
    payload: Dict[str, Any],
    *,
    config: Dict[str, Any],
    requested_variant_paths: List[str],
    resolved_variant_paths: List[str],
    engines: List[str],
    round1_max_variants: int,
    round2_max_variants: int,
    proxy_prune_threshold: float,
    candidate_engine_timeout_seconds: int,
    phase_b_timeout_seconds: int,
) -> Tuple[ExecutionManifest, Tuple[CandidateRecord, ...]]:
    """Fail closed if S06 runtime inputs diverge from the submission contract."""

    manifest = ExecutionManifest.from_dict(payload)
    expected_paths = tuple(
        _canonical_recipe_path(path) for path in requested_variant_paths
    )
    if manifest.config_hash != config["compiled_config_hash"]:
        raise ValueError("ExecutionManifest config_hash does not match compiled config")
    if manifest.task_type != config["task_type"]:
        raise ValueError("ExecutionManifest task_type does not match compiled config")
    if manifest.engines != tuple(engines):
        raise ValueError("ExecutionManifest engines do not match S06 engine_list")
    if manifest.recipe_paths != expected_paths:
        raise ValueError(
            "ExecutionManifest recipe_paths do not exactly match catalog input"
        )

    expected_recipe_ids: list[str] = []
    normalized_recipes: list[Dict[str, Any]] = []
    for resolved_path in resolved_variant_paths:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            raw_recipe = yaml.safe_load(handle) or {}
        normalized = normalize_recipe(
            raw_recipe, task_type=config["task_type"]
        )
        normalized_recipes.append(normalized)
        expected_recipe_ids.append(canonical_hash(normalized))
    if manifest.recipe_ids != tuple(expected_recipe_ids):
        raise ValueError(
            "ExecutionManifest recipe_ids do not match recipe file semantics"
        )

    expected_budgets = {
        "round1_max_variants": int(round1_max_variants),
        "round2_max_variants": int(round2_max_variants),
        "proxy_prune_threshold": float(proxy_prune_threshold),
        "candidate_engine_timeout_seconds": int(
            candidate_engine_timeout_seconds
        ),
        "phase_b_timeout_seconds": int(phase_b_timeout_seconds),
        "hpo_trials": int(config["phases"]["phase_c_hpo"]["n_trials"]),
        "hpo_timeout_seconds": int(
            config["phases"]["phase_c_hpo"]["timeout_seconds"]
        ),
    }
    if canonical_hash(manifest.budgets) != canonical_hash(expected_budgets):
        raise ValueError("ExecutionManifest budgets do not match S06 runtime inputs")

    candidate_payloads = payload.get("candidate_records")
    if not isinstance(candidate_payloads, list) or not candidate_payloads:
        raise ValueError("ExecutionManifest must include candidate_records")
    supplied_records = tuple(
        CandidateRecord.from_dict(item) for item in candidate_payloads
    )
    split_id = canonical_hash(config["split"])
    data_version = dataset_version_identity(config["dataset"])
    environment_hash = require_training_environment_hash(manifest)
    expected_records = tuple(
        CandidateRecord(
            task_type=manifest.task_type,
            recipe_id=recipe_id,
            recipe_hash=recipe_id,
            engine=engine,
            algorithm="engine_search",
            parameters=normalized,
            split_id=split_id,
            data_version=data_version,
            code_sha=manifest.code_sha,
            environment_hash=environment_hash,
        )
        for recipe_id, normalized in zip(
            expected_recipe_ids, normalized_recipes
        )
        for engine in manifest.engines
    )
    if tuple(record.candidate_id for record in expected_records) != (
        manifest.candidate_ids
    ):
        raise ValueError(
            "ExecutionManifest candidate_ids do not match canonical "
            "catalog-derived CandidateRecords"
        )
    if len(supplied_records) != len(expected_records):
        raise ValueError(
            "ExecutionManifest candidate_records do not cover recipe×engine exactly"
        )
    for index, (supplied, expected) in enumerate(
        zip(supplied_records, expected_records)
    ):
        if supplied.to_dict() != expected.to_dict():
            raise ValueError(
                "CandidateRecord does not match canonical catalog-derived "
                f"record at index {index}"
            )
    return manifest, expected_records


def require_training_environment_hash(manifest: ExecutionManifest) -> str:
    """Return the immutable training environment identity or fail closed."""

    environment_hash = str(
        manifest.environment_hashes.get("training") or ""
    ).strip()
    if not environment_hash:
        raise ValueError(
            "ExecutionManifest environment_hashes.training is required"
        )
    return environment_hash


def train_pycaret_variant(
    df: pd.DataFrame,
    variant: VariantConfig,
    target_column: str,
    task_type: str,
    time_budget: int = 300,
    random_seed: int = 42,
) -> Tuple[Any, Dict[str, Any], bool]:
    """Train models using PyCaret with variant configuration.
    
    Returns:
        (best_model, metrics_dict, timed_out_flag)
    """
    start_time = time.time()
    timed_out = False
    
    try:
        if task_type == "classification":
            from pycaret.classification import setup, compare_models, pull, add_metric
            from sklearn.metrics import balanced_accuracy_score
            
            # Get canonical model list (enforces MODEL_UNIVERSE, prevents PyCaret adding removed models)
            _include_models = get_model_list(task_type, "pycaret")
            print(f"      PyCaret include list: {len(_include_models)} models from MODEL_UNIVERSE")
            
            # Disable MLflow autologging to prevent conflicts
            # preprocess=False: data is ALREADY preprocessed by our variant pipeline
            # (encoding, scaling, imputation done in s03/s04). PyCaret re-preprocessing
            # causes double-encoding and degrades model performance.
            # fix_imbalance: when recipe requests SMOTE/ADASYN, enable PyCaret's built-in
            # per-fold resampling so models are evaluated on balanced data during CV.
            # PyCaret applies SMOTE only on each fold's train split (no leakage).
            _imb_cfg_pc = getattr(variant.stage3_preprocessing, 'imbalance_handling', None)
            _imb_method_pc = _imb_cfg_pc.method if _imb_cfg_pc else "none"
            _fix_imbalance = _imb_method_pc not in (None, "none")
            if _fix_imbalance:
                print(f"      ⚖️ PyCaret fix_imbalance=True ({_imb_method_pc}) — balanced CV enabled")
            setup(
                data=df,
                target=target_column,
                session_id=random_seed,
                fold=3,  # 3-fold CV (was defaulting to 10 — 3.3x speedup)
                preprocess=False,  # CRITICAL: avoid double-preprocessing
                normalize=False,         # K5: defense-in-depth
                transformation=False,    # K5: defense-in-depth
                fix_imbalance=_fix_imbalance,  # Apply SMOTE within CV folds when recipe requests it
                verbose=False,
                log_experiment=False,
                html=False
            )

            try:
                add_metric(
                    "balanced_accuracy",
                    "Balanced Accuracy",
                    balanced_accuracy_score,
                    target="pred",
                    greater_is_better=True,
                )
                print("      ✅ PyCaret custom metric registered: Balanced Accuracy")
            except Exception as _metric_err:
                print(f"      ⚠️ PyCaret balanced accuracy metric registration failed: {_metric_err}")
            
            # Train models with time budget (soft timeout)
            remaining_time = max(0.0, time_budget - (time.time() - start_time))
            # PyCaret accepts fractional minutes. Never round a lower configured
            # ceiling up to a one-minute floor.
            budget_minutes = max(1.0 / 60.0, remaining_time / 60.0)
            print(f"      PyCaret: remaining_time={remaining_time:.1f}s, budget={budget_minutes} minutes")
            
            if remaining_time < 30:
                timed_out = True
                return None, {
                    "primary_metric": 0.0,
                    "algorithm": "insufficient_time",
                    "runtime_sec": time.time() - start_time,
                    "timed_out": True,
                    "error": f"Insufficient time: {remaining_time:.1f}s"
                }, True
            
            sort_metric = get_primary_metric(task_type)
            try:
                best_model = compare_models(
                    include=_include_models,  # Enforce MODEL_UNIVERSE (all 14 models)
                    sort=sort_metric,
                    n_select=1,
                    budget_time=budget_minutes,
                    verbose=False
                )
            except Exception as _compare_metric_err:
                if sort_metric == "Balanced Accuracy":
                    print(f"      ⚠️ PyCaret sort by Balanced Accuracy failed: {_compare_metric_err}; falling back to AUC search")
                    sort_metric = "AUC"
                    best_model = compare_models(
                        include=_include_models,
                        sort=sort_metric,
                        n_select=1,
                        budget_time=budget_minutes,
                        verbose=False
                    )
                else:
                    raise
            
            # Check if we exceeded budget
            actual_runtime = time.time() - start_time
            if actual_runtime > time_budget:
                timed_out = True
            
            # Get leaderboard
            leaderboard = pull()
            primary_metric = sort_metric
            best_score = leaderboard[primary_metric].iloc[0]
            
            metrics = {
                "primary_metric": float(best_score),
                "accuracy": float(leaderboard.get("Accuracy", [0.0]).iloc[0]) if "Accuracy" in leaderboard else float(best_score),
                "algorithm": str(type(best_model).__name__),
                "runtime_sec": actual_runtime,
                "timed_out": timed_out,
                "n_models_trained": len(leaderboard)
            }
            
            # ── Extract ALL CV metrics from PyCaret leaderboard ──
            _PYCARET_METRIC_MAP = {
                "Balanced Accuracy": "balanced_accuracy",
                "Accuracy": "accuracy", "AUC": "auc",
                "Recall": "recall", "Prec.": "precision",
                "F1": "f1", "Kappa": "kappa", "MCC": "mcc",
            }
            for _pc_col, _m_key in _PYCARET_METRIC_MAP.items():
                if _pc_col in leaderboard.columns:
                    metrics[_m_key] = round(float(leaderboard[_pc_col].iloc[0]), 4)
            
            # ── SMOTE FULL-DATA RETRAINING ────────────────────────────────
            # Model selection used balanced CV via fix_imbalance=True above.
            # Now retrain the selected champion on the FULL training set with
            # SMOTE so the deployed model has maximum minority-class exposure.
            #
            # Any training-only resampling metadata stays attached to the
            # selected estimator; inference transforms are owned exclusively
            # by the fitted preprocessing graph in ModelBundle.
            # ─────────────────────────────────────────────────────────────
            _smote_label_encoders = {}  # col_name → fitted LabelEncoder
            _imb_cfg = getattr(variant.stage3_preprocessing, 'imbalance_handling', None)
            _imb_method = _imb_cfg.method if _imb_cfg else "none"
            if _imb_method and _imb_method != "none":
                try:
                    X_train_full = df.drop(columns=[target_column])
                    y_train_full = df[target_column]
                    _X_num = X_train_full.select_dtypes(include=[np.number])
                    # If non-numeric columns remain, label-encode them so SMOTE can proceed
                    if len(_X_num.columns) < len(X_train_full.columns):
                        from sklearn.preprocessing import LabelEncoder as _LE
                        _non_num_cols = X_train_full.select_dtypes(exclude=[np.number]).columns.tolist()
                        print(f"      ⚠️ SMOTE retrain: label-encoding {len(_non_num_cols)} non-numeric columns: {_non_num_cols[:5]}")
                        for _col in _non_num_cols:
                            _le = _LE()
                            X_train_full[_col] = _le.fit_transform(X_train_full[_col].astype(str))
                            _smote_label_encoders[_col] = _le
                    _sampler = None
                    if _imb_method == "smote":
                        from imblearn.over_sampling import SMOTE
                        _sampler = SMOTE(random_state=random_seed, n_jobs=-1)
                    elif _imb_method == "adasyn":
                        from imblearn.over_sampling import ADASYN
                        _sampler = ADASYN(random_state=random_seed, n_jobs=-1)
                    elif _imb_method == "smoteenn":
                        from imblearn.combine import SMOTEENN
                        _sampler = SMOTEENN(random_state=random_seed)
                    elif _imb_method == "smotetomek":
                        from imblearn.combine import SMOTETomek
                        _sampler = SMOTETomek(random_state=random_seed)
                    if _sampler is not None:
                        X_resampled, y_resampled = _sampler.fit_resample(X_train_full, y_train_full)
                        print(f"      ⚖️ SMOTE full-data retrain ({_imb_method}): {len(X_train_full)} → {len(X_resampled)} rows")
                        best_model.fit(X_resampled, y_resampled)
                        # Persist label encoders on the model so they survive
                        # joblib serialisation and can be applied at inference
                        if _smote_label_encoders:
                            best_model._smote_label_encoders = _smote_label_encoders
                            print(f"      💾 Saved {len(_smote_label_encoders)} SMOTE label encoders on model object")
                        metrics["smote_retrained"] = 1
                        metrics["smote_rows_after"] = len(X_resampled)
                        print(f"      ✅ Model retrained on resampled data ({len(X_resampled)} rows)")
                except Exception as _smote_err:
                    print(f"      ⚠️ SMOTE retraining failed: {_smote_err}, keeping model as-is")
                    metrics["smote_retrained"] = 0

            if "balanced_accuracy" not in metrics:
                try:
                    X_eval = df.drop(columns=[target_column])
                    y_eval = df[target_column]
                    y_pred_eval = best_model.predict(X_eval)
                    metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y_eval, y_pred_eval)), 4)
                    print(f"      ✅ PyCaret balanced_accuracy computed from selected model predictions: {metrics['balanced_accuracy']:.4f}")
                except Exception as _bal_err:
                    print(f"      ⚠️ PyCaret balanced_accuracy computation failed: {_bal_err}")

            if "balanced_accuracy" in metrics:
                metrics["primary_metric"] = metrics["balanced_accuracy"]
            
            return best_model, metrics, timed_out
            
        elif task_type == "regression":
            from pycaret.regression import setup, compare_models, pull
            
            # Get canonical model list (enforces MODEL_UNIVERSE, prevents PyCaret adding removed models)
            _include_models = get_model_list(task_type, "pycaret")
            print(f"      PyCaret include list: {len(_include_models)} models from MODEL_UNIVERSE")
            
            # preprocess=False: data is ALREADY preprocessed by our variant pipeline
            setup(
                data=df,
                target=target_column,
                session_id=random_seed,
                fold=3,  # 3-fold CV (was defaulting to 10 — 3.3x speedup)
                preprocess=False,  # CRITICAL: avoid double-preprocessing
                normalize=False,         # K5: defense-in-depth
                transformation=False,    # K5: defense-in-depth
                verbose=False,
                log_experiment=False,
                html=False
            )
            
            remaining_time = max(0.0, time_budget - (time.time() - start_time))
            budget_minutes = max(1, int(remaining_time // 60))
            print(f"      PyCaret: remaining_time={remaining_time:.1f}s, budget={budget_minutes} minutes")
            
            if remaining_time < 30:
                timed_out = True
                return None, {
                    "primary_metric": 0.0,
                    "algorithm": "insufficient_time",
                    "runtime_sec": time.time() - start_time,
                    "timed_out": True,
                    "error": f"Insufficient time: {remaining_time:.1f}s"
                }, True
            
            sort_metric = get_primary_metric(task_type)   # "R2" for regression
            best_model = compare_models(
                include=_include_models,  # Enforce MODEL_UNIVERSE (all 23 models)
                sort=sort_metric,         # Sort by R2 (explicit — matches s5a baseline)
                n_select=1,
                budget_time=budget_minutes,
                verbose=False
            )
            
            actual_runtime = time.time() - start_time
            if actual_runtime > time_budget:
                timed_out = True
            
            leaderboard = pull()
            primary_metric = sort_metric
            best_score = leaderboard[primary_metric].iloc[0]
            
            metrics = {
                "primary_metric": float(best_score),
                "r2": float(leaderboard.get("R2", [0.0]).iloc[0]) if "R2" in leaderboard else float(best_score),
                "algorithm": str(type(best_model).__name__),
                "runtime_sec": actual_runtime,
                "timed_out": timed_out,
                "n_models_trained": len(leaderboard)
            }

            # ── Extract ALL regression CV metrics from PyCaret leaderboard ──
            _PYCARET_REG_MAP = {
                "MAE": "mae", "MSE": "mse", "RMSE": "rmse",
                "R2": "r2", "RMSLE": "rmsle", "MAPE": "mape",
            }
            for _pc_col, _met_key in _PYCARET_REG_MAP.items():
                if _pc_col in leaderboard.columns:
                    metrics[_met_key] = round(float(leaderboard[_pc_col].iloc[0]), 4)

            return best_model, metrics, timed_out
            
        else:
            # Clustering remains PyCaret-only.  The outer killable process
            # enforces the same hard candidate-engine deadline.
            from pycaret.clustering import create_model, pull, setup

            df_cluster = df.copy()
            if target_column and target_column in df_cluster:
                df_cluster = df_cluster.drop(columns=[target_column])
            df_cluster = df_cluster.select_dtypes(include=[np.number])
            if df_cluster.empty or df_cluster.shape[1] == 0:
                raise ValueError("No numeric features available for clustering")
            remaining_time = max(0.0, time_budget - (time.time() - start_time))
            if remaining_time < 30:
                return None, {
                    "primary_metric": 0.0,
                    "algorithm": "insufficient_time",
                    "runtime_sec": time.time() - start_time,
                    "timed_out": True,
                    "error": f"Insufficient time: {remaining_time:.1f}s"
                }, True

            setup(
                data=df_cluster.astype(np.float64),
                session_id=random_seed,
                preprocess=False,
                normalize=False,
                transformation=False,
                verbose=False,
                log_experiment=False,
                html=False,
            )
            best_model = create_model(
                "kmeans",
                num_clusters=3,
                verbose=False,
            )
            leaderboard = pull()
            actual_runtime = time.time() - start_time
            timed_out = actual_runtime > time_budget
            silhouette_column = next(
                (
                    column
                    for column in leaderboard.columns
                    if "silhouette" in str(column).lower()
                ),
                None,
            )
            sil_score = (
                float(leaderboard[silhouette_column].iloc[0])
                if silhouette_column is not None and not leaderboard.empty
                else 0.0
            )
            metrics = {
                "primary_metric": sil_score,
                "silhouette_score": sil_score,
                "algorithm": str(type(best_model).__name__),
                "runtime_sec": actual_runtime,
                "timed_out": timed_out,
                "n_models_trained": 1,
                "engine": "pycaret",
            }
            return best_model, metrics, timed_out
            
    except Exception as e:
        print(f"❌ PyCaret training failed: {e}")
        return None, {
            "primary_metric": 0.0,
            "algorithm": "failed",
            "runtime_sec": time.time() - start_time,
            "timed_out": False,
            "error": str(e)
        }, False


def train_flaml_variant(
    df: pd.DataFrame,
    variant: VariantConfig,
    target_column: str,
    task_type: str,
    time_budget: int = 300,
    random_seed: int = 42,
) -> Tuple[Any, Dict[str, Any], bool]:
    """Train models using FLAML with variant configuration.
    
    Returns:
        (best_model, metrics_dict, timed_out_flag)
    """
    start_time = time.time()
    timed_out = False
    
    try:
        from flaml import AutoML
        
        # FLAML does not support clustering - skip with valid result
        if task_type == "clustering":
            print(f"      ⚠️  FLAML does not support clustering; skipping variant")
            return None, {
                "primary_metric": 0.0,
                "algorithm": "skipped",
                "runtime_sec": 0.0,
                "timed_out": False,
                "status": "skipped",
                "error": "FLAML does not support clustering task type"
            }, False
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        automl = AutoML()
        
        flaml_task = {
            "classification": "classification",
            "regression": "regression"
        }.get(task_type, "classification")
        
        # Check remaining time (soft timeout)
        remaining_time = max(0.0, time_budget - (time.time() - start_time))
        print(f"      FLAML budget: {min(remaining_time, time_budget):.1f} seconds")
        if remaining_time < 30:
            timed_out = True
            return None, {
                "primary_metric": 0.0,
                "algorithm": "insufficient_time",
                "runtime_sec": time.time() - start_time,
                "timed_out": True,
                "error": f"Insufficient time: {remaining_time:.1f}s"
            }, True
        
        automl.fit(
            X_train=X,
            y_train=y,
            task=flaml_task,
            time_budget=min(remaining_time, time_budget),
            metric="roc_auc" if task_type == "classification" else "r2",  # AUC handles class imbalance (matches s5b baseline)
            seed=random_seed,
            verbose=0,
            log_training_metric=False
        )
        
        actual_runtime = time.time() - start_time
        if actual_runtime > time_budget:
            timed_out = True
        
        # FIX: Prefer best_validation_score over best_loss (avoids inversion issues)
        if hasattr(automl, 'best_validation_score') and automl.best_validation_score is not None:
            primary_metric_value = float(automl.best_validation_score)
            metric_source = "best_validation_score"
        else:
            # Fallback to best_loss with convention (may need inversion)
            primary_metric_value = float(automl.best_loss) if task_type == "regression" else float(1 - automl.best_loss)
            metric_source = "best_loss_derived"
        
        metrics = {
            "primary_metric": primary_metric_value,
            "flaml_metric_source": metric_source,  # Track which metric used
            "algorithm": str(automl.best_estimator),
            "runtime_sec": actual_runtime,
            "timed_out": timed_out,
            "n_models_trained": len(automl.config_history) if hasattr(automl, 'config_history') else 1
        }
        
        # ── Track individual FLAML models tried during search ──
        # FLAML config_history: dict of {trial_id: (config, metric_value, ...)}
        flaml_individual_models = []
        if hasattr(automl, 'config_history') and automl.config_history:
            for trial_id, entry in automl.config_history.items():
                try:
                    if isinstance(entry, (list, tuple)):
                        config = entry[0] if len(entry) > 0 and isinstance(entry[0], dict) else {}
                        metric_val = float(entry[1]) if len(entry) > 1 else 0.0
                    elif isinstance(entry, dict):
                        config = entry
                        metric_val = 0.0
                    else:
                        continue
                    estimator = str(config.get('learner', config.get('ml_learner', 'unknown')))
                    flaml_individual_models.append({
                        "trial_id": int(trial_id),
                        "estimator": estimator,
                        "metric_value": round(abs(metric_val), 4),
                    })
                except Exception as e:
                    logger.debug("FLAML individual-trial parse failed: %s", e)
            print(f"      📊 FLAML tracked {len(flaml_individual_models)} individual model trials")
        metrics["flaml_individual_models"] = flaml_individual_models
        
        # ════════════════════════════════════════════════════════════════════
        # BATCH 3 FIX (UPDATED): Cross-validated metrics — TIME-AWARE.
        #
        # FLAML's best_validation_score is ALREADY cross-validated internally
        # (no resubstitution leakage).  The external cross_val_predict adds
        # detailed per-metric breakdown (F1, recall, precision, etc.) but it
        # trains the best model 3–6 extra times which can exceed the deadline.
        #
        # Strategy: check remaining wall-clock time after automl.fit().
        #   - >= 300s remaining → full 3-fold CV (predict + predict_proba)
        #   - >= 120s remaining → lightweight 2-fold CV (predict only)
        #   - <  120s remaining → skip CV; use FLAML's internal validation score
        # ══════════════════════════════════════════════════════════════════════
        remaining_after_fit = max(0.0, time_budget - (time.time() - start_time))
        
        if task_type == "classification":
            metrics["accuracy"] = metrics["primary_metric"]
            # ── Compute classification metrics (time-aware) ──
            if remaining_after_fit >= 120:
                cv_folds = 3 if remaining_after_fit >= 300 else 2
                try:
                    from sklearn.metrics import (
                        accuracy_score, f1_score, precision_score, recall_score,
                        cohen_kappa_score, matthews_corrcoef, roc_auc_score,
                        balanced_accuracy_score,
                    )
                    from sklearn.model_selection import cross_val_predict
                    from sklearn.base import clone as sklearn_clone

                    cv_model = sklearn_clone(automl.model)
                    y_pred = cross_val_predict(cv_model, X, y, cv=cv_folds, method='predict')
                    metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y, y_pred)), 4)
                    metrics["accuracy"] = round(float(accuracy_score(y, y_pred)), 4)
                    metrics["f1"] = round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4)
                    metrics["precision"] = round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4)
                    metrics["recall"] = round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4)
                    metrics["kappa"] = round(float(cohen_kappa_score(y, y_pred)), 4)
                    metrics["mcc"] = round(float(matthews_corrcoef(y, y_pred)), 4)
                    # AUC via cross-validated predict_proba (only if enough time)
                    remaining_for_proba = max(0.0, time_budget - (time.time() - start_time))
                    if remaining_for_proba >= 90:
                        try:
                            cv_model_proba = sklearn_clone(automl.model)
                            y_proba = cross_val_predict(cv_model_proba, X, y, cv=cv_folds, method='predict_proba')
                            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                                metrics["auc"] = round(float(roc_auc_score(y, y_proba[:, 1])), 4)
                            else:
                                metrics["auc"] = round(float(roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")), 4)
                        except Exception as e:
                            logger.debug("cross_val_predict AUC computation failed: %s", e)
                    else:
                        print(f"      ⏱️ Skipping AUC cross_val_predict (only {remaining_for_proba:.0f}s left)")
                    print(f"      ✅ FLAML metrics via {cv_folds}-fold cross_val_predict (no leakage)")
                except Exception as _sk_err:
                    print(f"      ⚠️ cross-validated metric computation failed (non-fatal): {_sk_err}")
            else:
                print(f"      ⏱️ Skipping cross_val_predict ({remaining_after_fit:.0f}s left < 120s); "
                      f"using FLAML internal validation score ({metrics['primary_metric']:.4f})")

            if "balanced_accuracy" not in metrics:
                try:
                    from sklearn.metrics import balanced_accuracy_score
                    y_pred_eval = automl.predict(X)
                    metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y, y_pred_eval)), 4)
                    print(f"      ✅ FLAML balanced_accuracy computed from selected model predictions: {metrics['balanced_accuracy']:.4f}")
                except Exception as _bal_err:
                    print(f"      ⚠️ FLAML balanced_accuracy computation failed: {_bal_err}")

            if "balanced_accuracy" in metrics:
                metrics["primary_metric"] = metrics["balanced_accuracy"]
        else:
            metrics["r2"] = metrics["primary_metric"]
            # ── Compute regression metrics (time-aware) ──
            if remaining_after_fit >= 120:
                cv_folds = 3 if remaining_after_fit >= 300 else 2
                try:
                    from sklearn.metrics import (
                        mean_absolute_error, mean_squared_error, r2_score,
                    )
                    from sklearn.model_selection import cross_val_predict
                    from sklearn.base import clone as sklearn_clone

                    cv_model = sklearn_clone(automl.model)
                    y_pred = cross_val_predict(cv_model, X, y, cv=cv_folds, method='predict')
                    metrics["r2"] = round(float(r2_score(y, y_pred)), 4)
                    metrics["mae"] = round(float(mean_absolute_error(y, y_pred)), 4)
                    metrics["mse"] = round(float(mean_squared_error(y, y_pred)), 4)
                    metrics["rmse"] = round(float(mean_squared_error(y, y_pred, squared=False)), 4)
                    print(f"      ✅ FLAML regression metrics via {cv_folds}-fold cross_val_predict (no leakage)")
                except Exception as _sk_err:
                    print(f"      ⚠️ cross-validated regression metric computation failed (non-fatal): {_sk_err}")
            else:
                print(f"      ⏱️ Skipping cross_val_predict ({remaining_after_fit:.0f}s left < 120s); "
                      f"using FLAML internal validation score ({metrics['primary_metric']:.4f})")
        
        return automl.model, metrics, timed_out
        
    except Exception as e:
        print(f"❌ FLAML training failed: {e}")
        return None, {
            "primary_metric": 0.0,
            "algorithm": "failed",
            "runtime_sec": time.time() - start_time,
            "timed_out": False,
            "error": str(e)
        }, False


def run_variant_with_nested_mlflow(
    variant: VariantConfig,
    df: pd.DataFrame,
    engine: str,
    target_column: str,
    task_type: str,
    time_budget: int,
    attempt_deadline: float,
    execution_id: str,
    search_candidate: CandidateRecord,
    random_seed: int,
    cv_folds: int,
    mlflow_parent_run_id: Optional[str],
    df_preprocessed: Optional[pd.DataFrame] = None  # FIX 2: Reuse cached transform
) -> Tuple[VariantResult, Any]:
    """Run one variant with one engine, nested MLflow tracking.
    
    HARDENED: Returns (VariantResult, trained_model) tuple.
    Hard budget enforced via attempt_deadline.
    
    Args:
        df_preprocessed: Optional preprocessed dataframe from Round0/Round1.
                        If provided, skips preprocessing step (avoids double transform).
    """
    
    run_name = f"variant_{variant.variant_id}_{engine}"
    start_time = time.time()
    trained_model = None
    
    try:
        # Hard budget guard: check if already exceeded
        if deadline_guard(attempt_deadline, "before_preprocessing"):
            return VariantResult(
                variant_id=variant.variant_id,
                engine=engine,
                algorithm="time_budget_exceeded",
                metrics={"primary_metric": 0.0},
                runtime_sec=time.time() - start_time,
                timed_out=True,
                failed=True,
                failure_reason="Time budget exceeded before preprocessing"
            ), None
        
        with mlflow.start_run(run_name=run_name, nested=True) as child_run:
            child_run_id = child_run.info.run_id
            mlflow.set_tag("execution_id", execution_id)
            # Log variant configuration
            mlflow.log_params({
                "variant_id": variant.variant_id,
                "engine": engine,
                "imputation": variant.stage3_preprocessing.imputation.method,
                "encoding": variant.stage3_preprocessing.encoding.categorical_method,
                "scaling": variant.stage3_preprocessing.scaling.method,
                "imbalance": variant.stage3_preprocessing.imbalance_handling.method if variant.stage3_preprocessing.imbalance_handling else "none",
                "feature_selection": variant.stage4_feature_engineering.feature_selection.method,
                "leakage_risk": check_leakage_risk(variant)
            })
            
            # Apply preprocessing with error handling (or reuse cached)
            if df_preprocessed is not None:
                print(f"  ♻️  Reusing preprocessed data from Round0/Round1...")
                df_processed = df_preprocessed
                n_features = df_processed.shape[1] - 1  # Exclude target
            else:
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm="preprocessing_contract_missing",
                    metrics={"primary_metric": 0.0},
                    runtime_sec=time.time() - start_time,
                    timed_out=False,
                    failed=True,
                    failure_reason=(
                        "Candidate execution requires the training-only fitted "
                        "preprocessing output"
                    ),
                    mlflow_run_id=child_run_id,
                ), None
            
            # Hard budget guard: check after preprocessing
            if deadline_guard(attempt_deadline, "after_preprocessing"):
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm="time_budget_exceeded",
                    metrics={"primary_metric": 0.0},
                    runtime_sec=time.time() - start_time,
                    timed_out=True,
                    failed=True,
                    failure_reason="Time budget exceeded after preprocessing",
                    mlflow_run_id=child_run_id,
                ), None
            
            # Train model with timeout enforcement
            print(f"  🏋️  Training with {engine} (budget: {time_budget}s)...")
            training_function = {
                "pycaret": train_pycaret_variant,
                "flaml": train_flaml_variant,
            }.get(engine)
            if training_function is None:
                raise ValueError(f"Unknown engine: {engine}")
            remaining_seconds = min(
                float(time_budget),
                float(CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS),
                max(0.0, attempt_deadline - time.time()),
            )
            try:
                model, metrics, timed_out = run_with_hard_timeout(
                    training_function,
                    df_processed,
                    variant,
                    target_column,
                    task_type,
                    remaining_seconds,
                    random_seed,
                    timeout_seconds=remaining_seconds,
                )
            except HardDeadlineExceeded as exc:
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm="hard_timeout",
                    metrics={"primary_metric": 0.0},
                    runtime_sec=time.time() - start_time,
                    timed_out=True,
                    failed=True,
                    failure_reason=str(exc),
                    leakage_risk=check_leakage_risk(variant),
                    n_features=n_features,
                    mlflow_run_id=child_run_id,
                ), None
            
            # Capture model for champion tracking
            trained_model = model
            
            # Validate and clean metrics
            metrics = validate_metrics(metrics)

            selected_parameters = {}
            if model is not None and hasattr(model, "get_params"):
                try:
                    selected_parameters = json.loads(
                        json.dumps(
                            model.get_params(deep=False),
                            sort_keys=True,
                            default=str,
                        )
                    )
                except Exception as exc:
                    return VariantResult(
                        variant_id=variant.variant_id,
                        engine=engine,
                        algorithm=metrics.get("algorithm", "unknown"),
                        metrics={"primary_metric": 0.0},
                        runtime_sec=time.time() - start_time,
                        timed_out=False,
                        failed=True,
                        failure_reason=(
                            "Selected model parameters are not serializable: "
                            f"{exc}"
                        ),
                        mlflow_run_id=child_run_id,
                    ), None
            realized_candidate = CandidateRecord(
                task_type=search_candidate.task_type,
                recipe_id=search_candidate.recipe_id,
                recipe_hash=search_candidate.recipe_hash,
                engine=search_candidate.engine,
                algorithm=str(metrics.get("algorithm") or type(model).__name__),
                parameters={
                    "recipe": search_candidate.to_dict()["parameters"],
                    "selected_model_parameters": selected_parameters,
                },
                split_id=search_candidate.split_id,
                data_version=search_candidate.data_version,
                code_sha=search_candidate.code_sha,
                environment_hash=search_candidate.environment_hash,
            )
            mlflow.set_tag("candidate_id", realized_candidate.candidate_id)
            mlflow.set_tag(
                "parent_search_candidate_id",
                search_candidate.candidate_id,
            )
            realized_candidate_payload = realized_candidate.to_dict()
            remaining_evaluation_seconds = min(
                float(CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS),
                max(0.0, attempt_deadline - time.time()),
            )
            if remaining_evaluation_seconds <= 0 or model is None or timed_out:
                terminal_timeout = bool(
                    timed_out or remaining_evaluation_seconds <= 0
                )
                terminal_reason = (
                    "No budget remains for common evaluation"
                    if terminal_timeout
                    else "Engine produced no selected estimator"
                )
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm=metrics.get("algorithm", "unknown"),
                    metrics={"primary_metric": 0.0},
                    runtime_sec=time.time() - start_time,
                    timed_out=terminal_timeout,
                    failed=True,
                    failure_reason=terminal_reason,
                    mlflow_run_id=child_run_id,
                    candidate_id=realized_candidate.candidate_id,
                    candidate_record={
                        **realized_candidate_payload,
                        "status": "timed_out" if terminal_timeout else "failed",
                        "timed_out": terminal_timeout,
                        "censored": terminal_timeout,
                        "failure_reason": terminal_reason,
                        "mlflow_run_id": child_run_id,
                    },
                ), None
            raw_features = (
                df.drop(columns=[target_column])
                if target_column and target_column in df
                else df.copy()
            )
            raw_target = (
                df[target_column]
                if target_column and target_column in df
                else None
            )
            evaluation_pipeline = build_fold_local_pipeline(
                FittedVariantPreprocessor(
                    variant.to_dict(),
                    random_seed=random_seed,
                ),
                model,
                recipe=variant.to_dict(),
                task_type=task_type,
                random_seed=random_seed,
            )
            evidence = evaluate_candidate(
                evaluation_pipeline,
                raw_features,
                raw_target,
                candidate_id=realized_candidate.candidate_id,
                engine=engine,
                spec=EvaluationSpec(
                    task_type=task_type,
                    seed=random_seed,
                    folds=int(cv_folds),
                    timeout_seconds=remaining_evaluation_seconds,
                    execution_id=execution_id,
                ),
                mlflow_parent_run_id=mlflow_parent_run_id,
                mlflow_child_run_id=child_run_id,
            )
            if not evidence.selectable:
                evaluator_timed_out = evidence.status in {
                    "timeout",
                    "timed_out",
                }
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm=metrics.get("algorithm", "unknown"),
                    metrics={
                        "primary_metric": 0.0,
                        "common_evaluator": evidence.to_dict(),
                    },
                    runtime_sec=time.time() - start_time,
                    timed_out=evaluator_timed_out,
                    failed=True,
                    failure_reason=(
                        evidence.failure_reason
                        or f"Common evaluation status: {evidence.status}"
                    ),
                    mlflow_run_id=child_run_id,
                    candidate_id=realized_candidate.candidate_id,
                    candidate_record={
                        **realized_candidate_payload,
                        "status": evidence.status,
                        "metrics": evidence.metrics,
                        "timed_out": evaluator_timed_out,
                        "censored": bool(evidence.censored),
                        "failure_reason": evidence.failure_reason,
                        "mlflow_run_id": child_run_id,
                    },
                ), None
            metrics = {
                **evidence.metrics,
                "primary_metric": float(evidence.selection_score),
                "algorithm": str(metrics.get("algorithm") or type(model).__name__),
                "runtime_sec": time.time() - start_time,
                "timed_out": False,
                "common_evaluator": evidence.to_dict(),
            }

            # Hard budget guard: check after training
            if deadline_guard(attempt_deadline, "after_training"):
                # Preserve real training metrics (model already trained)
                metrics["timed_out"] = True
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm=metrics.get("algorithm", "time_budget_exceeded"),
                    metrics=metrics,
                    runtime_sec=time.time() - start_time,
                    timed_out=True,
                    failed=True,
                    failure_reason="Time budget exceeded after training",
                    mlflow_run_id=child_run_id,
                ), None
            
            # Log metrics
            _mlflow_metrics = {
                "primary_metric": metrics.get("primary_metric", 0.0),
                "runtime_sec": metrics.get("runtime_sec", 0.0),
                "timed_out": 1.0 if timed_out else 0.0,
                "n_models_trained": metrics.get("n_models_trained", 1),
                "n_features": n_features
            }
            # Log all detailed classification/regression metrics
            for _mk in ("accuracy", "auc", "f1", "precision", "recall", "kappa", "mcc", "r2"):
                if _mk in metrics and isinstance(metrics[_mk], (int, float)):
                    _mlflow_metrics[_mk] = float(metrics[_mk])
            mlflow.log_metrics(_mlflow_metrics)
            
            # ── Log individual FLAML models as nested MLflow child runs ──
            # This allows users to see each model FLAML tried in Azure ML Studio
            if engine == "flaml" and metrics.get("flaml_individual_models"):
                for fm in metrics["flaml_individual_models"]:
                    fm_run_name = f"flaml_trial_{fm['trial_id']}_{fm['estimator']}"
                    try:
                        with mlflow.start_run(run_name=fm_run_name, nested=True):
                            mlflow.log_params({
                                "trial_id": fm["trial_id"],
                                "estimator": fm["estimator"],
                                "engine": "flaml",
                                "variant_id": variant.variant_id,
                            })
                            mlflow.log_metrics({
                                "metric_value": fm["metric_value"],
                            })
                    except Exception as _fm_err:
                        print(f"      ⚠️ Could not log FLAML trial {fm['trial_id']}: {_fm_err}")
            
            # Hard budget guard: check before model logging
            if deadline_guard(attempt_deadline, "before_model_logging"):
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm=metrics.get("algorithm", "time_budget_exceeded"),
                    metrics=metrics,
                    runtime_sec=time.time() - start_time,
                    timed_out=True,
                    failed=True,
                    failure_reason="Time budget exceeded before model logging",
                    mlflow_run_id=child_run_id,
                ), None
            
            # Return normalized result (no checkpoint marking here)
            return VariantResult(
                variant_id=variant.variant_id,
                engine=engine,
                algorithm=metrics.get("algorithm", "unknown"),
                metrics=metrics,
                runtime_sec=metrics.get("runtime_sec", 0.0),
                timed_out=timed_out,
                failed=(model is None or bool(timed_out)),
                failure_reason=(
                    metrics.get("error")
                    if model is None
                    else ("Engine reported timeout" if timed_out else None)
                ),
                leakage_risk=check_leakage_risk(variant),
                n_features=n_features,
                mlflow_run_id=child_run_id,
                candidate_id=realized_candidate.candidate_id,
                candidate_record={
                    **realized_candidate_payload,
                    "status": "success",
                    "metrics": evidence.metrics,
                    "timed_out": False,
                    "failure_reason": None,
                    "mlflow_run_id": child_run_id,
                },
            ), (None if timed_out else trained_model)
    
    except Exception as e:
        print(f"  ❌ Unexpected failure: {e}")
        # Return valid result (checkpoint marking happens in finally block of caller)
        return VariantResult(
            variant_id=variant.variant_id,
            engine=engine,
            algorithm="crashed",
            metrics={"primary_metric": 0.0},
            runtime_sec=time.time() - start_time,
            timed_out=False,
            failed=True,
            failure_reason=f"Unexpected error: {str(e)}"
        ), None


def main():
    parser = argparse.ArgumentParser(description="Phase B Variant Runner (HARDENED)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to main config YAML")
    parser.add_argument(
        "--execution_manifest",
        type=str,
        required=True,
        help="Immutable schema-v2 execution manifest",
    )
    parser.add_argument(
        "--split_manifest",
        type=str,
        required=True,
        help="Immutable Stage 2 split evidence",
    )
    parser.add_argument(
        "--candidate_catalog",
        type=str,
        required=False,
        help="Complete immutable recipe and CandidateRecord catalog uri_file",
    )
    parser.add_argument("--variants_json", type=str, required=False, help="JSON file containing list of variant paths")
    parser.add_argument(
        "--variants_list",
        type=str,
        required=False,
        help=(
            "Bounded compatibility-only comma-separated recipe paths; "
            "canonical submissions use --candidate_catalog"
        ),
    )
    parser.add_argument("--engine_list", type=str, required=True, help="Comma-separated engines (pycaret,flaml)")
    parser.add_argument("--dataset_in", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--leaderboard_out", type=str, required=False)
    parser.add_argument("--all_results_out", type=str, required=False)
    parser.add_argument("--champion_manifest_out", type=str, required=False)
    parser.add_argument("--execution_manifest_out", type=str, required=False)
    parser.add_argument("--split_manifest_out", type=str, required=False)
    parser.add_argument("--quality_decision_out", type=str, required=False)
    parser.add_argument("--time_budget_per_variant", type=int, default=300, help="Time budget per variant (seconds)")
    parser.add_argument("--flaml_min_budget", type=int, default=120,
                        help="Deprecated compatibility input; no floor is applied")
    parser.add_argument(
        "--phaseb_time_budget_sec",
        type=int,
        default=10800,
        help="Hard Phase B wall-clock ceiling",
    )
    
    # Signal artifact flags - DEFAULT TO TRUE (always enabled unless explicitly disabled via env vars)
    parser.add_argument("--enable_round0_feasibility", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Round 0 feasibility checks and artifact")
    parser.add_argument("--enable_round1_proxy", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Round 1 proxy leaderboard (transform-only)")
    parser.add_argument("--enable_elimination_report", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable detailed elimination report artifact")
    
    # V3-Proposed Planner flags
    parser.add_argument("--planner_enabled", action="store_true", default=False,
                        help="Enable V3-Proposed adaptive planner mode")
    parser.add_argument("--round1_max_variants", type=int, default=40,
                        help="Max variants for Round 1 proxy training")
    parser.add_argument("--round2_max_variants", type=int, default=8,
                        help="Max variants for Round 2 full training")
    parser.add_argument("--proxy_prune_threshold", type=float, default=0.50,
                        help="Proxy metric threshold for pruning (classification)")
    parser.add_argument("--diversity_min_hamming", type=int, default=2,
                        help="Min Hamming distance for diverse sampling")
    parser.add_argument("--cache_enabled", nargs="?", const=True, default=True, type=_parse_bool,
                        help="Enable preprocessing cache")
    
    args = parser.parse_args()
    if not 1 <= args.time_budget_per_variant <= CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS:
        raise ValueError(
            "time_budget_per_variant must be between 1 and "
            f"{CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS}"
        )
    if not 1 <= args.phaseb_time_budget_sec <= PHASE_B_TIMEOUT_CAP_SECONDS:
        raise ValueError(
            "phaseb_time_budget_sec must be between 1 and "
            f"{PHASE_B_TIMEOUT_CAP_SECONDS}"
        )
    if not 1 <= args.round1_max_variants <= ROUND1_MAX_VARIANTS_CAP:
        raise ValueError(
            f"round1_max_variants must be <= {ROUND1_MAX_VARIANTS_CAP}"
        )
    if not 1 <= args.round2_max_variants <= ROUND2_MAX_VARIANTS_CAP:
        raise ValueError(
            f"round2_max_variants must be <= {ROUND2_MAX_VARIANTS_CAP}"
        )
    if args.round2_max_variants > args.round1_max_variants:
        raise ValueError("Round 2 maximum cannot exceed Round 1 maximum")
    
    # Parse arguments
    engines = [e.strip().lower() for e in args.engine_list.split(",") if e.strip()]
    if not engines or len(engines) != len(set(engines)):
        raise ValueError("engine_list must contain unique, non-empty engines")
    
    # Canonical schema-v2 transport is the immutable catalog uri_file.  Keep the
    # former path inputs only for bounded/direct compatibility callers.
    candidate_catalog_payload = None
    supplied_sources = sum(
        bool(value)
        for value in (
            args.candidate_catalog,
            args.variants_json,
            args.variants_list,
        )
    )
    if supplied_sources != 1:
        raise ValueError(
            "Provide exactly one of --candidate_catalog, --variants_json, "
            "or --variants_list"
        )
    if args.candidate_catalog:
        candidate_catalog_payload, variant_paths = load_candidate_catalog(
            args.candidate_catalog
        )
    elif args.variants_json:
        with open(args.variants_json, 'r') as f:
            variant_paths = json.load(f)
    elif args.variants_list:
        if len(args.variants_list) > LEGACY_VARIANTS_LIST_MAX_CHARS:
            raise ValueError(
                "Legacy variants_list exceeds the bounded compatibility limit "
                f"of {LEGACY_VARIANTS_LIST_MAX_CHARS} characters; use "
                "--candidate_catalog"
            )
        variant_paths = [p.strip() for p in args.variants_list.split(",")]
    else:
        raise AssertionError("unreachable candidate input state")
    if (
        not isinstance(variant_paths, list)
        or not variant_paths
        or any(not isinstance(path, str) or not path.strip() for path in variant_paths)
    ):
        raise ValueError("Candidate input must contain non-empty recipe paths")
    requested_variant_paths = list(variant_paths)
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    variant_paths = resolve_recipe_paths(
        requested_variant_paths,
        project_root=_PROJECT_ROOT,
    )

    # Compile the same immutable schema-v2 artifact used by submission.
    with open(args.config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    config = compile_config(
        raw_config,
        source_name=Path(args.config_path).name,
    )
    minimum_comparable_candidates = int(
        config["metrics"]["min_comparable_candidates"]
    )
    task_type = config["task_type"]
    target_column = config["dataset"]["target_column"]
    delimiter = config["dataset"]["delimiter"]
    if tuple(engines) != tuple(config["phases"]["phase_b"]["engines"]):
        raise ValueError(
            "engine_list must exactly match compiled phases.phase_b.engines"
        )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    phase_b_started_at = time.time()
    phase_b_deadline = phase_b_started_at + args.phaseb_time_budget_sec

    with open(args.execution_manifest, "r", encoding="utf-8") as handle:
        execution_payload = json.load(handle)
    if candidate_catalog_payload is not None:
        validate_candidate_catalog_binding(
            execution_payload,
            candidate_catalog_payload,
        )
    execution_manifest, submitted_candidate_records = (
        validate_execution_manifest_for_run(
        execution_payload,
        config=config,
        requested_variant_paths=requested_variant_paths,
        resolved_variant_paths=variant_paths,
        engines=engines,
        round1_max_variants=args.round1_max_variants,
        round2_max_variants=args.round2_max_variants,
        proxy_prune_threshold=args.proxy_prune_threshold,
        candidate_engine_timeout_seconds=args.time_budget_per_variant,
        phase_b_timeout_seconds=args.phaseb_time_budget_sec,
        )
    )
    split_manifest = SplitManifest.from_json(
        Path(args.split_manifest).read_text(encoding="utf-8")
    )
    if split_manifest.task_type != task_type:
        raise ValueError("SplitManifest task_type does not match config")
    if split_manifest.random_seed != int(config["random_seed"]):
        raise ValueError("SplitManifest random_seed does not match config")
    if split_manifest.strategy != config["split"]["strategy"]:
        raise ValueError("SplitManifest strategy does not match config")
    if not split_manifest.locked_test or split_manifest.test_count <= 0:
        raise ValueError("SplitManifest must prove a non-empty locked test set")
    canonical_candidate_records = bind_candidate_records_to_runtime_split(
        submitted_candidate_records,
        split_manifest,
    )
    validated_execution_payload = {
        **execution_manifest.to_dict(),
        "candidate_records": [
            record.to_dict() for record in submitted_candidate_records
        ],
        "split_manifest": split_manifest.to_dict(),
        "runtime_split_id": split_manifest.split_id,
        "runtime_candidate_ids": [
            record.candidate_id for record in canonical_candidate_records
        ],
        "runtime_candidate_records": [
            record.to_dict() for record in canonical_candidate_records
        ],
    }
    atomic_write(
        output_path / "execution_manifest.json",
        json.dumps(
            validated_execution_payload,
            indent=2,
            sort_keys=True,
        ),
    )
    atomic_write(
        output_path / "split_manifest.json",
        split_manifest.to_json(indent=2),
    )
    
    print(f"\n{'='*80}")
    print(f"PHASE B VARIANT RUNNER (HARDENED)")
    print(f"{'='*80}")
    print(f"Variants to process: {len(variant_paths)}")
    print(f"Engines: {', '.join(engines)}")
    print(f"Total runs: {len(variant_paths) * len(engines)}")
    print(f"{'='*80}\n")
    
    phases_cfg = config.get("phases") or {}
    phase_b_cfg = phases_cfg.get("phase_b") or phases_cfg.get("phase_b_recipes") or {}
    print(f"Execution manifest: {execution_manifest.execution_id}")
    
    # ======== HARDENING FEATURE 6: DETERMINISTIC SEEDING ========
    set_deterministic_seed(int(config["random_seed"]))
    
    # Load dataset (respect config delimiter — e.g. bank_marketing uses ";")
    print(f"📂 Loading dataset from {args.dataset_in}")
    df = pd.read_csv(args.dataset_in, sep=delimiter)
    if df.empty or len(df) != split_manifest.train_count:
        raise ValueError(
            "Training-only input row count does not match SplitManifest"
        )
    print(f"   Training-only shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # ======== HARDENING FEATURE 4: DATA FINGERPRINTING ========
    data_fingerprint = compute_data_fingerprint(df)
    code_version = get_code_version()
    print(f"   Data fingerprint: {data_fingerprint['hash'][:12]}... (training only)")
    print(f"   Code version: {code_version}\n")
    
    # ======== V3-PROPOSED: PREPROCESSING CACHE ========
    preprocessing_cache = create_preprocessing_cache(
        enabled=args.cache_enabled,
        max_entries=50
    )
    print(f"🗄️  Preprocessing cache: {'ENABLED' if args.cache_enabled else 'DISABLED'}")
    
    # ======== V3-PROPOSED: ADAPTIVE PLANNER MODE ========
    variant_plan = None
    if args.planner_enabled:
        print(f"\n{'='*80}")
        print(f"V3-PROPOSED: ADAPTIVE PLANNER MODE ENABLED")
        print(f"{'='*80}")
        
        # Build EDA priors from dataset (simplified inline profiling)
        print(f"\n📊 Building EDA priors from dataset...")
        eda_priors = EdaPriors(
            missing_rate=float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            imbalance_ratio=float(df[target_column].value_counts(normalize=True).min()) if task_type == "classification" and target_column else 1.0,
            high_cardinality_cols=[col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() > 50],
            outlier_prevalence=0.0,  # Simplified - would need IQR calculation
            skewness_issues=sum(1 for col in df.select_dtypes(include=[np.number]).columns if abs(df[col].skew()) > 1),
            n_rows=df.shape[0],
            n_features=df.shape[1] - 1
        )
        print(f"   missing_rate: {eda_priors.missing_rate:.3f}")
        print(f"   imbalance_ratio: {eda_priors.imbalance_ratio:.3f}")
        print(f"   high_cardinality_cols: {len(eda_priors.high_cardinality_cols)}")
        print(f"   skewness_issues: {eda_priors.skewness_issues}")
        
        # Build planner config
        planner_config = {
            "round1_max_variants": args.round1_max_variants,
            "round2_max_variants": args.round2_max_variants,
            "proxy_prune_threshold": args.proxy_prune_threshold,
            "diversity_min_hamming_distance": args.diversity_min_hamming,
            "diversity_coverage_enabled": True,
            "round0_budget_per_variant_sec": 10,
            "round1_budget_per_variant_sec": 60,
            "round2_budget_per_variant_sec": args.time_budget_per_variant
        }
        
        # Load variant configs for scoring
        print(f"\n🔍 Scoring {len(variant_paths)} variants against EDA priors...")
        variant_configs = []
        scored_variants = []
        import yaml as yaml_loader
        
        for vpath in variant_paths:
            try:
                with open(vpath, 'r') as f:
                    vcfg = yaml_loader.safe_load(f)
                variant_configs.append(vcfg)
                
                # Score variant
                score, reasoning = score_variant_relevance(vcfg, eda_priors, task_type)
                stage3 = vcfg.get("stage3_preprocessing", {})
                stage4 = vcfg.get("stage4_feature_engineering", {})
                variant_id = vcfg.get("recipe_name") or vcfg.get("variant_metadata", {}).get("variant_id", Path(vpath).stem)
                
                scored_variants.append(VariantScore(
                    variant_id=variant_id,
                    variant_path=vpath,
                    relevance_score=score,
                    reasoning=reasoning,
                    preprocessing_hash=compute_preprocessing_hash(vcfg),
                    imputation=stage3.get("imputation", {}).get("method", "none"),
                    encoding=stage3.get("encoding", {}).get("categorical_method", "onehot"),
                    scaling=stage3.get("scaling", {}).get("method", "none"),
                    imbalance=stage3.get("imbalance_handling", {}).get("method", "none"),
                    feature_selection=stage4.get("feature_selection", {}).get("method", "none")
                ))
            except Exception as e:
                print(f"   ⚠️ Could not load/score variant {vpath}: {e}")
        
        # Diverse sample for Round 1
        print(f"\n🎯 Selecting top {args.round1_max_variants} diverse variants for Round 1...")
        round1_candidates = diverse_sample(scored_variants, args.round1_max_variants, args.diversity_min_hamming)
        
        # Build variant plan
        variant_plan = build_variant_plan(
            variant_configs=variant_configs,
            variant_paths=variant_paths,
            eda_priors=eda_priors,
            task_type=task_type,
            planner_config=planner_config
        )
        
        # Keep the planner shortlist advisory. The canonical funnel must screen
        # every catalog recipe for feasibility before applying its Round 1 cap.
        original_count = len(variant_paths)
        planner_proposed_paths = [v.variant_path for v in round1_candidates]
        
        print(f"\n📋 PLANNER RESULT:")
        print(f"   Original variants: {original_count}")
        print(f"   Proposed for Round 1: {len(planner_proposed_paths)}")
        print(f"   Top 5 by relevance score:")
        for v in sorted(round1_candidates, key=lambda x: x.relevance_score, reverse=True)[:5]:
            print(f"      - {v.variant_id}: score={v.relevance_score:.1f}")
        
        # Save variant_plan.json
        variant_plan_path = output_path / "variant_plan.json"
        with open(variant_plan_path, 'w') as f:
            from dataclasses import asdict as plan_asdict
            json.dump(plan_asdict(variant_plan) if hasattr(variant_plan, '__dataclass_fields__') else {
                "planner_version": "1.0",
                "eda_priors": plan_asdict(eda_priors),
                "shortlist": [{"variant_id": v.variant_id, "relevance_score": v.relevance_score, "reasoning": v.reasoning} for v in round1_candidates],
                "budget_allocation": variant_plan.budget_allocation if variant_plan else {}
            }, f, indent=2, default=str)
        print(f"   ✅ Saved variant_plan.json")
        print(f"{'='*80}\n")

    # Round 1 is a real execution gate.  Only its bounded, ranked survivors may
    # enter the expensive recipe×engine loop below.
    if not args.enable_round1_proxy:
        raise ValueError(
            "Round 1 proxy evaluation is required by the canonical recipe funnel"
        )
    # Profile every eligible catalog recipe against the training-only dataset,
    # then bound the expensive Round 1 proxy executions.  This preserves
    # data-aware ranking without a submission-side, data-blind truncation.
    catalog_eda = EdaPriors(
        missing_rate=float(
            df.isnull().sum().sum() / max(1, df.shape[0] * df.shape[1])
        ),
        imbalance_ratio=(
            float(df[target_column].value_counts(normalize=True).min())
            if task_type == "classification"
            and target_column
            and target_column in df
            else 1.0
        ),
        high_cardinality_cols=[
            column
            for column in df.select_dtypes(include=["object"]).columns
            if df[column].nunique() > 50
        ],
        outlier_prevalence=0.0,
        skewness_issues=sum(
            1
            for column in df.select_dtypes(include=[np.number]).columns
            if abs(df[column].skew()) > 1
        ),
        n_rows=df.shape[0],
        n_features=max(0, df.shape[1] - (1 if target_column in df else 0)),
    )
    catalog_profiles: list[dict[str, Any]] = []
    catalog_scores: list[VariantScore] = []
    for catalog_path in variant_paths:
        with open(catalog_path, "r", encoding="utf-8") as handle:
            catalog_recipe = yaml.safe_load(handle) or {}
        relevance_score, reasoning = score_variant_relevance(
            catalog_recipe,
            catalog_eda,
            task_type,
        )
        stage3 = catalog_recipe.get("stage3_preprocessing") or {}
        stage4 = catalog_recipe.get("stage4_feature_engineering") or {}
        variant_id = (
            catalog_recipe.get("recipe_name")
            or (catalog_recipe.get("variant_metadata") or {}).get("variant_id")
            or Path(catalog_path).stem
        )
        catalog_profiles.append(
            {
                "variant_id": variant_id,
                "variant_path": catalog_path,
                "relevance_score": float(relevance_score),
                "reasoning": reasoning,
                "semantic_hash": semantic_recipe_hash(
                    catalog_recipe,
                    task_type=task_type,
                ),
            }
        )
        catalog_scores.append(
            VariantScore(
                variant_id=variant_id,
                variant_path=catalog_path,
                relevance_score=relevance_score,
                reasoning=reasoning,
                preprocessing_hash=compute_preprocessing_hash(catalog_recipe),
                imputation=(stage3.get("imputation") or {}).get("method", "none"),
                encoding=(stage3.get("encoding") or {}).get(
                    "categorical_method", "onehot"
                ),
                scaling=(stage3.get("scaling") or {}).get("method", "none"),
                imbalance=(stage3.get("imbalance_handling") or {}).get(
                    "method", "none"
                ),
                feature_selection=(stage4.get("feature_selection") or {}).get(
                    "method", "none"
                ),
            )
        )
    profiled_count = len(catalog_scores)
    catalog_screening_reports: list[dict[str, Any]] = []
    round1_rejected: list[dict[str, Any]] = []
    for catalog_score in catalog_scores:
        variant_path = str(catalog_score.variant_path)
        require_phase_b_budget(phase_b_deadline, "round0_screen_start")
        validation_report = validate_variant_yaml(
            variant_path,
            task_type=task_type,
        )
        if not validation_report.get("valid"):
            reason = "; ".join(
                validation_report.get("errors")
                or ["variant validation failed"]
            )
            screening_report = {
                "variant_id": validation_report.get(
                    "variant_id", Path(variant_path).stem
                ),
                "variant_path": variant_path,
                "status": "quarantined",
                "reason": reason,
            }
            catalog_screening_reports.append(screening_report)
            round1_rejected.append(
                {
                    **screening_report,
                    "proxy_metric": 0.0,
                }
            )
            continue
        try:
            variant = load_variant(variant_path)
            validate_variant_for_task(variant, task_type)
            screening_report = {
                "variant_id": variant.variant_id,
                "variant_path": variant_path,
                "status": "pass",
                "reason": "",
            }
            if args.enable_round0_feasibility:
                remaining_budget = require_phase_b_budget(
                    phase_b_deadline,
                    "round0_feasibility_start",
                )
                (
                    transformed_train,
                    _,
                    y_train,
                    _,
                    _,
                ) = run_with_hard_timeout(
                    prepare_round1_proxy_partitions,
                    df,
                    variant,
                    target_column,
                    task_type,
                    5000,
                    int(config["random_seed"]),
                    timeout_seconds=min(
                        float(args.time_budget_per_variant),
                        remaining_budget,
                    ),
                )
                feasibility_frame = transformed_train.copy()
                if y_train is not None:
                    feasibility_frame[target_column] = np.asarray(y_train)
                _, feasibility = run_round0_feasibility(
                    df,
                    variant,
                    target_column,
                    df_transformed=feasibility_frame,
                )
                screening_report = {
                    **feasibility,
                    "variant_path": variant_path,
                }
            catalog_screening_reports.append(screening_report)
            if screening_report["status"] != "pass":
                round1_rejected.append(
                    {
                        "variant_id": variant.variant_id,
                        "variant_path": variant_path,
                        "status": "quarantined",
                        "proxy_metric": 0.0,
                        "reason": screening_report.get("reason")
                        or "Round 0 feasibility failed",
                    }
                )
        except HardDeadlineExceeded as exc:
            if time.time() >= phase_b_deadline:
                raise
            screening_report = {
                "variant_id": Path(variant_path).stem,
                "variant_path": variant_path,
                "status": "timeout",
                "reason": str(exc),
                "censored": True,
            }
            catalog_screening_reports.append(screening_report)
            round1_rejected.append(
                {
                    **screening_report,
                    "proxy_metric": 0.0,
                }
            )
        except Exception as exc:
            screening_report = {
                "variant_id": Path(variant_path).stem,
                "variant_path": variant_path,
                "status": "error",
                "reason": str(exc),
            }
            catalog_screening_reports.append(screening_report)
            round1_rejected.append(
                {
                    **screening_report,
                    "proxy_metric": 0.0,
                }
            )

    round1_selected = select_feasible_round1_candidates(
        catalog_scores,
        catalog_screening_reports,
        max_variants=args.round1_max_variants,
        diversity_min_hamming=args.diversity_min_hamming,
    )
    variant_paths = [item.variant_path for item in round1_selected]
    round1_funnel_reports: list[dict[str, Any]] = []
    round1_input_paths = list(variant_paths)
    for variant_path in round1_input_paths:
        require_phase_b_budget(phase_b_deadline, "round1_start")
        try:
            variant = load_variant(variant_path)
            validate_variant_for_task(variant, task_type)
            proxy_report = run_with_hard_timeout(
                run_round1_proxy,
                df,
                variant,
                target_column,
                task_type,
                5000,
                int(config["random_seed"]),
                timeout_seconds=min(
                    float(args.time_budget_per_variant),
                    require_phase_b_budget(
                        phase_b_deadline,
                        "round1_proxy_start",
                    ),
                ),
            )
            with open(variant_path, "r", encoding="utf-8") as handle:
                raw_recipe = yaml.safe_load(handle) or {}
            proxy_report.update(
                {
                    "variant_path": variant_path,
                    "semantic_hash": semantic_recipe_hash(
                        raw_recipe, task_type=task_type
                    ),
                }
            )
            round1_funnel_reports.append(proxy_report)
        except HardDeadlineExceeded:
            raise
        except Exception as exc:
            round1_rejected.append(
                {
                    "variant_id": Path(variant_path).stem,
                    "variant_path": variant_path,
                    "status": "error",
                    "proxy_metric": 0.0,
                    "reason": str(exc),
                }
            )

    variant_paths, ranked_rejections = rank_round2_proxy_survivors(
        round1_funnel_reports,
        task_type=task_type,
        configured_threshold=args.proxy_prune_threshold,
        max_variants=args.round2_max_variants,
    )
    round1_rejected.extend(ranked_rejections)
    funnel_evidence = {
        "schema_version": "2.0",
        "execution_id": execution_manifest.execution_id,
        "round1_input_count": len(round1_input_paths),
        "catalog_profiled_count": profiled_count,
        "catalog_feasible_count": sum(
            report.get("status") == "pass"
            for report in catalog_screening_reports
        ),
        "catalog_profiles": catalog_profiles,
        "catalog_screening_reports": catalog_screening_reports,
        "round1_max_variants": args.round1_max_variants,
        "round2_max_variants": args.round2_max_variants,
        "proxy_prune_threshold": args.proxy_prune_threshold,
        "round1_reports": round1_funnel_reports,
        "round2_variant_paths": variant_paths,
        "rejected": round1_rejected,
    }
    require_phase_b_budget(
        phase_b_deadline,
        "before_recipe_funnel_write",
    )
    atomic_write(
        output_path / "recipe_funnel.json",
        json.dumps(funnel_evidence, indent=2, sort_keys=True, default=str),
    )
    if not variant_paths:
        raise RuntimeError(
            "Recipe funnel produced no eligible Round 2 variants; "
            "see recipe_funnel.json"
        )
    print(
        f"🎯 Recipe funnel: {len(round1_input_paths)} Round 1 inputs → "
        f"{len(variant_paths)} Round 2 survivors"
    )

    # Checkpoint identity now reflects the bounded Round 2 candidate set.
    total_expected = len(variant_paths) * len(engines)
    checkpoint = CheckpointManager(output_path / "resume_state.json")
    progress = checkpoint.get_progress(total_expected=total_expected)
    print(
        f"\n📊 Checkpoint Progress: {progress['completed']}/"
        f"{progress['total']} completed"
    )
    if progress["completed"] > 0:
        print(
            "   ⏮️  Resuming from previous run "
            f"(last updated: {progress['last_updated']})"
        )
    print(
        f"   ⏱️  Time budget per candidate-engine: "
        f"{args.time_budget_per_variant}s\n"
    )
    
    # Disable autologging (do NOT change tracking URI — breaks MLflow hierarchy)
    try:
        mlflow.sklearn.autolog(disable=True)
    except Exception as e:
        logger.debug("mlflow.sklearn.autolog(disable=True) failed: %s", e)
    try:
        mlflow.autolog(disable=True)
    except Exception as e:
        logger.debug("mlflow.autolog(disable=True) failed: %s", e)
    # Preserve Azure ML's workspace tracking and registry endpoints exactly.
    # Rewriting the tracking URI or installing a local registry would sever the
    # nested candidate runs from the component run's immutable workspace lineage.
    
    # ======== AZURE ML PIPELINE COMPATIBILITY (CRITICAL) ========
    # Azure ML pipeline steps automatically create an MLflow run
    # Only create parent run if we're NOT already in an Azure ML pipeline context
    parent_run_created = False
    if mlflow.active_run() is None:
        # Standalone execution (local testing) - create parent run
        mlflow.start_run(run_name="phase_b_variant_runner")
        parent_run_created = True
        print(f"🔬 Started parent MLflow run: {mlflow.active_run().info.run_id}")
        print(f"   All variant×engine runs will nest under this parent\n")
    else:
        # Azure ML pipeline context - use existing step run as parent
        print(f"🔬 Using existing Azure ML pipeline step run as parent: {mlflow.active_run().info.run_id}")
        print(f"   All variant×engine runs will nest under this step\n")
    
    try:
        # Results collection
        all_results: List[VariantResult] = []
        
        # Champion-so-far tracking (avoid OOM from storing all models)
        primary_metric_name = get_primary_metric(task_type)
        best_score = float('-inf')  # Unified score (higher is better)
        best_key: Tuple[str, str] = (None, None)  # (variant_id, engine)
        best_model: Any = None
        best_preprocessor: Optional[FittedVariantPreprocessor] = None
        
        skipped_count = 0
        failed_count = 0
        
        # Phase B signal artifacts (optional)
        round0_feasibility_reports = [] if args.enable_round0_feasibility else None
        round1_proxy_reports = [] if args.enable_round1_proxy else None
        elimination_decisions = [] if args.enable_elimination_report else None
        variant_validation_reports: list[dict] = []
        variant_anomaly_reports: list[dict] = []
        variant_preprocessors: dict[str, FittedVariantPreprocessor] = {}
        
        print(f"🔍 Phase B Signals: Round0={'ON' if args.enable_round0_feasibility else 'OFF'}, Round1={'ON' if args.enable_round1_proxy else 'OFF'}, Elimination={'ON' if args.enable_elimination_report else 'OFF'}\n")
        
        # Process each variant
        for i, variant_path in enumerate(variant_paths, 1):
            print(f"\n{'='*80}")
            print(f"[{i}/{len(variant_paths)}] Processing variant: {Path(variant_path).name}")
            print(f"{'='*80}")
            
            try:
                validation_report = validate_variant_yaml(variant_path, task_type=task_type)
                variant_validation_reports.append(validation_report)
                if not validation_report.get("valid"):
                    error_text = "; ".join(validation_report.get("errors") or ["unknown validation error"])
                    print(f"   ❌ Variant YAML validation failed: {error_text}")
                    if args.enable_elimination_report:
                        elimination_decisions.append({
                            "variant_id": validation_report.get("variant_id", Path(variant_path).stem),
                            "engine": "all",
                            "stage": "variant_yaml_validation",
                            "reason": "variant_yaml_invalid",
                            "details": error_text,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        })
                    for engine in engines:
                        all_results.append(VariantResult(
                            variant_id=validation_report.get("variant_id", Path(variant_path).stem),
                            engine=engine,
                            algorithm="variant_yaml_invalid",
                            metrics={"primary_metric": 0.0},
                            runtime_sec=0.0,
                            timed_out=False,
                            failed=True,
                            failure_reason=error_text,
                        ))
                    failed_count += len(engines)
                    continue

                # Load and validate variant
                variant = load_variant(variant_path)
                validate_variant_for_task(variant, task_type)
                with open(variant_path, "r", encoding="utf-8") as recipe_handle:
                    variant_recipe_hash = semantic_recipe_hash(
                        yaml.safe_load(recipe_handle) or {},
                        task_type=task_type,
                    )
            
                print(f"   Variant ID: {variant.variant_id}")
                print(f"   Config: {variant.stage3_preprocessing.imputation.method}+{variant.stage3_preprocessing.encoding.categorical_method}+{variant.stage3_preprocessing.scaling.method}")

                preprocessing_started_at = time.time()
                try:
                    df_transformed, fitted_preprocessor = run_with_hard_timeout(
                        preprocess_training_data,
                        df,
                        variant,
                        target_column,
                        int(config["random_seed"]),
                        timeout_seconds=min(
                            CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS,
                            max(0.0, phase_b_deadline - time.time()),
                        ),
                    )
                except Exception as exc:
                    print(f"   ❌ Training-only preprocessing failed: {exc}")
                    failed_count += len(engines)
                    continue
                preprocessing_runtime_sec = (
                    time.time() - preprocessing_started_at
                )
                variant_preprocessors[variant.variant_id] = fitted_preprocessor

                # ==== INSERT POINT C: Round 0 Feasibility Check ====
                if preprocessing_cache.enabled:
                    print(
                        "   🔒 Preprocessing cache bypassed so the exact fitted "
                        "training transformer can be bundled"
                    )
                
                if args.enable_round0_feasibility:
                    print(f"   🔍 Round 0: Feasibility check (transform-only)...")
                    df_transformed, feas_report = run_round0_feasibility(
                        df,
                        variant,
                        target_column,
                        df_transformed=df_transformed,
                    )
                    round0_feasibility_reports.append(feas_report)
                    
                    if feas_report["status"] == "fail":
                        print(f"      ❌ FAILED: {feas_report['reason']} [{feas_report['transform_runtime_sec']:.2f}s]")
                        if args.enable_elimination_report:
                            elimination_decisions.append({
                                "variant_id": variant.variant_id,
                                "engine": "all",
                                "stage": "round0_feasibility",
                                "reason": "feasibility_fail",
                                "details": feas_report["reason"],
                                "timestamp": datetime.utcnow().isoformat() + "Z"
                            })
                        failed_count += len(engines)  # Count as failure for all engines
                        continue  # Skip engine loop for this variant
                    elif feas_report["status"] == "error":
                        print(f"      ⚠️  ERROR: {feas_report['reason']} [{feas_report['transform_runtime_sec']:.2f}s]")
                        if args.enable_elimination_report:
                            elimination_decisions.append({
                                "variant_id": variant.variant_id,
                                "engine": "all",
                                "stage": "round0_feasibility",
                                "reason": "feasibility_error",
                                "details": feas_report["reason"],
                                "timestamp": datetime.utcnow().isoformat() + "Z"
                            })
                        failed_count += len(engines)
                        continue
                    else:
                        print(f"      ✅ PASSED: {feas_report['n_features_after']} features [{feas_report['transform_runtime_sec']:.2f}s]")

                if df_transformed is not None and not any(
                    report.get("variant_id") == variant.variant_id
                    for report in variant_anomaly_reports
                ):
                    variant_anomaly_reports.append(build_variant_anomaly_report(variant, df_transformed, target_column))
                
                # ==== INSERT POINT D: Round 1 Proxy Leaderboard ====
                if args.enable_round1_proxy:
                    print(f"   📊 Round 1: Proxy leaderboard (SGD model on sampled data)...")
                    # Reuse transformed data from Round 0 if available
                    if df_transformed is None:
                        df_transformed = apply_variant_preprocessing(df, variant, target_column, apply_smote=False)
                    if not any(report.get("variant_id") == variant.variant_id for report in variant_anomaly_reports):
                        variant_anomaly_reports.append(build_variant_anomaly_report(variant, df_transformed, target_column))
                    
                    proxy_report = run_with_hard_timeout(
                        run_round1_proxy,
                        df,
                        variant,
                        target_column,
                        task_type,
                        5000,
                        int(config["random_seed"]),
                        timeout_seconds=min(
                            float(args.time_budget_per_variant),
                            require_phase_b_budget(
                                phase_b_deadline,
                                "signal_proxy_start",
                            ),
                        ),
                    )
                    round1_proxy_reports.append(proxy_report)
                    
                    if proxy_report["status"] == "fail":
                        print(f"      ⚠️  FAIL: {proxy_report['reason']} [proxy_metric={proxy_report['proxy_metric']:.4f}, {proxy_report['proxy_runtime_sec']:.2f}s]")
                    elif proxy_report["status"] == "error":
                        print(f"      ⚠️  ERROR: {proxy_report['reason']} [{proxy_report['proxy_runtime_sec']:.2f}s]")
                    else:
                        print(f"      ✅ proxy_metric={proxy_report['proxy_metric']:.4f} [{proxy_report['proxy_runtime_sec']:.2f}s]")
                
                # Run with each engine
                for engine in engines:
                    print(f"\n   🔧 Engine: {engine}")
                    search_candidate = next(
                        (
                            record
                            for record in canonical_candidate_records
                            if record.recipe_hash == variant_recipe_hash
                            and record.engine == engine
                        ),
                        None,
                    )
                    if search_candidate is None:
                        raise RuntimeError(
                            "No canonical search CandidateRecord for "
                            f"{variant.variant_id}::{engine}"
                        )
                
                    # ======== HARDENING FEATURE 1: SKIP IF ALREADY COMPLETED ========
                    if checkpoint.is_completed(variant.variant_id, engine):
                        print(f"   ⏭️  SKIPPED (already completed in previous run)")
                        skipped_count += 1
                        continue
                
                    # Single-point checkpointing: try/except/finally pattern
                    result = None
                    model = None
                    attempt_start = time.time()
                    remaining_phase_budget = phase_b_deadline - attempt_start
                    if remaining_phase_budget <= 0:
                        result = VariantResult(
                            variant_id=variant.variant_id,
                            engine=engine,
                            algorithm="phase_b_budget_exhausted",
                            metrics={"primary_metric": 0.0},
                            runtime_sec=0.0,
                            timed_out=True,
                            failed=True,
                            failure_reason=(
                                "Phase B wall-clock budget exhausted before "
                                "candidate-engine start"
                            ),
                        )
                        all_results.append(result)
                        checkpoint.mark_completed(variant.variant_id, engine)
                        failed_count += 1
                        if args.enable_elimination_report:
                            elimination_decisions.append(
                                {
                                    "variant_id": variant.variant_id,
                                    "engine": engine,
                                    "stage": "training",
                                    "reason": "phase_b_budget_exhausted",
                                    "details": result.failure_reason,
                                    "timestamp": datetime.utcnow().isoformat()
                                    + "Z",
                                }
                            )
                        continue
                    candidate_budget = min(
                        max(
                            0.0,
                            float(args.time_budget_per_variant)
                            - preprocessing_runtime_sec,
                        ),
                        remaining_phase_budget,
                    )
                    if candidate_budget <= 0:
                        result = VariantResult(
                            variant_id=variant.variant_id,
                            engine=engine,
                            algorithm="preprocessing_budget_exhausted",
                            metrics={"primary_metric": 0.0},
                            runtime_sec=preprocessing_runtime_sec,
                            timed_out=True,
                            failed=True,
                            failure_reason=(
                                "Training-only preprocessing consumed the "
                                "candidate-engine hard deadline"
                            ),
                        )
                        all_results.append(result)
                        checkpoint.mark_completed(variant.variant_id, engine)
                        failed_count += 1
                        continue
                    attempt_deadline = min(
                        phase_b_deadline,
                        attempt_start + candidate_budget,
                    )

                    try:
                        # ======== HARDENING FEATURE 2: HARD BUDGET GUARD ========
                        # FIX 2B: Pass cached df_transformed if available (from Round0/Round1)
                        result, model = run_variant_with_nested_mlflow(
                            variant=variant,
                            df=df,
                            engine=engine,
                            target_column=target_column,
                            task_type=task_type,
                            time_budget=candidate_budget,
                            attempt_deadline=attempt_deadline,
                            execution_id=execution_manifest.execution_id,
                            search_candidate=search_candidate,
                            random_seed=int(config["random_seed"]),
                            cv_folds=int(config["split"]["cv_folds"]),
                            mlflow_parent_run_id=(
                                mlflow.active_run().info.run_id
                                if mlflow.active_run()
                                else None
                            ),
                            df_preprocessed=df_transformed  # Reuse cached transform if available
                        )
                    except Exception as e:
                        # Outer safety net for catastrophic failures
                        print(f"   💥 CATASTROPHIC FAILURE: {e}")
                        result = VariantResult(
                            variant_id=variant.variant_id,
                            engine=engine,
                            algorithm="outer_exception",
                            metrics={"primary_metric": 0.0},
                            runtime_sec=time.time() - attempt_start,
                            timed_out=False,
                            failed=True,
                            failure_reason=f"Outer exception: {str(e)}"
                        )
                        model = None
                    finally:
                        # ======== SINGLE-POINT COMPLETION MARKING ========
                        # Guarantee result exists (fallback)
                        if result is None:
                            result = VariantResult(
                                variant_id=variant.variant_id,
                                engine=engine,
                                algorithm="result_not_set",
                                metrics={"primary_metric": 0.0},
                                runtime_sec=time.time() - attempt_start,
                                timed_out=False,
                                failed=True,
                                failure_reason="Result object not created"
                            )
                    
                        all_results.append(result)
                        checkpoint.mark_completed(variant.variant_id, engine)
                    
                        # Track champion-so-far (avoid OOM)
                        # MUST match the leaderboard's usable-results filter so
                        # best_key always agrees with champion_result. Weak but
                        # finite evidence remains eligible for S10 quality policy.
                        if model is not None and is_usable_phaseb_result(result):
                            current_score = get_result_score(result, primary_metric_name)
                            if current_score > best_score:
                                best_score = current_score
                                best_key = (variant.variant_id, engine)
                                best_model = model
                                best_preprocessor = variant_preprocessors.get(
                                    variant.variant_id
                                )
                                original_val = safe_float(result.metrics.get("primary_metric"))
                                print(f"   🏆 NEW CHAMPION: {variant.variant_id}::{engine} = {original_val:.4f} (score={current_score:.4f})")
                    
                        # ======== ENHANCED LOGGING ========
                        if result.failed:
                            print(f"   ❌ FAILED: {result.algorithm} - {result.failure_reason} [{result.runtime_sec:.1f}s]")
                            failed_count += 1
                            
                            # ==== INSERT POINT E: Track elimination decision ====
                            if args.enable_elimination_report:
                                elimination_decisions.append({
                                    "variant_id": variant.variant_id,
                                    "engine": engine,
                                    "stage": "training",
                                    "reason": "training_failed",
                                    "details": result.failure_reason,
                                    "timestamp": datetime.utcnow().isoformat() + "Z"
                                })
                            
                        elif result.timed_out:
                            print(f"   ⏱️  TIMED_OUT: metric={result.metrics.get('primary_metric', 0.0):.4f} [{result.runtime_sec:.1f}s]")
                        else:
                            print(f"   ✅ SUCCESS: {result.algorithm} | metric={result.metrics.get('primary_metric', 0.0):.4f} [{result.runtime_sec:.1f}s]")
        
            except HardDeadlineExceeded:
                raise
            except Exception as e:
                print(f"   ❌ Variant processing failed: {e}")
                failed_count += 1
                continue
        require_phase_b_budget(
            phase_b_deadline,
            "post_training_evidence",
        )
        valid_results = [
            result
            for result in all_results
            if is_usable_phaseb_result(result)
        ]
        usable_results = valid_results
        
        # Track eliminated timed-out/zero-metric variants
        if args.enable_elimination_report and len(usable_results) < len(valid_results):
            for r in valid_results:
                if r not in usable_results:
                    elimination_decisions.append({
                        "variant_id": r.variant_id,
                        "engine": r.engine,
                        "stage": "post_training",
                        "reason": "timed_out_or_zero_metric",
                        "details": f"timed_out={r.timed_out}, metric={r.metrics.get('primary_metric', 0.0):.4f}",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })

        if usable_results:
            print(f"\n✅ Found {len(valid_results)} usable results (filtered out {len([r for r in all_results if not r.failed]) - len(valid_results)} timed-out/non-finite runs)")

        if (
            count_distinct_phaseb_candidates(valid_results)
            < minimum_comparable_candidates
        ):
            print("\n❌ INSUFFICIENT COMPARABLE RESULTS - Cannot select a champion")
            print(
                "   Phase B requires at least "
                f"{minimum_comparable_candidates} distinct completed candidates."
            )
            results_path = output_path / "all_results.json"
            atomic_write(
                results_path,
                json.dumps(
                    [asdict(r) for r in all_results],
                    indent=2,
                    default=str,
                ),
            )
            write_variant_validation_report(
                output_path,
                variant_validation_reports,
            )
            require_valid_phaseb_results(
                valid_results,
                minimum_candidates=minimum_comparable_candidates,
            )
        
        # ======== HARDENING FEATURE 4: NORMALIZED RESULTS ========
        # Generate leaderboard using VariantResult schema with unified scoring
        # Use task-type-aware metric columns for uniform output structure
        metric_columns = get_metric_columns_for_task(task_type)
        leaderboard_data = []
        for r in valid_results:
            score = get_result_score(r, primary_metric_name)
            row = {
                "variant_id": r.variant_id,
                "engine": r.engine,
                "algorithm": r.algorithm,
                "primary_metric": r.metrics.get("primary_metric", 0.0),
            }
            # Add task-type-specific metric columns
            for col in metric_columns:
                row[col] = r.metrics.get(col, None)
            row.update({
                "score": score,  # Unified comparable score
                "runtime_sec": r.runtime_sec,
                "n_features": r.n_features,
                "leakage_risk": r.leakage_risk,
                "timed_out": r.timed_out,
                "failed": r.failed
            })
            leaderboard_data.append(row)
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Sort by unified score (descending = best first)
        leaderboard_df = leaderboard_df.sort_values("score", ascending=False)
        
        # Drop score column before saving (internal use only)
        leaderboard_df_export = leaderboard_df.drop(columns=["score"])
        leaderboard_path = output_path / "leaderboard.csv"
        
        # The hard Phase B budget includes every output and lineage write.
        final_deadline = phase_b_deadline
        require_phase_b_budget(final_deadline, "before_leaderboard_write")
        # ======== HARDENING FEATURE 3: ATOMIC LEADERBOARD WRITE ========
        leaderboard_temp = output_path / "leaderboard.csv.tmp"
        leaderboard_df_export.to_csv(leaderboard_temp, index=False)
        os.replace(str(leaderboard_temp), str(leaderboard_path))
        print(f"\n📊 Leaderboard saved (atomic): {leaderboard_path}")
        
        # Select champion using unified scoring (matches leaderboard sort)
        # Get variant_id + engine from top leaderboard row
        top_row = leaderboard_df.iloc[0]
        champion_result = None
        for r in valid_results:
            if r.variant_id == top_row["variant_id"] and r.engine == top_row["engine"]:
                champion_result = r
                break
        
        if champion_result is None:
            raise RuntimeError(
                "Leaderboard champion has no exact retained candidate result"
            )
        
        # Find corresponding variant config
        champion_variant_path = None
        for vp in variant_paths:
            v = load_variant(vp)
            if v.variant_id == champion_result.variant_id:
                champion_variant_path = vp
                break
        
        if champion_variant_path is None:
            raise RuntimeError(
                "Leaderboard champion recipe is absent from the bounded catalog"
            )
        champion_variant = load_variant(champion_variant_path)
        
            # ======== HARDENING FEATURE 3: STABLE CHAMPION MANIFEST CONTRACT ========
        # Store original (non-negated) metric value in manifest
        primary_metric_name_lower = get_primary_metric(task_type).lower().replace(" ", "_")
        original_metric_value = safe_float(
            champion_result.metrics.get("primary_metric") or 
            champion_result.metrics.get(primary_metric_name_lower)
        )
        champion_key = (champion_result.variant_id, champion_result.engine)
        if champion_key != best_key:
            raise RuntimeError(
                "Leaderboard champion does not match the exact retained model: "
                f"leaderboard={champion_key}, retained={best_key}"
            )
        if best_model is None or best_preprocessor is None:
            raise RuntimeError(
                "Exact champion model and fitted preprocessing graph must both "
                "be retained"
            )
        with open(
            champion_variant_path, "r", encoding="utf-8"
        ) as champion_recipe_handle:
            champion_recipe_hash = semantic_recipe_hash(
                yaml.safe_load(champion_recipe_handle) or {},
                task_type=task_type,
            )
        champion_search_candidate = next(
            (
                record
                for record in canonical_candidate_records
                if record.recipe_hash == champion_recipe_hash
                and record.engine == champion_result.engine
            ),
            None,
        )
        if champion_search_candidate is None:
            raise RuntimeError(
                "Champion has no canonical catalog-derived CandidateRecord"
            )
        if not champion_result.candidate_id:
            raise RuntimeError("Champion has no realized candidate identity")
        parent_run_id = (
            mlflow.active_run().info.run_id if mlflow.active_run() else None
        )
        raw_features = (
            df.drop(columns=[target_column])
            if target_column and target_column in df
            else df.copy()
        )
        labels = (
            tuple(
                sorted(
                    df[target_column].dropna().unique().tolist(),
                    key=lambda value: str(value),
                )
            )
            if task_type == "classification"
            and target_column
            and target_column in df
            else ()
        )
        model_bundle = ModelBundle(
            estimator=best_model,
            preprocessing=best_preprocessor,
            task_type=task_type,
            candidate_id=champion_result.candidate_id,
            input_schema=capture_input_schema(raw_features),
            recipe=champion_variant.to_dict(),
            selection_metrics=champion_result.metrics,
            final_test_metrics={},
            environment={
                "environment_hash": require_training_environment_hash(
                    execution_manifest
                ),
                "code_sha": execution_manifest.code_sha,
                "component": "s06_phaseb_variant_runner",
            },
            lineage={
                "execution_id": execution_manifest.execution_id,
                "split_id": split_manifest.split_id,
                "parent_run_id": parent_run_id,
                "candidate_run_id": champion_result.mlflow_run_id,
                "recipe_hash": champion_search_candidate.recipe_hash,
                "parent_search_candidate_id": (
                    champion_search_candidate.candidate_id
                ),
                "realized_candidate_id": champion_result.candidate_id,
            },
            dependencies=(
                "mlflow",
                "pandas",
                "scikit-learn",
                champion_result.engine,
            ),
            labels=labels,
            input_example=(
                raw_features.head(1).to_dict(orient="records")
                if not raw_features.empty
                else None
            ),
        )
        bundle_manifest = save_model_bundle(model_bundle, output_path)
        quality_decision = QualityDecision(
            decision="block",
            candidate_id=champion_result.candidate_id,
            evaluated_bundle_hash=model_bundle.bundle_id,
            metric_name=primary_metric_name_lower,
            metric_value=original_metric_value,
            threshold=None,
            registration_allowed=False,
            promotion_aliases=(),
            registration_tags={
                "quality_stage": "selection_only",
                "locked_test_evaluated": "false",
            },
            reasons=(
                "S10 owns the sole locked final-test evaluation",
            ),
        )
        atomic_write(
            output_path / "quality_decision.json",
            quality_decision.to_json(indent=2),
        )
        
        champion_manifest = ChampionManifest(
            variant_id=champion_result.variant_id,
            variant_path=champion_variant_path if champion_variant_path else "unknown",
            engine=champion_result.engine,
            algorithm=champion_result.algorithm,
            primary_metric_name=primary_metric_name_lower,
            primary_metric_value=original_metric_value,
            metrics=champion_result.metrics,
            preprocessing_config={
                "imputation": champion_variant.stage3_preprocessing.imputation.method if champion_variant else "unknown",
                "encoding": champion_variant.stage3_preprocessing.encoding.categorical_method if champion_variant else "unknown",
                "scaling": champion_variant.stage3_preprocessing.scaling.method if champion_variant else "unknown",
                "imbalance": champion_variant.stage3_preprocessing.imbalance_handling.method if (champion_variant and champion_variant.stage3_preprocessing.imbalance_handling) else "none",
                "feature_selection": champion_variant.stage4_feature_engineering.feature_selection.method if champion_variant else "unknown"
            },
            feature_engineering_config={
                "method": champion_variant.stage4_feature_engineering.feature_selection.method if champion_variant else "unknown",
                "n_features_selected": champion_result.n_features
            },
            data_fingerprint=data_fingerprint,
            code_version=code_version,
            timestamp=datetime.utcnow().isoformat() + "Z",
            leakage_risk=champion_result.leakage_risk,
            task_type=task_type,
            safety_net_review_required=False,
            review_status="locked_test_pending",
            registration_eligible=False,
            review_reason="S10 locked final-test evaluation is pending.",
            execution_id=execution_manifest.execution_id,
            candidate_id=champion_result.candidate_id,
            mlflow_parent_run_id=parent_run_id,
            mlflow_child_run_id=champion_result.mlflow_run_id,
            recipe=champion_variant.to_dict(),
            model_bundle_id=model_bundle.bundle_id,
        )
        
            # ======== HARDENING FEATURE 5: ATOMIC WRITES ========
        # Parent MLflow run already active (started at beginning of main())
        # All artifacts will be logged to this parent run
        
        manifest_path = output_path / "champion_manifest.json"
        require_phase_b_budget(final_deadline, "before_manifest_write")
        # Save champion manifest with stable schema
        atomic_write(manifest_path, json.dumps(asdict(champion_manifest), indent=2))
        print(f"🏆 Champion manifest saved: {manifest_path}")
        print(f"   Schema: ChampionManifest v1.1 (15 fields with primary_metric_name/value)")

        # Log to MLflow
        try:
            mlflow.log_artifact(str(manifest_path), "phase_b_outputs")
            print(f"   ☁️  Logged to MLflow: phase_b_outputs/champion_manifest.json")
        except Exception as e:
            print(f"   ⚠️  Could not log manifest to MLflow: {e}")

        # Save all results with normalized schema (atomic write)
        # CRITICAL: Always create this file (required by component YAML)
        require_phase_b_budget(final_deadline, "before_results_write")
        results_path = output_path / "all_results.json"
        atomic_write(results_path, json.dumps([asdict(r) for r in all_results], indent=2, default=str))
        print(f"📄 All results saved: {results_path}")
        print(f"   Schema: VariantResult (10 fields, normalized across engines)")
        write_variant_validation_report(output_path, variant_validation_reports)

        # Log to MLflow
        try:
            mlflow.log_artifact(str(results_path), "phase_b_outputs")
            print(f"   ☁️  Logged to MLflow: phase_b_outputs/all_results.json")
        except Exception as e:
            print(f"   ⚠️  Could not log all_results to MLflow: {e}")

        # ── Candidate Ledger ──────────────────────────────────────────────
        try:
            _ledger_rows = []
            for _idx, _r in enumerate(all_results):
                _st = "failed" if _r.failed else ("timed_out" if _r.timed_out else "ok")
                _norm = normalize_metrics(task_type, _r.metrics)
                _vid = _r.variant_id
                # Compute rank: 1-based by primary metric descending among valid
                _rank = None
                if not _r.failed and not _r.timed_out:
                    _valid_scores = [
                        (i, get_result_score(v, primary_metric_name))
                        for i, v in enumerate(all_results)
                        if not v.failed and not v.timed_out
                    ]
                    _valid_scores.sort(key=lambda x: x[1], reverse=True)
                    for _rk, (_vi, _) in enumerate(_valid_scores, 1):
                        if _vi == _idx:
                            _rank = _rk
                            break
                _is_champ = (
                    champion_result is not None
                    and _r.variant_id == champion_result.variant_id
                    and _r.engine == champion_result.engine
                )
                _row = make_row(
                    stage="phase_b", step_name="s06", engine=_r.engine,
                    candidate_id=f"{_vid}__{_r.engine}__{_r.algorithm}",
                    task_type=task_type,
                    dataset_id=data_fingerprint.get("hash", "")[:12] if data_fingerprint else "",
                    status=_st,
                    failure_reason=_r.failure_reason or "",
                    compute_time_sec=round(_r.runtime_sec, 2),
                    source_path="src/steps/s06_phaseb_variant_runner.py",
                    recipe_name=_vid,
                    candidate_rank=_rank,
                    is_stage_best=_is_champ,
                    **_norm,
                )
                _ledger_rows.append(_row)
            write_stage_table(
                _ledger_rows,
                csv_path=str(output_path / "s06_candidates.csv"),
                parquet_path=str(output_path / "s06_candidates.parquet"),
            )
            print(f"📒 Candidate ledger: {len(_ledger_rows)} rows → s06_candidates.csv")
        except Exception as _ledger_err:
            print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")
        
        require_phase_b_budget(final_deadline, "before_signal_artifacts")
        # ======== PHASE B SIGNAL ARTIFACTS ========
        print(f"\n{'='*80}")
        print(f"PHASE B SIGNAL ARTIFACTS")
        print(f"{'='*80}")
        
        # Artifact 1: Round 0 Feasibility Report
        if args.enable_round0_feasibility and round0_feasibility_reports:
            feas_df = pd.DataFrame(round0_feasibility_reports)
            feas_path = output_path / "round0_feasibility_report.csv"
            feas_df.to_csv(feas_path, index=False)
            print(f"🔍 Round 0 Feasibility Report: {feas_path}")
            print(f"   Variants: {len(feas_df)}")
            print(f"   Pass: {(feas_df['status'] == 'pass').sum()}, Fail: {(feas_df['status'] == 'fail').sum()}, Error: {(feas_df['status'] == 'error').sum()}")
            
            try:
                mlflow.log_artifact(str(feas_path), "phase_b_outputs")
                print(f"   ☁️  Logged to MLflow: phase_b_outputs/round0_feasibility_report.csv")
            except Exception as e:
                print(f"   ⚠️  Could not log to MLflow: {e}")
        
        # Artifact 2: Round 1 Proxy Leaderboard
        if args.enable_round1_proxy and round1_proxy_reports:
            proxy_df = pd.DataFrame(round1_proxy_reports)
            # Sort by proxy_metric descending (best first)
            proxy_df = proxy_df.sort_values("proxy_metric", ascending=False)
            proxy_path = output_path / "round1_proxy_leaderboard.csv"
            proxy_df.to_csv(proxy_path, index=False)
            print(f"📊 Round 1 Proxy Leaderboard: {proxy_path}")
            print(f"   Variants: {len(proxy_df)}")
            if len(proxy_df) > 0:
                print(f"   Top variant: {proxy_df.iloc[0]['variant_id']} (proxy_metric={proxy_df.iloc[0]['proxy_metric']:.4f})")
            
            try:
                mlflow.log_artifact(str(proxy_path), "phase_b_outputs")
                print(f"   ☁️  Logged to MLflow: phase_b_outputs/round1_proxy_leaderboard.csv")
            except Exception as e:
                print(f"   ⚠️  Could not log to MLflow: {e}")
        
        # Artifact 3: Elimination Report
        if args.enable_elimination_report and elimination_decisions:
            elim_report = {
                "summary": {
                    "total_attempts": len(all_results),
                    "total_eliminated": len(elimination_decisions),
                    "by_stage": {},
                    "by_reason": {}
                },
                "eliminated_variants": elimination_decisions
            }
            
            # FIX 5: Build per-variant summary map for audit trail
            variant_summary = {}
            for decision in elimination_decisions:
                vid = decision["variant_id"]
                stage = decision.get("stage", "unknown")
                reason = decision.get("reason", "unknown")
                
                # Aggregate by stage and reason
                elim_report["summary"]["by_stage"][stage] = elim_report["summary"]["by_stage"].get(stage, 0) + 1
                elim_report["summary"]["by_reason"][reason] = elim_report["summary"]["by_reason"].get(reason, 0) + 1
                
                # Build per-variant audit trail
                if vid not in variant_summary:
                    variant_summary[vid] = {
                        "final_status": "eliminated",
                        "stages": [],
                        "reasons": [],
                        "engines_affected": []
                    }
                variant_summary[vid]["stages"].append(stage)
                variant_summary[vid]["reasons"].append(reason)
                if decision.get("engine") and decision["engine"] not in variant_summary[vid]["engines_affected"]:
                    variant_summary[vid]["engines_affected"].append(decision["engine"])
            
            # Add variants that passed to summary (for completeness)
            for result in valid_results:
                if result.variant_id not in variant_summary:
                    variant_summary[result.variant_id] = {
                        "final_status": "passed",
                        "stages": ["training_complete"],
                        "reasons": ["success"],
                        "engines_affected": [result.engine]
                    }
            
            elim_report["variant_summary"] = variant_summary
            
            elim_path = output_path / "elimination_report.json"
            with open(elim_path, 'w') as f:
                json.dump(elim_report, f, indent=2)
            print(f"📋 Elimination Report: {elim_path}")
            print(f"   Total eliminated: {elim_report['summary']['total_eliminated']}/{elim_report['summary']['total_attempts']}")
            print(f"   By stage: {elim_report['summary']['by_stage']}")
            print(f"   By reason: {elim_report['summary']['by_reason']}")
            print(f"   Variant summary entries: {len(variant_summary)}")
            
            try:
                mlflow.log_artifact(str(elim_path), "phase_b_outputs")
                print(f"   ☁️  Logged to MLflow: phase_b_outputs/elimination_report.json")
            except Exception as e:
                print(f"   ⚠️  Could not log to MLflow: {e}")
            
            # FIX 5B: Log signal metrics to parent MLflow run for UI filtering
            try:
                round0_pass = len([r for r in round0_feasibility_reports if r["status"] == "pass"]) if round0_feasibility_reports else 0
                round0_fail = len([r for r in round0_feasibility_reports if r["status"] == "fail"]) if round0_feasibility_reports else 0
                round1_pass = len([r for r in round1_proxy_reports if r["status"] == "pass"]) if round1_proxy_reports else 0
                round1_warning = len([r for r in round1_proxy_reports if r["status"] == "warning"]) if round1_proxy_reports else 0
                
                mlflow.log_metrics({
                    "phase_b_round0_pass_count": round0_pass,
                    "phase_b_round0_fail_count": round0_fail,
                    "phase_b_round1_pass_count": round1_pass,
                    "phase_b_round1_warning_count": round1_warning,
                    "phase_b_eliminated_total": elim_report["summary"]["total_eliminated"],
                    "phase_b_variants_passed": len([v for v in variant_summary.values() if v["final_status"] == "passed"])
                })
                print(f"   ☁️  Logged signal metrics to MLflow parent run")
            except Exception as e:
                print(f"   ⚠️  Could not log signal metrics to MLflow: {e}")

        if variant_anomaly_reports:
            anomaly_path = output_path / "variant_anomaly_report.json"
            anomaly_csv_path = output_path / "variant_anomaly_report.csv"
            atomic_write(anomaly_path, json.dumps(variant_anomaly_reports, indent=2, default=str))
            pd.DataFrame(variant_anomaly_reports).to_csv(anomaly_csv_path, index=False)
            print(f"🧪 Variant anomaly report: {anomaly_path} / {anomaly_csv_path}")
            try:
                mlflow.log_artifact(str(anomaly_path), "phase_b_outputs")
                mlflow.log_artifact(str(anomaly_csv_path), "phase_b_outputs")
            except Exception as e:
                print(f"   ⚠️  Could not log variant anomaly report to MLflow: {e}")
        
        # S06 ends at selection and immutable raw-input bundle emission.  S10
        # alone mounts and evaluates the locked final-test partition.
        require_phase_b_budget(final_deadline, "before_final_manifest_update")
        validated_execution_payload["realized_candidate_records"] = [
            result.candidate_record
            for result in all_results
            if result.candidate_record is not None
        ]
        atomic_write(
            output_path / "execution_manifest.json",
            json.dumps(
                validated_execution_payload,
                indent=2,
                sort_keys=True,
                default=str,
            ),
        )
        for artifact in (
            leaderboard_path,
            results_path,
            manifest_path,
            output_path / "model_bundle.pkl",
            output_path / "model_bundle_manifest.json",
            output_path / "quality_decision.json",
        ):
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise RuntimeError(f"Required Phase B artifact missing: {artifact}")
        try:
            for artifact in (
                output_path / "model_bundle.pkl",
                output_path / "model_bundle_manifest.json",
                output_path / "quality_decision.json",
            ):
                mlflow.log_artifact(str(artifact), "phase_b_outputs")
        except Exception as exc:
            logger.warning("Phase B artifact logging failed: %s", exc)
        publication_map = {
            args.leaderboard_out: leaderboard_path,
            args.all_results_out: results_path,
            args.champion_manifest_out: manifest_path,
            args.execution_manifest_out: (
                output_path / "execution_manifest.json"
            ),
            args.split_manifest_out: output_path / "split_manifest.json",
            args.quality_decision_out: (
                output_path / "quality_decision.json"
            ),
        }
        for destination, source in publication_map.items():
            if destination:
                require_phase_b_budget(
                    final_deadline,
                    f"before_publish_{source.name}",
                )
                publish_uri_file(source, destination)
        print(
            "✅ Phase B completed at the selection boundary; locked final-test "
            "evaluation remains pending in S10."
        )
        return
    finally:
        # End parent run only if we created it (not in Azure ML pipeline context)
        # Azure ML pipeline will close the step run automatically
        if parent_run_created and mlflow.active_run():
            mlflow.end_run()
            print(f"🏁 Ended parent MLflow run\n")
        elif mlflow.active_run():
            print(f"🏁 Azure ML pipeline will close step run automatically\n")


if __name__ == "__main__":
    raise SystemExit(run_phase_b_cli())
