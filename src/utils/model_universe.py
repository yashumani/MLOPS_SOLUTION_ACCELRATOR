"""
Model Universe — Canonical model lists for PyCaret and FLAML.

Single source of truth for which models each engine should attempt per
task_type.  If a model is unavailable in the runtime environment it is
recorded as "skipped" in the coverage report (outputs/model_coverage.json).

Usage::

    from utils.model_universe import get_model_list, write_model_coverage

    include = get_model_list("classification", "pycaret")
    best = compare_models(sort="AUC", include=include)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Canonical model lists
# ──────────────────────────────────────────────────────────────────────────────

MODEL_UNIVERSE: Dict[str, List[str]] = {
    # ── Classification ────────────────────────────────────────────────────────
    "classification_pycaret": [
        "lr",        # Logistic Regression
        "knn",       # K-Nearest Neighbours
        "nb",        # Naive Bayes
        "dt",        # Decision Tree
        "ridge",     # Ridge Classifier
        "rf",        # Random Forest
        "qda",       # Quadratic Discriminant Analysis
        "ada",       # AdaBoost
        "gbc",       # Gradient Boosting
        "lda",       # Linear Discriminant Analysis
        "et",        # Extra Trees
        "xgboost",   # XGBoost
        "lightgbm",  # LightGBM
        "catboost",  # CatBoost
        # REMOVED: svm, rbfsvm, mlp — O(n²+), catastrophic on large datasets
    ],
    "classification_flaml": [
        "lgbm",
        "xgboost",
        "xgb_limitdepth",
        "catboost",
        "rf",
        "extra_tree",
        "lrl1",
        "lrl2",
        "kneighbor",
    ],

    # ── Regression ────────────────────────────────────────────────────────────
    "regression_pycaret": [
        "lr",        # Linear Regression
        "lasso",     # Lasso Regression
        "ridge",     # Ridge Regression
        "en",        # Elastic Net
        "lar",       # Least Angle Regression
        "llar",      # Lasso Least Angle Regression
        "omp",       # Orthogonal Matching Pursuit
        "br",        # Bayesian Ridge
        "ard",       # Automatic Relevance Determination
        "par",       # Passive Aggressive
        "ransac",    # RANSAC
        "tr",        # TheilSen
        "huber",     # Huber
        "kr",        # Kernel Ridge
        "knn",       # K-Nearest Neighbours
        "dt",        # Decision Tree
        "rf",        # Random Forest
        "et",        # Extra Trees
        "ada",       # AdaBoost
        "gbr",       # Gradient Boosting
        "xgboost",   # XGBoost
        "lightgbm",  # LightGBM
        "catboost",  # CatBoost
        # REMOVED: svm, mlp — O(n²+), catastrophic on large datasets
    ],
    "regression_flaml": [
        "lgbm",
        "xgboost",
        "xgb_limitdepth",
        "catboost",
        "rf",
        "extra_tree",
        "kneighbor",
    ],

    # ── Clustering ────────────────────────────────────────────────────────────
    "clustering_pycaret": [
        "kmeans",     # K-Means
        "ap",         # Affinity Propagation
        "meanshift",  # Mean Shift
        "sc",         # Spectral Clustering
        "hclust",     # Agglomerative Clustering
        "dbscan",     # DBSCAN
        "optics",     # OPTICS
        "birch",      # Birch
    ],
    "clustering_flaml": [],  # FLAML does not support clustering

    # ── Time-Series / Forecasting ─────────────────────────────────────────────
    # statsmodels-based models applied when Stage 1 detects time-series data.
    "forecasting_statsmodels": [
        "arima",               # ARIMA (Auto-Regressive Integrated Moving Average)
        "sarima",              # SARIMA (Seasonal ARIMA)
        "exponential_smoothing",  # Holt-Winters / ETS
        "theta",               # Theta method
        "naive",               # Seasonal naive baseline
        "ses",                 # Simple Exponential Smoothing
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# FAST_MODELS — Curated subset for Phase B variant steps.
# Excludes all O(n²) and multi-epoch models so each variant completes in
# <15 min even on 250K-row datasets.
# ──────────────────────────────────────────────────────────────────────────────

FAST_MODELS: Dict[str, List[str]] = {
    "classification_pycaret": [
        "lr", "nb", "dt", "ridge", "rf", "ada", "gbc",
        "lda", "et", "xgboost", "lightgbm",
        # EXCLUDED: knn (slow on big data), qda (singular cov), catboost (long)
    ],
    "regression_pycaret": [
        "lr", "lasso", "ridge", "en", "br", "huber",
        "dt", "rf", "et", "ada", "gbr", "xgboost", "lightgbm",
        # EXCLUDED: knn, kr (kernel ridge O(n³)), catboost (long), lar/llar/omp/ard/par/ransac/tr (niche)
    ],
    "clustering_pycaret": [
        "kmeans", "birch", "hclust",
        # EXCLUDED: dbscan/optics (eps tuning), ap/meanshift/sc (O(n²))
    ],
}


def get_fast_model_list(task_type: str, engine: str) -> List[str]:
    """Return the *fast* model list for Phase B variant steps.

    Falls back to the full MODEL_UNIVERSE list if no fast list is defined.
    """
    key = f"{task_type}_{engine}"
    fast = FAST_MODELS.get(key)
    if fast is not None:
        return list(fast)
    return get_model_list(task_type, engine)


def get_forecasting_models() -> List[str]:
    """Return the list of statsmodels-based forecasting models."""
    return list(MODEL_UNIVERSE.get("forecasting_statsmodels", []))


def get_model_list_for_data(
    task_type: str,
    engine: str,
    *,
    is_time_series: bool = False,
    fast_only: bool = False,
) -> Dict[str, List[str]]:
    """Return *all* applicable model lists given data characteristics.

    Returns a dict keyed by engine-type (e.g. ``pycaret``, ``flaml``,
    ``forecasting_statsmodels``) → model list.  The caller can iterate
    over engines and run them all.

    Parameters
    ----------
    task_type : str
        ``classification``, ``regression``, or ``clustering``.
    engine : str
        Primary ML engine (``pycaret`` or ``flaml``).
    is_time_series : bool
        If True, also include ``forecasting_statsmodels`` models.
    fast_only : bool
        If True, return the FAST_MODELS subset instead of the full list.
    """
    result: Dict[str, List[str]] = {}

    key = f"{task_type}_{engine}"
    if fast_only:
        models = FAST_MODELS.get(key, MODEL_UNIVERSE.get(key, []))
    else:
        models = MODEL_UNIVERSE.get(key, [])
    result[engine] = list(models)

    if is_time_series:
        result["forecasting_statsmodels"] = get_forecasting_models()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_model_list(task_type: str, engine: str) -> List[str]:
    """Return the canonical model list for a *task_type* + *engine* combo.

    Returns an empty list (and warns) if the combo is not defined.
    """
    key = f"{task_type}_{engine}"
    models = MODEL_UNIVERSE.get(key)
    if models is None:
        logger.warning("No model list for %s; returning empty", key)
        return []
    return list(models)


def pycaret_memory_plan(
    task_type: str, row_count: int, *, memory_budget_bytes: int | None = None,
) -> Dict[str, Any]:
    """Prune known dense-kernel allocations before starting model discovery."""
    if task_type not in {"classification", "regression"}:
        raise ValueError("Memory planning supports supervised PyCaret discovery only")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count <= 0:
        raise ValueError("Model feasibility requires a positive training row count")
    if memory_budget_bytes is None:
        import psutil

        available = int(psutil.virtual_memory().available)
        # Honor the job's cgroup allowance as well as host available memory.
        for limit_path, usage_path in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ):
            if Path(limit_path).is_file() and Path(usage_path).is_file():
                limit = Path(limit_path).read_text().strip()
                if limit != "max":
                    remaining = int(limit) - int(Path(usage_path).read_text().strip())
                    available = min(available, max(0, remaining))
        # Leave room for the parent process, data copies and Azure sidecars.
        memory_budget_bytes = available // 2
    if (not isinstance(memory_budget_bytes, int)
            or isinstance(memory_budget_bytes, bool) or memory_budget_bytes <= 0):
        raise ValueError("No positive memory budget is available for model discovery")
    models = get_model_list(task_type, "pycaret")
    excluded = []
    if "kr" in models:
        # Kernel Ridge forms a dense n-by-n Gram matrix. Account for the
        # matrix plus factorization/copy workspace at full-training refit size.
        estimated_bytes = row_count * row_count * 8 * 3
        if estimated_bytes > memory_budget_bytes:
            models.remove("kr")
            excluded.append({
                "model_id": "kr",
                "status": "skipped_memory_infeasible",
                "reason": "dense_kernel_estimate_exceeds_worker_budget",
                "estimated_peak_bytes": estimated_bytes,
            })
    return {
        "training_rows": row_count,
        "memory_budget_bytes": int(memory_budget_bytes),
        "included_models": models,
        "excluded_models": excluded,
        "n_jobs": 1,
    }


def check_model_availability(task_type: str, engine: str) -> Dict[str, Dict[str, Any]]:
    """Return per-model availability status.

    Each value is ``{"status": "available"|"skipped", "reason": "..."}``.
    Currently marks everything as available — extend when you have
    runtime-importable checks (e.g. ``pycaret.internal.preprocess``).
    """
    models = get_model_list(task_type, engine)
    coverage: Dict[str, Dict[str, Any]] = {}
    for m in models:
        coverage[m] = {"status": "available", "reason": ""}
    return coverage


def build_coverage_report(
    task_type: str, include_forecasting: bool = False,
    *, memory_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a combined coverage report for both engines.

    If *include_forecasting* is True, also report statsmodels forecasting models.
    """
    report: Dict[str, Any] = {"task_type": task_type, "engines": {}}
    engines_to_report = ["pycaret", "flaml"]
    if include_forecasting:
        engines_to_report.append("forecasting_statsmodels")
    for engine in engines_to_report:
        if engine == "forecasting_statsmodels":
            models = get_forecasting_models()
            avail = {m: {"status": "available", "reason": ""} for m in models}
        else:
            avail = check_model_availability(task_type, engine)
        if engine == "pycaret" and memory_plan is not None:
            for exclusion in memory_plan["excluded_models"]:
                model_id = exclusion["model_id"]
                if model_id not in avail:
                    raise ValueError(f"Memory plan model is outside the catalog: {model_id}")
                avail[model_id] = {key: value for key, value in exclusion.items() if key != "model_id"}
        total = len(avail)
        available = sum(1 for v in avail.values() if v["status"] == "available")
        skipped = total - available
        report["engines"][engine] = {
            "total": total,
            "available": available,
            "skipped": skipped,
            "models": avail,
        }
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Model Breakdown — per-model metrics for deep analysis
# ──────────────────────────────────────────────────────────────────────────────

# Standard columns for every model_breakdown.csv
BREAKDOWN_COLUMNS = [
    "step_name",      # e.g. s05a, s05b, s06_r1_pycaret
    "engine",         # pycaret | flaml | statsmodels
    "phase",          # baseline | phase_b | phase_c
    "variant",        # recipe/variant name; "baseline" for Phase A
    "model_name",     # e.g. lightgbm, xgboost, arima
    "rank",           # 1-based rank within the step
    "primary_metric", # value of the primary metric (AUC, R2, Silhouette…)
    "metric_name",    # name of that metric
    "fit_time_sec",   # seconds (if available)
    "task_type",
]


def build_pycaret_breakdown(
    leaderboard: "pd.DataFrame",
    *,
    step_name: str,
    phase: str,
    variant: str,
    task_type: str,
    metric_col: str = "AUC",
) -> "pd.DataFrame":
    """Convert a PyCaret ``pull()`` leaderboard into a model-breakdown DataFrame.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        The DataFrame returned by ``pycaret.X.pull()`` after ``compare_models()``.
    step_name : str
        Pipeline step identifier, e.g. ``s05a``.
    phase : str
        ``baseline`` or ``phase_b``.
    variant : str
        Recipe / variant name.
    task_type : str
        ``classification``, ``regression``, or ``clustering``.
    metric_col : str
        Primary metric column name in the leaderboard (default ``AUC``).
    """
    import pandas as _pd

    rows = []
    for rank_idx, (idx, row) in enumerate(leaderboard.iterrows(), start=1):
        model_name = str(idx) if isinstance(idx, str) else str(row.get("Model", f"model_{rank_idx}"))
        metric_val = float(row[metric_col]) if metric_col in row.index else None
        fit_time = float(row.get("TT (Sec)", row.get("Time", 0.0)))
        rows.append({
            "step_name": step_name,
            "engine": "pycaret",
            "phase": phase,
            "variant": variant,
            "model_name": model_name,
            "rank": rank_idx,
            "primary_metric": metric_val,
            "metric_name": metric_col,
            "fit_time_sec": round(fit_time, 2),
            "task_type": task_type,
        })
    return _pd.DataFrame(rows, columns=BREAKDOWN_COLUMNS)


def build_flaml_breakdown(
    automl: Any,
    *,
    step_name: str,
    phase: str,
    variant: str,
    task_type: str,
    metric_name: str = "auc",
    best_metric_value: float | None = None,
) -> "pd.DataFrame":
    """Build a model-breakdown DataFrame from FLAML's ``config_history``.

    Parameters
    ----------
    automl : flaml.AutoML
        The fitted AutoML instance.
    step_name, phase, variant, task_type : str
        Metadata for identification.
    metric_name : str
        Name of the optimised metric.
    best_metric_value : float | None
        The final best metric (used for the winning estimator row).
    """
    import pandas as _pd

    rows = []
    # Extract from config_history (dict: iteration_id → config dict)
    if hasattr(automl, "config_history") and automl.config_history:
        for rank_idx, (iter_key, cfg_val) in enumerate(automl.config_history.items(), start=1):
            if isinstance(cfg_val, dict):
                learner = cfg_val.get("learner",
                                      cfg_val.get("ml", {}).get("learner", "unknown")
                                      if isinstance(cfg_val.get("ml"), dict) else "unknown")
            else:
                learner = str(cfg_val)
            rows.append({
                "step_name": step_name,
                "engine": "flaml",
                "phase": phase,
                "variant": variant,
                "model_name": learner,
                "rank": rank_idx,
                "primary_metric": None,  # FLAML doesn't store per-iter metrics easily
                "metric_name": metric_name,
                "fit_time_sec": None,
                "task_type": task_type,
            })

    # Always ensure at least the best estimator is recorded
    if not rows or (hasattr(automl, "best_estimator") and automl.best_estimator):
        rows.insert(0, {
            "step_name": step_name,
            "engine": "flaml",
            "phase": phase,
            "variant": variant,
            "model_name": getattr(automl, "best_estimator", "unknown"),
            "rank": 1,
            "primary_metric": best_metric_value,
            "metric_name": metric_name,
            "fit_time_sec": None,
            "task_type": task_type,
        })

    return _pd.DataFrame(rows, columns=BREAKDOWN_COLUMNS)


def write_model_breakdown(
    df: "pd.DataFrame",
    output_dir: str,
    filename: str = "model_breakdown.csv",
) -> str:
    """Write a model-breakdown DataFrame to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    df.to_csv(path, index=False)
    logger.info("📊 Model breakdown written → %s  (%d rows)", path, len(df))
    return str(path)


def write_model_coverage(report: Dict[str, Any], output_dir: str) -> str:
    """Write ``model_coverage.json`` to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model_coverage.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("📊 Model coverage written → %s", path)
    return str(path)
