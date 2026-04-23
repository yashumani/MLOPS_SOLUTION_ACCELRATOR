"""
AIM Tournament — Adaptive Indicator-gated Multi-metric Tournament.

Provides multi-metric evaluation, per-metric ranking, Pareto frontier
identification, weighted utility scoring, and per-metric top-K reports.

Usage in final_evaluation.py::

    from utils.aim_tournament import run_aim_tournament
    df = pd.read_csv("outputs/all_candidates.csv")
    enriched = run_aim_tournament(df, task_type, "outputs")

Design principles:
  - Filesystem-only — no MLflow or azureml:// dependency
  - Backward compatible — if metric columns are missing they are skipped
  - Pareto frontier uses true non-dominated sorting (not proxy ranking)
  - Utility score uses rank-percentile normalisation (distribution-free)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Metric catalog
# ──────────────────────────────────────────────────────────────────────────────

METRIC_CATALOG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "classification": {
        "accuracy":  {"direction": "maximize", "default_weight": 0.15},
        "roc_auc":   {"direction": "maximize", "default_weight": 0.25},
        "f1":        {"direction": "maximize", "default_weight": 0.20},
        "precision": {"direction": "maximize", "default_weight": 0.15},
        "recall":    {"direction": "maximize", "default_weight": 0.15},
        "logloss":   {"direction": "minimize", "default_weight": 0.10},
    },
    "regression": {
        "r2":   {"direction": "maximize", "default_weight": 0.35},
        "rmse": {"direction": "minimize", "default_weight": 0.35},
        "mae":  {"direction": "minimize", "default_weight": 0.30},
    },
    "clustering": {
        "silhouette":       {"direction": "maximize", "default_weight": 0.40},
        "davies_bouldin":   {"direction": "minimize", "default_weight": 0.30},
        "calinski_harabasz": {"direction": "maximize", "default_weight": 0.30},
    },
}

# Column names as they appear in the candidate ledger
METRIC_LEDGER_COLS: Dict[str, List[str]] = {
    "classification": ["accuracy", "f1", "precision", "recall", "roc_auc", "logloss"],
    "regression":     ["r2", "rmse", "mae"],
    "clustering":     ["silhouette", "davies_bouldin", "calinski_harabasz"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true,
    y_pred,
    task_type: str = "classification",
    y_prob=None,
    X_data=None,
) -> Dict[str, float]:
    """Compute the full metric vector for *task_type*.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (classification/regression) **or** feature matrix (clustering).
    y_pred : array-like
        Predicted labels/values (classification/regression) **or** cluster assignments (clustering).
    task_type : str
    y_prob : array-like, optional
        Predicted probabilities for classification (needed for AUC / logloss).
    X_data : array-like, optional
        Original feature matrix — used for clustering metrics if *y_true* is labels.

    Returns
    -------
    dict  — keys are canonical ledger column names.
    """
    from sklearn import metrics as skm

    result: Dict[str, float] = {}

    if task_type == "classification":
        result["accuracy"] = float(skm.accuracy_score(y_true, y_pred))
        result["f1"] = float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0))
        result["precision"] = float(skm.precision_score(y_true, y_pred, average="weighted", zero_division=0))
        result["recall"] = float(skm.recall_score(y_true, y_pred, average="weighted", zero_division=0))
        if y_prob is not None:
            try:
                if hasattr(y_prob, "ndim") and y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    result["roc_auc"] = float(skm.roc_auc_score(y_true, y_prob[:, 1]))
                elif hasattr(y_prob, "ndim") and y_prob.ndim == 2:
                    result["roc_auc"] = float(skm.roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
                else:
                    result["roc_auc"] = float(skm.roc_auc_score(y_true, y_prob))
            except Exception:
                result["roc_auc"] = float("nan")
            try:
                result["logloss"] = float(skm.log_loss(y_true, y_prob))
            except Exception:
                result["logloss"] = float("nan")
        else:
            result["roc_auc"] = float("nan")
            result["logloss"] = float("nan")

    elif task_type == "regression":
        result["r2"] = float(skm.r2_score(y_true, y_pred))
        result["rmse"] = float(math.sqrt(skm.mean_squared_error(y_true, y_pred)))
        result["mae"] = float(skm.mean_absolute_error(y_true, y_pred))

    elif task_type == "clustering":
        # For clustering: y_true = feature matrix, y_pred = cluster labels
        data_matrix = X_data if X_data is not None else y_true
        labels = y_pred
        try:
            result["silhouette"] = float(skm.silhouette_score(data_matrix, labels))
        except Exception:
            result["silhouette"] = float("nan")
        try:
            result["davies_bouldin"] = float(skm.davies_bouldin_score(data_matrix, labels))
        except Exception:
            result["davies_bouldin"] = float("nan")
        try:
            result["calinski_harabasz"] = float(skm.calinski_harabasz_score(data_matrix, labels))
        except Exception:
            result["calinski_harabasz"] = float("nan")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Catalog helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_metric_columns(task_type: str) -> List[str]:
    """Metric column names for *task_type* as they appear in the ledger."""
    return list(METRIC_LEDGER_COLS.get(task_type, []))


def get_metric_directions(task_type: str) -> Dict[str, str]:
    """Return ``{metric_name: 'maximize'|'minimize'}``."""
    catalog = METRIC_CATALOG.get(task_type, {})
    return {k: v["direction"] for k, v in catalog.items()}


def get_default_weights(task_type: str, primary_metric: Optional[str] = None) -> Dict[str, float]:
    """Metric weights, optionally boosted for *primary_metric*."""
    catalog = METRIC_CATALOG.get(task_type, {})
    weights = {k: v["default_weight"] for k, v in catalog.items()}
    if primary_metric and primary_metric in weights:
        weights[primary_metric] *= 1.5
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# Ranking
# ──────────────────────────────────────────────────────────────────────────────

def add_rank_columns(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Add ``rank_<metric>`` columns using dense ranking (1 = best)."""
    directions = get_metric_directions(task_type)
    for metric, direction in directions.items():
        if metric not in df.columns:
            continue
        numeric = pd.to_numeric(df[metric], errors="coerce")
        ascending = direction == "minimize"
        df[f"rank_{metric}"] = (
            numeric.rank(method="dense", ascending=ascending, na_option="bottom")
            .astype("Int64")
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Pareto frontier
# ──────────────────────────────────────────────────────────────────────────────

def find_pareto_frontier(df: pd.DataFrame, task_type: str) -> pd.Series:
    """Return a boolean Series marking Pareto-optimal (non-dominated) rows.

    A candidate is non-dominated if no other candidate is strictly better
    in **all** metrics simultaneously.
    """
    directions = get_metric_directions(task_type)
    metric_cols = [c for c in directions if c in df.columns]

    if not metric_cols:
        return pd.Series([False] * len(df), index=df.index)

    # Build value matrix where HIGHER = BETTER (negate minimize metrics)
    vals = (
        df[metric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(float("-inf"))
        .values.copy()
    )
    for i, col in enumerate(metric_cols):
        if directions[col] == "minimize":
            vals[:, i] = -vals[:, i]

    n = len(vals)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i  →  j >= i everywhere AND j > i somewhere
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                is_pareto[i] = False
                break

    return pd.Series(is_pareto, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# Utility score  (rank-percentile approach — distribution-free)
# ──────────────────────────────────────────────────────────────────────────────

def add_utility_scores(
    df: pd.DataFrame,
    task_type: str,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Add ``utility_score`` and ``utility_rank`` columns.

    Utility is computed via rank-percentile normalisation: for each metric
    the rank is converted to a [0, 1] percentile (1 = best) then
    multiplied by the weight.
    """
    if weights is None:
        weights = get_default_weights(task_type)

    directions = get_metric_directions(task_type)
    n = len(df)
    scores = pd.Series(0.0, index=df.index)
    total_w = 0.0

    for metric, w in weights.items():
        if metric not in df.columns:
            continue
        numeric = pd.to_numeric(df[metric], errors="coerce")
        ascending = directions.get(metric) == "minimize"
        ranks = numeric.rank(method="average", ascending=ascending, na_option="bottom")
        percentile = 1.0 - (ranks - 1) / max(n - 1, 1)
        scores = scores + w * percentile
        total_w += w

    df["utility_score"] = (scores / total_w if total_w > 0 else scores).round(6)
    df["utility_rank"] = (
        df["utility_score"]
        .rank(method="dense", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Top-K tables
# ──────────────────────────────────────────────────────────────────────────────

def write_topk_tables(
    df: pd.DataFrame,
    task_type: str,
    output_dir: str,
    k: int = 10,
) -> List[str]:
    """Write ``top_{k}_{metric}.csv`` files into ``{output_dir}/topk/``."""
    out = Path(output_dir) / "topk"
    out.mkdir(parents=True, exist_ok=True)

    directions = get_metric_directions(task_type)
    written: List[str] = []

    for metric in directions:
        if metric not in df.columns:
            continue
        ascending = directions[metric] == "minimize"
        numeric = pd.to_numeric(df[metric], errors="coerce")
        sorted_df = df.loc[numeric.sort_values(ascending=ascending, na_position="last").index]
        topk = sorted_df.head(k)

        path = out / f"top_{k}_{metric}.csv"
        topk.to_csv(path, index=False)
        written.append(str(path))
        logger.info("  📊 Top-%d %s → %s", k, metric, path)

    return written


# ──────────────────────────────────────────────────────────────────────────────
# Pareto report
# ──────────────────────────────────────────────────────────────────────────────

def write_pareto_report(df: pd.DataFrame, task_type: str, output_dir: str) -> str:
    """Write ``pareto_frontier.csv`` and ``pareto_summary.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pareto_mask = find_pareto_frontier(df, task_type)
    pareto_df = df[pareto_mask].copy()

    path = out / "pareto_frontier.csv"
    pareto_df.to_csv(path, index=False)

    summary: Dict[str, Any] = {
        "total_candidates": len(df),
        "pareto_size": int(pareto_mask.sum()),
        "pareto_fraction": round(float(pareto_mask.sum()) / max(len(df), 1), 4),
        "pareto_candidates": (
            pareto_df["candidate_id"].tolist()
            if "candidate_id" in pareto_df.columns
            else []
        ),
    }

    summary_path = out / "pareto_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "  🏆 Pareto frontier: %d/%d candidates → %s",
        summary["pareto_size"],
        summary["total_candidates"],
        path,
    )
    return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Full AIM-Tournament pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_aim_tournament(
    df: pd.DataFrame,
    task_type: str,
    output_dir: str,
    k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    primary_metric: Optional[str] = None,
) -> pd.DataFrame:
    """Run the complete AIM-Tournament enrichment on a candidate DataFrame.

    Steps:
      1. Add per-metric rank columns
      2. Identify Pareto frontier
      3. Compute utility scores (rank-percentile, weighted)
      4. Write per-metric top-K tables
      5. Write Pareto frontier report
      6. Write enriched ``all_candidates_ranked.csv``

    Returns the enriched DataFrame.
    """
    print("=" * 80)
    print("🏆 AIM-TOURNAMENT: Multi-metric ranking and Pareto analysis")
    print("=" * 80)

    if weights is None and primary_metric:
        weights = get_default_weights(task_type, primary_metric)

    # 1. Per-metric ranks
    df = add_rank_columns(df, task_type)
    print(f"  ✅ Added rank columns for {task_type}")

    # 2. Pareto frontier
    df["pareto_optimal"] = find_pareto_frontier(df, task_type)
    pareto_count = int(df["pareto_optimal"].sum())
    print(f"  ✅ Pareto frontier: {pareto_count}/{len(df)} candidates")

    # 3. Utility scores
    df = add_utility_scores(df, task_type, weights)
    best_util = df["utility_score"].max()
    print(f"  ✅ Utility scores computed (top: {best_util:.4f})")

    # 4. Top-K tables
    written = write_topk_tables(df, task_type, output_dir, k=k)
    print(f"  ✅ Wrote {len(written)} top-{k} metric tables")

    # 5. Pareto report
    write_pareto_report(df, task_type, output_dir)

    # 6. Write enriched ledger
    enriched_path = Path(output_dir) / "all_candidates_ranked.csv"
    df.to_csv(enriched_path, index=False)
    print(f"  ✅ Enriched ledger → {enriched_path}")

    try:
        import pyarrow  # noqa: F401
        parquet_path = Path(output_dir) / "all_candidates_ranked.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
    except Exception:
        pass

    print("=" * 80)
    return df
