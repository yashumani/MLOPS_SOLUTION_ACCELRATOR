"""
Drift Detection Utility — PSI (Population Stability Index)

Computes per-feature PSI between a reference (training) and test distribution.
Supports both numeric and categorical features. Provides baseline statistics
generation and retraining cadence recommendation.

PSI Thresholds (industry standard):
  - < 0.1  → No significant drift (green)
  - 0.1–0.25 → Moderate drift (yellow)
  - > 0.25 → Significant drift (red)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# PSI thresholds
PSI_GREEN = 0.1
PSI_YELLOW = 0.25

# Small constant to avoid log(0) and division by zero
EPS = 1e-6


def compute_psi_numeric(reference: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    """Compute PSI for a single numeric feature using equal-width bins from reference.

    PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)
    where P_i = proportion in bin i for reference, Q_i for test.

    Args:
        reference: 1D array of reference (training) values.
        test: 1D array of test values.
        n_bins: Number of equal-width bins.

    Returns:
        PSI score (float >= 0). Lower = less drift.
    """
    # Defensive: cast bool/object arrays to float64 so np.isnan and arithmetic work.
    # Modern stage4 one-hot encoders return bool dtype, which breaks numpy
    # subtract on quantile interpolation downstream.
    if reference.dtype == bool:
        reference = reference.astype(np.float64)
    if test.dtype == bool:
        test = test.astype(np.float64)
    ref = reference[~np.isnan(reference)]
    tst = test[~np.isnan(test)]

    if len(ref) < n_bins or len(tst) < n_bins:
        logger.warning("Insufficient data for PSI bins; returning 0.0")
        return 0.0

    # Create bins from reference distribution
    min_val = ref.min()
    max_val = ref.max()
    if min_val == max_val:
        return 0.0  # Constant feature — no drift possible

    bins = np.linspace(min_val, max_val, n_bins + 1)
    # Extend edges to capture test values outside reference range
    bins[0] = min(bins[0], tst.min()) - EPS
    bins[-1] = max(bins[-1], tst.max()) + EPS

    ref_counts = np.histogram(ref, bins=bins)[0].astype(float)
    tst_counts = np.histogram(tst, bins=bins)[0].astype(float)

    # Convert to proportions with smoothing
    ref_pct = (ref_counts + EPS) / (ref_counts.sum() + EPS * n_bins)
    tst_pct = (tst_counts + EPS) / (tst_counts.sum() + EPS * n_bins)

    psi = np.sum((tst_pct - ref_pct) * np.log(tst_pct / ref_pct))
    return float(max(psi, 0.0))


def compute_psi_categorical(reference: np.ndarray, test: np.ndarray) -> float:
    """Compute PSI for a single categorical feature using category bins.

    Each unique category is treated as a bin.

    Args:
        reference: 1D array of reference (training) categorical values.
        test: 1D array of test categorical values.

    Returns:
        PSI score (float >= 0).
    """
    ref_series = pd.Series(reference).dropna()
    tst_series = pd.Series(test).dropna()

    if len(ref_series) == 0 or len(tst_series) == 0:
        return 0.0

    # Get all categories from both sets
    all_cats = set(ref_series.unique()) | set(tst_series.unique())
    if len(all_cats) == 0:
        return 0.0

    ref_counts = ref_series.value_counts()
    tst_counts = tst_series.value_counts()

    psi = 0.0
    for cat in all_cats:
        ref_pct = (ref_counts.get(cat, 0) + EPS) / (len(ref_series) + EPS * len(all_cats))
        tst_pct = (tst_counts.get(cat, 0) + EPS) / (len(tst_series) + EPS * len(all_cats))
        psi += (tst_pct - ref_pct) * np.log(tst_pct / ref_pct)

    return float(max(psi, 0.0))


def compute_feature_psi(
    ref_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute PSI for each feature between reference and test DataFrames.

    Automatically detects numeric vs categorical features.

    Args:
        ref_df: Reference (training) features DataFrame.
        test_df: Test features DataFrame.
        n_bins: Number of bins for numeric features.

    Returns:
        Dict mapping feature_name → PSI score.
    """
    common_cols = [c for c in ref_df.columns if c in test_df.columns]
    psi_scores = {}

    for col in common_cols:
        ref_vals = ref_df[col].values
        tst_vals = test_df[col].values

        if pd.api.types.is_numeric_dtype(ref_df[col]):
            # Cast bool to float to avoid numpy subtract errors in quantile interp.
            if ref_df[col].dtype == bool:
                ref_vals = ref_vals.astype(np.float64)
                tst_vals = tst_vals.astype(np.float64)
            psi_scores[col] = compute_psi_numeric(ref_vals, tst_vals, n_bins=n_bins)
        else:
            psi_scores[col] = compute_psi_categorical(ref_vals, tst_vals)

    return psi_scores


def compute_baseline_statistics(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Compute per-feature baseline statistics for future drift monitoring.

    For numeric features: mean, std, min, max, quantiles (5th, 25th, 50th, 75th, 95th).
    For categorical features: value_counts (top 50 categories), n_unique.

    Args:
        df: DataFrame of features (no target).
        feature_cols: Subset of columns to profile. If None, uses all.

    Returns:
        Dict mapping feature_name → statistics dict.
    """
    cols = feature_cols if feature_cols else list(df.columns)
    stats = {}

    for col in cols:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            # Cast bool to int8 — bool series breaks pandas .quantile() in
            # newer numpy (`subtract` not supported on bool).
            if vals.dtype == bool:
                vals = vals.astype(np.int8)
            if len(vals) == 0:
                stats[col] = {"type": "numeric", "count": 0}
                continue
            quantiles = vals.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
            stats[col] = {
                "type": "numeric",
                "count": int(len(vals)),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "quantiles": {f"p{int(k*100)}": float(v) for k, v in quantiles.items()},
                "missing_rate": float(df[col].isna().mean()),
            }
        else:
            vc = df[col].value_counts().head(50)
            stats[col] = {
                "type": "categorical",
                "count": int(df[col].notna().sum()),
                "n_unique": int(df[col].nunique()),
                "top_categories": {str(k): int(v) for k, v in vc.items()},
                "missing_rate": float(df[col].isna().mean()),
            }

    return stats


def compute_stability_score(
    psi_scores: Dict[str, float],
    n_rows: int,
    n_features: int,
    imbalance_ratio: Optional[float],
    feature_volatility: float,
) -> Tuple[int, Dict[str, dict]]:
    """Compute overall stability score (0–100) from data characteristics.

    Components (weighted):
        - Self-check PSI (40%): Mean per-feature PSI → lower = more stable
        - Dataset size (20%): log(n_rows) normalized → larger = more stable
        - Feature complexity (20%): n_features/n_rows ratio → lower = more stable
        - Class balance (10%): imbalance_ratio → closer to 1.0 = more stable
        - Feature volatility (10%): Mean CV across numeric features → lower = more stable

    Args:
        psi_scores: Per-feature PSI scores from self-check.
        n_rows: Number of rows in dataset.
        n_features: Number of features.
        imbalance_ratio: min_class/max_class ratio (None for clustering/regression).
        feature_volatility: Mean coefficient of variation across numeric features.

    Returns:
        Tuple of (stability_score 0-100, component_details dict).
    """
    # 1. Self-check PSI score (40%)
    mean_psi = np.mean(list(psi_scores.values())) if psi_scores else 0.0
    # PSI 0 → 100, PSI 0.25+ → 0
    psi_score = max(0, min(100, 100 * (1 - mean_psi / PSI_YELLOW)))

    # 2. Dataset size score (20%)
    # log10(100)=2 → low, log10(1M)=6 → high; normalize to 0-100
    log_rows = np.log10(max(n_rows, 1))
    size_score = max(0, min(100, (log_rows - 2) / 4 * 100))

    # 3. Feature complexity score (20%)
    # ratio 0.001 → high score, ratio 1.0 → low score
    complexity_ratio = n_features / max(n_rows, 1)
    complexity_score = max(0, min(100, 100 * (1 - min(complexity_ratio * 100, 1.0))))

    # 4. Class balance score (10%)
    if imbalance_ratio is not None:
        balance_score = max(0, min(100, imbalance_ratio * 100))
    else:
        balance_score = 75  # Neutral for clustering/regression

    # 5. Feature volatility score (10%)
    # CV 0 → 100, CV 2+ → 0
    volatility_score = max(0, min(100, 100 * (1 - min(feature_volatility / 2, 1.0))))

    # Weighted combination
    total = (
        0.40 * psi_score
        + 0.20 * size_score
        + 0.20 * complexity_score
        + 0.10 * balance_score
        + 0.10 * volatility_score
    )
    stability = int(round(total))

    components = {
        "self_check_psi": {"raw": round(mean_psi, 6), "score": round(psi_score, 1), "weight": 0.40},
        "dataset_size": {"raw": n_rows, "score": round(size_score, 1), "weight": 0.20},
        "feature_complexity": {"raw": round(complexity_ratio, 6), "score": round(complexity_score, 1), "weight": 0.20},
        "class_balance": {"raw": round(imbalance_ratio, 4) if imbalance_ratio is not None else None, "score": round(balance_score, 1), "weight": 0.10},
        "feature_volatility": {"raw": round(feature_volatility, 4), "score": round(volatility_score, 1), "weight": 0.10},
    }

    return stability, components


def determine_retraining_cadence(stability_score: int) -> Tuple[str, int, str]:
    """Map stability score to retraining cadence recommendation.

    Args:
        stability_score: Overall stability score (0–100).

    Returns:
        Tuple of (cadence_name, recommended_days, rationale).
    """
    if stability_score >= 80:
        return (
            "quarterly",
            90,
            "Stable dataset with low drift risk. Large data volume and consistent feature distributions "
            "suggest minimal distribution shift between retraining cycles.",
        )
    elif stability_score >= 60:
        return (
            "monthly",
            30,
            "Moderate complexity with some feature volatility. Monthly retraining balances freshness "
            "with compute cost for datasets of this profile.",
        )
    elif stability_score >= 40:
        return (
            "biweekly",
            14,
            "Notable feature complexity or volatility detected. Bi-weekly monitoring recommended "
            "to catch distribution shifts before model degradation.",
        )
    else:
        return (
            "weekly",
            7,
            "High drift risk due to small dataset size, high feature volatility, or class imbalance. "
            "Weekly monitoring and frequent retraining recommended.",
        )


def classify_feature_drift(psi: float) -> str:
    """Classify a single feature PSI into green/yellow/red."""
    if psi < PSI_GREEN:
        return "green"
    elif psi < PSI_YELLOW:
        return "yellow"
    else:
        return "red"


def compute_feature_volatility(df: pd.DataFrame) -> float:
    """Compute mean coefficient of variation across numeric features.

    CV = std / |mean|. Higher CV means more volatile features.

    Args:
        df: Features DataFrame (no target).

    Returns:
        Mean CV across numeric features. Returns 0 if no numeric features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0.0

    cvs = []
    for col in numeric_cols:
        vals = df[col].dropna()
        mean_val = vals.mean()
        if abs(mean_val) > EPS:
            cvs.append(vals.std() / abs(mean_val))

    return float(np.mean(cvs)) if cvs else 0.0
