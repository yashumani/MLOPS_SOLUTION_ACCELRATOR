"""Generate synthetic datasets with controlled drift for testing.

Provides three presets — **no drift**, **mild drift**, and **severe drift** —
applied to numeric features, categorical columns, prediction columns, and
label distributions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Public factory ──────────────────────────────────────────────

def generate_drifted_data(
    reference_df: pd.DataFrame,
    drift_level: str = "none",
    seed: int = 42,
    n_rows: Optional[int] = None,
    target_column: Optional[str] = None,
    prediction_column: Optional[str] = None,
) -> pd.DataFrame:
    """Return a copy of *reference_df* with synthetic drift injected.

    Parameters
    ----------
    reference_df : pd.DataFrame
        The clean reference (baseline) dataset.
    drift_level : str
        ``"none"`` | ``"mild"`` | ``"severe"``.
    seed : int
        Random seed for reproducibility.
    n_rows : int or None
        Number of rows in the output.  Defaults to ``len(reference_df)``.
    target_column : str or None
        Name of the target/label column (for label-drift injection).
    prediction_column : str or None
        Name of the prediction column (for prediction-drift injection).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the requested drift applied.
    """
    rng = np.random.default_rng(seed)
    n = n_rows or len(reference_df)
    df = reference_df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    if drift_level == "none":
        logger.info("Generating no-drift variant (%d rows)", n)
        return df

    params = _DRIFT_PARAMS[drift_level]
    logger.info("Generating %s-drift variant (%d rows)", drift_level, n)

    # Exclude target/prediction from generic numeric shift — they are
    # handled separately via _flip_labels / _shift_prediction_column.
    exclude = [c for c in [target_column, prediction_column] if c]
    df = _shift_numeric_features(df, rng, params, exclude_cols=exclude)
    df = _inject_categorical_noise(df, rng, params)

    if prediction_column and prediction_column in df.columns:
        df = _shift_prediction_column(df, rng, params, prediction_column)

    if target_column and target_column in df.columns:
        df = _flip_labels(df, rng, params, target_column)

    return df


# ── Drift intensity presets ─────────────────────────────────────

_DRIFT_PARAMS: Dict[str, Dict[str, float]] = {
    "mild": {
        "mean_shift_factor": 0.5,
        "noise_std_factor": 0.2,
        "category_flip_rate": 0.05,
        "label_flip_rate": 0.05,
        "prediction_shift_factor": 0.3,
    },
    "severe": {
        "mean_shift_factor": 2.0,
        "noise_std_factor": 0.8,
        "category_flip_rate": 0.25,
        "label_flip_rate": 0.20,
        "prediction_shift_factor": 1.5,
    },
}


# ── Internal transforms ────────────────────────────────────────

def _shift_numeric_features(
    df: pd.DataFrame,
    rng: np.random.Generator,
    params: Dict[str, float],
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Shift means and add noise to numeric columns."""
    exclude = set(exclude_cols or [])
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    for col in numeric_cols:
        col_std = df[col].std()
        if col_std == 0 or np.isnan(col_std):
            continue
        shift = params["mean_shift_factor"] * col_std
        noise = rng.normal(loc=0, scale=params["noise_std_factor"] * col_std, size=len(df))
        df[col] = df[col] + shift + noise
    return df


def _inject_categorical_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    params: Dict[str, float],
) -> pd.DataFrame:
    """Randomly replace categorical values with other valid categories."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    flip_rate = params["category_flip_rate"]

    for col in cat_cols:
        cats = df[col].dropna().unique().tolist()
        if len(cats) < 2:
            continue
        mask = rng.random(len(df)) < flip_rate
        replacements = rng.choice(cats, size=int(mask.sum()))
        df.loc[mask, col] = replacements
    return df


def _shift_prediction_column(
    df: pd.DataFrame,
    rng: np.random.Generator,
    params: Dict[str, float],
    pred_col: str,
) -> pd.DataFrame:
    """Shift the prediction column distribution."""
    if pd.api.types.is_numeric_dtype(df[pred_col]):
        unique_vals = df[pred_col].dropna().unique()
        # Binary/integer predictions → flip a fraction of labels
        if len(unique_vals) <= 10 and all(float(v).is_integer() for v in unique_vals):
            flip_rate = params["prediction_shift_factor"] * 0.2  # scale down
            flip_rate = min(flip_rate, 0.5)  # cap at 50%
            labels = unique_vals.tolist()
            if len(labels) >= 2:
                mask = rng.random(len(df)) < flip_rate
                for idx in np.where(mask)[0]:
                    current = df.iloc[idx][pred_col]
                    df.iat[idx, df.columns.get_loc(pred_col)] = rng.choice(
                        [l for l in labels if l != current]
                    )
        else:
            # Continuous predictions → shift distribution
            col_std = df[pred_col].std()
            if col_std > 0:
                shift = params["prediction_shift_factor"] * col_std
                df[pred_col] = df[pred_col] + shift
    else:
        # Categorical predictions — flip some labels
        cats = df[pred_col].dropna().unique().tolist()
        if len(cats) >= 2:
            mask = rng.random(len(df)) < params["label_flip_rate"]
            df.loc[mask, pred_col] = rng.choice(cats, size=int(mask.sum()))
    return df


def _flip_labels(
    df: pd.DataFrame,
    rng: np.random.Generator,
    params: Dict[str, float],
    target_col: str,
) -> pd.DataFrame:
    """Randomly flip a fraction of target labels."""
    flip_rate = params["label_flip_rate"]
    labels = df[target_col].dropna().unique().tolist()
    if len(labels) < 2:
        return df

    mask = rng.random(len(df)) < flip_rate
    df.loc[mask, target_col] = rng.choice(labels, size=int(mask.sum()))
    return df
