"""Deterministic train/holdout partition metadata shared by stages 3 and 4."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_COLUMN = "__mlops_split__"
ROW_ID_COLUMN = "__mlops_row_id__"
TRAIN_PARTITION = "train"
HOLDOUT_PARTITION = "holdout"
_VALID_PARTITIONS = {TRAIN_PARTITION, HOLDOUT_PARTITION}


def ensure_holdout_partition(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    task_type: str,
    holdout_fraction: float,
    random_seed: int,
    split_strategy: str = "random",
    time_column: str | None = None,
) -> pd.DataFrame:
    """Return a copy with a validated deterministic split assignment."""
    result = df.copy().reset_index(drop=True)
    if ROW_ID_COLUMN in result.columns:
        if result[ROW_ID_COLUMN].isna().any():
            raise ValueError(f"{ROW_ID_COLUMN} must identify every row")
        if not result[ROW_ID_COLUMN].is_unique:
            raise ValueError(f"{ROW_ID_COLUMN} must be unique")
    else:
        identity_source = result.drop(
            columns=[SPLIT_COLUMN],
            errors="ignore",
        )
        result[ROW_ID_COLUMN] = (
            pd.util.hash_pandas_object(identity_source, index=True)
            .astype("uint64")
            .astype(str)
        )
        if not result[ROW_ID_COLUMN].is_unique:
            raise ValueError("Could not assign unique canonical row identities")
    if SPLIT_COLUMN in result.columns:
        if result[SPLIT_COLUMN].isna().any():
            raise ValueError(f"{SPLIT_COLUMN} must assign every row")
        labels = set(result[SPLIT_COLUMN].astype(str).unique())
        if not labels or not labels.issubset(_VALID_PARTITIONS):
            raise ValueError(
                f"{SPLIT_COLUMN} must contain only {sorted(_VALID_PARTITIONS)}"
            )
        if not (result[SPLIT_COLUMN] == TRAIN_PARTITION).any():
            raise ValueError("Holdout partition metadata contains no training rows")
        if not (result[SPLIT_COLUMN] == HOLDOUT_PARTITION).any():
            raise ValueError("Holdout partition metadata contains no holdout rows")
        return result

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if len(result) < 2:
        raise ValueError("At least two rows are required for a train/holdout split")

    normalized_strategy = split_strategy.strip().lower()
    if normalized_strategy == "time":
        normalized_strategy = "chronological"
    if normalized_strategy not in {"random", "stratified", "chronological"}:
        raise ValueError(
            "split_strategy must be 'random', 'stratified', or "
            "'time'/'chronological'"
        )

    if normalized_strategy == "chronological":
        if not time_column or time_column not in result.columns:
            raise ValueError(
                "A valid time_column is required for a chronological holdout split"
            )
        timestamps = pd.to_datetime(result[time_column], errors="coerce", utc=True)
        if timestamps.isna().any():
            raise ValueError(
                f"time_column {time_column!r} contains missing or invalid timestamps"
            )
        holdout_count = max(1, int(math.ceil(len(result) * holdout_fraction)))
        if holdout_count >= len(result):
            raise ValueError("Chronological split must leave at least one training row")
        ordered_index = timestamps.sort_values(kind="stable").index
        holdout_index = ordered_index[-holdout_count:]
        result[SPLIT_COLUMN] = TRAIN_PARTITION
        result.loc[holdout_index, SPLIT_COLUMN] = HOLDOUT_PARTITION
        return result

    if normalized_strategy == "stratified" and task_type != "classification":
        raise ValueError("stratified split requires task_type='classification'")
    if normalized_strategy == "stratified" and (
        not target_col or target_col not in result.columns
    ):
        raise ValueError("stratified split requires a valid target column")

    stratify = None
    if task_type == "classification" and target_col and target_col in result.columns:
        target = result[target_col]
        class_counts = target.value_counts(dropna=False)
        if len(class_counts) > 1 and (class_counts >= 2).all():
            stratify = target
        elif normalized_strategy == "stratified":
            raise ValueError(
                "stratified split requires at least two rows in every class"
            )

    try:
        train_index, holdout_index = train_test_split(
            result.index,
            test_size=holdout_fraction,
            random_state=random_seed,
            stratify=stratify,
        )
    except ValueError:
        if normalized_strategy == "stratified":
            raise
        # Very small class counts can make stratification impossible even when
        # each class has two rows. Preserve a deterministic split instead.
        train_index, holdout_index = train_test_split(
            result.index,
            test_size=holdout_fraction,
            random_state=random_seed,
            stratify=None,
        )

    result[SPLIT_COLUMN] = HOLDOUT_PARTITION
    result.loc[train_index, SPLIT_COLUMN] = TRAIN_PARTITION
    result.loc[holdout_index, SPLIT_COLUMN] = HOLDOUT_PARTITION
    return result


def training_mask(df: pd.DataFrame) -> pd.Series:
    """Return a validated boolean mask selecting training rows."""
    if SPLIT_COLUMN not in df.columns:
        raise ValueError(f"Missing required partition metadata column {SPLIT_COLUMN}")
    mask = df[SPLIT_COLUMN].astype(str).eq(TRAIN_PARTITION)
    if not mask.any() or mask.all():
        raise ValueError("Partition metadata must contain train and holdout rows")
    return mask
