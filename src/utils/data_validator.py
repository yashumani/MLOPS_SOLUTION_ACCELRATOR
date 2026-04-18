import pandas as pd
from typing import List


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")


def drop_high_cardinality(
    df: pd.DataFrame, categorical_columns: List[str], max_unique: int = 100
) -> tuple[pd.DataFrame, List[str]]:
    """Drop categorical columns that exceed the uniqueness threshold.

    Returns the filtered dataframe and the list of dropped columns so callers can log metrics.
    """
    dropped: List[str] = []
    for col in categorical_columns:
        if col in df.columns and df[col].nunique() > max_unique:
            dropped.append(col)
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped
