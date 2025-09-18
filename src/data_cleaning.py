
"""
Data cleaning module.

Handles missing values, duplicate rows, and outlier treatment.
"""

import pandas as pd
from sklearn.impute import SimpleImputer


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling missing values and duplicates.

    Args:
        df (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle missing values: numeric columns -> median, categorical -> mode
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(numeric_cols) > 0:
        imp_num = SimpleImputer(strategy="median")
        df[numeric_cols] = imp_num.fit_transform(df[numeric_cols])

    if len(categorical_cols) > 0:
        imp_cat = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = imp_cat.fit_transform(df[categorical_cols])

    return df
