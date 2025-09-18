
"""
Feature engineering module.

Handles encoding, scaling, and feature selection (Boruta).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Tuple


def engineer_features(df: pd.DataFrame, target: str, task_type: str) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Apply transformations and select important features.

    Args:
        df (pd.DataFrame): Input data including target column.
        target (str): Name of the target column.
        task_type (str): Problem type (classification, regression, clustering).

    Returns:
        Tuple[pd.DataFrame, pd.Series, Pipeline]: Processed features, target, and the transformation pipeline.
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # Define preprocessing pipelines
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Convert to dense DataFrame
    X_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
    )

    # Feature selection using Boruta
    if task_type == 'classification':
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    else:
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

    selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
    selector.fit(X_df.values, y.values)

    # Select important features
    X_selected = X_df.loc[:, selector.support_]

    return X_selected.reset_index(drop=True), y.reset_index(drop=True), preprocessor
