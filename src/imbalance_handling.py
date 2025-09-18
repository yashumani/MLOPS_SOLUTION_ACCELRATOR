
"""
Imbalance detection and handling module.

Applies SMOTE to classification datasets if needed.
"""

import pandas as pd
from typing import Tuple
from collections import Counter
from imblearn.over_sampling import SMOTE


def detect_imbalance(y: pd.Series, threshold: float = 0.2) -> bool:
    """Detect whether the target variable is imbalanced.

    Args:
        y (pd.Series): Target variable.
        threshold (float): Ratio threshold below which to consider data imbalanced.

    Returns:
        bool: True if imbalanced, False otherwise.
    """
    counts = Counter(y)
    majority = max(counts.values())
    minority = min(counts.values())
    imbalance_ratio = minority / majority
    return imbalance_ratio < threshold


def apply_smote(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance the classes.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Balanced feature matrix and target variable.
    """
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
