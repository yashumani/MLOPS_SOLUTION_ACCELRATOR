
"""
Model training module using AutoML libraries (PyCaret and FLAML).
"""

import pandas as pd
from typing import Dict, Any, Tuple

# PyCaret imports for AutoML
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull
from pycaret.clustering import setup as clu_setup, create_model as clu_create, pull as clu_pull

# FLAML AutoML
from flaml import AutoML


def train_with_pycaret(X: pd.DataFrame, y: pd.Series, task: str, metric: str) -> Tuple[Any, Any]:
    """Train models using PyCaret AutoML.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        task (str): Task type.
        metric (str): Primary metric for evaluation.

    Returns:
        Tuple[Any, Any]: Best model and results DataFrame.
    """
    df = pd.concat([X, y], axis=1)
    target_name = y.name

    if task == 'classification':
        s = cls_setup(data=df, target=target_name, silent=True, session_id=42)
        best_model = cls_compare(sort=metric)
        results = cls_pull()
    elif task == 'regression':
        s = reg_setup(data=df, target=target_name, silent=True, session_id=42)
        best_model = reg_compare(sort=metric)
        results = reg_pull()
    elif task == 'clustering':
        s = clu_setup(data=df, silent=True, session_id=42)
        best_model = clu_create('kmeans')
        results = clu_pull()
    else:
        raise ValueError(f"Unsupported task type: {task}")

    return best_model, results


def train_with_flaml(X: pd.DataFrame, y: pd.Series, task: str, metric: str) -> Tuple[Any, Dict[str, Any]]:
    """Train models using FLAML AutoML.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        task (str): 'classification' or 'regression'.
        metric (str): Primary metric.

    Returns:
        Tuple[Any, Dict[str, Any]]: Best model and best configuration.
    """
    automl = AutoML()
    settings = {
        "time_budget": 600,  # seconds
        "metric": metric,
        "task": task,
        "log_file_name": "flaml.log",
    }
    automl.fit(X_train=X, y_train=y, **settings)
    return automl.model, automl.best_config


def train_models(X: pd.DataFrame, y: pd.Series, task: str, metric: str) -> Dict[str, Any]:
    """Train models using PyCaret and FLAML.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        task (str): Task type.
        metric (str): Primary metric.

    Returns:
        Dict[str, Any]: Dictionary containing results from PyCaret and FLAML.
    """
    results = {}

    # PyCaret training
    try:
        best_model, res_df = train_with_pycaret(X, y, task, metric)
        results['pycaret'] = {
            'best_model': best_model,
            'results': res_df
        }
    except Exception as e:
        results['pycaret'] = {
            'error': str(e)
        }

    # FLAML training (only for classification/regression)
    if task in ['classification', 'regression']:
        try:
            model, config = train_with_flaml(X, y, task, metric)
            results['flaml'] = {
                'best_model': model,
                'best_config': config
            }
        except Exception as e:
            results['flaml'] = {
                'error': str(e)
            }
    return results
