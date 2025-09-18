
"""
MLflow utilities for experiment logging.
"""

import mlflow
from typing import Dict, Any


def init_mlflow_experiment(experiment_name: str, tracking_uri: str) -> None:
    """Initialize MLflow experiment.

    Args:
        experiment_name (str): Name of the experiment.
        tracking_uri (str): Tracking URI for MLflow.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow.

    Args:
        params (Dict[str, Any]): Parameters dictionary.
    """
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: Dict[str, float]) -> None:
    """Log metrics to MLflow.

    Args:
        metrics (Dict[str, float]): Metrics dictionary.
    """
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
