
"""
Configuration loader and validator.

This module loads a YAML configuration file and validates required fields.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Required fields
    required = ["industry", "dataset_path", "task_type", "primary_metric", "mlflow_tracking_uri"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration fields: {missing}")

    # Validate task type
    valid_tasks = ["classification", "regression", "clustering"]
    if config["task_type"] not in valid_tasks:
        raise ValueError(f"Invalid task_type '{config['task_type']}'. Valid options: {valid_tasks}")

    # Set defaults
    config.setdefault("imbalance_handling", False)
    config.setdefault("azure_ml", {})
    config.setdefault("drift_monitoring", {
        "enabled": True,
        "psi_green": 0.1,
        "psi_yellow": 0.25,
        "concept_drift_threshold": 0.05,
        "cadence_override": None,
    })
    return config
