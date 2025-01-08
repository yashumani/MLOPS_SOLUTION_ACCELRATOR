# FILE: /linear_regression_project/src/config.py

import yaml
import os

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

# Ensure the configuration file exists
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    if config is None:
        raise ValueError(f"Configuration file is empty or invalid: {config_path}")