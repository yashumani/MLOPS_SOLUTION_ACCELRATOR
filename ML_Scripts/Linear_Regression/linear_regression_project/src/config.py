# FILE: /ML_Scripts/Linear_Regression/linear_regression_project/config.py

import yaml
import os

# Define the path to the config.yaml file relative to the project root
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))

# Ensure the configuration file exists
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    if config is None:
        raise ValueError(f"Configuration file is empty or invalid: {config_path}")

# Update paths in the config to be absolute paths
config['log_path'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config['log_path']))
config['data_path'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config['data_path']))
config['new_data_path'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config['new_data_path']))
config['reports_path'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config['reports_path']))
if 'report_path' in config['eda_tool']:
    config['eda_tool']['report_path'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config['eda_tool']['report_path']))