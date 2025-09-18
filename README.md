
# Savvy Minds MLOps Solution Accelerator (Version 1)

This repository contains a **config-driven MLOps solution accelerator** designed for Savvy Minds. It offers an automated pipeline for data preparation, feature engineering, automated model training, and evaluation. Deployment and monitoring are handled by separate teams and are out of scope for this version.

## Key Features

- **Config-driven**: All pipeline settings (industry, task type, metrics, etc.) are defined in a YAML configuration file.
- **Modular architecture**: Each stage of the pipeline (ingestion, validation, cleaning, feature engineering, training, evaluation) is implemented as a separate module.
- **AutoML training**: Utilizes PyCaret and FLAML to automatically train and tune a suite of algorithms appropriate for the selected task type.
- **Balanced and feature-aware**: Includes steps for handling class imbalance using SMOTE and feature selection using Boruta.
- **Experiment tracking**: All experiments, metrics, and artifacts are logged to MLflow for reproducibility and analysis.
- **Industry-agnostic**: Template structure allows users to specify which industry and dataset to run the pipeline on via a configuration file.

## Getting Started

### 1. Install dependencies

Ensure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Define your configuration

Create a YAML configuration file in the `config/` directory (e.g., `config/example_config.yaml`). You can use the provided `sample_config.yaml` as a template. This configuration should specify:

- **industry**: The industry for which the pipeline is run (e.g., finance, healthcare).
- **dataset_path**: Path to the dataset CSV file.
- **task_type**: Type of problem (`classification`, `regression`, or `clustering`).
- **primary_metric**: Metric to use for model evaluation (e.g., `f1`, `rmse`).
- **imbalance_handling**: Boolean indicating whether to apply SMOTE for classification tasks.
- **mlflow_tracking_uri**: Path or URI for the MLflow tracking server.

### 3. Run the pipeline

From the repository root, run:

```bash
python src/main.py --config config/your_config.yaml
```

This will execute the entire pipeline: ingestion, validation, cleaning, SMOTE balancing, feature engineering, training via PyCaret/FLAML, evaluation, and MLflow logging.

### 4. View results

After running the pipeline:

- Model artifacts and metrics are logged in MLflow. You can launch the MLflow UI using:

```bash
mlflow ui --backend-store-uri <mlflow_tracking_uri> --host 0.0.0.0 --port 5000
```

- The best pipeline and configuration (the "recipe") are printed to the console and saved as an artifact in the `artifacts/` directory.

### Structure Overview

```
|
|-- src/                          # Core source code for each pipeline stage
|    |-- config_loader.py         # Loads and validates YAML configuration files
|    |-- data_ingestion.py        # Handles dataset loading from various sources
|    |-- data_validation.py       # Implements schema validation via Pandera / Great Expectations
|    |-- data_cleaning.py         # Cleans dataset (missing values, duplicates, outliers)
|    |-- imbalance_handling.py    # Detects and handles class imbalance via SMOTE
|    |-- feature_engineering.py   # Transforms features, encodes categorical variables, applies Boruta
|    |-- model_training.py        # Runs AutoML via PyCaret or FLAML
|    |-- evaluation.py            # Evaluates models using appropriate metrics
|    |-- mlflow_utils.py          # Utility functions for MLflow logging
|    |-- main.py                  # Orchestrates the entire pipeline
|
|-- config/
|    |-- sample_config.yaml        # Example configuration file for guidance
|
|-- docs/
|    |-- design_overview.md        # In-depth architecture and workflow documentation
|    |-- azure_devops_integration.md # Guide on integrating this repo with Azure DevOps
|    |-- usage_guide.md            # Detailed usage instructions and best practices
|
|-- requirements.txt              # List of required Python packages
|
|-- artifacts/                    # Output artifacts (best recipes, models)
|
```

Feel free to raise issues or improvements as you utilize this framework for your projects.
