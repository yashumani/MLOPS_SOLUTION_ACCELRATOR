
# Usage Guide

This guide describes how to configure, execute, and troubleshoot the Savvy Minds MLOps solution accelerator.

## Prerequisites

- Python 3.8 or higher.
- Basic familiarity with YAML, Python, and ML concepts.
- A dataset in CSV format with a clear target column for supervised learning tasks.
- Optional: MLflow installed globally if you wish to run the UI (`pip install mlflow`).

## Creating a Configuration File

1. Copy the example configuration from `config/sample_config.yaml` to a new file (e.g., `config/my_run_config.yaml`).
2. Fill out the following fields:
   - `industry`: Choose one of your organization’s supported industries.
   - `dataset_path`: Path to your dataset CSV file.
   - `task_type`: Specify `classification`, `regression`, or `clustering`.
   - `primary_metric`: Choose the metric to optimize (e.g., `f1` for classification). See the evaluation module documentation for available metrics.
   - `imbalance_handling`: Boolean (`true` or `false`) indicating whether to apply SMOTE on imbalanced datasets.
   - `mlflow_tracking_uri`: Directory or server URL for MLflow. Local runs can use a directory like `mlruns`.
3. Optionally fill in Azure ML details (subscription ID, resource group, workspace name) if you plan to register datasets or use AML compute for training in future iterations.

## Running the Pipeline

To execute the pipeline:

```bash
python src/main.py --config config/my_run_config.yaml
```

The orchestrator will perform the following:

1. Load configuration and validate it.
2. Ingest the dataset.
3. Run initial data validation checks.
4. Clean the data (handle missing values, duplicates, outliers).
5. Validate the cleaned data.
6. Detect class imbalance and apply SMOTE if enabled and the task is classification.
7. Perform feature engineering, including Boruta feature selection.
8. Validate the engineered data.
9. Train models using PyCaret or FLAML (AutoML) based on the task type.
10. Evaluate models using the specified primary metric and log results to MLflow.
11. Identify and save the best pipeline configuration.

## Understanding the Results

- **MLflow experiments**: All runs and their metrics are logged to MLflow. You can view them via the UI.
- **Console outputs**: The script prints intermediate progress and the best model details.
- **Saved artifacts**: The best pipeline recipe and any evaluation artifacts are saved in the `artifacts/` directory.

## Troubleshooting

- **Missing dependencies**: Ensure all packages listed in `requirements.txt` are installed.
- **Config errors**: If the script fails at startup, check your YAML configuration for syntax errors or missing fields.
- **Invalid metrics**: Make sure the primary metric you select is appropriate for the task type.

## Extending the Framework

- **Adding new features**: Implement new functions in the relevant module (e.g., new pre-processing techniques in `feature_engineering.py`). Update the orchestrator to call your function.
- **Custom metrics**: Add custom evaluation metrics by importing functions from `sklearn.metrics` or writing your own. Update `evaluation.py` to include these metrics.
- **Additional data sources**: For ingestion from other sources (e.g., databases, APIs), extend `data_ingestion.py` with new methods.

If you encounter issues or wish to contribute, create an issue in the repository or contact the maintainers.
