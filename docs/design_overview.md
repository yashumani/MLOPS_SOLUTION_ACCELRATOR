
# Design Overview

This document provides a high-level overview of the MLOps solution accelerator's architecture.

## Pipeline Stages

1. **Configuration Input**: A user-defined YAML config file specifies the industry, dataset, task type, and various options. It also includes details for Azure ML integration (if needed).

2. **Data Ingestion**: The system reads the dataset from the specified path or remote location (e.g., Azure ML datastore). The ingestion module registers the dataset in Azure ML, although the main repository does not handle deployment.

3. **Data Validation**: Utilizes Pandera or Great Expectations to ensure the input data meets required schemas (e.g., data types, missing values, allowed ranges). This happens multiple times: after ingestion, after cleaning, and after feature engineering.

4. **Data Cleaning**: Removes duplicates, handles missing values via imputation or removal, and detects outliers. The cleaned data is validated again.

5. **Imbalance Handling** (Classification tasks): Detects imbalanced class distribution; applies SMOTE via imbalanced-learn. The balanced dataset is logged in MLflow.

6. **Feature Engineering**: Includes encoding categorical variables (one-hot or target encoding), scaling numeric columns, and performing feature selection with Boruta. Optionally, polynomial features can be added based on config settings. The engineered dataset is validated again.

7. **Model Training**: Uses AutoML libraries (PyCaret and FLAML) to train and tune models for the given task. PyCaret automatically selects algorithms and hyperparameters, while FLAML provides light-weight hyperparameter optimization. All runs are recorded in MLflow.

8. **Model Evaluation**: Compares candidate models using a primary metric defined in the config (e.g., F1 score for classification, RMSE for regression). Secondary metrics are also recorded. The best pipeline configuration is identified and its recipe is saved.

9. **MLflow Tracking**: All parameters, metrics, models, and artifacts (e.g., plots, validation reports) are logged in MLflow for traceability and reproducibility.

## Extensibility

- **Additional Featurizers**: New feature engineering modules can be added by implementing functions in `feature_engineering.py` and updating the orchestrator in `main.py`.
- **Alternative Imbalance Techniques**: If desired, other imbalance handling methods (e.g., ADASYN) can be integrated in `imbalance_handling.py`.
- **Task Types**: Clustering is supported but limited to algorithms available in PyCaret or FLAML. Extend evaluation metrics accordingly.
- **Azure ML Integration**: The config includes placeholders for Azure ML workspace details. Future versions may connect to Azure ML for training and deployment.

## Modularity & Best Practices

- Each module (ingestion, validation, cleaning, etc.) is self-contained, promoting unit testing and reuse.
- Functions are documented via docstrings specifying inputs, outputs, and behavior.
- The orchestrator in `main.py` ensures that pipeline stages can be executed sequentially or individually (useful for debugging).
- Logging via Python `logging` library captures runtime information; MLflow handles experiment tracking.
