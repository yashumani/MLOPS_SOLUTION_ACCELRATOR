# Machine Learning Pipeline Project

This project implements a comprehensive machine learning pipeline that handles various tasks such as classification, clustering, regression, forecasting, recommendation systems, and anomaly detection. The pipeline includes data loading, cleaning, exploratory data analysis (EDA), feature engineering, data splitting, model training, hyperparameter tuning, model evaluation, model deployment, and drift detection. It incorporates multiple AutoML libraries, advanced feature engineering, systematic logging with MLflow, and benchmarking across different libraries.

## Directory Structure

- `src/`: Contains the source code.
  - `__init__.py`: Makes the `src` directory a package.
  - `clear_reports_directory.py`: Clears the reports directory.
  - `clean_data.py`: Cleans the dataset.
  - `config.py`: Loads the configuration.
  - `data_ingestion.py`: Loads the dataset.
  - `data_cleaning.py`: Cleans the dataset.
  - `drift_detection.py`: Detects data drift.
  - `eda.py`: Performs exploratory data analysis.
  - `feature_engineering.py`: Performs feature engineering.
  - `feature_selection.py`: Selects relevant features.
  - `generate_eda_report.py`: Generates EDA reports.
  - `get_logger.py`: Sets up logging.
  - `hyperparameter_tuning.py`: Tunes hyperparameters.
  - `load_data.py`: Loads the dataset.
  - `load_model.py`: Loads the model.
  - `main.py`: Main script to run the pipeline.
  - `model_deployment.py`: Deploys the model.
  - `model_evaluation.py`: Evaluates the model.
  - `split_data.py`: Splits the dataset.
  - `utils.py`: Utility functions.
  - `visualize_model_performance.py`: Visualizes model performance.
  - `explain_model_predictions.py`: Explains model predictions.
- `requirements.txt`: Lists the required packages.
- `README.md`: Provides an overview of the project.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ML_Pipeline
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Configure the project settings in `src/config.yaml`.
2. Run the main application:
   ```
   python src/main.py
   ```

## Features

- **Data Loading**: Load datasets from specified paths.
- **Data Cleaning**: Remove duplicates and impute missing values using advanced techniques.
- **Exploratory Data Analysis**: Generate EDA reports using various tools (Sweetviz, ydata-profiling, D-Tale).
- **Feature Engineering**: Encode categorical variables, normalize numeric variables, and select relevant features.
- **Model Selection**: Select the best model using multiple AutoML libraries (FLAML, PyCaret).
- **Hyperparameter Tuning**: Optimize model parameters using Optuna.
- **Model Evaluation**: Evaluate model performance using metrics like MSE and R² Score, and visualize true vs. predicted values.
- **Model Deployment**: Save and deploy the best model.
- **Data Drift Detection**: Detect data drift using statistical tests or monitoring model performance over time.
- **Logging and Tracking**: Log all runs and experiments using MLflow, and track changes in pipeline parameters and their effects on performance.

## End-to-End Pipeline

### 1. Data Ingestion (`data_ingestion.py`)
- Load the dataset from the specified path.
- Log the data loading process.

### 2. Data Cleaning (`data_cleaning.py`)
- Remove duplicates from the dataset.
- Impute missing values using KNN imputation.
- Log the data cleaning process.

### 3. Feature Engineering (`feature_engineering.py`)
- Encode categorical variables using Label Encoding.
- Normalize numeric variables using StandardScaler.
- Log the feature engineering process.

### 4. Model Selection (`model_selection.py`)
- Select the best model using AutoML libraries (FLAML, PyCaret).
- Log the model selection process.

### 5. Hyperparameter Tuning (`hyperparameter_tuning.py`)
- Optimize model parameters using Optuna.
- Log the hyperparameter tuning process.
- Save performance metrics and predictions to CSV files.

### 6. Model Evaluation (`model_evaluation.py`)
- Evaluate the model's performance using metrics like Mean Squared Error (MSE) and R² Score.
- Generate a scatter plot of true vs. predicted values.
- Log the model evaluation process.

### 7. Model Deployment (`model_deployment.py`)
- Save the best model to a specified path using `joblib`.
- Log the model deployment process.

### 8. Drift Detection (`drift_detection.py`)
- Implement data drift detection logic (e.g., using statistical tests or monitoring model performance over time).
- Log the data drift detection process.

### 9. Exploratory Data Analysis (EDA)
- Generate EDA reports using various tools (Sweetviz, ydata-profiling, D-Tale).
- Perform EDA on performance metrics and predictions files.

### 10. Logging and Tracking
- Log all runs and experiments using MLflow.
- Track changes in pipeline parameters and their effects on performance.

## Conclusion

This comprehensive and modular pipeline handles various machine learning tasks, incorporates multiple AutoML libraries, and supports advanced feature engineering and systematic logging. The pipeline is designed to be problem-agnostic, allowing you to drop in a dataset and a problem type, and run the pipeline from ingestion to deployment automatically.