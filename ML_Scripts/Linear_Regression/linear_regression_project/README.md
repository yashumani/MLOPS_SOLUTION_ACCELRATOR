# Linear Regression Project

This project performs linear regression using various tools and libraries. It includes data loading, cleaning, exploratory data analysis (EDA), feature engineering, data splitting, model training, hyperparameter tuning, and model evaluation.

## Directory Structure

- `src/`: Contains the source code.
  - `__init__.py`: Makes the `src` directory a package.
  - `clear_reports_directory.py`: Clears the reports directory.
  - `clean_data.py`: Cleans the dataset.
  - `config.py`: Loads the configuration.
  - `domain_based_imputation.py`
  - `eda.py`
  - `feature_engineering.py`
  - `feature_selection.py`
  - `generate_eda_report.py`
  - `get_logger.py`
  - `hyperparameter_tuning.py`
  - `load_data.py`
  - `load_model.py`
  - `main.py`
  - `split_data.py`
  - `visualize_model_performance.py`
  - `explain_model_predictions.py`
- `requirements.txt`
- `README.md`

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd linear_regression_project
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
- **Data Cleaning**: Remove duplicates and impute missing values.
- **Exploratory Data Analysis**: Generate EDA reports using various tools.
- **Feature Engineering**: Normalize data and select relevant features.
- **Model Training**: Train regression models using PyCaret and FLAML.
- **Hyperparameter Tuning**: Optimize model parameters using Optuna.
- **Model Evaluation**: Visualize model performance and explain predictions.


