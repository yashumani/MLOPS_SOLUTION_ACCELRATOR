import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, tune_model as reg_tune_model, predict_model as reg_predict_model
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import logging
import os
import optuna
import sweetviz as sv

# Configure logging
log_path = 'C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Linear_Regression/Reports/logs.log'
logging.basicConfig(
    filename=log_path, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/datasets/College.csv"
TARGET_COLUMN = "Grad.Rate"  # Replace with the target column name in your dataset
REPORTS_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Linear_Regression/Reports"

# Ensure the reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)

# Step 1: Data Collection
def load_data(path):
    logging.info("Loading dataset...")
    df = pd.read_csv(path)
    logging.info("Dataset loaded successfully.")
    return df

# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(df, title="EDA"):
    logging.info(f"--- {title} ---")
    logging.info("Generating Sweetviz report...")
    report = sv.analyze(df)
    report.show_html(os.path.join(REPORTS_PATH, f"{title}_report.html"))
    logging.info("Sweetviz report generated successfully.")

# Step 3: Data Cleaning
def clean_data(df):
    logging.info("--- Data Cleaning ---")
    logging.info("Initial dataset shape: %s", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()
    logging.info("Shape after removing duplicates: %s", df.shape)

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    logging.info("Shape after encoding categorical variables: %s", df.shape)

    # Advanced imputation using MICE
    imputer = IterativeImputer()
    df[df.columns] = imputer.fit_transform(df)
    logging.info("Shape after advanced imputation: %s", df.shape)

    # Remove outliers using percentiles
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lower_percentile = df[col].quantile(0.01)
            upper_percentile = df[col].quantile(0.99)
            df = df[(df[col] >= lower_percentile) & (df[col] <= upper_percentile)]
            logging.info("Shape after removing outliers in %s: %s", col, df.shape)

    logging.info("Dataset shape after cleaning: %s", df.shape)
    return df

# Step 4: Feature Engineering
def feature_engineering(df):
    logging.info("--- Feature Engineering ---")
    # Add your feature engineering steps here
    return df

# Step 5: Hyperparameter Tuning using Optuna
def objective(trial, train_df, test_df):
    model_name = trial.suggest_categorical("model", ["pycaret", "flaml"])
    if model_name == "pycaret":
        reg_setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False)
        best_model = reg_compare_models()
        tuned_model = reg_tune_model(best_model, optimize="R2")
        predictions = reg_predict_model(tuned_model, data=test_df)
        r2 = r2_score(test_df[TARGET_COLUMN], predictions['prediction_label'])
    else:
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]
        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=300)
        predictions = automl.predict(X_test)
        r2 = r2_score(y_test, predictions)
    return r2

def hyperparameter_tuning(train_df, test_df):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_df, test_df), n_trials=10)
    return study.best_trial

# Step 6: Model Building using PyCaret and FLAML
def build_model(train_df, test_df, best_trial):
    if best_trial.params["model"] == "pycaret":
        reg_setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False)
        best_model = reg_compare_models()
        tuned_model = reg_tune_model(best_model, optimize="R2")
        predictions = reg_predict_model(tuned_model, data=test_df)
        r2 = r2_score(test_df[TARGET_COLUMN], predictions['prediction_label'])
        rmse = np.sqrt(mean_squared_error(test_df[TARGET_COLUMN], predictions['prediction_label']))
        return tuned_model, r2, rmse, "PyCaret"
    else:
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]
        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=300)
        predictions = automl.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return automl, r2, rmse, "FLAML"

# Function to visualize model performance
def visualize_model_performance(model, test_df, model_name):
    logging.info("Visualizing %s Model Performance", model_name)

    if model_name == "PyCaret":
        predictions = reg_predict_model(model, data=test_df)
        y_true = test_df[TARGET_COLUMN]
        y_pred = predictions['prediction_label']
    else:
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_true = test_df[TARGET_COLUMN]
        y_pred = model.predict(X_test)

    # Scatter plot of true vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} Model: True vs Predicted Values')
    plt.savefig(os.path.join(REPORTS_PATH, f"{model_name}_true_vs_predicted.png"))
    plt.close()

    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title(f'{model_name} Model: Residuals Distribution')
    plt.savefig(os.path.join(REPORTS_PATH, f"{model_name}_residuals_distribution.png"))
    plt.close()

# Main Function
def main():
    # Load dataset
    df = load_data(DATA_PATH)

    # Perform EDA before cleaning
    perform_eda(df, title="EDA Before Cleaning")

    # Clean the dataset
    df = clean_data(df)

    # Perform EDA after cleaning
    perform_eda(df, title="EDA After Cleaning")

    # Feature Engineering
    df = feature_engineering(df)

    # Split dataset into training and testing subsets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)

    # Hyperparameter Tuning
    best_trial = hyperparameter_tuning(train_df, test_df)

    # Build the best model
    best_model, best_r2, best_rmse, best_model_name = build_model(train_df, test_df, best_trial)

    # Visualize the performance of the best model
    visualize_model_performance(best_model, test_df, best_model_name)

    # Log and print the results
    logging.info("Best Model: %s", best_model_name)
    logging.info("Best Model R²: %s", best_r2)
    logging.info("Best Model RMSE: %s", best_rmse)

    print(f"\nWinner: {best_model_name}")
    print(f"Best Model R²: {best_r2}")
    print(f"Best Model RMSE: {best_rmse}")

if __name__ == "__main__":
    main()
