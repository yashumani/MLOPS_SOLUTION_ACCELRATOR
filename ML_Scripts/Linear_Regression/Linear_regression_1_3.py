import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import setup, compare_models, pull, create_model, tune_model
from flaml import AutoML
import optuna

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv"
TARGET_COLUMN = "medv"  # Replace with the target column name in your dataset

# Step 1: Data Cleaning
def clean_data(df):
    print("\n--- Data Cleaning ---")
    # Remove duplicates
    df = df.drop_duplicates()
    # Fill missing values
    df = df.fillna(df.median())
    
    # Remove outliers using percentiles
    for col in df.columns:
        lower_percentile = df[col].quantile(0.01)
        upper_percentile = df[col].quantile(0.99)
        df = df[(df[col] >= lower_percentile) & (df[col] <= upper_percentile)]
    
    print(f"Dataset shape after cleaning: {df.shape}")
    return df

# Step 2: PyCaret Linear Regression with Hyperparameter Tuning
def pycaret_linear_regression(df):
    print("\n--- PyCaret Linear Regression ---")
    # Setup PyCaret
    setup(data=df, target=TARGET_COLUMN, session_id=123, verbose=False)
    best_model = compare_models()
    print("\nBest Model from PyCaret (Pre-Tuning):")
    print(best_model)
    
    # Tune the best model
    tuned_model = tune_model(best_model, optimize="R2")
    print("\nTuned Model from PyCaret:")
    print(tuned_model)
    
    return tuned_model

# Step 3: FLAML Linear Regression with Hyperparameter Tuning
def flaml_linear_regression(df):
    print("\n--- FLAML Linear Regression ---")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Initialize AutoML
    automl = AutoML()
    
    # Fit the AutoML model
    automl.fit(X_train=X, y_train=y, task="regression", time_budget=300)
    print("\nBest Model from FLAML (Pre-Tuning):")
    print(automl.best_estimator)
    print(f"Best Config: {automl.best_config}")
    print(f"Best Loss: {automl.best_loss}")
    
    # Hyperparameter Optimization with Optuna
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        
        # Fit FLAML with custom parameters
        automl.fit(X_train=X, y_train=y, task="regression", estimator_list=["lgbm"], time_budget=30, **params)
        return automl.best_loss
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print("\nBest Parameters from Optuna:")
    print(study.best_params)
    return automl

# Main Function
def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    
    # PyCaret Linear Regression
    pycaret_model = pycaret_linear_regression(df)
    
    # FLAML Linear Regression
    flaml_model = flaml_linear_regression(df)

if __name__ == "__main__":
    main()
