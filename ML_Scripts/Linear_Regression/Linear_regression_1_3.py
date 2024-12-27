import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import setup, compare_models, pull, create_model, tune_model, predict_model
from flaml import AutoML
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

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
def pycaret_linear_regression(train_df, test_df):
    print("\n--- PyCaret Linear Regression ---")
    # Setup PyCaret
    setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False)
    best_model = compare_models()
    print("\nBest Model from PyCaret (Pre-Tuning):")
    print(best_model)
    
    # Tune the best model
    tuned_model = tune_model(best_model, optimize="R2")
    print("\nTuned Model from PyCaret:")
    print(tuned_model)
    
    # Predict on test set
    predictions = predict_model(tuned_model, data=test_df)
    r2 = r2_score(test_df[TARGET_COLUMN], predictions['prediction_label'])
    rmse = np.sqrt(mean_squared_error(test_df[TARGET_COLUMN], predictions['prediction_label']))
    
    print(f"PyCaret R²: {r2}")
    print(f"PyCaret RMSE: {rmse}")
    
    return tuned_model, r2, rmse

# Step 3: FLAML Linear Regression with Hyperparameter Tuning
def flaml_linear_regression(train_df, test_df):
    print("\n--- FLAML Linear Regression ---")
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    
    # Initialize AutoML
    automl = AutoML()
    
    # Fit the AutoML model
    automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=300)
    print("\nBest Model from FLAML (Pre-Tuning):")
    print(automl.best_estimator)
    print(f"Best Config: {automl.best_config}")
    print(f"Best Loss: {automl.best_loss}")
    
    # Predict on test set
    predictions = automl.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"FLAML R²: {r2}")
    print(f"FLAML RMSE: {rmse}")
    
    return automl, r2, rmse

# Main Function
def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    
    # Split dataset into training and testing subsets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    
    # PyCaret Linear Regression
    pycaret_model, pycaret_r2, pycaret_rmse = pycaret_linear_regression(train_df, test_df)
    
    # FLAML Linear Regression
    flaml_model, flaml_r2, flaml_rmse = flaml_linear_regression(train_df, test_df)
    
    # Model Comparison
    print("\n--- Model Comparison ---")
    print(f"PyCaret R²: {pycaret_r2}, RMSE: {pycaret_rmse}")
    print(f"FLAML R²: {flaml_r2}, RMSE: {flaml_rmse}")
    
    if pycaret_r2 > flaml_r2:
        print("\nWinner: PyCaret")
    else:
        print("\nWinner: FLAML")

if __name__ == "__main__":
    main()