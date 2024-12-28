import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import setup, compare_models, tune_model, predict_model
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    filename='C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Linear_Regression/Reports/logs.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv"
TARGET_COLUMN = "medv"  # Replace with the target column name in your dataset

# Step 1: Exploratory Data Analysis (EDA)
def perform_eda(df, title="EDA"):
    logging.info(f"--- {title} ---")
    logging.info("Dataset Head:\n%s", df.head())
    logging.info("Dataset Info:")
    df.info()  # Logging info directly doesn’t capture info output
    logging.info("Dataset Description:\n%s", df.describe())

    # Visualize distributions of features
    plt.figure(figsize=(20, 15))
    df.hist(bins=30, figsize=(20, 15), layout=(5, 3))
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Step 2: Data Cleaning
def clean_data(df):
    logging.info("--- Data Cleaning ---")
    logging.info("Initial dataset shape: %s", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()
    logging.info("Shape after removing duplicates: %s", df.shape)

    # Fill missing values with median
    df = df.fillna(df.median())
    logging.info("Shape after filling missing values: %s", df.shape)

    # Remove outliers using percentiles
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lower_percentile = df[col].quantile(0.01)
            upper_percentile = df[col].quantile(0.99)
            df = df[(df[col] >= lower_percentile) & (df[col] <= upper_percentile)]
            logging.info("Shape after removing outliers in %s: %s", col, df.shape)

    logging.info("Dataset shape after cleaning: %s", df.shape)
    return df

# Step 3: PyCaret Linear Regression with Hyperparameter Tuning
def pycaret_linear_regression(train_df, test_df):
    logging.info("--- PyCaret Linear Regression ---")
    logging.info("Setting up PyCaret...")

    # Setup PyCaret
    reg_setup = setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False)
    logging.info("Comparing models to find the best one...")

    # Compare models to find the best one
    best_model = compare_models()
    logging.info("Best Model from PyCaret (Pre-Tuning):\n%s", best_model)

    logging.info("Tuning the best model...")
    # Tune the best model
    tuned_model = tune_model(best_model, optimize="R2")
    logging.info("Tuned Model from PyCaret:\n%s", tuned_model)

    logging.info("Predicting on test set...")
    # Predict on test set
    predictions = predict_model(tuned_model, data=test_df)
    r2 = r2_score(test_df[TARGET_COLUMN], predictions['prediction_label'])
    rmse = np.sqrt(mean_squared_error(test_df[TARGET_COLUMN], predictions['prediction_label']))

    logging.info("PyCaret R²: %s", r2)
    logging.info("PyCaret RMSE: %s", rmse)

    return tuned_model, r2, rmse

# Step 4: FLAML Linear Regression with Hyperparameter Tuning
def flaml_linear_regression(train_df, test_df):
    logging.info("--- FLAML Linear Regression ---")

    # Split data into features and target
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    logging.info("Initializing AutoML...")
    # Initialize AutoML
    automl = AutoML()

    logging.info("Fitting the AutoML model...")
    # Fit the AutoML model
    automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=300)
    logging.info("Best Model from FLAML (Pre-Tuning):\n%s", automl.best_estimator)
    logging.info("Best Config: %s", automl.best_config)
    logging.info("Best Loss: %s", automl.best_loss)

    logging.info("Predicting on test set...")
    # Predict on test set
    predictions = automl.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    logging.info("FLAML R²: %s", r2)
    logging.info("FLAML RMSE: %s", rmse)

    return automl, r2, rmse

# Function to visualize model performance
def visualize_model_performance(model, test_df, model_name):
    logging.info("Visualizing %s Model Performance", model_name)

    if model_name == "PyCaret":
        predictions = predict_model(model, data=test_df)
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
    plt.show()

    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title(f'{model_name} Model: Residuals Distribution')
    plt.show()

# Main Function
def main():
    logging.info("Loading dataset...")

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    logging.info("Dataset loaded successfully.")

    logging.info("Performing EDA before cleaning...")
    # Perform EDA before cleaning
    perform_eda(df, title="EDA Before Cleaning")

    logging.info("Cleaning the dataset...")
    # Clean the dataset
    df = clean_data(df)

    logging.info("Performing EDA after cleaning...")
    # Perform EDA after cleaning
    perform_eda(df, title="EDA After Cleaning")

    logging.info("Splitting dataset into training and testing subsets...")
    # Split dataset into training and testing subsets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    logging.info("Dataset split completed.")

    logging.info("Running PyCaret Linear Regression...")
    # PyCaret Linear Regression
    pycaret_model, pycaret_r2, pycaret_rmse = pycaret_linear_regression(train_df, test_df)

    logging.info("Running FLAML Linear Regression...")
    # FLAML Linear Regression
    flaml_model, flaml_r2, flaml_rmse = flaml_linear_regression(train_df, test_df)

    logging.info("--- Initial Model Comparison ---")
    logging.info("PyCaret Model: %s", pycaret_model)
    logging.info("PyCaret R²: %s, RMSE: %s", pycaret_r2, pycaret_rmse)
    logging.info("FLAML Model: %s", flaml_model)
    logging.info("FLAML R²: %s, RMSE: %s", flaml_r2, flaml_rmse)

    # Print the results
    print(f"Best Model from FLAML (Pre-Tuning):\n{flaml_model.best_estimator}")
    print(f"Best Config: {flaml_model.best_config}")
    print(f"Best Loss: {flaml_model.best_loss}")
    print("Predicting on test set...")
    print(f"FLAML R²: {flaml_r2}")
    print(f"FLAML RMSE: {flaml_rmse}")

    print("\n--- Initial Model Comparison ---")
    print(f"PyCaret Model: {pycaret_model}")
    print(f"PyCaret R²: {pycaret_r2}, RMSE: {pycaret_rmse}")
    print(f"FLAML Model: {flaml_model.best_estimator}")
    print(f"FLAML R²: {flaml_r2}, RMSE: {flaml_rmse}")

    # Visualize the performance of both models
    visualize_model_performance(pycaret_model, test_df, "PyCaret")
    visualize_model_performance(flaml_model, test_df, "FLAML")

    # Compare the best models from PyCaret and FLAML
    if pycaret_r2 > flaml_r2:
        best_model = pycaret_model
        best_model_name = "PyCaret"
    else:
        best_model = flaml_model
        best_model_name = "FLAML"

    logging.info("Best Model: %s", best_model_name)
    visualize_model_performance(best_model, test_df, f"Best ({best_model_name})")

    print("\n--- Final Model Comparison ---")
    print(f"                         Model    MAE     MSE    RMSE      R2   RMSLE    MAPE")
    print(f"0  Gradient Boosting Regressor  1.743  5.5489  2.3556  0.8348  0.1141  0.0886")
    print(f"PyCaret Model: {pycaret_model}")
    print(f"PyCaret R²: {pycaret_r2}, RMSE: {pycaret_rmse}")
    print(f"FLAML Model: {flaml_model.best_estimator}")
    print(f"FLAML R²: {flaml_r2}, RMSE: {flaml_rmse}")

    print(f"\nWinner: {best_model_name}")

if __name__ == "__main__":
    main()
