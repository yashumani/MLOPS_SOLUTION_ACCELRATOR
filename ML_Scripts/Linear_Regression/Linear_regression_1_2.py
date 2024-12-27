import os
import pandas as pd
import numpy as np
import h2o
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from h2o.automl import H2OAutoML

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/"
TARGET_COLUMN = "target"
DEGREE = 2
REPORTS_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Reports/"
os.makedirs(REPORTS_PATH, exist_ok=True)
h2o.init()

# MLflow setup
mlflow.set_tracking_uri("file:///C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/MLflow_Tracking")
mlflow.set_experiment("AutoML Benchmark")

# Load Dataset
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath.")
        return None

# Data Cleaning
def clean_data(df):
    print("\n--- Data Cleaning ---")
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        df = df.dropna()
    for col in df.select_dtypes(include=[np.number]).columns:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Feature Engineering
def advanced_feature_engineering(X, degree=DEGREE):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly

# Preprocessing
def preprocess_data(df, target_column, degree=DEGREE):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_transformed = advanced_feature_engineering(X, degree=degree)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    return X_scaled, y

# AutoML with H2O
def run_h2o_automl(X_train, y_train, X_test, max_models=10):
    feature_columns = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Create H2OFrame with consistent column names
    train = h2o.H2OFrame(pd.concat([pd.DataFrame(X_train, columns=feature_columns), pd.DataFrame(y_train, columns=[TARGET_COLUMN])], axis=1))
    test = h2o.H2OFrame(pd.DataFrame(X_test, columns=feature_columns))

    # Train AutoML
    aml = H2OAutoML(max_models=max_models, seed=42)
    aml.train(y=TARGET_COLUMN, training_frame=train)

    # Display leaderboard
    leaderboard = aml.leaderboard.as_data_frame()
    print(leaderboard)
    return aml.leader

# Benchmark AutoML Libraries
def benchmark_automl_libraries(df, target_column):
    df = clean_data(df)
    X, y = preprocess_data(df, target_column, degree=DEGREE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = {}
    
    # H2O AutoML
    with mlflow.start_run(run_name="H2O_AutoML"):
        best_model_h2o = run_h2o_automl(X_train, y_train, X_test)
        h2o_test = h2o.H2OFrame(pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_train.shape[1])]))
        y_pred_h2o = best_model_h2o.predict(h2o_test).as_data_frame().values.ravel()
        mae, r2 = mean_absolute_error(y_test, y_pred_h2o), r2_score(y_test, y_pred_h2o)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        results["H2O"] = {"mae": mae, "r2": r2}

    return results

# Main Workflow
def main():
    df = load_data(os.path.join(DATA_PATH, "classification_dataset.csv"))  # Replace with an appropriate dataset
    if df is not None:
        results = benchmark_automl_libraries(df, TARGET_COLUMN)
        print("Benchmark Results:", results)

if __name__ == "__main__":
    main()
