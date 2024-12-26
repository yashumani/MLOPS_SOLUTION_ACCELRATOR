import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, classification_report, mean_absolute_error
from h2o.automl import H2OAutoML
import h2o
from fastapi import FastAPI
import pickle
from scipy.stats import ks_2samp
from kaggle.api.kaggle_api_extended import KaggleApi
import datasets
import mlflow
import mlflow.sklearn
import mlflow.h2o

# Run the script and check the output. The script will load the Boston Housing dataset, preprocess the data, run an H2O AutoML model, evaluate the model, and monitor data drift. The data drift report will be saved in the Reports folder.
# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/"
TARGET_COLUMN = "medv"
DEGREE = 2
ALPHA = 1.0
REPORTS_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Linear_Regression/Reports/"
os.makedirs(REPORTS_PATH, exist_ok=True)
h2o.init()

# Fetch Datasets
def fetch_kaggle_dataset(dataset_name, save_path=DATA_PATH):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=save_path, unzip=True)
    print(f"Dataset {dataset_name} downloaded to {save_path}.")

def fetch_huggingface_dataset(dataset_name):
    dataset = datasets.load_dataset(dataset_name)
    print(f"Dataset {dataset_name} loaded.")
    return dataset

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
    if (missing_values > 0):
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
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_poly)
    return X_reduced

# Preprocessing
def preprocess_data(df, target_column, degree=DEGREE):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_reduced = advanced_feature_engineering(X, degree=degree)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    return X_scaled, y

# AutoML with H2O
def run_h2o_automl(X_train, y_train, X_test, max_models=10):
    train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
    test = h2o.H2OFrame(pd.DataFrame(X_test, columns=train.columns[:-1]))
    aml = H2OAutoML(max_models=max_models, seed=42)
    aml.train(y=TARGET_COLUMN, training_frame=train)
    leaderboard = aml.leaderboard.as_data_frame()
    print(leaderboard)
    return aml.leader

# Evaluation
def evaluate_model(y_true, y_pred, problem_type="regression"):
    if problem_type == "regression":
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"MAE: {mae:.4f}, R²: {r2:.4f}")
        return mae, r2
    elif problem_type == "classification":
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))
        return report

# Drift Detection
def monitor_data_drift(reference_data, current_data):
    drift_report = {}
    for column in reference_data.columns:
        stat, p_value = ks_2samp(reference_data[column], current_data[column])
        drift_report[column] = {"statistic": stat, "p_value": p_value}
    with open(os.path.join(REPORTS_PATH, "data_drift_report.txt"), "w") as f:
        for column, metrics in drift_report.items():
            f.write(f"{column}: {metrics}\n")
    print("Data drift report saved.")

app = FastAPI()

@app.post("/predict/")
def predict(data: dict):
    model = pickle.load(open("best_model.pkl", "rb"))
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}

# Main Workflow
def main():
    mlflow.set_experiment("Boston Housing Experiment")
    with mlflow.start_run():
        df = load_data(os.path.join(DATA_PATH, "BostonHousing.csv"))
        if df is not None:
            if TARGET_COLUMN not in df.columns:
                print(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
                return
            df = clean_data(df)
            X, y = preprocess_data(df, TARGET_COLUMN, degree=DEGREE)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Log parameters
            mlflow.log_param("degree", DEGREE)
            mlflow.log_param("alpha", ALPHA)

            # Run H2O AutoML
            best_model = run_h2o_automl(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test))

            # Evaluate the model
            y_pred = best_model.predict(h2o.H2OFrame(X_test)).as_data_frame().values.ravel()
            mae, r2 = evaluate_model(y_test, y_pred, problem_type="regression")

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log model
            mlflow.h2o.log_model(best_model, "model")

            # Monitor Data Drift
            monitor_data_drift(pd.DataFrame(X_train), pd.DataFrame(X_test))

if __name__ == "__main__":
    main()
