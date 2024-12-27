import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from flaml import AutoML
import optuna

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv"
TARGET_COLUMN = "medv"  # Replace with the target column name in your dataset

# Step 1: Data Loading
def load_data(filepath):
    """
    Load dataset from a specified file path.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath.")
        return None

# Step 2: Data Cleaning
def clean_data(df):
    """
    Clean the dataset by handling missing values, duplicate rows, and outliers.
    """
    print("\n--- Data Cleaning ---")

    # Handle missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df = df.fillna(df.median())
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows. Removing...")
        df = df.drop_duplicates()

    print(f"Dataset shape after cleaning: {df.shape}")
    return df

# Step 3: EDA
def perform_eda(df, target_column):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.
    """
    print("\n--- Exploratory Data Analysis ---")
    
    # Summary Statistics
    print("\nDataset Overview:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())

    # Target Variable Distribution
    print("\nTarget Variable Distribution:")
    sns.histplot(data=df, x=target_column, kde=True)
    plt.title("Target Variable Distribution")
    plt.show()

    # Correlation Heatmap
    print("\nCorrelation Heatmap:")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Pairplot for selected features
    print("\nPairplot of Selected Features:")
    selected_features = df.select_dtypes(include=["float64", "int64"]).columns[:5]  # Limit to 5 variables
    sns.pairplot(df[selected_features])
    plt.show()

# Step 4: FLAML Automated Machine Learning
def run_flaml_workflow(X_train, y_train, X_test, y_test):
    """
    Use FLAML to perform automated model selection and training.
    """
    print("\n--- FLAML Workflow ---")

    automl = AutoML()

    # Fit the AutoML model
    print("\nTraining with FLAML...")
    automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=300)

    # Best model details
    print("\nBest Model Details:")
    print(f"Best Model: {automl.best_estimator}")
    print(f"Best Config: {automl.best_config}")
    print(f"Best Loss: {automl.best_loss}")

    # Evaluate on the test set
    predictions = automl.predict(X_test)
    print("\nModel Evaluation:")
    print(f"RMSE: {mean_squared_error(y_test, predictions, squared=False):.4f}")
    print(f"R^2: {r2_score(y_test, predictions):.4f}")

    return automl

# Step 5: Hyperparameter Optimization
def optimize_hyperparameters(X_train, y_train):
    """
    Optimize hyper-parameters using Optuna.
    """
    print("\n--- Hyper-parameter Optimization with Optuna ---")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        }
        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=300, estimator_list=['lgbm'], **params)
        return automl.best_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("\nBest Hyper-parameters:")
    print(study.best_params)
    return study.best_params

# Main Workflow
def main():
    df = load_data(DATA_PATH)
    if df is not None:
        df = clean_data(df)
        perform_eda(df, TARGET_COLUMN)

        # Prepare data for modeling
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Run FLAML workflow
        automl = run_flaml_workflow(X_train, y_train, X_test, y_test)

        # Hyperparameter optimization
        best_params = optimize_hyperparameters(X_train, y_train)

if __name__ == "__main__":
    main()
