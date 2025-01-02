import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, tune_model as reg_tune_model, predict_model as reg_predict_model
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
from scipy.stats import zscore, iqr, shapiro, skew
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import logging
import os
import optuna
import sweetviz as sv
import shutil

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

# Configuration for outlier detection and handling
config = {
    "outlier_detection": {
        "univariate": {
            "zscore_threshold": 3,
            "iqr_multiplier": 1.5
        },
        "multivariate": {
            "mahalanobis_threshold": 3
        },
        "ml_models": {
            "isolation_forest": {"contamination": 0.01},
            "one_class_svm": {"nu": 0.01, "kernel": "rbf"}
        }
    },
    "outlier_handling": {
        "strategy": "remove",  # Options: "remove", "cap", "impute", "separate"
        "imputation_method": "knn"  # Options: "knn", "mean", "median", "regression"
    },
    "industry_specific": {
        "finance": {"retain_anomalies": True},
        "healthcare": {"retain_rare_cases": True, "drop_erroneous": True},
        "retail": {"retain_spikes": True, "drop_inventory_anomalies": True},
        "manufacturing": {"retain_anomalies": True}
    }
}
# Function to clear the Reports directory
def clear_reports_directory(path):
    logging.info("Clearing the Reports directory...")
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f"Failed to delete {file_path}. Reason: {e}")
    logging.info("Reports directory cleared.")

# Step 1: Data Collection
# Step 1: Data Collection
def load_data(path):
    logging.info("Loading dataset...")
    df = pd.read_csv(path)
    logging.info("Dataset loaded successfully.")
    
    # Check if the target column exists
    if TARGET_COLUMN not in df.columns:
        logging.error(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
    
    # Check if there are enough numeric features for regression
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) < 2:  # At least one feature and the target column
        logging.error("Not enough numeric features for regression.")
        raise ValueError("Not enough numeric features for regression.")
    
    # Handle categorical variables
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_features:
        logging.info("Categorical features found: %s", categorical_features)
        df = pd.get_dummies(df, drop_first=True)
        logging.info("Categorical features encoded.")
    
    # Remove constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    if constant_columns:
        logging.info("Constant columns found and removed: %s", constant_columns)
        df = df.drop(columns=constant_columns)
    
    logging.info("Dataset is compatible for regression.")
    print(f"Data Loading Summary:\nTotal rows and columns: {df.shape}\nConstant columns removed: {len(constant_columns)}\nCategorical columns encoded: {len(categorical_features)}")
    return df

# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(df, title="EDA"):
    logging.info(f"--- {title} ---")
    logging.info("Generating Sweetviz report...")
    report = sv.analyze(df)
    report.show_html(os.path.join(REPORTS_PATH, f"{title}_report.html"), open_browser=False)
    logging.info("Sweetviz report generated successfully.")
    
    # Generate a single box plot for numeric columns
    logging.info("Generating box plot for numeric columns...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df[numeric_columns])
        plt.title('Box Plot of Numeric Columns')
        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(REPORTS_PATH, f"{title}_box_plot_numeric_columns.png"))
        plt.close()
        logging.info("Box plot generated successfully.")
    else:
        logging.warning("No numeric columns found in the dataset.")
    
    # Generate pairplot for numeric columns
    logging.info("Generating pairplot...")
    if numeric_columns:
        sns.pairplot(df[numeric_columns])
        plt.savefig(os.path.join(REPORTS_PATH, f"{title}_pairplot.png"))
        plt.close()
        logging.info("Pairplot generated successfully.")
    else:
        logging.warning("No numeric columns found for pairplot.")
    
    # Print summary statistics
    summary_stats = df.describe().T
    print(f"EDA Summary ({title}):\n{summary_stats[['mean', '50%', 'std']].to_string()}\nNumber of visualizations generated: {1 if numeric_columns else 0}")

# Step 3: Data Cleaning
def clean_data(df):
    logging.info("--- Data Cleaning ---")
    logging.info("Initial dataset shape: %s", df.shape)
    initial_shape = df.shape

    # Data Quality Assessment
    logging.info("Performing data quality assessment...")
    # Rule-based checks
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logging.warning("Missing values found in the dataset:\n%s", missing_values)

    # Remove duplicates
    df = df.drop_duplicates()
    logging.info("Shape after removing duplicates: %s", df.shape)
    duplicates_removed = initial_shape[0] - df.shape[0]

    # Data Deduplication with Clustering
    logging.info("Performing data deduplication with clustering...")
    # Example: Using fuzzy matching for text columns
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].astype(str).apply(lambda x: x.encode('utf-8').decode('utf-8'))  # Ensure all text columns are strings
        unique_values = df[col].unique()
        matches = process.extractBests(unique_values, unique_values, scorer=fuzz.token_sort_ratio, score_cutoff=90)
        for match in matches:
            if match[0] != match[1]:
                df[col] = df[col].replace(match[1], match[0])
    logging.info("Shape after deduplication: %s", df.shape)

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    logging.info("Shape after encoding categorical variables: %s", df.shape)

    # Advanced imputation using KNN
    logging.info("Performing advanced imputation using KNN...")
    imputer = KNNImputer()
    df[df.columns] = imputer.fit_transform(df)
    logging.info("Shape after advanced imputation: %s", df.shape)

    # Anomaly Detection
    logging.info("Performing anomaly detection...")
    iso_forest = IsolationForest(**config["outlier_detection"]["ml_models"]["isolation_forest"])
    outliers = iso_forest.fit_predict(df)
    df = df[outliers == 1]
    logging.info("Shape after removing anomalies: %s", df.shape)
    anomalies_removed = initial_shape[0] - df.shape[0]

    # Identify distribution type and clean data accordingly
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_skewness = skew(df[col])
            logging.info(f"Column {col} skewness: {col_skewness}")
            
            if col_skewness > 1:
                logging.info(f"Column {col} is right skewed. Applying log transformation.")
                df[col] = np.log1p(df[col])
            elif col_skewness < -1:
                logging.info(f"Column {col} is left skewed. Applying square root transformation.")
                df[col] = np.sqrt(df[col])
            else:
                logging.info(f"Column {col} is approximately normally distributed.")
            
            # Check for normal distribution and remove outliers using percentiles
            stat, p = shapiro(df[col])
            if p > 0.05:
                logging.info(f"Column {col} is normally distributed (p-value: {p}).")
            else:
                logging.warning(f"Column {col} is not normally distributed (p-value: {p}).")
            
            lower_percentile = df[col].quantile(0.01)
            upper_percentile = df[col].quantile(0.99)
            df = df[(df[col] >= lower_percentile) & (df[col] <= upper_percentile)]
            logging.info("Shape after removing outliers in %s: %s", col, df.shape)

    logging.info("Dataset shape after cleaning: %s", df.shape)
    print(f"Data Cleaning Summary:\nDuplicates removed: {duplicates_removed}\nAnomalies removed: {anomalies_removed}\nMissing values imputed using KNN")
    return df

# Step 4: Feature Engineering
def feature_engineering(df):
    logging.info("--- Feature Engineering ---")
    # Feature Engineering for Normalization
    logging.info("Performing feature engineering for normalization...")
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)
# Step 5: Enhanced Feature Engineering with GridSearchCV
def enhance_feature_engineering_with_gridsearch(df, target_column):
    logging.info("--- Enhanced Feature Engineering with GridSearchCV ---")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('transformer', PowerTransformer()),
        ('pca', PCA()),
        ('model', Ridge())
    ])
    
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler(), None],
        'transformer': [PowerTransformer(), None],
        'pca': [PCA(n_components=0.95), None],
        'model': [Ridge(), RandomForestRegressor()],
        'model__alpha': [0.1, 1.0, 10.0] if isinstance(pipeline.named_steps['model'], Ridge) else [None],
        'model__max_depth': [3, 5, 10] if isinstance(pipeline.named_steps['model'], RandomForestRegressor) else [None]
    }
    
    scoring = {
        'r2': make_scorer(r2_score),
        'rmse': make_scorer(mean_squared_error, squared=False),
        'mae': make_scorer(mean_absolute_error),
        'mape': make_scorer(mean_absolute_percentage_error),
        'explained_variance': make_scorer(explained_variance_score)
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, refit='r2', cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    
    best_pipeline = grid_search.best_estimator_
    metrics_dict = {
        'R² Score': grid_search.best_score_,
        'RMSE': mean_squared_error(y, best_pipeline.predict(X), squared=False),
        'MAE': mean_absolute_error(y, best_pipeline.predict(X)),
        'MAPE': mean_absolute_percentage_error(y, best_pipeline.predict(X)),
        'Explained Variance': explained_variance_score(y, best_pipeline.predict(X))
    }
    
    logging.info("Best parameters: %s", grid_search.best_params_)
    logging.info("Performance metrics: %s", metrics_dict)
    
    print(f"Enhanced Feature Engineering Summary:\nBest parameters: {grid_search.best_params_}\nPerformance metrics: {metrics_dict}")
    
    return best_pipeline, metrics_dict

# Step 4: Feature Engineering
def feature_engineering(df):
    logging.info("--- Feature Engineering ---")
    # Feature Engineering for Normalization
    logging.info("Performing feature engineering for normalization...")
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)
    print("Feature Engineering Summary:\nApplied StandardScaler for normalization.")
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
        automl = AutoML()
        automl.fit(X_train, y_train, task="regression")
        predictions = automl.predict(X_test)
        r2 = r2_score(test_df[TARGET_COLUMN], predictions)
    return r2

def hyperparameter_tuning(train_df, test_df):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_df, test_df), n_trials=10)
    best_trial = study.best_trial
    print(f"Hyperparameter Tuning Summary:\nBest hyperparameters: {best_trial.params}\nBest R² score: {best_trial.value}")
    return best_trial

# Function to visualize model performance
def visualize_model_performance(pycaret_model, flaml_model, test_df):
    logging.info("Visualizing Model Performance")

    # PyCaret Model
    predictions = reg_predict_model(pycaret_model, data=test_df)
    y_true = test_df[TARGET_COLUMN]
    y_pred = predictions['prediction_label']

    # Scatter plot of true vs predicted values for PyCaret
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('PyCaret Model: True vs Predicted Values')
    plt.savefig(os.path.join(REPORTS_PATH, "PyCaret_true_vs_predicted.png"))
    plt.close()

    # Residuals distribution for PyCaret
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('PyCaret Model: Residuals Distribution')
    plt.savefig(os.path.join(REPORTS_PATH, "PyCaret_residuals_distribution.png"))
    plt.close()

    # FLAML Model
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_true = test_df[TARGET_COLUMN]
    y_pred = flaml_model.predict(X_test)

    # Scatter plot of true vs predicted values for FLAML
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('FLAML Model: True vs Predicted Values')
    plt.savefig(os.path.join(REPORTS_PATH, "FLAML_true_vs_predicted.png"))
    plt.close()

    # Residuals distribution for FLAML
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('FLAML Model: Residuals Distribution')
    plt.savefig(os.path.join(REPORTS_PATH, "FLAML_residuals_distribution.png"))
    plt.close()

# Main Function
def main():
    try:
        print("Starting the linear regression process...")
        logging.info("Starting the linear regression process...")
        
        # Load dataset
        print("Step 1: Loading the dataset...")
        df = load_data(DATA_PATH)
        print("Dataset loaded successfully.")
        
        # Clear the Reports directory
        clear_reports_directory(REPORTS_PATH)
        
        # Perform EDA before cleaning
        print("Step 2: Performing EDA before cleaning...")
        perform_eda(df, title="EDA Before Cleaning")
        print("EDA report generated before cleaning.")
        
        # Clean the dataset
        print("Step 3: Cleaning the dataset...")
        df = clean_data(df)
        print("Dataset cleaned successfully.")
        
        # Perform EDA after cleaning
        print("Step 4: Performing EDA after cleaning...")
        perform_eda(df, title="EDA After Cleaning")
        print("EDA report generated after cleaning.")
        
        # Feature Engineering
        print("Step 5: Performing feature engineering...")
        df = feature_engineering(df)
        print("Feature engineering completed.")
        
        # Perform EDA after feature engineering
        print("Step 6: Performing EDA after feature engineering...")
        perform_eda(df, title="EDA After Feature Engineering")
        print("EDA report generated after feature engineering.")
        
        # Split dataset into training and testing subsets
        print("Step 7: Splitting the dataset into training and testing subsets...")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
        print("Dataset split into training and testing subsets.")
        
        # Hyperparameter Tuning
        print("Step 8: Tuning hyperparameters using Optuna...")
        best_trial = hyperparameter_tuning(train_df, test_df)
        print(f"Hyperparameter tuning completed. Best trial: {best_trial}")
        
        visualize_model_performance(tuned_model if best_model_name == "PyCaret" else None, best_model if best_model_name == "FLAML" else None, test_df)
        print("Step 9: Building the best model...")
        if best_trial.params["model"] == "pycaret":
            reg_setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False)
            best_model = reg_compare_models()
            tuned_model = reg_tune_model(best_model, optimize="R2")
            best_model_name = "PyCaret"
        else:
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            automl = AutoML()
            automl.fit(X_train, y_train, task="regression")
            best_model = automl
            best_model_name = "FLAML"
        print(f"Best model built using {best_model_name}.")
        
        # Visualize the performance of the best model
        print("Step 10: Visualizing the performance of the best model...")
        visualize_model_performance(best_model, test_df, best_model_name)
        print("Model performance visualized.")
        
        # Log and print the results
        logging.info("Best Model: %s", best_model_name)
        if best_model_name == "PyCaret":
            predictions = reg_predict_model(tuned_model, data=test_df)
            y_true = test_df[TARGET_COLUMN]
            y_pred = predictions['prediction_label']
        else:
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_true = test_df[TARGET_COLUMN]
            y_pred = best_model.predict(X_test)
        
        best_r2 = r2_score(y_true, y_pred)
        best_rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        logging.info("Best Model R²: %s", best_r2)
        logging.info("Best Model RMSE: %s", best_rmse)
        
        print(f"\nWinner: {best_model_name}")
        print(f"Best Model R²: {best_r2}")
        print(f"Best Model RMSE: {best_rmse}")
        
        # Recommendations
        print("\nRecommendations:")
        if best_model_name == "PyCaret":
            print("The PyCaret model performed better based on the R² score. It is recommended to use the PyCaret model for regression.")
        else:
            print("The FLAML model performed better based on the R² score. It is recommended to use the FLAML model for regression.")
        print("Further tuning and validation can be performed to ensure the robustness of the selected model.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
