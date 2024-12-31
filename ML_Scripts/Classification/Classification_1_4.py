import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models, tune_model as clf_tune_model, predict_model as clf_predict_model
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import logging
import os
import optuna
import sweetviz as sv

# Configure logging
log_path = 'C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Classification/Reports/logs.log'
logging.basicConfig(
    filename=log_path, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/datasets/BreastCancer.csv"
TARGET_COLUMN = "Class"  # Replace with the target column name in your dataset
REPORTS_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Classification/Reports"

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
    report.show_html(os.path.join(REPORTS_PATH, f"{title}_report.html"), open_browser=False)
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
        clf_setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False)
        best_model = clf_compare_models()
        tuned_model = clf_tune_model(best_model, optimize="Accuracy")
        predictions = clf_predict_model(tuned_model, data=test_df)
        accuracy = accuracy_score(test_df[TARGET_COLUMN], predictions['prediction_label'])
    else:
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]
        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task="classification", time_budget=300)
        predictions = automl.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
    return accuracy

def hyperparameter_tuning(train_df, test_df):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_df, test_df), n_trials=10)
    return study.best_trial

# Step 6: Model Building using PyCaret and FLAML
def build_model(train_df, test_df, best_trial):
    if best_trial.params["model"] == "pycaret":
        clf_setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False)
        best_model = clf_compare_models()
        tuned_model = clf_tune_model(best_model, optimize="Accuracy")
        predictions = clf_predict_model(tuned_model, data=test_df)
        accuracy = accuracy_score(test_df[TARGET_COLUMN], predictions['prediction_label'])
        return tuned_model, accuracy, "PyCaret"
    else:
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]
        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task="classification", time_budget=300)
        predictions = automl.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return automl, accuracy, "FLAML"

# Function to visualize model performance
def visualize_model_performance(model, test_df, model_name):
    logging.info("Visualizing %s Model Performance", model_name)

    if model_name == "PyCaret":
        predictions = clf_predict_model(model, data=test_df)
        y_true = test_df[TARGET_COLUMN]
        y_pred = predictions['prediction_label']
    else:
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_true = test_df[TARGET_COLUMN]
        y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Model: Confusion Matrix')
    plt.savefig(os.path.join(REPORTS_PATH, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(REPORTS_PATH, f"{model_name}_classification_report.csv"))

# Main Function
def main():
    try:
        print("Starting the classification process...")
        logging.info("Starting the classification process...")
        
        # Load dataset
        print("Step 1: Loading the dataset...")
        df = load_data(DATA_PATH)
        print("Dataset loaded successfully.")
        
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
        
        # Split dataset into training and testing subsets
        print("Step 6: Splitting the dataset into training and testing subsets...")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
        print("Dataset split into training and testing subsets.")
        
        # Hyperparameter Tuning
        print("Step 7: Tuning hyperparameters using Optuna...")
        best_trial = hyperparameter_tuning(train_df, test_df)
        print(f"Hyperparameter tuning completed. Best trial: {best_trial}")
        
        # Build the best model
        print("Step 8: Building the best model...")
        best_model, best_accuracy, best_model_name = build_model(train_df, test_df, best_trial)
        print(f"Best model built using {best_model_name}.")
        
        # Visualize the performance of the best model
        print("Step 9: Visualizing the performance of the best model...")
        visualize_model_performance(best_model, test_df, best_model_name)
        print("Model performance visualized.")
        
        # Log and print the results
        logging.info("Best Model: %s", best_model_name)
        logging.info("Best Model Accuracy: %s", best_accuracy)
        
        print(f"\nWinner: {best_model_name}")
        print(f"Best Model Accuracy: {best_accuracy}")
        
        # Recommendations
        print("\nRecommendations:")
        if best_model_name == "PyCaret":
            print("The PyCaret model performed better based on the accuracy score. It is recommended to use the PyCaret model for classification.")
        else:
            print("The FLAML model performed better based on the accuracy score. It is recommended to use the FLAML model for classification.")
        print("Further tuning and validation can be performed to ensure the robustness of the selected model.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
