# FILE: /linear_regression_project/src/main.py

import os
import pandas as pd
from data_ingestion import load_data
from data_cleaning import clean_data
from feature_engineering import feature_engineering
from model_selection import select_model
from hyperparameter_tuning import hyperparameter_tuning
from model_evaluation import evaluate_model
from model_deployment import deploy_model
from drift_detection import detect_drift
from get_logger import get_logger
from config import config

def main():
    try:
        print("Executing main.py")
        logger = get_logger('main')
        logger.info("Starting the machine learning pipeline...")

        # Load dataset
        df = load_data()
        logger.info("Dataset loaded successfully.")
        print("Columns after loading data:", df.columns)

        # Clean the dataset
        df = clean_data(df)
        logger.info("Dataset cleaned successfully.")
        print("Columns after cleaning data:", df.columns)

        # Perform feature engineering
        df = feature_engineering(df, config)
        logger.info("Feature engineering completed successfully.")
        print("Columns after feature engineering:", df.columns)

        # Split dataset into training and testing subsets
        train_df, test_df = split_data(df, config)
        print("Columns in training data:", train_df.columns)
        print("Columns in testing data:", test_df.columns)

        # Model selection
        best_model = select_model(train_df, config)
        logger.info("Model selection completed successfully.")

        # Hyperparameter tuning
        best_trial, best_model = hyperparameter_tuning(train_df, test_df, config)
        logger.info("Hyperparameter tuning completed successfully.")

        # Model evaluation
        X_test = test_df.drop(columns=[config['target_column']])
        y_test = test_df[config['target_column']]
        mse, r2 = evaluate_model(best_model, X_test, y_test)
        logger.info("Model evaluation completed successfully.")

        # Model deployment
        deploy_model(best_model)
        logger.info("Model deployment completed successfully.")

        # Data drift detection
        detect_drift(df)
        logger.info("Data drift detection completed successfully.")

        # Perform EDA on performance metrics and predictions files
        performance_metrics_path = os.path.join(config['reports_path'], 'performance_metrics.csv')
        predictions_path = os.path.join(config['reports_path'], 'predictions.csv')

        if os.path.exists(performance_metrics_path):
            performance_metrics_df = pd.read_csv(performance_metrics_path)
            perform_eda(performance_metrics_df, title="Performance Metrics EDA")

        if os.path.exists(predictions_path):
            predictions_df = pd.read_csv(predictions_path)
            perform_eda(predictions_df, title="Predictions EDA")

        # Print the best model performance and best performance metric
        best_performance_metric = best_trial.value
        best_model_params = best_trial.params
        logger.info(f"Best Model Performance: {best_performance_metric}")
        logger.info(f"Best Model Parameters: {best_model_params}")
        print(f"Best Model Performance: {best_performance_metric}")
        print(f"Best Model Parameters: {best_model_params}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()