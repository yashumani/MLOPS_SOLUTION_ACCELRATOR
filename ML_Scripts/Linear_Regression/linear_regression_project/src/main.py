# FILE: /linear_regression_project/src/main.py

import os
import pandas as pd
from clear_reports_directory import clear_reports_directory
from clean_data import clean_data
from load_data import load_data
from eda import perform_eda
from feature_engineering import feature_engineering
from split_data import split_data
from hyperparameter_tuning import hyperparameter_tuning
from visualize_model_performance import visualize_model_performance
from explain_model_predictions import explain_model_predictions
from get_logger import get_logger
from config import config

def main():
    try:
        print("Executing main.py")
        logger = get_logger('main')
        logger.info("Starting the linear regression process...")

        # Load dataset
        df = load_data(config['data_path'])
        logger.info("Dataset loaded successfully.")
        print("Columns after loading data:", df.columns)

        # Clear the Reports directory
        clear_reports_directory(config['reports_path'])

        # Perform EDA before cleaning
        perform_eda(df, title="EDA Before Cleaning")

        # Clean the dataset
        df = clean_data(df)
        logger.info("Dataset cleaned successfully.")
        print("Columns after cleaning data:", df.columns)

        # Perform EDA after cleaning
        perform_eda(df, title="EDA After Cleaning")

        # Feature Engineering
        df = feature_engineering(df, config)
        print("Columns after feature engineering:", df.columns)

        # Split dataset into training and testing subsets
        train_df, test_df = split_data(df, config)
        print("Columns in training data:", train_df.columns)
        print("Columns in testing data:", test_df.columns)

        # Hyperparameter Tuning
        best_trial, best_model = hyperparameter_tuning(train_df, test_df, config)

        # Visualize the performance of the best model
        visualize_model_performance(best_trial, test_df)

        # Explain model predictions
        explain_model_predictions(best_model, test_df)

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