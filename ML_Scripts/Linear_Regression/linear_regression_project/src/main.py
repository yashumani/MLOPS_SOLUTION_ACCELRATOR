# FILE: /linear_regression_project/src/main.py

import os
import sys
import pandas as pd
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from clear_reports_directory import clear_reports_directory
from clean_data import clean_data
from load_data import load_data
from eda import perform_eda
from feature_engineering import feature_engineering
from split_data import split_data
from model_selection import select_model
from hyperparameter_tuning import hyperparameter_tuning
from model_evaluation import evaluate_model
from model_deployment import deploy_model
from drift_detection import detect_drift
from get_logger import get_logger
from config import config

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def main():
    logger = get_logger('main')
    run = None
    try:
        print("Executing main.py")
        logger.info("Starting the machine learning pipeline...")

        # Set up Azure ML workspace
        try:
            ws = Workspace.from_config()
        except Exception as e:
            logger.error(f"Failed to load Azure ML workspace configuration: {e}")
            raise

        # Load the Custom Environment
        env = Environment.get(workspace=ws, name="env_ml_pipeline")

        # Configure the Script to Use the Environment
        script_config = ScriptRunConfig(
            source_directory=os.path.abspath(os.path.dirname(__file__)),
            script="main.py",
            compute_target="computemlpipeline",
            environment=env
        )

        # Submit Experiment
        experiment = Experiment(workspace=ws, name=config['azureml']['experiment_name'])
        run = experiment.submit(script_config)
        run.wait_for_completion(show_output=True)

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

        # Log hyperparameters and metrics
        run.log("best_r2_score", best_trial.value)
        for param, value in best_trial.params.items():
            run.log(param, value)

        # Model evaluation
        X_test = test_df.drop(columns=[config['target_column']])
        y_test = test_df[config['target_column']]
        mse, r2 = evaluate_model(best_model, X_test, y_test)
        logger.info("Model evaluation completed successfully.")
        run.log("mse", mse)
        run.log("r2_score", r2)

        # Model deployment
        deploy_model(best_model)
        logger.info("Model deployment completed successfully.")

        # Data drift detection
        detect_drift(df)
        logger.info("Data drift detection completed successfully.")

        # Perform EDA on performance metrics and predictions
        performance_metrics_path = os.path.join(config['reports_path'], 'performance_metrics.csv')
        predictions_path = os.path.join(config['reports_path'], 'predictions.csv')

        if os.path.exists(performance_metrics_path):
            performance_metrics_df = pd.read_csv(performance_metrics_path)
            perform_eda(performance_metrics_df, title="Performance Metrics EDA")
            run.upload_file(name='performance_metrics.csv', path_or_stream=performance_metrics_path)

        if os.path.exists(predictions_path):
            predictions_df = pd.read_csv(predictions_path)
            perform_eda(predictions_df, title="Predictions EDA")
            run.upload_file(name='predictions.csv', path_or_stream=predictions_path)

        # Print best model performance and parameters
        best_performance_metric = best_trial.value
        best_model_params = best_trial.params
        logger.info(f"Best Model Performance: {best_performance_metric}")
        logger.info(f"Best Model Parameters: {best_model_params}")
        print(f"Best Model Performance: {best_performance_metric}")
        print(f"Best Model Parameters: {best_model_params}")

        # Complete the run
        run.complete()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        if run:
            run.fail()

if __name__ == "__main__":
    main()