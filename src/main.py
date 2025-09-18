
"""
Orchestrates the entire MLOps pipeline.

Loads configuration, runs data ingestion, validation, cleaning, feature engineering, AutoML training, evaluation,
and logs results with MLflow.
"""

import argparse
import logging
import os
import json

import pandas as pd
import mlflow

from config_loader import load_config
from data_ingestion import load_data
from data_validation import validate_schema
from data_cleaning import clean_data
from imbalance_handling import detect_imbalance, apply_smote
from feature_engineering import engineer_features
from model_training import train_models
from evaluation import rank_models
from mlflow_utils import init_mlflow_experiment, log_params


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_pipeline(config_path: str) -> None:
    """Run the entire MLOps pipeline.

    Args:
        config_path (str): Path to YAML configuration file.
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize MLflow
    experiment_name = f"{config['industry']}_{config['task_type']}_{config['primary_metric']}"
    init_mlflow_experiment(experiment_name, config['mlflow_tracking_uri'])

    with mlflow.start_run():
        log_params(config)

        # Data ingestion
        df, dataset_name = load_data(config['dataset_path'])
        logging.info(f"Dataset '{dataset_name}' loaded. Shape: {df.shape}")

        # Initial validation
        try:
            validate_schema(df)
            logging.info('Initial data validation passed.')
        except Exception as e:
            logging.error(f'Data validation failed: {e}')
            raise

        # Cleaning
        df_clean = clean_data(df)
        logging.info(f'Data cleaned. Shape: {df_clean.shape}')

        # Validation after cleaning
        try:
            validate_schema(df_clean)
            logging.info('Validation after cleaning passed.')
        except Exception as e:
            logging.error(f'Validation after cleaning failed: {e}')
            raise

        # Identify target column (assumed to be last column)
        target_col = df_clean.columns[-1]

        # Imbalance handling for classification
        if config['task_type'] == 'classification' and config.get('imbalance_handling', False):
            y = df_clean[target_col]
            X = df_clean.drop(columns=[target_col])
            if detect_imbalance(y):
                logging.info('Imbalance detected. Applying SMOTE...')
                X_bal, y_bal = apply_smote(X, y)
                df_clean = pd.concat([X_bal, y_bal], axis=1)
                logging.info(f'After SMOTE, data shape: {df_clean.shape}')
            else:
                logging.info('No imbalance detected.')

        # Feature engineering
        X_transformed, y_transformed, preprocessor = engineer_features(df_clean, target_col, config['task_type'])
        logging.info(f'Feature engineering complete. Transformed shape: {X_transformed.shape}')

        # Model training
        models = train_models(X_transformed, y_transformed, config['task_type'], config['primary_metric'])
        logging.info('Model training completed.')

        # Model evaluation and selection
        best_key, best_model = rank_models(models, config['primary_metric'], config['task_type'])
        logging.info(f'Best model type: {best_key}')

        # Save best recipe
        recipe = {
            'task_type': config['task_type'],
            'primary_metric': config['primary_metric'],
            'feature_pipeline': str(preprocessor),
            'model_library': best_key,
        }
        recipe_dir = 'artifacts'
        os.makedirs(recipe_dir, exist_ok=True)
        recipe_path = os.path.join(recipe_dir, f"best_recipe_{experiment_name}.json")
        with open(recipe_path, 'w') as fp:
            json.dump(recipe, fp, indent=4)

        mlflow.log_artifact(recipe_path)
        logging.info(f'Best recipe saved to {recipe_path}')
        logging.info('Pipeline execution completed.')


def main():
    parser = argparse.ArgumentParser(description='Run the Savvy Minds MLOps pipeline')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == '__main__':
    main()
