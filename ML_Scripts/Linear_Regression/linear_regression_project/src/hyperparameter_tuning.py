# FILE: /linear_regression_project/src/hyperparameter_tuning.py

from flaml import AutoML
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import optuna
from optuna.samplers import TPESampler
from get_logger import get_logger
from config import config
import pandas as pd
import os
import time

def hyperparameter_tuning(train_df, test_df, config):
    print("Executing hyperparameter_tuning.py")
    logger = get_logger('hyperparameter_tuning')
    logger.info("Starting hyperparameter tuning...")

    print("Columns in training data before tuning:", train_df.columns)
    print("Columns in testing data before tuning:", test_df.columns)

    X_train = train_df.drop(columns=[config['target_column']])
    y_train = train_df[config['target_column']]
    X_test = test_df.drop(columns=[config['target_column']])
    y_true = test_df[config['target_column']]

    performance_metrics = []
    predictions = pd.DataFrame(y_true).reset_index(drop=True)

    def objective(trial):
        model_name = trial.suggest_categorical('model', ['ridge', 'flaml'])
        logger.info(f"Evaluating model: {model_name}")
        start_time = time.time()
        
        if model_name == 'ridge':
            alpha = trial.suggest_float('alpha', 0.01, 10.0)
            logger.info(f"Ridge alpha: {alpha}")
            model = Ridge(alpha=alpha)
        else:
            automl = AutoML()
            automl_settings = {
                "time_budget": config['flaml']['time_budget'],
                "metric": config['flaml']['metric'],
                "task": config['flaml']['task'],
                "log_file_name": 'flaml.log',
                "verbose": 0,
                "estimator_list": config['flaml']['estimator_list'],
                "n_jobs": -1,
                "mem_thres": 0.8,  # Set memory threshold to 80%
                "learner_kwargs": {
                    "lgbm": {
                        "num_leaves": 31,
                        "max_depth": -1,
                        "n_estimators": 100
                    }
                }
            }
            logger.info(f"AutoML settings: {automl_settings}")
            automl.fit(X_train, y_train, **automl_settings)
            model = automl

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_true, y_pred)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"R2 score: {r2}")
        logger.info(f"Time taken: {duration} seconds")

        # Record performance metrics and parameters
        performance_metrics.append({
            'trial': trial.number,
            'model': model_name,
            'alpha': alpha if model_name == 'ridge' else None,
            'r2_score': r2,
            'time_taken': duration,
            'parameters': trial.params
        })

        # Record predictions
        predictions[f'pred_{model_name}_trial_{trial.number}'] = y_pred

        return r2

    # Use TPESampler with a seed for reproducibility
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config['optuna']['n_trials'])
    best_trial = study.best_trial
    logger.info(f"Hyperparameter Tuning Summary:\nBest hyperparameters: {best_trial.params}\nBest R² score: {best_trial.value}")

    # Retrieve the best model
    if best_trial.params['model'] == 'ridge':
        best_model = Ridge(alpha=best_trial.params['alpha'])
        best_model.fit(X_train, y_train)
    else:
        best_model = AutoML()
        best_model.fit(X_train, y_train, **automl_settings)

    # Save performance metrics and parameters to CSV
    performance_metrics_df = pd.DataFrame(performance_metrics)
    performance_metrics_path = os.path.join(config['reports_path'], 'performance_metrics.csv')
    performance_metrics_df.to_csv(performance_metrics_path, index=False)
    logger.info(f"Performance metrics saved to {performance_metrics_path}")

    # Save predictions to CSV
    predictions_path = os.path.join(config['reports_path'], 'predictions.csv')
    predictions.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")

    return best_trial, best_model