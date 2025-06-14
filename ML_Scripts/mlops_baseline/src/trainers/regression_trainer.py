# src/trainers/regression_trainer.py

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import mlflow
import logging
import re
import inspect
from typing import Dict, Any, List, Optional

# Assuming plotting_utils.py is in src/utils/
from ..utils import plotting_utils 

logger = logging.getLogger(__name__)

# --- Helper Function ---
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes column names for compatibility."""
    df_copy = df.copy()
    sanitized_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in df_copy.columns]
    seen = {}; final_cols = []
    for col in sanitized_cols:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)
    df_copy.columns = final_cols
    return df_copy

class NoOpLabelEncoder:
    """A dummy label encoder for regression tasks to maintain a consistent artifact structure."""
    def fit(self, y): return self
    def transform(self, y): return y
    def inverse_transform(self, y): return y
    @property
    def classes_(self): return None

# --- Optuna Space Functions for Regression ---
def get_ridge_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_float("ridge_alpha", 1e-3, 1e2, log=True)
    return {}

def get_elasticnet_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_float("elasticnet_alpha", 1e-3, 1e2, log=True)
    trial.suggest_float("elasticnet_l1_ratio", 0.01, 0.99)
    return {}

def get_svr_space(trial: optuna.trial.Trial) -> dict:
    kernel = trial.suggest_categorical("svr_kernel", ["linear", "rbf", "poly", "sigmoid"])
    trial.suggest_float("svr_C", 1e-2, 1e3, log=True)
    trial.suggest_float("svr_epsilon", 1e-2, 1.0, log=True)
    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma_choice = trial.suggest_categorical("svr_gamma_choice", ["scale", "auto", "specific"])
        if gamma_choice == "specific":
            trial.suggest_float("svr_gamma_specific", 1e-4, 1e-1, log=True)
    if kernel == "poly":
        trial.suggest_int("svr_degree", 2, 5)
    return {}

def get_random_forest_regressor_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("rfr_n_estimators", 50, 300, step=50)
    if trial.suggest_categorical("rfr_use_max_depth", [True, False]):
        trial.suggest_int("rfr_max_depth", 3, 20, log=True)
    trial.suggest_int("rfr_min_samples_split", 2, 20)
    trial.suggest_int("rfr_min_samples_leaf", 1, 20)
    return {}

def get_xgb_regressor_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("xgb_n_estimators", 50, 300)
    trial.suggest_int("xgb_max_depth", 2, 10)
    trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
    trial.suggest_float("xgb_subsample", 0.6, 1.0)
    trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0)
    trial.suggest_float("xgb_gamma", 0, 5)
    return {}

def get_lgbm_regressor_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("lgbm_n_estimators", 50, 500)
    trial.suggest_float("lgbm_learning_rate", 0.01, 0.2, log=True)
    trial.suggest_int("lgbm_num_leaves", 20, 150)
    trial.suggest_int("lgbm_max_depth", -1, 15)
    trial.suggest_float("lgbm_subsample", 0.6, 1.0)
    trial.suggest_float("lgbm_colsample_bytree", 0.6, 1.0)
    return {}

def get_catboost_regressor_space(trial: optuna.trial.Trial) -> dict:
     trial.suggest_int("cat_iterations", 50, 700)
     trial.suggest_float("cat_learning_rate", 0.01, 0.3, log=True)
     trial.suggest_int("cat_depth", 3, 10)
     trial.suggest_float("cat_l2_leaf_reg", 1e-3, 10.0, log=True)
     return {}

def get_knn_regressor_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("knn_n_neighbors", 3, 30)
    trial.suggest_categorical("knn_weights", ["uniform", "distance"])
    trial.suggest_int("knn_p", 1, 2)
    return {}

MODEL_CONFIGS_REGRESSION = {
    "Ridge": {"model_class": Ridge, "space_func": get_ridge_space, "fixed_params": {}, "requires_scaling": True},
    "ElasticNet": {"model_class": ElasticNet, "space_func": get_elasticnet_space, "fixed_params": {"max_iter": 2000}, "requires_scaling": True},
    "SVR": {"model_class": SVR, "space_func": get_svr_space, "fixed_params": {}, "requires_scaling": True},
    "RandomForestRegressor": {"model_class": RandomForestRegressor, "space_func": get_random_forest_regressor_space, "fixed_params": {}, "requires_scaling": False},
    "XGBRegressor": {"model_class": XGBRegressor, "space_func": get_xgb_regressor_space, "fixed_params": {"objective": "reg:squarederror"}, "requires_scaling": False},
    "LGBMRegressor": {"model_class": LGBMRegressor, "space_func": get_lgbm_regressor_space, "fixed_params": {"verbose": -1}, "requires_scaling": False},
    "CatBoostRegressor": {"model_class": CatBoostRegressor, "space_func": get_catboost_regressor_space, "fixed_params": {"verbose": 0}, "requires_scaling": False},
    "KNeighborsRegressor": {"model_class": KNeighborsRegressor, "space_func": get_knn_regressor_space, "fixed_params": {}, "requires_scaling": True}
}

METRIC_DIRECTIONS = {
    "rmse": "minimize",
    "mae": "minimize",
    "r2": "maximize"
}

def _prepare_model_params(optuna_trial_params: dict, base_fixed_params: dict, model_alias: str, rnd_state_global: int, model_class_ref: type) -> dict:
    """Prepares model parameters by merging Optuna suggestions with fixed params and handling model-specific logic."""
    params = base_fixed_params.copy()
    
    param_map = {
        "Ridge": {"ridge_alpha": "alpha"},
        "ElasticNet": {"elasticnet_alpha": "alpha", "elasticnet_l1_ratio": "l1_ratio"},
        "SVR": {"svr_C": "C", "svr_epsilon": "epsilon", "svr_kernel": "kernel", "svr_gamma_specific": "gamma", "svr_degree": "degree"},
        "RandomForestRegressor": {"rfr_n_estimators": "n_estimators", "rfr_max_depth": "max_depth", "rfr_min_samples_split": "min_samples_split", "rfr_min_samples_leaf": "min_samples_leaf"},
        "XGBRegressor": {"xgb_n_estimators": "n_estimators", "xgb_max_depth": "max_depth", "xgb_learning_rate": "learning_rate", "xgb_subsample": "subsample", "xgb_colsample_bytree": "colsample_bytree", "xgb_gamma": "gamma"},
        "LGBMRegressor": {"lgbm_n_estimators": "n_estimators", "lgbm_learning_rate": "learning_rate", "lgbm_num_leaves": "num_leaves", "lgbm_max_depth": "max_depth", "lgbm_subsample": "subsample", "lgbm_colsample_bytree": "colsample_bytree"},
        "CatBoostRegressor": {"cat_iterations": "iterations", "cat_learning_rate": "learning_rate", "cat_depth": "depth", "cat_l2_leaf_reg": "l2_leaf_reg"},
        "KNeighborsRegressor": {"knn_n_neighbors": "n_neighbors", "knn_weights": "weights", "knn_p": "p"}
    }
    
    current_map = param_map.get(model_alias, {})
    for optuna_key, model_key in current_map.items():
        if optuna_key in optuna_trial_params:
            params[model_key] = optuna_trial_params[optuna_key]
    
    if model_alias == "SVR" and "svr_gamma_choice" in optuna_trial_params:
        gamma_choice_val = optuna_trial_params["svr_gamma_choice"]
        if gamma_choice_val != "specific":
            params["gamma"] = gamma_choice_val
            
    if model_alias == "RandomForestRegressor" and not optuna_trial_params.get("rfr_use_max_depth", True):
        params["max_depth"] = None
    
    try:
        if "random_state" in inspect.signature(model_class_ref).parameters and "random_state" not in params:
            params["random_state"] = rnd_state_global
    except Exception as e:
        logger.debug(f"Could not dynamically add random_state for {model_alias}: {e}")
        
    return params

def train_regression_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model_alias: str, model_config: dict, n_trials_optuna: int,
    artifacts_path: str, rnd_state: int,
    primary_metric: str
):
    """Trains, tunes, and evaluates a single regression model."""
    logger.info(f"Processing model: {model_alias}")
    model_class = model_config["model_class"]
    space_func = model_config["space_func"]
    base_fixed_params = model_config.get("fixed_params", {}).copy()
    requires_scaling = model_config.get("requires_scaling", False)

    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    scaler = None
    
    # FIX: Only scale numerical columns to avoid dtype errors with boolean/OHE columns
    if requires_scaling:
        logger.info(f"Applying StandardScaler for {model_alias}...")
        scaler = StandardScaler()
        numerical_cols_to_scale = X_train_processed.select_dtypes(include=np.number).columns.tolist()
        
        if numerical_cols_to_scale:
            X_train_processed[numerical_cols_to_scale] = scaler.fit_transform(X_train_processed[numerical_cols_to_scale])
            X_test_processed[numerical_cols_to_scale] = scaler.transform(X_test_processed[numerical_cols_to_scale])
            logger.info(f"Scaling complete for {len(numerical_cols_to_scale)} numerical columns.")
        else:
            logger.warning(f"Scaling required for {model_alias}, but no numerical columns were found to scale.")
    
    feature_names_for_model = list(X_train_processed.columns)
    if model_alias in ["LGBMRegressor", "CatBoostRegressor"]:
        logger.info(f"Sanitizing feature names for {model_alias}...")
        X_train_processed = sanitize_feature_names(X_train_processed)
        X_test_processed = sanitize_feature_names(X_test_processed)
        feature_names_for_model = list(X_train_processed.columns)

    def objective(trial):
        space_func(trial)
        optuna_trial_params = trial.params
        params_for_model = _prepare_model_params(optuna_trial_params, base_fixed_params, model_alias, rnd_state, model_class)
        model = model_class(**params_for_model)
        
        X_trial_train, X_trial_val, y_trial_train, y_trial_val = train_test_split(
            X_train_processed, y_train, test_size=0.25, random_state=rnd_state
        )
        try:
            model.fit(X_trial_train, y_trial_train)
            preds = model.predict(X_trial_val)
            
            # Calculate score based on the selected primary_metric
            if primary_metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_trial_val, preds))
            elif primary_metric == 'mae':
                score = mean_absolute_error(y_trial_val, preds)
            elif primary_metric == 'r2':
                score = r2_score(y_trial_val, preds)
            else:
                score = np.sqrt(mean_squared_error(y_trial_val, preds)) # Default to RMSE

            if METRIC_DIRECTIONS.get(primary_metric) == "minimize":
                return -score
                
        except Exception as e:
            logger.warning(f"Optuna trial failed for {model_alias}: {e}")
            return float('-inf') if METRIC_DIRECTIONS.get(primary_metric) == "maximize" else float('inf')
        return score

    study_direction = METRIC_DIRECTIONS.get(primary_metric, "maximize")
    study = optuna.create_study(direction=study_direction, study_name=f"Optuna_{model_alias}_{primary_metric}")
    with mlflow.start_run(run_name=f"Optuna_{model_alias}_Regression", nested=True):
        study.optimize(objective, n_trials=n_trials_optuna, n_jobs=1, gc_after_trial=True)
        best_params_raw = study.best_trial.params
        best_value = study.best_value
        
        score_display = f"{best_value:.4f}" if isinstance(best_value, (int, float)) and np.isfinite(best_value) else str(best_value)
        logger.info(f"Best {primary_metric} for {model_alias} from Optuna: {score_display}")
        mlflow.log_params({f"best_optuna_{k}": v for k, v in best_params_raw.items()})
        if best_value is not None and np.isfinite(best_value):
            mlflow.log_metric(f"best_optuna_{primary_metric}_val", best_value)

    final_model_hyperparams = _prepare_model_params(best_params_raw, base_fixed_params, model_alias, rnd_state, model_class)
    final_model = model_class(**final_model_hyperparams)
    logger.info(f"Retraining {model_alias} with best parameters...")
    final_model.fit(X_train_processed, y_train)
    
    y_pred_test = final_model.predict(X_test_processed)
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "r2": r2_score(y_test, y_pred_test)
    }
    logger.info(f"Final metrics for {model_alias} on test set: {metrics}")
    
    return final_model, scaler, metrics, best_params_raw, study, X_train_processed, y_pred_test


def run_regression_pipeline(
    X_train_df: pd.DataFrame, y_train_series: pd.Series, X_test_df: pd.DataFrame, y_test_series: pd.Series, 
    n_trials_optuna: int, artifacts_path_base: str, rnd_state_global: int, primary_metric: str
):
    """Main function to run the regression training pipeline for all configured models."""
    logger.info("--- Starting Regression Training Pipeline ---")
    
    le = NoOpLabelEncoder()
    label_encoder_path = os.path.join(artifacts_path_base, "label_encoder_regression.joblib")
    joblib.dump(le, label_encoder_path)
    logger.info(f"NoOpLabelEncoder saved to {label_encoder_path}")
    if mlflow.active_run():
        mlflow.log_artifact(label_encoder_path, "label_encoder")

    all_model_results = {}
    for model_alias, config in MODEL_CONFIGS_REGRESSION.items():
        logger.info(f"\n--- Processing model: {model_alias} (regression) ---")
        
        with mlflow.start_run(run_name=f"Train_{model_alias}_Regression", nested=True) as child_run:
            mlflow.set_tag("model_type", model_alias); mlflow.set_tag("task_type", "regression")
            mlflow.log_param("primary_optimization_metric", primary_metric)
            
            final_model, fitted_scaler, metrics, best_hyperparams, optuna_study, \
            X_train_model_input, y_pred_test_final = train_regression_model(
                X_train_df, y_train_series, X_test_df, y_test_series, model_alias, 
                config, n_trials_optuna, artifacts_path_base, rnd_state_global, primary_metric
            )
            
            run_id = child_run.info.run_id
            all_model_results[model_alias] = {"model": final_model, "scaler": fitted_scaler, "metrics": metrics, "params": best_hyperparams, "mlflow_run_id": run_id}
            
            if final_model:
                mlflow.log_params(best_hyperparams)
                mlflow.log_metrics(metrics)
                
                joblib.dump(final_model, os.path.join(artifacts_path_base, f"{model_alias.lower()}_regression_model.joblib"))
                if fitted_scaler:
                    joblib.dump(fitted_scaler, os.path.join(artifacts_path_base, f"{model_alias.lower()}_regression_scaler.joblib"))
                    mlflow.log_artifact(os.path.join(artifacts_path_base, f"{model_alias.lower()}_regression_scaler.joblib"), "scaler")
                
                plotting_utils.log_feature_importance_plot(final_model, X_train_model_input.columns.tolist(), model_alias, "regression", artifacts_path_base)
                plotting_utils.log_actual_vs_predicted_plot(y_test_series, y_pred_test_final, model_alias, artifacts_path_base)
                plotting_utils.log_residuals_plot(y_test_series, y_pred_test_final, model_alias, artifacts_path_base)
                if optuna_study:
                    plotting_utils.log_optuna_visualizations(optuna_study, model_alias, "regression", artifacts_path_base)
                
                # ... (MLflow model logging with signature would go here) ...
    
    return all_model_results