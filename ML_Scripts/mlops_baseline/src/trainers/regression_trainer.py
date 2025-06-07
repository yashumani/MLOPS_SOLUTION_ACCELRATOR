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
import inspect # For robust random_state checking

# Assuming plotting_utils.py is in src/utils/
from ..utils import plotting_utils 

logger = logging.getLogger(__name__)

# --- Helper Function (Consider moving to a common utils.py) ---
def sanitize_feature_names(df_in: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes column names for LightGBM and CatBoost compatibility."""
    df = df_in.copy()
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    df.columns = [re.sub(r"_+", "_", col) for col in df.columns]
    df.columns = [col.strip("_") for col in df.columns]
    cols = pd.Series(df.columns)
    if cols.duplicated().any():
        logger.warning(f"Duplicate column names found after sanitization, attempting to rename: {cols[cols.duplicated()].unique().tolist()}")
        new_names = {}
        new_columns_list = []
        for item in df.columns: 
            col_str = str(item) 
            if col_str in new_names:
                new_names[col_str] += 1
                new_columns_list.append(f"{col_str}_{new_names[col_str]}")
            else:
                new_names[col_str] = 0
                new_columns_list.append(col_str)
        df.columns = new_columns_list
    return df

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
        gamma_choice = trial.suggest_categorical("svr_gamma_choice", ["scale", "auto", "specific_gamma"])
        if gamma_choice == "specific_gamma": # Check against the suggested value
            trial.suggest_float("svr_gamma_specific", 1e-4, 1e-1, log=True)
    if kernel == "poly":
        trial.suggest_int("svr_degree", 2, 5)
    return {}

def get_random_forest_regressor_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("rfr_n_estimators", 50, 300, step=50)
    if trial.suggest_categorical("rfr_use_max_depth", [True, False]):
        trial.suggest_int("rfr_max_depth", 3, 20, log=True)
    # else max_depth will not be in trial.params if not suggested
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
    trial.suggest_int("lgbm_max_depth", -1, 15) # -1 means no limit
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

def _prepare_model_params(optuna_trial_params: dict, base_fixed_params: dict, model_alias: str, rnd_state_global: int, model_class_ref: type) -> dict:
    """ Prepares model parameters by merging Optuna suggestions with fixed params and handling model-specific logic. """
    params = base_fixed_params.copy()
    
    # Map Optuna suggested names to model's expected parameter names
    param_mapping = {
        "Ridge": {"ridge_alpha": "alpha"},
        "ElasticNet": {"elasticnet_alpha": "alpha", "elasticnet_l1_ratio": "l1_ratio"},
        "SVR": {"svr_C": "C", "svr_epsilon": "epsilon", "svr_kernel": "kernel", 
                "svr_gamma_specific": "gamma", "svr_degree": "degree"}, # svr_gamma_choice handled separately
        "RandomForestRegressor": {"rfr_n_estimators": "n_estimators", "rfr_max_depth": "max_depth", 
                                  "rfr_min_samples_split": "min_samples_split", "rfr_min_samples_leaf": "min_samples_leaf"},
        "XGBRegressor": {"xgb_n_estimators": "n_estimators", "xgb_max_depth": "max_depth", 
                         "xgb_learning_rate": "learning_rate", "xgb_subsample": "subsample", 
                         "xgb_colsample_bytree": "colsample_bytree", "xgb_gamma": "gamma"},
        "LGBMRegressor": {"lgbm_n_estimators": "n_estimators", "lgbm_learning_rate": "learning_rate", 
                          "lgbm_num_leaves": "num_leaves", "lgbm_max_depth": "max_depth", 
                          "lgbm_subsample": "subsample", "lgbm_colsample_bytree": "colsample_bytree"},
        "CatBoostRegressor": {"cat_iterations": "iterations", "cat_learning_rate": "learning_rate", 
                              "cat_depth": "depth", "cat_l2_leaf_reg": "l2_leaf_reg"},
        "KNeighborsRegressor": {"knn_n_neighbors": "n_neighbors", "knn_weights": "weights", "knn_p": "p"}
    }

    current_map = param_mapping.get(model_alias, {})
    for optuna_key, model_key in current_map.items():
        if optuna_key in optuna_trial_params:
            params[model_key] = optuna_trial_params[optuna_key]
    
    # Handle specific logic like SVR's gamma_choice
    if model_alias == "SVR" and "svr_gamma_choice" in optuna_trial_params:
        gamma_choice_val = optuna_trial_params["svr_gamma_choice"]
        if gamma_choice_val == "specific_gamma":
            if "svr_gamma_specific" in optuna_trial_params: # It should be if choice was specific_gamma
                 params["gamma"] = optuna_trial_params["svr_gamma_specific"]
            # else gamma remains as potentially set by fixed_params or default
        elif gamma_choice_val in ["scale", "auto"]:
            params["gamma"] = gamma_choice_val
        # if svr_gamma_choice was not 'specific_gamma', svr_gamma_specific is ignored and 'gamma' key might not be set by optuna_trial_params
        # so we ensure it is set if kernel needs it
        if params.get("kernel") in ["rbf", "poly", "sigmoid"] and "gamma" not in params:
            params["gamma"] = "scale" # Default for these kernels

    # Add global random_state if model supports it and not already set by fixed_params or optuna_params
    try:
        # Check if model can be instantiated to inspect its params
        # This is a bit safer than model_class_ref().get_params() if __init__ has required args
        sig = inspect.signature(model_class_ref)
        if "random_state" in sig.parameters and "random_state" not in params:
            params["random_state"] = rnd_state_global
    except Exception as e:
        logger.debug(f"Could not dynamically add random_state for {model_alias} via inspect: {e}")
        # Fallback: if model has fixed_params and random_state is in it, it would already be there.
        # This is mostly for models where random_state is not in fixed_params but is acceptable.

    # Clean up any Optuna-specific keys that were not directly mapped if they differ from model params
    # This step is mostly a safeguard; explicit mapping is preferred.
    optuna_prefixed_keys = [k for k in params if k.startswith("optuna_") or \
                            k.startswith("logreg_") or k.startswith("rf_") or \
                            k.startswith("xgb_") or k.startswith("lgbm_") or \
                            k.startswith("cat_") or k.startswith("svc_") or k.startswith("knn_") or \
                            k.startswith("rfr_") or k.startswith("ridge_") or k.startswith("elasticnet_")]
    
    # Only remove if the base name (e.g., 'alpha' from 'ridge_alpha') isn't a direct model param.
    # The mapping above should handle most cases. This is more for keys that Optuna might add
    # but are not model params at all.
    for key in optuna_prefixed_keys:
        base_key_candidate = key.split('_', 1)[1] if '_' in key else key # Simplistic base key
        if base_key_candidate not in model_class_ref().get_params() and key in params:
            # If a mapped version (e.g. 'alpha') is already in params, don't remove the optuna key
            # if its value was already used. This needs careful thought.
            # For now, if it's an optuna_prefixed key that IS NOT a direct model param, it could be removed.
            # However, the explicit pop in the mapping is safer.
            # This cleanup is risky if not done carefully, safer to ensure mapping is exhaustive.
            pass


    # Ensure max_depth for RandomForestRegressor is None if use_max_depth was false
    if model_alias == "RandomForestRegressor" and not optuna_trial_params.get("rfr_use_max_depth", True):
        params["max_depth"] = None
        
    return params

def train_regression_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model_alias: str, model_config: dict, n_trials_optuna: int,
    artifacts_path: str, rnd_state: int
):
    logger.info(f"Processing model: {model_alias}")
    model_class = model_config["model_class"]
    space_func = model_config["space_func"]
    base_fixed_params = model_config.get("fixed_params", {}).copy()
    requires_scaling = model_config.get("requires_scaling", False)

    current_X_train_df = X_train.copy()
    current_X_test_df = X_test.copy()
    scaler = None
    original_columns = list(X_train.columns) 

    if requires_scaling:
        logger.info(f"Applying StandardScaler for {model_alias}...")
        scaler = StandardScaler()
        current_X_train_scaled_np = scaler.fit_transform(current_X_train_df)
        current_X_test_scaled_np = scaler.transform(current_X_test_df)
        current_X_train_df = pd.DataFrame(current_X_train_scaled_np, columns=original_columns, index=X_train.index)
        current_X_test_df = pd.DataFrame(current_X_test_scaled_np, columns=original_columns, index=X_test.index)
        logger.info(f"Scaling complete for {model_alias}.")

    X_train_processed = current_X_train_df.copy()
    X_test_processed = current_X_test_df.copy()
    feature_names_for_model = original_columns 

    if model_alias in ["LGBMRegressor", "CatBoostRegressor"]:
        logger.info(f"Sanitizing feature names for {model_alias}...")
        X_train_processed = sanitize_feature_names(X_train_processed)
        X_test_processed = sanitize_feature_names(X_test_processed)
        feature_names_for_model = X_train_processed.columns.tolist()

    def objective(trial):
        space_func(trial) # Defines suggestions on the trial object
        optuna_trial_params = trial.params 

        params_for_model_instantiation = _prepare_model_params(
            optuna_trial_params, base_fixed_params, model_alias, rnd_state, model_class
        )
        model = model_class(**params_for_model_instantiation)
        
        X_trial_train, X_trial_val, y_trial_train, y_trial_val = train_test_split(
            X_train_processed, y_train, test_size=0.25, random_state=rnd_state
        )
        try:
            model.fit(X_trial_train, y_trial_train)
            preds = model.predict(X_trial_val)
            mse = mean_squared_error(y_trial_val, preds) 
            rmse = np.sqrt(mse)                          
            score = -rmse 
        except Exception as e:
            logger.warning(f"Optuna trial for {model_alias} with params {params_for_model_instantiation} failed during fit/predict: {e}")
            return float('-inf') 
        return score

    study_name = f"ModelGarden_AutoML_Regression_v1_{model_alias}_opt_study"
    logger.info(f"--- Optimizing: {model_alias} (regression) with Optuna ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=False)
    
    with mlflow.start_run(run_name=f"Optuna_{model_alias}_Regression", nested=True) as optuna_run:
        mlflow.set_tag("mlflow.runName", f"Optuna Tuning - {model_alias} (Regression)")
        mlflow.log_param("model_alias", model_alias); mlflow.log_param("n_trials_optuna", n_trials_optuna)
        
        def callback(study, trial_data):
            mlflow.log_metric(f"trial_{trial_data.number}_neg_rmse_val", trial_data.value if trial_data.value is not None else float('-inf'), step=trial_data.number)
            # Log parameters as suggested by Optuna (with Optuna's internal names)
            for key, value in trial_data.params.items(): 
                mlflow.log_param(f"trial_{trial_data.number}_{key}", value)

        study.optimize(objective, n_trials=n_trials_optuna, n_jobs=1, callbacks=[callback], gc_after_trial=True)

        best_params_from_optuna_raw = study.best_trial.params # These are Optuna's internal names
        best_value_neg_rmse = study.best_value if study.best_value is not None else float('-inf')
        
        neg_rmse_display = f"{best_value_neg_rmse:.4f}" if isinstance(best_value_neg_rmse, (int, float)) and np.isfinite(best_value_neg_rmse) else str(best_value_neg_rmse)
        actual_rmse_val = -best_value_neg_rmse if best_value_neg_rmse != float('-inf') else float('inf')
        rmse_display = f"{actual_rmse_val:.4f}" if isinstance(actual_rmse_val, (int, float)) and np.isfinite(actual_rmse_val) else str(actual_rmse_val)
        logger.info(f"Best negative RMSE for {model_alias} from Optuna (validation split): {neg_rmse_display} (RMSE: {rmse_display})")

        mlflow.log_metric("best_optuna_neg_rmse_val", best_value_neg_rmse)
        if best_value_neg_rmse != float('-inf'): 
            mlflow.log_metric("best_optuna_rmse_val", -best_value_neg_rmse)
        mlflow.log_params({f"best_optuna_raw_{k}": v for k, v in best_params_from_optuna_raw.items()})

    final_model_hyperparams = _prepare_model_params(
        best_params_from_optuna_raw, base_fixed_params, model_alias, rnd_state, model_class
    )
    final_model = model_class(**final_model_hyperparams)
    
    logger.info(f"Retraining {model_alias} with best parameters on full X_train_processed...")
    final_model.fit(X_train_processed, y_train)
    logger.info("Retraining complete.")

    y_pred_test = final_model.predict(X_test_processed)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    metrics = {
        "rmse": np.sqrt(mse_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "r2": r2_score(y_test, y_pred_test)
    }
    logger.info(f"Final metrics for {model_alias} on test set: {metrics}")
    
    return final_model, scaler, metrics, best_params_from_optuna_raw, study, X_train_processed, y_pred_test


def run_regression_pipeline(
    X_train_df: pd.DataFrame, 
    y_train_series: pd.Series, 
    X_test_df: pd.DataFrame, 
    y_test_series: pd.Series, 
    n_trials_optuna: int,
    artifacts_path_base: str,
    rnd_state_global: int
):
    logger.info("--- Starting Regression Training Pipeline ---")
    
    le = NoOpLabelEncoder()
    label_encoder_filename = "label_encoder_regression.joblib"
    label_encoder_path = os.path.join(artifacts_path_base, label_encoder_filename)
    joblib.dump(le, label_encoder_path)
    logger.info(f"NoOpLabelEncoder saved for regression task to {label_encoder_path}")
    
    if mlflow.active_run():
        try: mlflow.log_artifact(label_encoder_path, "label_encoder")
        except Exception as e: logger.warning(f"Could not log NoOpLabelEncoder to MLflow: {e}")

    all_model_results = {}
    train_columns_path_global = os.path.join(artifacts_path_base, "train_columns.json")

    for model_alias, config in MODEL_CONFIGS_REGRESSION.items():
        logger.info(f"\n--- Processing model: {model_alias} (regression) ---")
        
        current_model_config = config.copy() 
        if "fixed_params" not in current_model_config: current_model_config["fixed_params"] = {}
        
        with mlflow.start_run(run_name=f"Train_{model_alias}_Regression", nested=True) as child_run:
            mlflow.set_tag("model_type", model_alias); mlflow.set_tag("task_type", "regression")
            
            final_model, fitted_scaler, metrics, best_hyperparams_optuna_raw, \
            optuna_study, X_train_model_input, y_pred_test_final = train_regression_model(
                X_train_df.copy(), y_train_series.copy(), X_test_df.copy(), y_test_series.copy(), 
                model_alias, current_model_config, n_trials_optuna,
                artifacts_path_base, rnd_state_global
            )
            
            all_model_results[model_alias] = {
                "model": final_model, "scaler": fitted_scaler,
                "metrics": metrics, "params": best_hyperparams_optuna_raw
            }
            
            if final_model:
                mlflow.log_params(best_hyperparams_optuna_raw)
                mlflow.log_metrics(metrics)
                
                model_filename = f"{model_alias.lower()}_regression_model.joblib"
                model_path = os.path.join(artifacts_path_base, model_filename)
                joblib.dump(final_model, model_path)
                
                input_example_df_for_signature = X_train_model_input.head(5) if not X_train_model_input.empty else None
                signature, input_example_log = None, None
                if input_example_df_for_signature is not None and not input_example_df_for_signature.empty:
                    try:
                        example_prediction = final_model.predict(input_example_df_for_signature)
                        signature = mlflow.models.infer_signature(input_example_df_for_signature, example_prediction)
                        input_example_log = input_example_df_for_signature.iloc[[0]].to_dict(orient='records')[0]
                    except Exception as sig_ex:
                        logger.warning(f"Could not generate MLflow signature/input_example for {model_alias}: {sig_ex}")

                mlflow.sklearn.log_model(
                    sk_model=final_model, 
                    artifact_path=f"{model_alias.lower()}_regression_model", 
                    registered_model_name=f"AutoML_Regression_{model_alias}",
                    signature=signature,
                    input_example=input_example_log
                )
                logger.info(f"Saved final {model_alias} model locally to {model_path}")
                logger.info(f"Logged and registered final {model_alias} model to MLflow as AutoML_Regression_{model_alias}")

                if fitted_scaler:
                    scaler_filename = f"{model_alias.lower()}_regression_scaler.joblib"
                    scaler_path = os.path.join(artifacts_path_base, scaler_filename)
                    joblib.dump(fitted_scaler, scaler_path)
                    mlflow.log_artifact(scaler_path, "scaler")
                    logger.info(f"Logged fitted scaler for {model_alias} to MLflow.")
                    logger.info(f"Saved fitted scaler for {model_alias} locally to {scaler_path}")
                
                plotting_utils.log_feature_importance_plot(final_model, X_train_model_input.columns.tolist(), model_alias, "regression", artifacts_path_base)
                plotting_utils.log_actual_vs_predicted_plot(y_test_series, y_pred_test_final, model_alias, artifacts_path_base)
                plotting_utils.log_residuals_plot(y_test_series, y_pred_test_final, model_alias, artifacts_path_base)
                if optuna_study:
                    plotting_utils.log_optuna_visualizations(optuna_study, model_alias, "regression", artifacts_path_base)
            
            if os.path.exists(train_columns_path_global):
                 mlflow.log_artifact(train_columns_path_global, "feature_schema_from_prep")
            if os.path.exists(label_encoder_path):
                 mlflow.log_artifact(label_encoder_path, "label_encoder")
    
    return all_model_results