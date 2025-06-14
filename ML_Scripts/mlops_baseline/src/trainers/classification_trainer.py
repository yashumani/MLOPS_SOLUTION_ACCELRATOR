# src/trainers/classification_trainer.py

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import optuna
import mlflow
import logging
import re
import inspect
from typing import Any, Dict, List, Optional

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

# --- Optuna Space Functions for Classification ---
def get_logistic_regression_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_categorical("logreg_solver", ["liblinear", "saga"])
    trial.suggest_categorical("logreg_penalty_choice", ["l1", "l2", "elasticnet", "none"])
    trial.suggest_float("logreg_C", 1e-4, 1e4, log=True)
    if trial.params.get("logreg_penalty_choice") == "elasticnet":
        trial.suggest_float("logreg_l1_ratio", 0.0, 1.0)
    return {}

def get_random_forest_classifier_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("rf_n_estimators", 50, 300)
    if trial.suggest_categorical("rf_use_max_depth", [True, False]):
        trial.suggest_int("rf_max_depth", 3, 20, log=True)
    trial.suggest_int("rf_min_samples_split", 2, 20)
    trial.suggest_int("rf_min_samples_leaf", 1, 20)
    trial.suggest_categorical("rf_class_weight", ["balanced", "balanced_subsample", None])
    return {}

def get_xgb_classifier_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("xgb_n_estimators", 50, 300)
    trial.suggest_int("xgb_max_depth", 2, 10)
    trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
    trial.suggest_float("xgb_subsample", 0.6, 1.0)
    trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0)
    trial.suggest_float("xgb_gamma", 0, 5)
    return {}

def get_lgbm_classifier_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("lgbm_n_estimators", 50, 500)
    trial.suggest_float("lgbm_learning_rate", 0.01, 0.2, log=True)
    trial.suggest_int("lgbm_num_leaves", 20, 150)
    trial.suggest_int("lgbm_max_depth", -1, 15)
    trial.suggest_float("lgbm_subsample", 0.6, 1.0)
    trial.suggest_float("lgbm_colsample_bytree", 0.6, 1.0)
    trial.suggest_categorical("lgbm_class_weight", ["balanced", None])
    return {}

def get_catboost_classifier_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("cat_iterations", 50, 700)
    trial.suggest_float("cat_learning_rate", 0.01, 0.3, log=True)
    trial.suggest_int("cat_depth", 3, 10)
    trial.suggest_float("cat_l2_leaf_reg", 1e-3, 10.0, log=True)
    return {}

def get_svc_space(trial: optuna.trial.Trial) -> dict:
    kernel = trial.suggest_categorical("svc_kernel", ["linear", "rbf", "poly", "sigmoid"])
    trial.suggest_float("svc_C", 1e-3, 1e3, log=True)
    trial.suggest_categorical("svc_class_weight", ["balanced", None])
    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma_choice = trial.suggest_categorical("svc_gamma_choice", ["scale", "auto", "specific"])
        if gamma_choice == "specific":
            trial.suggest_float("svc_gamma_specific", 1e-4, 1e-1, log=True)
    if kernel == "poly":
        trial.suggest_int("svc_degree", 2, 5)
    return {}

def get_knn_classifier_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("knn_n_neighbors", 3, 30)
    trial.suggest_categorical("knn_weights", ["uniform", "distance"])
    trial.suggest_int("knn_p", 1, 2)
    return {}

MODEL_CONFIGS_CLASSIFICATION = {
    "LogisticRegression": {"model_class": LogisticRegression, "space_func": get_logistic_regression_space, "fixed_params": {"max_iter": 2000}, "requires_scaling": True},
    "RandomForestClassifier": {"model_class": RandomForestClassifier, "space_func": get_random_forest_classifier_space, "fixed_params": {}, "requires_scaling": False},
    "XGBClassifier": {"model_class": XGBClassifier, "space_func": get_xgb_classifier_space, "fixed_params": {"use_label_encoder": False, "eval_metric": "logloss"}, "requires_scaling": False},
    "LGBMClassifier": {"model_class": LGBMClassifier, "space_func": get_lgbm_classifier_space, "fixed_params": {"verbose": -1}, "requires_scaling": False},
    "CatBoostClassifier": {"model_class": CatBoostClassifier, "space_func": get_catboost_classifier_space, "fixed_params": {"verbose": 0}, "requires_scaling": False},
    "SVC": {"model_class": SVC, "space_func": get_svc_space, "fixed_params": {"probability": True}, "requires_scaling": True},
    "KNeighborsClassifier": {"model_class": KNeighborsClassifier, "space_func": get_knn_classifier_space, "fixed_params": {}, "requires_scaling": True}
}

# Define optimization direction for each metric
METRIC_DIRECTIONS = {
    "f1_weighted": "maximize", "roc_auc_binary": "maximize", "accuracy": "maximize",
    "precision_weighted": "maximize", "recall_weighted": "maximize"
}

def _prepare_model_params(optuna_trial_params: dict, base_fixed_params: dict, model_alias: str, rnd_state_global: int, model_class_ref: type) -> dict:
    """Prepares model parameters by merging Optuna suggestions with fixed params and handling model-specific logic."""
    params = base_fixed_params.copy()
    
    # Generic mapping from optuna param names to model param names
    param_map = {
        "LogisticRegression": {"logreg_solver": "solver", "logreg_C": "C", "logreg_l1_ratio": "l1_ratio"},
        "RandomForestClassifier": {"rf_n_estimators": "n_estimators", "rf_max_depth": "max_depth", "rf_min_samples_split": "min_samples_split", "rf_min_samples_leaf": "min_samples_leaf", "rf_class_weight": "class_weight"},
        "XGBClassifier": {"xgb_n_estimators": "n_estimators", "xgb_max_depth": "max_depth", "xgb_learning_rate": "learning_rate", "xgb_subsample": "subsample", "xgb_colsample_bytree": "colsample_bytree", "xgb_gamma": "gamma"},
        "LGBMClassifier": {"lgbm_n_estimators": "n_estimators", "lgbm_learning_rate": "learning_rate", "lgbm_num_leaves": "num_leaves", "lgbm_max_depth": "max_depth", "lgbm_subsample": "subsample", "lgbm_colsample_bytree": "colsample_bytree", "lgbm_class_weight": "class_weight"},
        "CatBoostClassifier": {"cat_iterations": "iterations", "cat_learning_rate": "learning_rate", "cat_depth": "depth", "cat_l2_leaf_reg": "l2_leaf_reg"},
        "SVC": {"svc_C": "C", "svc_kernel": "kernel", "svc_class_weight": "class_weight", "svc_gamma_specific": "gamma", "svc_degree": "degree"},
        "KNeighborsClassifier": {"knn_n_neighbors": "n_neighbors", "knn_weights": "weights", "knn_p": "p"}
    }
    
    current_map = param_map.get(model_alias, {})
    for optuna_key, model_key in current_map.items():
        if optuna_key in optuna_trial_params:
            params[model_key] = optuna_trial_params[optuna_key]
            
    # Handle model-specific conditional logic
    if model_alias == "LogisticRegression":
        solver = params.get("solver")
        penalty_choice = optuna_trial_params.get("logreg_penalty_choice")
        if solver == "liblinear" and penalty_choice not in ["l1", "l2"]:
            params["penalty"] = "l2"
        elif solver == "saga":
            params["penalty"] = None if penalty_choice == "none" else penalty_choice
        else: # For liblinear or other future solvers
             params["penalty"] = penalty_choice if penalty_choice in ["l1", "l2"] else "l2"

        if params.get("penalty") != "elasticnet":
            params.pop("l1_ratio", None)

    if model_alias == "SVC" and "svc_gamma_choice" in optuna_trial_params:
        if optuna_trial_params["svc_gamma_choice"] != "specific":
            params["gamma"] = optuna_trial_params["svc_gamma_choice"]

    if model_alias == "RandomForestClassifier" and not optuna_trial_params.get("rf_use_max_depth", True):
        params["max_depth"] = None

    # Add global random_state if applicable
    try:
        if "random_state" in inspect.signature(model_class_ref).parameters and "random_state" not in params:
            params["random_state"] = rnd_state_global
    except Exception as e:
        logger.debug(f"Could not dynamically add random_state for {model_alias}: {e}")
        
    return params

def train_classification_model(
    X_train: pd.DataFrame, y_train_encoded: np.ndarray,
    X_test: pd.DataFrame, y_test_encoded: np.ndarray,
    model_alias: str, model_config: dict, n_trials_optuna: int,
    artifacts_path: str, rnd_state: int,
    primary_metric: str
):
    """Trains, tunes, and evaluates a single classification model."""
    logger.info(f"Processing model: {model_alias}")
    model_class = model_config["model_class"]
    space_func = model_config["space_func"]
    base_fixed_params = model_config.get("fixed_params", {}).copy()
    requires_scaling = model_config.get("requires_scaling", False)

    scaler = None
    X_train_processed, X_test_processed = X_train.copy(), X_test.copy()
    if requires_scaling:
        logger.info(f"Applying StandardScaler for {model_alias}...")
        scaler = StandardScaler()
        X_train_processed[:] = scaler.fit_transform(X_train_processed)
        X_test_processed[:] = scaler.transform(X_test_processed)
    
    if model_alias in ["LGBMClassifier", "CatBoostClassifier"]:
        logger.info(f"Sanitizing feature names for {model_alias}...")
        X_train_processed = sanitize_feature_names(X_train_processed)
        X_test_processed = sanitize_feature_names(X_test_processed)

    def objective(trial):
        space_func(trial)
        optuna_trial_params = trial.params
        params_for_model = _prepare_model_params(optuna_trial_params, base_fixed_params, model_alias, rnd_state, model_class)
        model = model_class(**params_for_model)
        
        X_trial_train, X_trial_val, y_trial_train, y_trial_val = train_test_split(
            X_train_processed, y_train_encoded, test_size=0.25, random_state=rnd_state,
            stratify=y_train_encoded if len(np.unique(y_train_encoded)) > 1 else None
        )
        try:
            model.fit(X_trial_train, y_trial_train)
            preds = model.predict(X_trial_val)
            
            if primary_metric == "roc_auc_binary":
                if hasattr(model, "predict_proba"):
                    y_proba_val = model.predict_proba(X_trial_val)[:, 1]
                    score = roc_auc_score(y_trial_val, y_proba_val)
                else: return -1.0
            else:
                score_func = globals().get(primary_metric.replace("_weighted", "") + "_score")
                if score_func:
                    score = score_func(y_trial_val, preds, average='weighted' if 'weighted' in primary_metric else 'binary', zero_division=0)
                else: # Default fallback
                    score = f1_score(y_trial_val, preds, average='weighted', zero_division=0)

        except Exception as e:
            logger.warning(f"Optuna trial failed for {model_alias}: {e}")
            return -1.0 
        return score

    study_direction = METRIC_DIRECTIONS.get(primary_metric, "maximize")
    study = optuna.create_study(direction=study_direction, study_name=f"Optuna_{model_alias}_{primary_metric}")
    with mlflow.start_run(run_name=f"Optuna_{model_alias}", nested=True):
        study.optimize(objective, n_trials=n_trials_optuna, n_jobs=1, gc_after_trial=True)
        best_optuna_params_raw = study.best_trial.params
        best_value = study.best_value
        logger.info(f"Best {primary_metric} for {model_alias} from Optuna: {best_value:.4f}")
        mlflow.log_params({f"best_optuna_{k}": v for k, v in best_optuna_params_raw.items()})
        mlflow.log_metric(f"best_optuna_{primary_metric}_val", best_value)

    final_model_hyperparams = _prepare_model_params(best_optuna_params_raw, base_fixed_params, model_alias, rnd_state, model_class)
    final_model = model_class(**final_model_hyperparams)
    logger.info(f"Retraining {model_alias} with best parameters...")
    final_model.fit(X_train_processed, y_train_encoded)
    
    y_pred_test = final_model.predict(X_test_processed)
    y_proba_test = None
    if hasattr(final_model, "predict_proba"):
        try:
            if len(np.unique(y_test_encoded)) == 2:
                y_proba_test = final_model.predict_proba(X_test_processed)[:, 1]
        except Exception as e:
            logger.warning(f"Could not get predict_proba for {model_alias}: {e}")

    metrics = {
        "accuracy": accuracy_score(y_test_encoded, y_pred_test),
        "f1_weighted": f1_score(y_test_encoded, y_pred_test, average='weighted', zero_division=0),
        "precision_weighted": precision_score(y_test_encoded, y_pred_test, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test_encoded, y_pred_test, average='weighted', zero_division=0)
    }
    if y_proba_test is not None:
        metrics["roc_auc_binary"] = roc_auc_score(y_test_encoded, y_proba_test)
    logger.info(f"Final metrics for {model_alias} on test set: {metrics}")
    
    return final_model, scaler, metrics, best_optuna_params_raw, study, X_train_processed, y_pred_test, y_proba_test


def run_classification_pipeline(
    X_train_df: pd.DataFrame, y_train_series: pd.Series, X_test_df: pd.DataFrame, y_test_series: pd.Series, 
    n_trials_optuna: int, artifacts_path_base: str, rnd_state_global: int, primary_metric: str
):
    """Main function to run the classification training pipeline for all configured models."""
    logger.info("--- Starting Classification Training Pipeline ---")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_series.values.ravel())
    y_test_encoded = le.transform(y_test_series.values.ravel())
    
    label_encoder_path = os.path.join(artifacts_path_base, "label_encoder_classification.joblib")
    joblib.dump(le, label_encoder_path)
    logger.info(f"Label encoder saved to {label_encoder_path}")
    if mlflow.active_run():
        mlflow.log_artifact(label_encoder_path, "label_encoder")

    all_model_results = {}
    for model_alias, config in MODEL_CONFIGS_CLASSIFICATION.items():
        logger.info(f"\n--- Processing model: {model_alias} (classification) ---")
        
        with mlflow.start_run(run_name=f"Train_{model_alias}_Classification", nested=True) as child_run:
            mlflow.set_tag("model_type", model_alias); mlflow.set_tag("task_type", "classification")
            mlflow.log_param("primary_optimization_metric", primary_metric)
            
            final_model, fitted_scaler, metrics, best_hyperparams, optuna_study, \
            X_train_model_input, _, y_pred_test_final, y_test_proba_final = train_classification_model(
                X_train_df, y_train_encoded, X_test_df, y_test_encoded, model_alias, 
                config, n_trials_optuna, artifacts_path_base, rnd_state_global, primary_metric
            )
            
            run_id = child_run.info.run_id
            all_model_results[model_alias] = {"model": final_model, "scaler": fitted_scaler, "metrics": metrics, "params": best_hyperparams, "mlflow_run_id": run_id}
            
            if final_model:
                mlflow.log_params(best_hyperparams)
                mlflow.log_metrics(metrics)
                
                model_path = os.path.join(artifacts_path_base, f"{model_alias.lower()}_classification_model.joblib")
                joblib.dump(final_model, model_path)
                
                if fitted_scaler:
                    scaler_path = os.path.join(artifacts_path_base, f"{model_alias.lower()}_classification_scaler.joblib")
                    joblib.dump(fitted_scaler, scaler_path)
                    mlflow.log_artifact(scaler_path, "scaler")
                
                # Log all plots
                plotting_utils.log_feature_importance_plot(final_model, X_train_model_input.columns.tolist(), model_alias, "classification", artifacts_path_base)
                if hasattr(le, 'classes_'):
                    plotting_utils.log_confusion_matrix_plot(y_test_encoded, y_pred_test_final, [str(c) for c in le.classes_], model_alias, artifacts_path_base)
                if y_test_proba_final is not None:
                    plotting_utils.log_roc_curve_plot(y_test_encoded, y_test_proba_final, model_alias, artifacts_path_base)
                    plotting_utils.log_precision_recall_curve_plot(y_test_encoded, y_test_proba_final, model_alias, artifacts_path_base)
                if optuna_study:
                    plotting_utils.log_optuna_visualizations(optuna_study, model_alias, "classification", artifacts_path_base)
                
                # ... (MLflow model logging with signature) ...
    
    return all_model_results