from __future__ import annotations
import warnings
import sys
import json
import optuna
import os
from pathlib import Path
from datetime import datetime
import argparse
import re # For sanitizing feature names

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# --- Model-specific imports ---
# REGRESSORS
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb # For xgb.XGBRegressor and xgb.XGBClassifier
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# CLASSIFIERS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ───────── CONFIG ───────── #
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PREPARED_DATA_PATH = ARTIFACTS_DIR / "prepared.parquet"
FEATURETOOLS_MATRIX_PATH = ARTIFACTS_DIR / "featuretools_matrix.parquet"
PREP_MANIFEST_PATH = ARTIFACTS_DIR / "prep_manifest.json"

RND_STATE = 42
TEST_SIZE = 0.2
N_TRIALS_OPTUNA = 20

# === Sanitization Function ===
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in df_copy.columns]
    return df_copy

# === Define Hyperparameter Space Functions ===

# --- Regression Space Functions ---
def get_rf_regressor_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

def get_xgb_regressor_optuna_space(trial: optuna.trial.Trial) -> dict: # Corrected name
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.8), "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 4.0),
    }

def get_ridge_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}

def get_lgbm_regressor_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150), "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True), "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

def get_catboost_regressor_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 700), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 10), "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    }

def get_svr_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True), "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-2, 1e0, log=True), "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
    }

def get_knn_regressor_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]), "p": trial.suggest_int("p", 1, 2),
    }

def get_elasticnet_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True), "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}

# --- Classification Space Functions ---
def get_logistic_optuna_space(trial: optuna.trial.Trial) -> dict:
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
    if solver == "liblinear": penalty_val = trial.suggest_categorical("penalty", ["l1", "l2"])
    elif solver == "saga": penalty_val = trial.suggest_categorical("penalty", ["l1", "l2"])
    else: penalty_val = "l2"
    return {"C": trial.suggest_float("C", 1e-4, 1e2, log=True), "solver": solver, "penalty": penalty_val}

def get_rf_classifier_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
    }

def get_xgb_classifier_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }

def get_lgbm_classifier_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150), "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }

def get_catboost_classifier_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 700), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 10), "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    }

def get_svc_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True), "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]), "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }

def get_knn_classifier_optuna_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]), "p": trial.suggest_int("p", 1, 2),
    }

# === MODEL_CONFIGS Dictionary - Task Specific ===
MODEL_CONFIGS = {
    "regression": {
        "RandomForestRegressor": {"model_class": RandomForestRegressor, "space_func": get_rf_regressor_optuna_space, "fixed_params": {"random_state": RND_STATE, "n_jobs": -1}, "requires_scaling": False,},
        "XGBRegressor": {"model_class": xgb.XGBRegressor, "space_func": get_xgb_regressor_optuna_space, "fixed_params": {"random_state": RND_STATE, "n_jobs": -1, "objective": "reg:squarederror", "tree_method": "hist",}, "requires_scaling": False,},
        "Ridge": {"model_class": Ridge, "space_func": get_ridge_optuna_space, "fixed_params": {"random_state": RND_STATE}, "requires_scaling": True,},
        "LGBMRegressor": {"model_class": LGBMRegressor, "space_func": get_lgbm_regressor_optuna_space, "fixed_params": {"random_state": RND_STATE, "n_jobs": -1, "verbose": -1}, "requires_scaling": False,},
        "CatBoostRegressor": {"model_class": CatBoostRegressor, "space_func": get_catboost_regressor_optuna_space, "fixed_params": {"random_state": RND_STATE, "verbose": 0}, "requires_scaling": False,},
        "SVR": {"model_class": SVR, "space_func": get_svr_optuna_space, "fixed_params": {}, "requires_scaling": True,},
        "KNeighborsRegressor": {"model_class": KNeighborsRegressor, "space_func": get_knn_regressor_optuna_space, "fixed_params": {"n_jobs": -1}, "requires_scaling": True,},
        "ElasticNet": {"model_class": ElasticNet, "space_func": get_elasticnet_optuna_space, "fixed_params": {"random_state": RND_STATE, "max_iter": 2000}, "requires_scaling": True}
    },
    "classification": {
        "LogisticRegression": {"model_class": LogisticRegression, "space_func": get_logistic_optuna_space, "fixed_params": {"random_state": RND_STATE, "max_iter": 2000}, "requires_scaling": True,},
        "RandomForestClassifier": {"model_class": RandomForestClassifier, "space_func": get_rf_classifier_optuna_space, "fixed_params": {"random_state": RND_STATE, "n_jobs": -1}, "requires_scaling": False,},
        "XGBClassifier": {"model_class": xgb.XGBClassifier, "space_func": get_xgb_classifier_optuna_space, "fixed_params": {"random_state": RND_STATE, "n_jobs": -1, "use_label_encoder": False, "eval_metric": "logloss"}, "requires_scaling": False,},
        "LGBMClassifier": {"model_class": LGBMClassifier, "space_func": get_lgbm_classifier_optuna_space, "fixed_params": {"random_state": RND_STATE, "n_jobs": -1, "verbose": -1}, "requires_scaling": False,},
        "CatBoostClassifier": {"model_class": CatBoostClassifier, "space_func": get_catboost_classifier_optuna_space, "fixed_params": {"random_state": RND_STATE, "verbose": 0}, "requires_scaling": False,},
        "SVC": {"model_class": SVC, "space_func": get_svc_optuna_space, "fixed_params": {"random_state": RND_STATE, "probability": True}, "requires_scaling": True,},
        "KNeighborsClassifier": {"model_class": KNeighborsClassifier, "space_func": get_knn_classifier_optuna_space, "fixed_params": {"n_jobs": -1}, "requires_scaling": True,},
    }
}
# ────────────────────────── #

# === Metric Calculation Functions ===
# ... (calculate_regression_metrics and calculate_classification_metrics - code remains the same as previous full version) ...
def calculate_regression_metrics(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}

def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None, num_classes=None, average_method='weighted'):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        f"f1_{average_method}": f1_score(y_true, y_pred, average=average_method, zero_division=0),
        f"precision_{average_method}": precision_score(y_true, y_pred, average=average_method, zero_division=0),
        f"recall_{average_method}": recall_score(y_true, y_pred, average=average_method, zero_division=0),
    }
    if y_pred_proba is not None and num_classes is not None:
        try:
            if num_classes == 2: metrics["roc_auc_binary"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            elif num_classes > 2:
                metrics["roc_auc_ovo_weighted"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='weighted')
                metrics["roc_auc_ovr_weighted"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e: print(f"Warning: Could not calculate ROC AUC: {e}")
    return metrics

# === Generic Optuna Objective Function ===
# ... (generic_objective function - code remains the same as previous full version with sanitization logic) ...
def generic_objective(
    trial: optuna.trial.Trial, model_alias: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    task_type: str, current_model_config_for_task: dict, num_classes_for_clf: int | None = None
) -> float:
    model_entry = current_model_config_for_task[model_alias]
    model_class = model_entry["model_class"]
    space_func = model_entry["space_func"]
    fixed_params_config = model_entry["fixed_params"].copy()

    if task_type == "classification":
        if model_alias == "XGBClassifier" and "objective" not in fixed_params_config and num_classes_for_clf is not None:
            fixed_params_config["objective"] = "binary:logistic" if num_classes_for_clf == 2 else "multi:softprob"

    with mlflow.start_run(
        run_name=f"{model_alias}_trial_{trial.number}", nested=True,
        tags={"model_family": model_alias, "optuna_trial_number": str(trial.number), "task_type": task_type}
    ) as trial_run:
        hyperparams_to_tune = space_func(trial)
        current_model_params = {**fixed_params_config, **hyperparams_to_tune}
        
        X_train_to_fit = X_train.copy()
        X_test_to_eval = X_test.copy()
        model_name_for_sanitize_check = model_alias
        
        if model_name_for_sanitize_check in ["LGBMClassifier", "CatBoostClassifier", "LGBMRegressor", "CatBoostRegressor"]:
            print(f"Sanitizing feature names for {model_alias} trial...")
            X_train_to_fit = sanitize_feature_names(X_train_to_fit)
            X_test_to_eval = sanitize_feature_names(X_test_to_eval)

        model = model_class(**current_model_params)
        primary_metric_name_for_optuna = "rmse" if task_type == "regression" else "f1_weighted"
        score_to_optimize = float('inf') if task_type == "regression" else 0.0

        try:
            model.fit(X_train_to_fit, y_train)
            y_pred_labels = model.predict(X_test_to_eval)
            eval_metrics = {}
            if task_type == "regression":
                eval_metrics = calculate_regression_metrics(y_test, y_pred_labels)
                score_to_optimize = eval_metrics["rmse"]
            elif task_type == "classification":
                y_pred_proba = None
                if hasattr(model, "predict_proba"):
                    try: y_pred_proba = model.predict_proba(X_test_to_eval)
                    except Exception as e: print(f"Warning: Could not get predict_proba for {model_alias}: {e}")
                eval_metrics = calculate_classification_metrics(y_test, y_pred_labels, y_pred_proba, num_classes_for_clf)
                score_to_optimize = eval_metrics.get(primary_metric_name_for_optuna, 0.0)
            
            mlflow.log_params(hyperparams_to_tune)
            mlflow.log_metrics(eval_metrics)
            mlflow.set_tag("model_class", model_class.__name__)
        except Exception as e:
            print(f"Trial {trial.number} for {model_alias} failed: {e}")
            mlflow.set_tag("trial_status", "failed")
            mlflow.log_param("error_message", str(e))
            return float('inf') if task_type == "regression" else 0.0 
        return score_to_optimize

# ... (load_prep_manifest function as before) ...
def load_prep_manifest() -> dict:
    if not PREP_MANIFEST_PATH.exists(): sys.exit(f"[ERROR] Manifest not found: {PREP_MANIFEST_PATH}.")
    with open(PREP_MANIFEST_PATH, 'r') as f: manifest = json.load(f)
    return manifest

# ---------- Main Training Orchestration ----------
# ... (main function - code remains the same as previous full version with sanitization, ensuring the try-except for version retrieval is correct) ...
def main(task_type: str, use_dfs_features: bool = False) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if task_type == "regression":
        current_model_config_for_task = MODEL_CONFIGS["regression"]
        PRIMARY_OPTIMIZATION_METRIC = "rmse"; OPTIMIZATION_DIRECTION = "minimize"
        EXPERIMENT_NAME_SUFFIX = "Regression_v1"; REGISTERED_MODEL_NAME_BASE_PREFIX = "AutoML_Regression"
    elif task_type == "classification":
        current_model_config_for_task = MODEL_CONFIGS["classification"]
        PRIMARY_OPTIMIZATION_METRIC = "f1_weighted"; OPTIMIZATION_DIRECTION = "maximize"
        EXPERIMENT_NAME_SUFFIX = "Classification_v1"; REGISTERED_MODEL_NAME_BASE_PREFIX = "AutoML_Classification"
    else: sys.exit(f"Unsupported task_type: {task_type}.")

    FINAL_EXPERIMENT_NAME = f"ModelGarden_AutoML_{EXPERIMENT_NAME_SUFFIX}"
    mlflow.set_experiment(FINAL_EXPERIMENT_NAME); ARTIFACTS_DIR.mkdir(exist_ok=True)
    prep_manifest = load_prep_manifest(); TARGET_COL = prep_manifest["target_column"]
    INDEX_COL_DFS = prep_manifest.get("index_column_dfs", "index_col_for_dfs")
    print(f"Starting training pipeline for TASK TYPE: {task_type.upper()}")
    print("Loading data...");
    if not PREPARED_DATA_PATH.exists(): sys.exit(f"[ERROR] Base prepared data not found.")
    df_source_for_y = pd.read_parquet(PREPARED_DATA_PATH)
    if TARGET_COL not in df_source_for_y.columns: sys.exit(f"[ERROR] Target column not found.")
    
    num_classes = None; label_encoder_filename = f"label_encoder_{task_type.lower()}.joblib"
    if task_type == "classification":
        le = LabelEncoder(); df_source_for_y[TARGET_COL] = le.fit_transform(df_source_for_y[TARGET_COL])
        num_classes = len(le.classes_); print(f"Target label encoded. Classes: {num_classes}")
        joblib.dump(le, ARTIFACTS_DIR / label_encoder_filename); print(f"Label encoder saved.")
    
    df_source_for_y = df_source_for_y.set_index(INDEX_COL_DFS); y = df_source_for_y[TARGET_COL]
    if use_dfs_features:
        if not prep_manifest["dfs_feature_matrix_path"] or not FEATURETOOLS_MATRIX_PATH.exists(): sys.exit(f"[ERROR] DFS matrix not found.")
        print(f"Using DFS features..."); X_features = pd.read_parquet(FEATURETOOLS_MATRIX_PATH); X_features = X_features.set_index(INDEX_COL_DFS)
    else:
        print(f"Using features from prepared.parquet..."); X_features = pd.read_parquet(PREPARED_DATA_PATH); X_features = X_features.set_index(INDEX_COL_DFS)
        if TARGET_COL in X_features.columns: X_features = X_features.drop(columns=[TARGET_COL])
    X, y = X_features.align(y, join='inner', axis=0)
    if X.empty or y.empty: sys.exit("[ERROR] X or y empty after alignment.")
    print(f"Data loaded: X {X.shape}, y {y.shape}")
    stratify_split = y if task_type == "classification" and num_classes is not None and num_classes > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RND_STATE, stratify=stratify_split)
    print(f"Split: X_train {X_train.shape}, X_test {X_test.shape}")
    training_columns_original_names = list(X_train.columns) # Save original names before any sanitization in main loop
    with open(ARTIFACTS_DIR / "train_columns.json", "w") as f: json.dump(training_columns_original_names, f, indent=2)
    print(f"Training columns (original) saved.")

    print(f"\nStarting Optuna (Optimizing for: {PRIMARY_OPTIMIZATION_METRIC}, Direction: {OPTIMIZATION_DIRECTION})...")
    all_models_optuna_results = {} 
    for model_alias, config in current_model_config_for_task.items():
        print(f"\n--- Optimizing: {model_alias} ({task_type}) ---")
        X_train_processed = X_train.copy(); X_test_processed = X_test.copy(); fitted_scaler = None
        if config.get("requires_scaling", False):
            print(f"Applying StandardScaler for {model_alias}..."); scaler = StandardScaler()
            X_train_processed_np = scaler.fit_transform(X_train_processed)
            X_test_processed_np = scaler.transform(X_test_processed)
            X_train_processed = pd.DataFrame(X_train_processed_np, columns=X_train.columns, index=X_train.index)
            X_test_processed = pd.DataFrame(X_test_processed_np, columns=X_test.columns, index=X_test.index)
            fitted_scaler = scaler; print("Scaling complete.")

        optuna_study_name = f"{FINAL_EXPERIMENT_NAME}_{model_alias}_opt_study"
        with mlflow.start_run(run_name=f"OptunaStudy_{model_alias}", nested=True, tags={"model_family_to_tune": model_alias, "task_type": task_type}) as optuna_parent_run:
            mlflow.log_param("model_alias_for_tuning", model_alias); mlflow.log_param("n_trials_optuna", N_TRIALS_OPTUNA)
            mlflow.log_param("primary_optimization_metric", PRIMARY_OPTIMIZATION_METRIC)
            if fitted_scaler: mlflow.log_param("scaling_applied", True); mlflow.log_params(fitted_scaler.get_params())
            study = optuna.create_study(direction=OPTIMIZATION_DIRECTION, study_name=optuna_study_name)
            study.optimize(
                lambda trial: generic_objective(
                    trial, model_alias, X_train_processed, y_train, X_test_processed, y_test,
                    task_type, current_model_config_for_task, num_classes
                ),
                n_trials=N_TRIALS_OPTUNA, show_progress_bar=True,
            )
            all_models_optuna_results[model_alias] = {
                "best_tuned_params": study.best_trial.params, "best_optuna_score_val": study.best_value, 
                "fitted_scaler": fitted_scaler, "optuna_parent_run_id": optuna_parent_run.info.run_id
            }
            print(f"Best {PRIMARY_OPTIMIZATION_METRIC} for {model_alias} from Optuna: {study.best_value:.4f}")
            mlflow.log_metric(f"best_optuna_{PRIMARY_OPTIMIZATION_METRIC}_val", study.best_value) 
            mlflow.log_params(study.best_trial.params)

    print("\n--- Optuna Optimization Phase Complete ---")
    print("Collected Optuna results for all models:")
    for model_alias, results in all_models_optuna_results.items():
        print(f"  {model_alias}: Best Optuna Score ({PRIMARY_OPTIMIZATION_METRIC}) = {results[f'best_optuna_score_val']:.4f}, Tuned Params = {results['best_tuned_params']}")
        if results['fitted_scaler']: print(f"    Scaler was used for {model_alias}.")

    print("\n--- Final Model Retraining, Evaluation, and Registration ---")
    final_model_evaluations = {} 
    mlflow_client = mlflow.MlflowClient()

    for model_alias, optuna_results_data in all_models_optuna_results.items():
        print(f"\n--- Processing final model for: {model_alias} ({task_type}) ---")
        model_config_entry = current_model_config_for_task[model_alias]
        model_class = model_config_entry["model_class"]
        fixed_params_config = model_config_entry["fixed_params"].copy()
        best_tuned_params = optuna_results_data["best_tuned_params"]
        fitted_scaler_from_tuning = optuna_results_data["fitted_scaler"]

        if task_type == "classification":
            if model_alias == "XGBClassifier" and "objective" not in fixed_params_config and num_classes is not None:
                fixed_params_config["objective"] = "binary:logistic" if num_classes == 2 else "multi:softprob"
        full_best_params = {**fixed_params_config, **best_tuned_params}
        
        X_train_for_final_retrain = X_train.copy() # Start with original X_train
        X_test_for_final_eval = X_test.copy()     # Start with original X_test
        
        # This variable will hold the data to be fed into the final model
        # It will be scaled if a scaler was used, and then names sanitized if model is LGBM/CatBoost
        final_X_train_to_fit = X_train.copy() 
        final_X_test_to_eval = X_test.copy()

        if fitted_scaler_from_tuning:
            print(f"Applying stored scaler for {model_alias} for final training and evaluation...")
            final_X_train_to_fit_np = fitted_scaler_from_tuning.transform(X_train.copy())
            final_X_train_to_fit = pd.DataFrame(final_X_train_to_fit_np, columns=X_train.columns, index=X_train.index)
            final_X_test_to_eval_np = fitted_scaler_from_tuning.transform(X_test.copy())
            final_X_test_to_eval = pd.DataFrame(final_X_test_to_eval_np, columns=X_test.columns, index=X_test.index)

        # Sanitize feature names for specific models AFTER scaling (if any)
        # The model will be trained on these sanitized names.
        # The signature and input_example should also use these sanitized names.
        if model_alias in ["LGBMClassifier", "CatBoostClassifier", "LGBMRegressor", "CatBoostRegressor"]:
            print(f"Sanitizing feature names for final fit/eval of {model_alias}...")
            final_X_train_to_fit = sanitize_feature_names(final_X_train_to_fit)
            final_X_test_to_eval = sanitize_feature_names(final_X_test_to_eval)
            # If names are sanitized here, train_columns.json (saved earlier with original names)
            # will NOT match what these specific models expect. This needs to be handled in API for these models.

        print(f"Retraining {model_alias} with best parameters on full X_train...")
        final_model = model_class(**full_best_params)
        final_model.fit(final_X_train_to_fit, y_train) 
        print("Retraining complete.")
        
        y_pred_labels_final_test = final_model.predict(final_X_test_to_eval)
        final_test_metrics_dict = {}
        if task_type == "regression":
            final_test_metrics_dict = calculate_regression_metrics(y_test, y_pred_labels_final_test)
        elif task_type == "classification":
            y_pred_proba_final_test = None
            if hasattr(final_model, "predict_proba"):
                try: y_pred_proba_final_test = final_model.predict_proba(final_X_test_to_eval)
                except Exception as e: print(f"Warning: Could not get predict_proba for final {model_alias}: {e}")
            final_test_metrics_dict = calculate_classification_metrics(y_test, y_pred_labels_final_test, y_pred_proba_final_test, num_classes)
        
        primary_metric_final_value = final_test_metrics_dict.get(PRIMARY_OPTIMIZATION_METRIC, 0.0 if OPTIMIZATION_DIRECTION == "maximize" else float('inf'))
        print(f"Final metrics for {model_alias} on test set: {final_test_metrics_dict}")

        run_name_final_best = f"Best_{model_alias}"
        registered_model_name_specific = f"{REGISTERED_MODEL_NAME_BASE_PREFIX}_{model_alias}"

        with mlflow.start_run(run_name=run_name_final_best, tags={"model_family": model_alias, "status": "final_best_model", "task_type": task_type}) as final_run:
            mlflow.log_params(full_best_params)
            for metric_name, metric_value in final_test_metrics_dict.items(): mlflow.log_metric(f"final_test_{metric_name}", metric_value)
            mlflow.set_tag("optuna_parent_run_id", optuna_results_data["optuna_parent_run_id"])
            if fitted_scaler_from_tuning:
                scaler_artifact_path_temp = "fitted_scaler.joblib"
                joblib.dump(fitted_scaler_from_tuning, scaler_artifact_path_temp)
                mlflow.log_artifact(scaler_artifact_path_temp, artifact_path="scaler")
                os.remove(scaler_artifact_path_temp); print("Logged fitted scaler to MLflow.")
            
            # Use the (potentially) sanitized data for signature and input example
            sample_for_sig_X = final_X_test_to_eval.head(min(5, final_X_test_to_eval.shape[0])) if not final_X_test_to_eval.empty else final_X_train_to_fit.head(min(5, final_X_train_to_fit.shape[0]))
            signature = None; input_example_data = None
            if not sample_for_sig_X.empty:
                try:
                    predictions_sample = final_model.predict(sample_for_sig_X)
                    signature = infer_signature(sample_for_sig_X, predictions_sample)
                    input_example_data = sample_for_sig_X.iloc[[0]].to_dict(orient="records")[0]
                except Exception as e: print(f"Warning: Could not infer signature for final {model_alias}: {e}")
            
            print(f"Logging final {model_alias} model to MLflow and registering as {registered_model_name_specific}...");
            model_log_info = mlflow.sklearn.log_model(
                sk_model=final_model, artifact_path="model", signature=signature,
                input_example=input_example_data, registered_model_name=registered_model_name_specific
            )
            new_model_version_str = None
            try: 
                latest_versions = mlflow_client.get_latest_versions(registered_model_name_specific)
                if latest_versions: 
                    new_model_version_str = latest_versions[0].version
            except Exception as e: 
                print(f"Warning: Error retrieving latest version for {registered_model_name_specific}: {e}")

            if new_model_version_str:
                print(f"Registered final {model_alias} as {registered_model_name_specific} version {new_model_version_str}.")
                final_model_evaluations[model_alias] = {
                    "model_name_registered": registered_model_name_specific, "version": new_model_version_str,
                    "primary_metric_test": primary_metric_final_value, "all_metrics_test": final_test_metrics_dict,
                    "mlflow_run_id": final_run.info.run_id
                }
            else: print(f"Warning: Final {model_alias} registered but version retrieval failed.")
        
        model_joblib_path = ARTIFACTS_DIR / f"{model_alias.lower()}_{task_type.lower()}_model.joblib"
        joblib.dump(final_model, model_joblib_path); print(f"Saved final {model_alias} model locally to {model_joblib_path}")
        if fitted_scaler_from_tuning:
            scaler_joblib_path = ARTIFACTS_DIR / f"{model_alias.lower()}_{task_type.lower()}_scaler.joblib"
            joblib.dump(fitted_scaler_from_tuning, scaler_joblib_path); print(f"Saved fitted scaler for {model_alias} locally to {scaler_joblib_path}")

    if final_model_evaluations:
        valid_for_promotion = {k: v for k, v in final_model_evaluations.items() if "version" in v and "primary_metric_test" in v}
        if not valid_for_promotion: print("\nNo models available for promotion.")
        else:
            if OPTIMIZATION_DIRECTION == "maximize": overall_winner_alias = max(valid_for_promotion, key=lambda k: valid_for_promotion[k]["primary_metric_test"])
            else: overall_winner_alias = min(valid_for_promotion, key=lambda k: valid_for_promotion[k]["primary_metric_test"])
            winner_details = valid_for_promotion[overall_winner_alias]
            print(f"\n🏆 Overall Winner for {task_type.upper()} based on Test Set {PRIMARY_OPTIMIZATION_METRIC}: {overall_winner_alias}")
            print(f"   Registered Name: {winner_details['model_name_registered']}, Version: {winner_details['version']}, {PRIMARY_OPTIMIZATION_METRIC}: {winner_details['primary_metric_test']:.4f}")
            print(f"Promoting {winner_details['model_name_registered']} v{winner_details['version']} to Staging...")
            try:
                mlflow_client.transition_model_version_stage(name=winner_details['model_name_registered'], version=winner_details['version'], stage="Staging", archive_existing_versions=True)
                print(f"✅ Promoted to Staging.")
            except Exception as e: print(f"❌ Error during model promotion: {e}")
    else: print("\nNo models evaluated for promotion.")

    print(f"\n=== Final Model Leaderboard (Test Set {PRIMARY_OPTIMIZATION_METRIC}) ===")
    if final_model_evaluations:
        sorted_models = sorted(final_model_evaluations.items(), key=lambda item: item[1]["primary_metric_test"], reverse=(OPTIMIZATION_DIRECTION == "maximize"))
        for model_alias_sorted, details in sorted_models:
            version_display = details.get("version", "N/A")
            print(f"  {model_alias_sorted} (Registered: {details.get('model_name_registered', 'N/A')} v{version_display}): Test {PRIMARY_OPTIMIZATION_METRIC} = {details['primary_metric_test']:.4f}")
    else: print("No models to display.")
    print(f"\nTraining pipeline for {task_type.upper()} (All Steps) complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Pipeline - Task Aware")
    parser.add_argument(
        "--task_type", type=str, required=True, choices=["regression", "classification"],
        help="Type of ML task: 'regression' or 'classification'"
    )
    parser.add_argument(
        "--use_dfs", action="store_true",
        help="Use features generated by Featuretools DFS from prep_pipeline.",
    )
    args = parser.parse_args()
    main(task_type=args.task_type, use_dfs_features=args.use_dfs)