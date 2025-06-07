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
import inspect # Keep for robust random_state checking

# Assuming plotting_utils.py is in src/utils/
from ..utils import plotting_utils 

logger = logging.getLogger(__name__)

# --- Helper Function ---
def sanitize_feature_names(df_in: pd.DataFrame) -> pd.DataFrame:
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

# --- Optuna Space Functions for Classification ---
def get_logistic_regression_space(trial: optuna.trial.Trial) -> dict:
    # Optuna will suggest these parameter names.
    # We will map them to actual model parameters in _prepare_model_params.
    trial.suggest_categorical("logreg_solver", ["liblinear", "saga"])
    trial.suggest_categorical("logreg_penalty_choice", ["l1", "l2", "elasticnet", "none"]) # Broader choice, will be filtered by solver
    trial.suggest_float("logreg_C", 1e-4, 1e4, log=True)
    trial.suggest_float("logreg_l1_ratio", 0.0, 1.0) # Only used if penalty is elasticnet
    return {} # Optuna uses trial.params

def get_random_forest_classifier_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("rf_n_estimators", 50, 300)
    trial.suggest_int("rf_max_depth", 3, 20, log=True) if trial.suggest_categorical("rf_use_max_depth", [True, False]) else None
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
    trial.suggest_float("svc_C", 1e-3, 1e3, log=True)
    kernel = trial.suggest_categorical("svc_kernel", ["linear", "rbf", "poly", "sigmoid"])
    trial.suggest_categorical("svc_class_weight", ["balanced", None])
    if kernel in ["rbf", "poly", "sigmoid"]:
        trial.suggest_categorical("svc_gamma_choice", ["scale", "auto", "specific_gamma"])
        if trial.params.get("svc_gamma_choice") == "specific_gamma": # Check if key exists
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

def _prepare_model_params(optuna_trial_params: dict, base_fixed_params: dict, model_alias: str, rnd_state_global: int, model_class_ref: type) -> dict:
    """ Prepares model parameters by merging Optuna suggestions with fixed params and handling model-specific logic. """
    params = base_fixed_params.copy() # Start with fixed_params
    
    # Map Optuna suggested names to model's expected parameter names
    if model_alias == "LogisticRegression":
        params["solver"] = optuna_trial_params.get("logreg_solver")
        params["C"] = optuna_trial_params.get("logreg_C")
        
        penalty_choice = optuna_trial_params.get("logreg_penalty_choice")
        
        # Solver and penalty compatibility
        if params["solver"] == "liblinear":
            if penalty_choice not in ["l1", "l2"]:
                params["penalty"] = "l2" # Default for liblinear
            else:
                params["penalty"] = penalty_choice
        elif params["solver"] == "saga":
            if penalty_choice == "none":
                params["penalty"] = None
            elif penalty_choice == "elasticnet":
                params["penalty"] = "elasticnet"
                params["l1_ratio"] = optuna_trial_params.get("logreg_l1_ratio")
            elif penalty_choice in ["l1", "l2"]:
                params["penalty"] = penalty_choice
            else: # Default for saga if invalid choice
                params["penalty"] = "l2"
        else: # Should not happen based on categorical choices
            params["penalty"] = "l2" 

        # Clean up l1_ratio if not elasticnet
        if params.get("penalty") != "elasticnet" and "l1_ratio" in params:
            del params["l1_ratio"]

    elif model_alias == "RandomForestClassifier":
        params["n_estimators"] = optuna_trial_params.get("rf_n_estimators")
        if optuna_trial_params.get("rf_use_max_depth"):
            params["max_depth"] = optuna_trial_params.get("rf_max_depth")
        else:
            params["max_depth"] = None
        params["min_samples_split"] = optuna_trial_params.get("rf_min_samples_split")
        params["min_samples_leaf"] = optuna_trial_params.get("rf_min_samples_leaf")
        params["class_weight"] = optuna_trial_params.get("rf_class_weight")

    elif model_alias == "XGBClassifier":
        params["n_estimators"] = optuna_trial_params.get("xgb_n_estimators")
        params["max_depth"] = optuna_trial_params.get("xgb_max_depth")
        params["learning_rate"] = optuna_trial_params.get("xgb_learning_rate")
        params["subsample"] = optuna_trial_params.get("xgb_subsample")
        params["colsample_bytree"] = optuna_trial_params.get("xgb_colsample_bytree")
        params["gamma"] = optuna_trial_params.get("xgb_gamma")

    elif model_alias == "LGBMClassifier":
        params["n_estimators"] = optuna_trial_params.get("lgbm_n_estimators")
        params["learning_rate"] = optuna_trial_params.get("lgbm_learning_rate")
        params["num_leaves"] = optuna_trial_params.get("lgbm_num_leaves")
        params["max_depth"] = optuna_trial_params.get("lgbm_max_depth")
        params["subsample"] = optuna_trial_params.get("lgbm_subsample")
        params["colsample_bytree"] = optuna_trial_params.get("lgbm_colsample_bytree")
        params["class_weight"] = optuna_trial_params.get("lgbm_class_weight")

    elif model_alias == "CatBoostClassifier":
        params["iterations"] = optuna_trial_params.get("cat_iterations")
        params["learning_rate"] = optuna_trial_params.get("cat_learning_rate")
        params["depth"] = optuna_trial_params.get("cat_depth")
        params["l2_leaf_reg"] = optuna_trial_params.get("cat_l2_leaf_reg")

    elif model_alias == "SVC":
        params["C"] = optuna_trial_params.get("svc_C")
        params["kernel"] = optuna_trial_params.get("svc_kernel")
        params["class_weight"] = optuna_trial_params.get("svc_class_weight")
        if params["kernel"] in ["rbf", "poly", "sigmoid"]:
            if optuna_trial_params.get("svc_gamma_choice") == "specific_gamma":
                params["gamma"] = optuna_trial_params.get("svc_gamma_specific")
            else: # 'scale' or 'auto'
                params["gamma"] = optuna_trial_params.get("svc_gamma_choice") 
        if params["kernel"] == "poly":
            params["degree"] = optuna_trial_params.get("svc_degree")

    elif model_alias == "KNeighborsClassifier":
        params["n_neighbors"] = optuna_trial_params.get("knn_n_neighbors")
        params["weights"] = optuna_trial_params.get("knn_weights")
        params["p"] = optuna_trial_params.get("knn_p")
    else: # Fallback for any other model, assumes direct mapping
        params.update(optuna_trial_params)

    # Add global random_state if model supports it and not already specifically set
    # Check against the actual model class constructor parameters
    try:
        sig = inspect.signature(model_class_ref)
        if "random_state" in sig.parameters and "random_state" not in params:
            # Only add if not already defined by fixed_params or optuna_params
            # Some optuna spaces might define it, if so, that takes precedence.
            params["random_state"] = rnd_state_global
    except Exception as e:
        logger.debug(f"Could not dynamically add/check random_state for {model_alias}: {e}")
        
    # Add class_weight for LogisticRegression and SVC if Optuna didn't set it and it's not in fixed_params
    if model_alias in ["LogisticRegression", "SVC"] and "class_weight" not in params:
        if optuna_trial_params.get(f"{model_alias.lower()}_class_weight", "not_set") is None: # Check if Optuna explicitly set it to None
            params["class_weight"] = None
        else: # Default to balanced if Optuna didn't suggest it or suggested None via a choice that wasn't "None"
            params["class_weight"] = 'balanced'


    return params


def train_classification_model(
    X_train: pd.DataFrame, y_train_encoded: np.ndarray,
    X_test: pd.DataFrame, y_test_encoded: np.ndarray,
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

    if model_alias in ["LGBMClassifier", "CatBoostClassifier"]:
        logger.info(f"Sanitizing feature names for {model_alias}...")
        X_train_processed = sanitize_feature_names(X_train_processed)
        X_test_processed = sanitize_feature_names(X_test_processed)
        feature_names_for_model = X_train_processed.columns.tolist()

    def objective(trial):
        # Optuna space_func defines suggestions ON the trial object
        space_func(trial)
        optuna_trial_params = trial.params # These have Optuna-specific names

        params_for_model_instantiation = _prepare_model_params(
            optuna_trial_params, 
            base_fixed_params, 
            model_alias, 
            rnd_state, # Use rnd_state passed to train_classification_model for trial consistency
            model_class
        )
        
        model = model_class(**params_for_model_instantiation)
        
        X_trial_train, X_trial_val, y_trial_train, y_trial_val = train_test_split(
            X_train_processed, y_train_encoded, test_size=0.25, random_state=rnd_state,
            stratify=y_train_encoded if len(np.unique(y_train_encoded)) > 1 else None
        )
        
        try:
            model.fit(X_trial_train, y_trial_train)
            preds = model.predict(X_trial_val)
            score = f1_score(y_trial_val, preds, average='weighted', zero_division=0)
        except Exception as e:
            logger.warning(f"Optuna trial for {model_alias} with params {params_for_model_instantiation} failed during fit/predict: {e}")
            return -1.0 
        return score

    study_name = f"ModelGarden_AutoML_Classification_v1_{model_alias}_opt_study"
    logger.info(f"--- Optimizing: {model_alias} (classification) with Optuna ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=False)
    
    with mlflow.start_run(run_name=f"Optuna_{model_alias}_Classification", nested=True) as optuna_run:
        mlflow.set_tag("mlflow.runName", f"Optuna Tuning - {model_alias}")
        mlflow.log_param("model_alias", model_alias)
        mlflow.log_param("n_trials_optuna", n_trials_optuna)
        
        def callback(study, trial_data):
            mlflow.log_metric(f"trial_{trial_data.number}_f1_weighted_val", trial_data.value if trial_data.value is not None else -1.0, step=trial_data.number)
            # Log actual Optuna param names
            for key, value in trial_data.params.items():
                mlflow.log_param(f"trial_{trial_data.number}_optuna_{key}", value)


        study.optimize(objective, n_trials=n_trials_optuna, n_jobs=1, callbacks=[callback], gc_after_trial=True)

        best_optuna_params_raw = study.best_trial.params # These are Optuna's internal names
        best_value_f1 = study.best_value if study.best_value is not None else -1.0
        logger.info(f"Best f1_weighted for {model_alias} from Optuna (validation split): {best_value_f1:.4f}")
        mlflow.log_metric("best_optuna_f1_weighted_val", best_value_f1)
        mlflow.log_params({f"best_optuna_raw_{k}": v for k, v in best_optuna_params_raw.items()}) # Log raw Optuna params

    # Prepare hyperparams for final model instantiation using the helper
    final_model_hyperparams = _prepare_model_params(
        best_optuna_params_raw, 
        base_fixed_params, 
        model_alias, 
        rnd_state, # Use rnd_state for final model
        model_class
    )
            
    final_model = model_class(**final_model_hyperparams)
    
    logger.info(f"Retraining {model_alias} with best parameters on full X_train_processed...")
    final_model.fit(X_train_processed, y_train_encoded)
    logger.info("Retraining complete.")

    y_pred_test = final_model.predict(X_test_processed)
    y_proba_test = None
    if hasattr(final_model, "predict_proba"):
        try:
            y_proba_test_all_classes = final_model.predict_proba(X_test_processed)
            if y_proba_test_all_classes.shape[1] == 2: # Binary classification
                y_proba_test = y_proba_test_all_classes[:, 1]
            elif y_proba_test_all_classes.shape[1] > 2: # Multiclass
                 # For multiclass ROC AUC, need to decide strategy (OvR, etc.)
                 # Or log probabilities for all classes if metric supports it
                logger.info(f"Multiclass probabilities detected for {model_alias}. ROC AUC binary might not be directly applicable.")
                # y_proba_test could be set to y_proba_test_all_classes for other multiclass metrics
            else: # Single class probability (unlikely but handle)
                y_proba_test = y_proba_test_all_classes[:, 0]

        except Exception as e_proba:
            logger.warning(f"Could not get predict_proba for {model_alias}: {e_proba}")
    
    metrics = {
        "accuracy": accuracy_score(y_test_encoded, y_pred_test),
        "f1_weighted": f1_score(y_test_encoded, y_pred_test, average='weighted', zero_division=0),
        "precision_weighted": precision_score(y_test_encoded, y_pred_test, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test_encoded, y_pred_test, average='weighted', zero_division=0)
    }
    if y_proba_test is not None and len(np.unique(y_test_encoded)) == 2 :
        try:
            metrics["roc_auc_binary"] = roc_auc_score(y_test_encoded, y_proba_test)
        except ValueError as e_roc:
            logger.warning(f"Could not compute ROC AUC for {model_alias}: {e_roc}")
            metrics["roc_auc_binary"] = 0.0 
    else:
        metrics["roc_auc_binary"] = 0.0 
    
    logger.info(f"Final metrics for {model_alias} on test set: {metrics}")
    
    # Return Optuna's raw best params for logging, and processed data for plotting.
    return final_model, scaler, metrics, best_optuna_params_raw, study, X_train_processed, X_test_processed, y_pred_test, y_proba_test


def run_classification_pipeline(
    X_train_df: pd.DataFrame, 
    y_train_series: pd.Series, 
    X_test_df: pd.DataFrame, 
    y_test_series: pd.Series, 
    n_trials_optuna: int,
    artifacts_path_base: str,
    rnd_state_global: int
):
    logger.info("--- Starting Classification Training Pipeline ---")
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_series.values.ravel() if isinstance(y_train_series, pd.Series) else y_train_series)
    y_test_encoded = le.transform(y_test_series.values.ravel() if isinstance(y_test_series, pd.Series) else y_test_series)
    
    logger.info(f"Target label encoded. Classes: {list(le.classes_)} mapped to {list(range(len(le.classes_)))}")
    
    label_encoder_filename = "label_encoder_classification.joblib"
    label_encoder_path = os.path.join(artifacts_path_base, label_encoder_filename)
    joblib.dump(le, label_encoder_path)
    logger.info(f"Label encoder saved to {label_encoder_path}")
    
    if mlflow.active_run():
        try: mlflow.log_artifact(label_encoder_path, "label_encoder")
        except Exception as e: logger.warning(f"Could not log label_encoder_path to MLflow: {e}")

    all_model_results = {}
    train_columns_path_global = os.path.join(artifacts_path_base, "train_columns.json") 

    for model_alias, config in MODEL_CONFIGS_CLASSIFICATION.items():
        logger.info(f"\n--- Processing model: {model_alias} (classification) ---")
        
        current_model_config = config.copy() 
        if "fixed_params" not in current_model_config: current_model_config["fixed_params"] = {}
        
        with mlflow.start_run(run_name=f"Train_{model_alias}_Classification", nested=True) as child_run:
            mlflow.set_tag("model_type", model_alias)
            mlflow.set_tag("task_type", "classification")
            
            final_model, fitted_scaler, metrics, best_hyperparams_optuna_raw, optuna_study, \
            X_train_model_input, X_test_model_input, y_pred_test_final, y_test_proba_final = train_classification_model(
                X_train_df.copy(), y_train_encoded, X_test_df.copy(), y_test_encoded, 
                model_alias, current_model_config, n_trials_optuna,
                artifacts_path_base, rnd_state_global
            )
            
            all_model_results[model_alias] = {
                "model": final_model, "scaler": fitted_scaler,
                "metrics": metrics, "params": best_hyperparams_optuna_raw # Store Optuna's raw params
            }
            
            if final_model:
                mlflow.log_params(best_hyperparams_optuna_raw) # Log Optuna's raw params
                mlflow.log_metrics(metrics)
                
                model_filename = f"{model_alias.lower()}_classification_model.joblib"
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
                    artifact_path=f"{model_alias.lower()}_classification_model", 
                    registered_model_name=f"AutoML_Classification_{model_alias}",
                    signature=signature,
                    input_example=input_example_log
                )
                logger.info(f"Saved final {model_alias} model locally to {model_path}")
                logger.info(f"Logged and registered final {model_alias} model to MLflow as AutoML_Classification_{model_alias}")

                if fitted_scaler:
                    scaler_filename = f"{model_alias.lower()}_classification_scaler.joblib"
                    scaler_path = os.path.join(artifacts_path_base, scaler_filename)
                    joblib.dump(fitted_scaler, scaler_path)
                    mlflow.log_artifact(scaler_path, "scaler")
                    logger.info(f"Logged fitted scaler for {model_alias} to MLflow.")
                    logger.info(f"Saved fitted scaler for {model_alias} locally to {scaler_path}")
                
                # Log plots
                plotting_utils.log_feature_importance_plot(final_model, X_train_model_input.columns.tolist(), model_alias, "classification", artifacts_path_base)
                if hasattr(le, 'classes_'):
                    plotting_utils.log_confusion_matrix_plot(y_test_encoded, y_pred_test_final, [str(c) for c in le.classes_], model_alias, artifacts_path_base)
                if y_test_proba_final is not None and len(np.unique(y_test_encoded)) == 2: # Ensure binary for ROC/PR
                    plotting_utils.log_roc_curve_plot(y_test_encoded, y_test_proba_final, model_alias, artifacts_path_base)
                    plotting_utils.log_precision_recall_curve_plot(y_test_encoded, y_test_proba_final, model_alias, artifacts_path_base)
                if optuna_study:
                    plotting_utils.log_optuna_visualizations(optuna_study, model_alias, "classification", artifacts_path_base)
            
            if os.path.exists(train_columns_path_global):
                 mlflow.log_artifact(train_columns_path_global, "feature_schema_from_prep")
            if os.path.exists(label_encoder_path):
                 mlflow.log_artifact(label_encoder_path, "label_encoder")
    
    return all_model_results