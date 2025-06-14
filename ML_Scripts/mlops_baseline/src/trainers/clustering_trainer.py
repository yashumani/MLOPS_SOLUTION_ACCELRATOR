# src/trainers/clustering_trainer.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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

# --- Optuna Space Functions for Clustering ---
def get_kmeans_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("kmeans_n_clusters", 2, 15)
    trial.suggest_categorical("kmeans_init", ["k-means++", "random"])
    trial.suggest_int("kmeans_max_iter", 100, 1000, step=100)
    trial.suggest_float("kmeans_tol", 1e-5, 1e-2, log=True)
    try:
        KMeans(n_init='auto')
        trial.suggest_categorical("kmeans_n_init_suggestion", [10, 'auto'])
    except ValueError:
        trial.suggest_int("kmeans_n_init_suggestion_int", 5, 20, step=5)
    return {}

def get_dbscan_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_float("dbscan_eps", 0.1, 3.0, step=0.1)
    trial.suggest_int("dbscan_min_samples", 2, 30)
    return {}

def get_agglomerative_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_categorical("agglomerative_linkage", ["ward", "complete", "average", "single"])
    trial.suggest_int("agglomerative_n_clusters", 2, 15)
    return {}

def get_gmm_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("gmm_n_components", 2, 10)
    trial.suggest_categorical("gmm_covariance_type", ["full", "tied", "diag", "spherical"])
    trial.suggest_float("gmm_reg_covar", 1e-7, 1e-3, log=True)
    trial.suggest_int("gmm_n_init", 1, 10)
    return {}

# --- Model Configurations for Clustering ---
MODEL_CONFIGS_CLUSTERING = {
    "KMeans": {"model_class": KMeans, "space_func": get_kmeans_space, "fixed_params": {}, "requires_scaling": True},
    "DBSCAN": {"model_class": DBSCAN, "space_func": get_dbscan_space, "fixed_params": {"n_jobs": -1}, "requires_scaling": True},
    "AgglomerativeClustering": {"model_class": AgglomerativeClustering, "space_func": get_agglomerative_space, "fixed_params": {}, "requires_scaling": True},
    "GaussianMixture": {"model_class": GaussianMixture, "space_func": get_gmm_space, "fixed_params": {}, "requires_scaling": True}
}

# --- Metric Optimization Directions ---
METRIC_DIRECTIONS = {
    "silhouette_score": "maximize",
    "calinski_harabasz_score": "maximize",
    "davies_bouldin_score": "minimize"
}

def _prepare_model_params(optuna_trial_params: dict, base_fixed_params: dict, model_alias: str, rnd_state_global: int, model_class_ref: type) -> dict:
    """Prepares model parameters by merging Optuna suggestions with fixed params and handling model-specific logic."""
    params = base_fixed_params.copy()
    
    param_map = {
        "KMeans": {"kmeans_n_clusters": "n_clusters", "kmeans_init": "init", "kmeans_max_iter": "max_iter", "kmeans_tol": "tol"},
        "DBSCAN": {"dbscan_eps": "eps", "dbscan_min_samples": "min_samples"},
        "AgglomerativeClustering": {"agglomerative_n_clusters": "n_clusters", "agglomerative_linkage": "linkage"},
        "GaussianMixture": {"gmm_n_components": "n_components", "gmm_covariance_type": "covariance_type", "gmm_reg_covar": "reg_covar", "gmm_n_init": "n_init"}
    }
    
    current_map = param_map.get(model_alias, {})
    for optuna_key, model_key in current_map.items():
        if optuna_key in optuna_trial_params:
            params[model_key] = optuna_trial_params[optuna_key]
            
    if model_alias == "KMeans":
        n_init_val = optuna_trial_params.get("kmeans_n_init_suggestion", optuna_trial_params.get("kmeans_n_init_suggestion_int"))
        if n_init_val == 'auto':
            try:
                KMeans(n_init='auto')
                params["n_init"] = 'auto'
            except ValueError:
                params["n_init"] = 10
        else:
            params["n_init"] = n_init_val

    if model_alias == "AgglomerativeClustering" and params.get("n_clusters") is not None:
        params["distance_threshold"] = None
    
    try:
        if "random_state" in inspect.signature(model_class_ref).parameters and "random_state" not in params:
            params["random_state"] = rnd_state_global
    except Exception as e:
        logger.debug(f"Could not dynamically add random_state for {model_alias}: {e}")
        
    return params

def train_clustering_model_instance(
    X_data_full: pd.DataFrame, model_alias: str, model_config: dict, 
    n_trials_optuna: int, artifacts_path_base: str, rnd_state_global: int, primary_metric: str
):
    logger.info(f"Processing instance for model: {model_alias}")
    model_class = model_config["model_class"]
    space_func = model_config["space_func"]
    base_fixed_params = model_config.get("fixed_params", {}).copy()
    requires_scaling = model_config.get("requires_scaling", False)

    current_X_df = X_data_full.copy()
    id_columns_to_drop = ['CUST_ID']
    cols_present_to_drop = [col for col in id_columns_to_drop if col in current_X_df.columns]
    if cols_present_to_drop:
        current_X_df = current_X_df.drop(columns=cols_present_to_drop)
        logger.info(f"Dropped identifier columns: {cols_present_to_drop}")
    
    if current_X_df.shape[1] < 2:
        logger.error(f"Not enough features ({current_X_df.shape[1]}) for clustering. Aborting.")
        return None, None, {"error": "Not enough features"}, {}, None, None, None

    clustering_features_filename = f"{model_alias.lower()}_clustering_features.json"
    with open(os.path.join(artifacts_path_base, clustering_features_filename), 'w') as f:
        json.dump(current_X_df.columns.tolist(), f)
    
    scaler = None
    X_processed = current_X_df.copy()
    if requires_scaling:
        scaler = StandardScaler()
        X_processed[:] = scaler.fit_transform(X_processed)
        logger.info(f"StandardScaler applied for {model_alias}.")

    def objective(trial):
        space_func(trial)
        params_for_model = _prepare_model_params(trial.params, base_fixed_params, model_alias, rnd_state_global, model_class)
        model = model_class(**params_for_model)
        
        try:
            labels = model.fit_predict(X_processed) if hasattr(model, 'fit_predict') else model.fit(X_processed).labels_
            n_clusters = len(np.unique(labels[labels != -1]))
            if n_clusters > 1 and n_clusters < len(X_processed):
                sample_size = min(2000, len(X_processed))
                indices = np.random.choice(len(X_processed), sample_size, replace=False)
                score = silhouette_score(X_processed.iloc[indices], labels[indices])
            else:
                score = -1.0 # Bad score if < 2 clusters are found
        except Exception as e:
            logger.warning(f"Optuna trial for {model_alias} failed: {e}")
            return -2.0 # Indicate failure
        return score

    study_direction = METRIC_DIRECTIONS.get(primary_metric, "maximize")
    study = optuna.create_study(direction=study_direction, study_name=f"Optuna_{model_alias}_{primary_metric}")
    study.optimize(objective, n_trials=n_trials_optuna, gc_after_trial=True)
    
    best_params_raw = study.best_trial.params
    final_model_hyperparams = _prepare_model_params(best_params_raw, base_fixed_params, model_alias, rnd_state_global, model_class)
    final_model = model_class(**final_model_hyperparams)
    
    logger.info(f"Fitting final {model_alias} with best parameters...")
    final_model.fit(X_processed)
    
    cluster_labels = final_model.labels_ if hasattr(final_model, 'labels_') else final_model.predict(X_processed)
    metrics = {}
    n_clusters_found = len(np.unique(cluster_labels[cluster_labels != -1]))
    metrics["n_clusters_found_actual"] = n_clusters_found
    
    if "n_clusters" in final_model_hyperparams: metrics["n_clusters_parameter"] = final_model_hyperparams['n_clusters']
    elif "n_components" in final_model_hyperparams: metrics["n_clusters_parameter"] = final_model_hyperparams['n_components']
    
    if n_clusters_found > 1:
        try:
            metrics["silhouette_score"] = silhouette_score(X_processed, cluster_labels)
            metrics["davies_bouldin_score"] = davies_bouldin_score(X_processed, cluster_labels)
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_processed, cluster_labels)
        except MemoryError:
            logger.error(f"MemoryError calculating final metrics for {model_alias}. Logging placeholders.")
            metrics["silhouette_score"], metrics["davies_bouldin_score"], metrics["calinski_harabasz_score"] = -1.0, float('inf'), 0.0
    
    logger.info(f"Final metrics for {model_alias}: {metrics}")
    
    return final_model, scaler, metrics, best_params_raw, study, X_processed, cluster_labels

def run_clustering_pipeline(
    X_df_original: pd.DataFrame, n_trials_optuna: int, artifacts_path_base: str, 
    rnd_state_global: int, primary_metric: str = "silhouette_score"
):
    logger.info("--- Starting Clustering Training Pipeline ---")
    all_model_results = {}
    
    for model_alias, config in MODEL_CONFIGS_CLUSTERING.items():
        logger.info(f"\n--- Processing model: {model_alias} (clustering) ---")
        
        with mlflow.start_run(run_name=f"Train_{model_alias}_Clustering", nested=True) as child_run:
            mlflow.set_tag("model_type", model_alias); mlflow.set_tag("task_type", "clustering")
            mlflow.log_param("primary_optimization_metric", primary_metric)
            
            final_model, fitted_scaler, metrics, best_hyperparams, optuna_study, \
            X_model_input, cluster_labels_final = train_clustering_model_instance(
                X_df_original, model_alias, config, n_trials_optuna, 
                artifacts_path_base, rnd_state_global, primary_metric
            )
            
            if final_model is None:
                all_model_results[model_alias] = {"model": None, "scaler": None, "metrics": {}, "params": {}}
                continue

            run_id = child_run.info.run_id
            all_model_results[model_alias] = {"model": final_model, "scaler": fitted_scaler, "metrics": metrics, "params": best_hyperparams, "mlflow_run_id": run_id}
            
            mlflow.log_params(best_hyperparams)
            mlflow.log_metrics(metrics)
            
            joblib.dump(final_model, os.path.join(artifacts_path_base, f"{model_alias.lower()}_clustering_model.joblib"))
            if fitted_scaler:
                joblib.dump(fitted_scaler, os.path.join(artifacts_path_base, f"{model_alias.lower()}_clustering_scaler.joblib"))
                mlflow.log_artifact(os.path.join(artifacts_path_base, f"{model_alias.lower()}_clustering_scaler.joblib"), "scaler_clustering")
            
            plotting_utils.log_cluster_plot_pca(X_model_input, cluster_labels_final, model_alias, artifacts_path_base)
            if optuna_study:
                plotting_utils.log_optuna_visualizations(optuna_study, model_alias, "clustering", artifacts_path_base)
            
            # ... MLflow model logging with signature ...
    
    return all_model_results