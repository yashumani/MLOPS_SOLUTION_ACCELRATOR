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

# --- Optuna Space Functions for Clustering ---
# These functions define the parameters Optuna will tune.
# The names used in trial.suggest_xxx will be the keys in trial.params.

def get_kmeans_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_int("kmeans_n_clusters", 2, 15)
    trial.suggest_categorical("kmeans_init", ["k-means++", "random"])
    trial.suggest_int("kmeans_max_iter", 100, 1000, step=100)
    trial.suggest_float("kmeans_tol", 1e-5, 1e-2, log=True)
    try:
        KMeans(n_init='auto') # Check if 'auto' is valid
        trial.suggest_categorical("kmeans_n_init_suggestion", [10, 'auto'])
    except ValueError:
        trial.suggest_int("kmeans_n_init_suggestion_int", 5, 20, step=5)
    return {} # Optuna uses trial object, no need to return params dict here

def get_dbscan_space(trial: optuna.trial.Trial) -> dict:
    trial.suggest_float("eps", 0.1, 3.0, step=0.1)
    trial.suggest_int("min_samples", 2, 30)
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
    "AgglomerativeClustering": {"model_class": AgglomerativeClustering, "space_func": get_agglomerative_space, "fixed_params": {}, "requires_scaling": True}, # distance_threshold handled in _prepare_model_params
    "GaussianMixture": {"model_class": GaussianMixture, "space_func": get_gmm_space, "fixed_params": {}, "requires_scaling": True}
}

def _prepare_model_params(optuna_trial_params: dict, base_fixed_params: dict, model_alias: str, rnd_state_global: int, model_class_ref: type) -> dict:
    """
    Merges Optuna suggested parameters with fixed parameters and remaps names for model instantiation.
    optuna_trial_params are the parameters directly from trial.params or study.best_trial.params.
    """
    # Start with base fixed params, then update with Optuna's suggestions
    # Optuna suggestions (from trial.params) will use the names defined in trial.suggest_xxx
    params_for_model = base_fixed_params.copy()
    
    # Map Optuna-specific names to actual model parameter names
    if model_alias == "KMeans":
        params_for_model["n_clusters"] = optuna_trial_params.get("kmeans_n_clusters")
        params_for_model["init"] = optuna_trial_params.get("kmeans_init")
        params_for_model["max_iter"] = optuna_trial_params.get("kmeans_max_iter")
        params_for_model["tol"] = optuna_trial_params.get("kmeans_tol")
        
        n_init_val = optuna_trial_params.get("kmeans_n_init_suggestion", optuna_trial_params.get("kmeans_n_init_suggestion_int"))
        if n_init_val == 'auto':
            try: KMeans(n_init='auto') # Test if 'auto' is acceptable
            except ValueError: n_init_val = 10
        params_for_model["n_init"] = n_init_val

    elif model_alias == "DBSCAN":
        params_for_model["eps"] = optuna_trial_params.get("eps")
        params_for_model["min_samples"] = optuna_trial_params.get("min_samples")

    elif model_alias == "AgglomerativeClustering":
        params_for_model["n_clusters"] = optuna_trial_params.get("agglomerative_n_clusters")
        params_for_model["linkage"] = optuna_trial_params.get("agglomerative_linkage")
        # If n_clusters is provided, distance_threshold must be None for AgglomerativeClustering
        if params_for_model.get("n_clusters") is not None:
            params_for_model["distance_threshold"] = None
        # If linkage is 'ward', n_clusters cannot be None. Optuna space should ensure this.
        if params_for_model.get("linkage") == "ward" and params_for_model.get("n_clusters") is None:
            logger.warning(f"Ward linkage for AgglomerativeClustering requires n_clusters. Optuna suggested None. Defaulting to 2.")
            params_for_model["n_clusters"] = 2 # Fallback

    elif model_alias == "GaussianMixture":
        params_for_model["n_components"] = optuna_trial_params.get("gmm_n_components")
        params_for_model["covariance_type"] = optuna_trial_params.get("gmm_covariance_type")
        params_for_model["reg_covar"] = optuna_trial_params.get("gmm_reg_covar")
        params_for_model["n_init"] = optuna_trial_params.get("gmm_n_init")
    else: # For any other model, assume direct mapping if not specified
        params_for_model.update(optuna_trial_params)

    # Add global random_state if model supports it and not already set by optuna or fixed_params
    try:
        # Check against the actual model class constructor parameters
        sig = inspect.signature(model_class_ref)
        if "random_state" in sig.parameters and "random_state" not in params_for_model:
            params_for_model["random_state"] = rnd_state_global
    except Exception as e:
        logger.debug(f"Could not dynamically add random_state for {model_alias}: {e}")
        
    # Remove any optuna-specific prefixed keys that were not mapped, if any remain
    # (though the above mapping should handle all defined ones)
    final_params = {k: v for k, v in params_for_model.items() if not k.startswith("optuna_") and not k.endswith("_optuna") and not k.endswith("_optuna_int")}
    # Ensure mapped keys are present
    if model_alias == "KMeans":
        if "n_clusters_kmeans" in optuna_trial_params: final_params["n_clusters"] = optuna_trial_params["n_clusters_kmeans"]
        if "init_kmeans" in optuna_trial_params: final_params["init"] = optuna_trial_params["init_kmeans"]
        if "max_iter_kmeans" in optuna_trial_params: final_params["max_iter"] = optuna_trial_params["max_iter_kmeans"]
        if "tol_kmeans" in optuna_trial_params: final_params["tol"] = optuna_trial_params["tol_kmeans"]
        n_init_val = optuna_trial_params.get("kmeans_n_init_suggestion", optuna_trial_params.get("kmeans_n_init_suggestion_int"))
        if n_init_val == 'auto':
            try: KMeans(n_init='auto')
            except ValueError: n_init_val = 10
        final_params["n_init"] = n_init_val

    elif model_alias == "AgglomerativeClustering":
        if "agglomerative_n_clusters" in optuna_trial_params: final_params["n_clusters"] = optuna_trial_params["agglomerative_n_clusters"]
        if "agglomerative_linkage" in optuna_trial_params: final_params["linkage"] = optuna_trial_params["agglomerative_linkage"]
        if "n_clusters" in final_params and final_params["n_clusters"] is not None:
            final_params["distance_threshold"] = None

    elif model_alias == "GaussianMixture":
        if "gmm_n_components" in optuna_trial_params: final_params["n_components"] = optuna_trial_params["gmm_n_components"]
        if "gmm_covariance_type" in optuna_trial_params: final_params["covariance_type"] = optuna_trial_params["gmm_covariance_type"]
        if "gmm_reg_covar" in optuna_trial_params: final_params["reg_covar"] = optuna_trial_params["gmm_reg_covar"]
        if "gmm_n_init" in optuna_trial_params: final_params["n_init"] = optuna_trial_params["gmm_n_init"]
        
    return final_params


def train_clustering_model_instance(
    X_data_full: pd.DataFrame, 
    model_alias: str, 
    model_config: dict, 
    n_trials_optuna: int, 
    artifacts_path_base: str, 
    rnd_state_global: int
):
    logger.info(f"Processing instance for model: {model_alias}")
    model_class = model_config["model_class"]
    space_func = model_config["space_func"]
    base_fixed_params = model_config.get("fixed_params", {}).copy()
    requires_scaling = model_config.get("requires_scaling", False)

    current_X_df = X_data_full.copy()
    logger.info(f"Initial features for {model_alias} before ID drop: {current_X_df.columns.tolist()[:10]}... (Total: {len(current_X_df.columns)})")

    id_columns_to_drop = ['CUST_ID'] 
    cols_present_to_drop = [col for col in id_columns_to_drop if col in current_X_df.columns]
    if cols_present_to_drop:
        logger.info(f"Dropping identifier columns for clustering model {model_alias}: {cols_present_to_drop}")
        current_X_df = current_X_df.drop(columns=cols_present_to_drop)
    
    if current_X_df.empty or current_X_df.shape[1] < 1:
        logger.error(f"Not enough features ({current_X_df.shape[1]}) for clustering model {model_alias}. Aborting.")
        return None, None, {"error": "Not enough features for clustering"}, {}, None, None, None

    clustering_features_used = current_X_df.columns.tolist()
    clustering_features_filename = f"{model_alias.lower()}_clustering_features.json"
    clustering_features_filepath = os.path.join(artifacts_path_base, clustering_features_filename)
    try:
        with open(clustering_features_filepath, 'w') as f: json.dump(clustering_features_used, f)
        if mlflow.active_run(): mlflow.log_artifact(clustering_features_filepath, "clustering_feature_schema")
        logger.info(f"Clustering input features for {model_alias} saved and logged.")
    except Exception as e: logger.error(f"Error saving/logging clustering input features for {model_alias}: {e}")

    scaler = None
    X_processed_for_optuna = current_X_df.copy() 
    if requires_scaling:
        logger.info(f"Applying StandardScaler for {model_alias}...")
        scaler = StandardScaler()
        scaled_np = scaler.fit_transform(X_processed_for_optuna)
        X_processed_for_optuna = pd.DataFrame(scaled_np, columns=X_processed_for_optuna.columns, index=X_processed_for_optuna.index)
        logger.info(f"Scaling complete for {model_alias}.")

    def objective(trial):
        space_func(trial) # This defines the suggestions for the trial object
        optuna_trial_params = trial.params # These are the raw suggested params by Optuna

        params_for_model_instantiation = _prepare_model_params(
            optuna_trial_params, 
            base_fixed_params, 
            model_alias, 
            rnd_state_global,
            model_class 
        )
        
        model = model_class(**params_for_model_instantiation)
        
        try:
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X_processed_for_optuna)
            else: # For models like AgglomerativeClustering that might not have fit_predict
                model.fit(X_processed_for_optuna)
                labels = model.labels_
            
            unique_labels = np.unique(labels)
            n_clusters_found_trial = len(unique_labels[unique_labels != -1]) if -1 in unique_labels else len(unique_labels)

            if n_clusters_found_trial > 1 and n_clusters_found_trial < X_processed_for_optuna.shape[0]:
                sample_size_optuna = min(2000, len(X_processed_for_optuna)) 
                if len(X_processed_for_optuna) > sample_size_optuna:
                    logger.debug(f"Trial {trial.number}: Calculating Silhouette score on a sample of {sample_size_optuna} data points.")
                    # Ensure consistent sampling for comparability if desired
                    indices = np.random.RandomState(rnd_state_global).choice(X_processed_for_optuna.index, size=sample_size_optuna, replace=False)
                    X_sample = X_processed_for_optuna.loc[indices]
                    labels_sample = labels[X_processed_for_optuna.index.get_indexer(indices)]

                    unique_labels_sample = np.unique(labels_sample)
                    n_clusters_sample = len(unique_labels_sample[unique_labels_sample != -1]) if -1 in unique_labels_sample else len(unique_labels_sample)
                    if n_clusters_sample > 1 and n_clusters_sample < len(X_sample):
                         score = silhouette_score(X_sample, labels_sample)
                    else:
                        logger.debug(f"Trial {trial.number} (Sampled): Silhouette score not computed (found {n_clusters_sample} clusters).")
                        score = -1.0 
                else:
                    score = silhouette_score(X_processed_for_optuna, labels)
            else:
                logger.warning(f"Optuna trial {trial.number} for {model_alias}: Silhouette score not computed (found {n_clusters_found_trial} clusters for {X_processed_for_optuna.shape[0]} samples). Returning -1.0.")
                score = -1.0 
        except MemoryError:
            logger.error(f"Optuna trial {trial.number} for {model_alias} with params {params_for_model_instantiation} caused MemoryError. Returning -2.0")
            return -2.0 
        except ValueError as ve:
             logger.warning(f"Optuna trial {trial.number} for {model_alias} with params {params_for_model_instantiation} raised ValueError: {ve}. Returning -1.5")
             score = -1.5 
        except Exception as e:
            logger.warning(f"Optuna trial {trial.number} for {model_alias} with params {params_for_model_instantiation} failed: {e}", exc_info=False)
            return -2.0 
        return score

    study_name = f"ModelGarden_AutoML_Clustering_v1_{model_alias}_opt_study"
    logger.info(f"--- Optimizing: {model_alias} (clustering) with Optuna ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=False)
    
    with mlflow.start_run(run_name=f"Optuna_{model_alias}_Clustering", nested=True):
        mlflow.set_tag("mlflow.runName", f"Optuna Tuning - {model_alias} (Clustering)")
        mlflow.log_param("model_alias", model_alias); mlflow.log_param("n_trials_optuna", n_trials_optuna)
        
        def callback(study, trial_data):
            mlflow.log_metric(f"trial_{trial_data.number}_silhouette_val", trial_data.value if trial_data.value is not None else -2.0, step=trial_data.number)
            for key, value in trial_data.params.items(): mlflow.log_param(f"trial_{trial_data.number}_{key}", value)

        study.optimize(objective, n_trials=n_trials_optuna, n_jobs=1, callbacks=[callback], gc_after_trial=True)

        best_params_from_optuna_trial = study.best_trial.params # These are Optuna's internal names
        best_value_silhouette = study.best_value if study.best_value is not None else -2.0
        logger.info(f"Best Silhouette Score for {model_alias} from Optuna (validation): {best_value_silhouette:.4f}")
        mlflow.log_metric("best_optuna_silhouette_score_val", best_value_silhouette)
        mlflow.log_params({f"best_optuna_{k}": v for k, v in best_params_from_optuna_trial.items()})

    # Prepare hyperparams for final model instantiation using the helper
    final_model_hyperparams = _prepare_model_params(
        best_params_from_optuna_trial, 
        base_fixed_params, 
        model_alias, 
        rnd_state_global,
        model_class
    )
            
    final_model = model_class(**final_model_hyperparams)
    
    logger.info(f"Fitting final {model_alias} with best parameters on full processed dataset...")
    if hasattr(final_model, 'fit_predict') and model_alias != "GaussianMixture":
        cluster_labels = final_model.fit_predict(X_processed_for_optuna)
    else:
        final_model.fit(X_processed_for_optuna)
        cluster_labels = final_model.labels_ if hasattr(final_model, 'labels_') else final_model.predict(X_processed_for_optuna)
    logger.info("Final model fitting complete.")
    
    metrics = {}
    unique_labels_final = np.unique(cluster_labels)
    n_clusters_found_actual = len(unique_labels_final[unique_labels_final != -1]) if -1 in unique_labels_final else len(unique_labels_final)
    metrics["n_clusters_found_actual"] = n_clusters_found_actual
    
    if "n_clusters" in final_model_hyperparams: metrics["n_clusters_parameter"] = final_model_hyperparams['n_clusters']
    elif "n_components" in final_model_hyperparams: metrics["n_clusters_parameter"] = final_model_hyperparams['n_components']
    
    if n_clusters_found_actual > 1 and n_clusters_found_actual < X_processed_for_optuna.shape[0]:
        logger.info(f"Calculating final Silhouette score for {model_alias} on full data...")
        try: metrics["silhouette_score"] = silhouette_score(X_processed_for_optuna, cluster_labels)
        except MemoryError: logger.error(f"MemoryError: Silhouette score for {model_alias}. Logging -1."); metrics["silhouette_score"] = -1.0
        except Exception as e_sil: logger.error(f"Error Silhouette: {e_sil}. Logging -1."); metrics["silhouette_score"] = -1.0

        try: metrics["davies_bouldin_score"] = davies_bouldin_score(X_processed_for_optuna, cluster_labels)
        except MemoryError: logger.error(f"MemoryError: Davies-Bouldin for {model_alias}. Logging inf."); metrics["davies_bouldin_score"] = float('inf')
        except Exception as e_db: logger.error(f"Error Davies-Bouldin: {e_db}. Logging inf."); metrics["davies_bouldin_score"] = float('inf')

        try: metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_processed_for_optuna, cluster_labels)
        except MemoryError: logger.error(f"MemoryError: Calinski-Harabasz for {model_alias}. Logging 0."); metrics["calinski_harabasz_score"] = 0.0
        except Exception as e_ch: logger.error(f"Error Calinski-Harabasz: {e_ch}. Logging 0."); metrics["calinski_harabasz_score"] = 0.0
    else:
        metrics["silhouette_score"] = -1.0; metrics["davies_bouldin_score"] = float('inf'); metrics["calinski_harabasz_score"] = 0.0
        logger.warning(f"Final model for {model_alias} resulted in {n_clusters_found_actual} actual cluster(s). Metrics might be uninformative or invalid.")
    
    logger.info(f"Final metrics for {model_alias} (clustering): {metrics}")
    
    return final_model, scaler, metrics, best_params_from_optuna_trial, study, X_processed_for_optuna, cluster_labels


def run_clustering_pipeline(
    X_df_original: pd.DataFrame, 
    n_trials_optuna: int, 
    artifacts_path_base: str, 
    rnd_state_global: int
):
    logger.info("--- Starting Clustering Training Pipeline ---")
    all_model_results = {}
    os.makedirs(artifacts_path_base, exist_ok=True)
    
    train_columns_path_global = os.path.join(artifacts_path_base, "train_columns.json")
    if os.path.exists(train_columns_path_global) and mlflow.active_run():
        try: mlflow.log_artifact(train_columns_path_global, "input_schema_from_prep")
        except Exception as e: logger.warning(f"Could not log global_train_columns.json: {e}")

    for model_alias, config in MODEL_CONFIGS_CLUSTERING.items():
        logger.info(f"\n--- Processing model: {model_alias} (clustering) ---")
        
        current_model_config = config.copy() 
        if "fixed_params" not in current_model_config: current_model_config["fixed_params"] = {}
        
        with mlflow.start_run(run_name=f"Train_{model_alias}_Clustering", nested=True) as child_run:
            mlflow.set_tag("model_type", model_alias); mlflow.set_tag("task_type", "clustering")
            
            final_model, fitted_scaler, metrics, best_hyperparams_optuna, optuna_study, \
            X_model_input, cluster_labels_final = train_clustering_model_instance(
                X_df_original.copy(), model_alias, current_model_config, 
                n_trials_optuna, artifacts_path_base, rnd_state_global
            )
            
            if final_model is None:
                logger.error(f"Skipping MLflow logging and saving for {model_alias} due to training error.")
                all_model_results[model_alias] = {"model": None, "scaler": None, "metrics": {}, "params": {}}
                continue

            all_model_results[model_alias] = {
                "model": final_model, "scaler": fitted_scaler,
                "metrics": metrics, "params": best_hyperparams_optuna 
            }
            
            mlflow.log_params(best_hyperparams_optuna) 
            mlflow.log_metrics(metrics)
            
            model_filename = f"{model_alias.lower()}_clustering_model.joblib"
            model_path = os.path.join(artifacts_path_base, model_filename)
            joblib.dump(final_model, model_path)
            
            input_example_df = X_model_input.head(5) if X_model_input is not None and not X_model_input.empty else None
            signature, input_example_log = None, None
            if input_example_df is not None and not input_example_df.empty:
                try:
                    example_output = None
                    if hasattr(final_model, 'predict') and callable(getattr(final_model, 'predict')):
                        example_output = final_model.predict(input_example_df)
                    elif hasattr(final_model, 'labels_') and final_model.labels_ is not None:
                        if len(cluster_labels_final) >= len(input_example_df):
                             example_output = cluster_labels_final[:len(input_example_df)]
                    
                    if example_output is not None:
                        signature = mlflow.models.infer_signature(input_example_df, pd.Series(example_output, name="cluster_label"))
                        input_example_log = input_example_df.iloc[[0]].to_dict(orient='records')[0] if not input_example_df.empty else None
                except Exception as sig_ex:
                    logger.warning(f"Could not generate MLflow signature/input_example for {model_alias}: {sig_ex}")
            
            mlflow.sklearn.log_model(
                sk_model=final_model, 
                artifact_path=f"{model_alias.lower()}_clustering_model", 
                registered_model_name=f"AutoML_Clustering_{model_alias}",
                signature=signature,
                input_example=input_example_log
            )
            logger.info(f"Saved final {model_alias} model locally to {model_path}")
            logger.info(f"Logged and registered final {model_alias} model to MLflow as AutoML_Clustering_{model_alias}")

            if fitted_scaler:
                scaler_filename = f"{model_alias.lower()}_clustering_scaler.joblib"
                scaler_path = os.path.join(artifacts_path_base, scaler_filename)
                joblib.dump(fitted_scaler, scaler_path)
                mlflow.log_artifact(scaler_path, "scaler_clustering")
                logger.info(f"Logged fitted scaler for {model_alias} to MLflow.")
                logger.info(f"Saved fitted scaler for {model_alias} locally to {scaler_path}")
            
            if X_model_input is not None and not X_model_input.empty and cluster_labels_final is not None:
                 plotting_utils.log_cluster_plot_pca(X_model_input, cluster_labels_final, model_alias, artifacts_path_base)
            if optuna_study:
                plotting_utils.log_optuna_visualizations(optuna_study, model_alias, "clustering", artifacts_path_base)
    
    return all_model_results