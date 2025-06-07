# src/train_pipeline.py (Orchestrator)
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import json
import os
import logging

# --- Add project root to Python path for robust imports ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.trainers import classification_trainer, regression_trainer, clustering_trainer
    # Assuming plotting_utils is used within trainers, it will be imported there.
except ImportError as e:
    print(f"Error during initial imports: {e}")
    print("Ensure 'trainers' and 'utils' are valid packages within 'src' or adjust PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    # Fallback for direct execution if structure is flatter or for specific IDE issues
    # This can be removed if the above sys.path modification works consistently
    if 'trainers' not in sys.modules:
        try:
            from trainers import classification_trainer, regression_trainer, clustering_trainer
        except ImportError:
            print("Could not import trainers. Please check your project structure and PYTHONPATH.")
            sys.exit(1)

# --- Constants ---
RND_STATE = 42
ARTIFACTS_PATH = "artifacts" # Relative to project root
PREPARED_DATA_PATH = os.path.join(ARTIFACTS_PATH, "prepared.parquet")
TRAIN_COLUMNS_PATH = os.path.join(ARTIFACTS_PATH, "train_columns.json")
# LABEL_ENCODER_BASE_PATH is handled within each task-specific trainer

# --- Configure Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="MLOps Main Training Pipeline Orchestrator")
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["classification", "regression", "clustering"],
                        help="Type of ML task: 'classification', 'regression', or 'clustering'")
    parser.add_argument("--n_trials_optuna", type=int, default=10,
                        help="Number of Optuna trials for HPO per model. Default: 10")
    
    args = parser.parse_args()
    TASK_TYPE = args.task_type.lower()
    N_TRIALS_OPTUNA = args.n_trials_optuna

    logger.info(f"Starting MLOps training pipeline for TASK TYPE: {TASK_TYPE.upper()}")

    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    # --- Data Loading ---
    logger.info(f"Loading prepared data from {PREPARED_DATA_PATH}...")
    if not os.path.exists(PREPARED_DATA_PATH):
        logger.error(f"Prepared data not found at {PREPARED_DATA_PATH}. "
                     f"Please run 'python src/prep_pipeline.py --input data/<your_data>.csv --target <your_target_or_id>' first.")
        sys.exit(1)
    df = pd.read_parquet(PREPARED_DATA_PATH)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")

    # Load target column name from prep_manifest.json
    TARGET_COLUMN = None # Will remain None for clustering if not in manifest
    manifest_path = os.path.join(ARTIFACTS_PATH, "prep_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        TARGET_COLUMN = manifest.get("target_column") # This might be CUST_ID for clustering if specified during prep
        logger.info(f"Target column from manifest: {TARGET_COLUMN}")
    else:
        logger.warning(f"Preparation manifest {manifest_path} not found. Target column cannot be determined from manifest.")
        if TASK_TYPE != "clustering":
            logger.error("Manifest is required for supervised tasks to identify the target column.")
            sys.exit(1)

    # Prepare X and y
    X: pd.DataFrame
    y: Union[pd.Series, None] = None

    if TASK_TYPE == "clustering":
        X = df.copy() 
        # TARGET_COLUMN might be an ID column if specified during prep; clustering_trainer will handle dropping it.
        logger.info(f"Clustering task: Using all {X.shape[1]} columns from prepared data as initial features for trainer.")
    elif TARGET_COLUMN and TARGET_COLUMN in df.columns:
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    else:
        logger.error(f"Target column '{TARGET_COLUMN}' not found in prepared data or not specified for supervised task '{TASK_TYPE}'.")
        sys.exit(1)
        
    # Save the list of feature columns (from prepared.parquet, target potentially removed for supervised)
    # This train_columns.json is crucial for the API.
    try:
        feature_names_for_json = X.columns.tolist()
        with open(TRAIN_COLUMNS_PATH, 'w') as f:
            json.dump(feature_names_for_json, f)
        logger.info(f"Base training feature columns saved to {TRAIN_COLUMNS_PATH}. Count: {len(feature_names_for_json)}")
    except Exception as e:
        logger.error(f"Error saving training columns: {e}")

    # Split data for supervised tasks
    X_train, X_test, y_train, y_test = None, None, None, None
    if TASK_TYPE in ["classification", "regression"] and y is not None:
        stratify_on = None
        if TASK_TYPE == "classification":
            # Ensure y is suitable for stratification (e.g., not all unique values for small classes)
            y_for_stratify = y.dropna()
            unique_classes, counts = np.unique(y_for_stratify, return_counts=True)
            if len(unique_classes) > 1 and all(c >= 2 for c in counts): # Stratification needs at least 2 samples per class
                stratify_on = y 
            else:
                logger.warning(f"Stratification not possible due to insufficient samples per class (found classes: {dict(zip(unique_classes, counts))}) or single class in y. Proceeding without stratification.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RND_STATE, stratify=stratify_on
        )
        logger.info(f"Data split: X_train ({X_train.shape}), X_test ({X_test.shape}), y_train ({y_train.shape}), y_test ({y_test.shape})")
    elif TASK_TYPE == "clustering":
        X_train = X # Pass the full X dataset to the clustering trainer
        logger.info(f"Clustering task: Full dataset X (shape: {X_train.shape}) passed to trainer.")

    # --- MLflow Experiment Setup ---
    experiment_name = f"ModelGarden_AutoML_{TASK_TYPE.upper()}_Pipeline_v2"
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: '{experiment_name}'")
    except Exception as e:
        logger.error(f"Error setting MLflow experiment '{experiment_name}': {e}. Ensure MLflow server is accessible or local setup is correct.")
        sys.exit(1)

    all_model_results = {}

    with mlflow.start_run(run_name=f"Main_Pipeline_Run_{TASK_TYPE.upper()}") as parent_run:
        mlflow.log_param("task_type", TASK_TYPE)
        mlflow.log_param("n_trials_per_model", N_TRIALS_OPTUNA)
        mlflow.log_param("dataset_feature_count_initial", X.shape[1])
        mlflow.log_param("dataset_total_rows_initial", df.shape[0])
        if TASK_TYPE in ["classification", "regression"] and X_train is not None:
            mlflow.log_param("train_set_rows", X_train.shape[0])
            mlflow.log_param("test_set_rows", X_test.shape[0])
        mlflow.set_tag("pipeline_version", "2.2_final_trainers") # Updated version
        
        if os.path.exists(TRAIN_COLUMNS_PATH):
            mlflow.log_artifact(TRAIN_COLUMNS_PATH, "input_schema_from_prep")
        
        # Task-specific label encoders are saved and logged within their respective trainer modules

        if TASK_TYPE == "classification":
            all_model_results = classification_trainer.run_classification_pipeline(
                X_train, y_train, X_test, y_test, N_TRIALS_OPTUNA, ARTIFACTS_PATH, RND_STATE
            )
        elif TASK_TYPE == "regression":
            all_model_results = regression_trainer.run_regression_pipeline(
                 X_train, y_train, X_test, y_test, N_TRIALS_OPTUNA, ARTIFACTS_PATH, RND_STATE
            )
        elif TASK_TYPE == "clustering":
            all_model_results = clustering_trainer.run_clustering_pipeline(
                X_train, N_TRIALS_OPTUNA, ARTIFACTS_PATH, RND_STATE # Pass X_train (which is full X for clustering)
            )
        else:
            logger.error(f"Unsupported task_type: {TASK_TYPE}")
            sys.exit(1)

        # --- Final Model Selection and Promotion ---
        best_model_alias_overall = None
        if TASK_TYPE == "classification":
            primary_metric_name = "f1_weighted"
            best_score_overall = -1.0 
            higher_is_better = True
        elif TASK_TYPE == "regression":
            primary_metric_name = "rmse" 
            best_score_overall = float('inf')
            higher_is_better = False
        elif TASK_TYPE == "clustering":
            primary_metric_name = "silhouette_score"
            best_score_overall = -2.0 # Silhouette score range is -1 to 1
            higher_is_better = True
        else:
            logger.error(f"Cannot determine primary metric for unknown task type: {TASK_TYPE}")
            primary_metric_name = "unknown_metric"
            best_score_overall = 0 
            higher_is_better = True

        for model_alias, results in all_model_results.items():
            if results.get("model") and results.get("metrics"):
                current_score = results["metrics"].get(primary_metric_name)
                if current_score is not None and not (isinstance(current_score, float) and (np.isnan(current_score) or np.isinf(current_score))):
                    if higher_is_better:
                        if current_score > best_score_overall:
                            best_score_overall = current_score
                            best_model_alias_overall = model_alias
                    else: # lower is better (like RMSE)
                        if current_score < best_score_overall:
                            best_score_overall = current_score
                            best_model_alias_overall = model_alias
        
        if best_model_alias_overall:
            mlflow.set_tag("best_model_overall_alias", best_model_alias_overall)
            mlflow.log_metric(f"best_overall_{primary_metric_name}", best_score_overall)
            logger.info(f"\n🏆 Overall Winner for {TASK_TYPE.upper()} based on Test Set {primary_metric_name}: {best_model_alias_overall}")
            logger.info(f"   Score: {best_score_overall:.4f}")

            try:
                client = mlflow.tracking.MlflowClient()
                model_name_for_registry = f"AutoML_{TASK_TYPE.capitalize()}_{best_model_alias_overall}"
                
                latest_versions = client.get_latest_versions(model_name_for_registry, stages=["None", "Staging", "Production", "Archived"])
                
                if latest_versions:
                    latest_version_obj = sorted(latest_versions, key=lambda v: int(v.version), reverse=True)[0]
                    latest_version_number = latest_version_obj.version
                    
                    logger.info(f"Attempting to promote {model_name_for_registry} version {latest_version_number} to Staging...")
                    client.transition_model_version_stage(
                        name=model_name_for_registry,
                        version=latest_version_number,
                        stage="Staging",
                        archive_existing_versions=True 
                    )
                    logger.info(f"✅ Promoted {model_name_for_registry} v{latest_version_number} to Staging.")
                else:
                    logger.warning(f"No versions found for model {model_name_for_registry} in the registry. Promotion to Staging skipped. Model may have just been registered in a child run.")
            except mlflow.exceptions.RestException as e:
                 if "RESOURCE_DOES_NOT_EXIST" in str(e) or "Registered model not found" in str(e):
                     logger.warning(f"Model {model_name_for_registry} not found in registry. Cannot promote. Ensure it was registered by the trainer.")
                 else:
                     logger.error(f"MLflow RestException promoting model {model_name_for_registry} to Staging: {e}", exc_info=False)
            except Exception as e:
                logger.error(f"General error promoting model {model_name_for_registry} to Staging: {e}", exc_info=True)
        else:
            logger.warning(f"Could not determine the best model for {TASK_TYPE}. No model promoted to Staging.")
            
        logger.info(f"\n=== Final Model Leaderboard (Based on Primary Metric: {primary_metric_name}) ===")
        
        valid_results_for_leaderboard = []
        for m, r in all_model_results.items():
            if r.get("model") and r.get("metrics"):
                score = r["metrics"].get(primary_metric_name)
                if score is not None and not (isinstance(score, float) and (np.isnan(score) or np.isinf(score))):
                    valid_results_for_leaderboard.append((m, score))
                else:
                    logger.warning(f"Model {m} had an invalid score ({score}) for primary metric and will be excluded from leaderboard.")


        if valid_results_for_leaderboard:
            sorted_results_list = sorted(
                valid_results_for_leaderboard,
                key=lambda item: item[1],
                reverse=higher_is_better
            )
            for model_name, score in sorted_results_list:
                registered_name_display = f"AutoML_{TASK_TYPE.capitalize()}_{model_name}"
                version_info = ""
                try:
                    client = mlflow.tracking.MlflowClient()
                    versions = client.get_latest_versions(registered_name_display, stages=["None", "Staging", "Production"]) # Fetch from all stages
                    if versions:
                         latest_ver_obj = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                         version_info = f" (latest v{latest_ver_obj.version}, stage: {latest_ver_obj.current_stage})"
                except Exception: pass
                
                score_display = f"{score:.4f}" # Already filtered for valid numerics
                logging.info(f"  {model_name} (Registered: {registered_name_display}{version_info}): Metric {primary_metric_name} = {score_display}")
        else:
            logger.info("  No models successfully trained and evaluated with valid primary metric to display on the leaderboard.")

    logger.info(f"\nTraining pipeline for {TASK_TYPE.upper()} (All Steps) complete.")

if __name__ == "__main__":
    main()