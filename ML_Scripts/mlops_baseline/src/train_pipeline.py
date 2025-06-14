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
from typing import Union, Dict, Any

# --- Add project root to Python path for robust imports ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.trainers import classification_trainer, regression_trainer, clustering_trainer
except ImportError:
    from trainers import classification_trainer, regression_trainer, clustering_trainer

# --- Constants ---
RND_STATE = 42

# --- Configure Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MLOps Main Training Pipeline Orchestrator")
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["classification", "regression", "clustering"],
                        help="Type of ML task")
    parser.add_argument("--n_trials_optuna", type=int, default=10,
                        help="Number of Optuna trials for HPO per model.")
    parser.add_argument("--primary_metric", type=str, default=None,
                        help="The primary metric to optimize for during HPO.")
    parser.add_argument("--artifacts_path", type=str, default="artifacts", 
                        help="Path to load artifacts from and save new ones to.")
    
    args = parser.parse_args()
    TASK_TYPE = args.task_type.lower()
    N_TRIALS_OPTUNA = args.n_trials_optuna
    ARTIFACTS_PATH = args.artifacts_path

    # --- DYNAMIC ARTIFACT PATHS ---
    PREPARED_DATA_PATH = os.path.join(ARTIFACTS_PATH, "prepared.parquet")
    RESULTS_FILE_PATH = os.path.join(ARTIFACTS_PATH, "final_results.json")
    
    PRIMARY_METRIC = args.primary_metric
    if PRIMARY_METRIC is None:
        if TASK_TYPE == "classification": PRIMARY_METRIC = "f1_weighted"
        elif TASK_TYPE == "regression": PRIMARY_METRIC = "rmse"
        elif TASK_TYPE == "clustering": PRIMARY_METRIC = "silhouette_score"
        logger.info(f"No primary metric specified. Defaulting to '{PRIMARY_METRIC}' for {TASK_TYPE} task.")
    else:
        logger.info(f"Primary metric for optimization specified: '{PRIMARY_METRIC}'")

    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    logger.info(f"Loading prepared data from {PREPARED_DATA_PATH}...")
    df = pd.read_parquet(PREPARED_DATA_PATH)
    
    TARGET_COLUMN = None
    manifest_path = os.path.join(ARTIFACTS_PATH, "prep_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        TARGET_COLUMN = manifest.get("target_column")
        logger.info(f"Target column from manifest: {TARGET_COLUMN}")
    elif TASK_TYPE != "clustering":
        logger.error(f"Manifest not found and is required for supervised task '{TASK_TYPE}'.")
        sys.exit(1)

    X: pd.DataFrame
    y: Union[pd.Series, None] = None

    if TASK_TYPE == "clustering":
        X = df.copy()
    elif TARGET_COLUMN and TARGET_COLUMN in df.columns:
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
    else:
        logger.error(f"Target column '{TARGET_COLUMN}' not found for supervised task '{TASK_TYPE}'.")
        sys.exit(1)
        
    # --- FIX: Use stratify for classification tasks ---
    X_train, X_test, y_train, y_test = (X, None, y, None) if TASK_TYPE == "clustering" else train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=RND_STATE, 
        stratify=(y if TASK_TYPE == "classification" and y is not None and y.nunique() > 1 and all(y.value_counts() >= 2) else None)
    )
    logger.info(f"Data prepared for trainer. X_train shape: {X_train.shape}")
    
    mlflow.set_experiment(f"ModelGarden_AutoML_{TASK_TYPE.upper()}_Pipeline_v2")
    
    all_model_results: Dict[str, Dict[str, Any]] = {}

    with mlflow.start_run(run_name=f"Main_Pipeline_Run_{Path(ARTIFACTS_PATH).name}", nested=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"Main MLflow parent run started. Run ID: {parent_run_id}")
        mlflow.log_param("primary_optimization_metric", PRIMARY_METRIC)
        mlflow.log_param("source_artifacts", ARTIFACTS_PATH)
        
        if TASK_TYPE == "classification":
            all_model_results = classification_trainer.run_classification_pipeline(X_train, y_train, X_test, y_test, N_TRIALS_OPTUNA, ARTIFACTS_PATH, RND_STATE, PRIMARY_METRIC)
        elif TASK_TYPE == "regression":
            all_model_results = regression_trainer.run_regression_pipeline(X_train, y_train, X_test, y_test, N_TRIALS_OPTUNA, ARTIFACTS_PATH, RND_STATE, PRIMARY_METRIC)
        elif TASK_TYPE == "clustering":
            all_model_results = clustering_trainer.run_clustering_pipeline(X_train, N_TRIALS_OPTUNA, ARTIFACTS_PATH, RND_STATE, PRIMARY_METRIC)
        
        best_model_alias_overall, best_score_overall = None, None
        higher_is_better = True
        
        try:
            if TASK_TYPE == "classification":
                higher_is_better = classification_trainer.METRIC_DIRECTIONS.get(PRIMARY_METRIC, True)
            elif TASK_TYPE == "regression":
                higher_is_better = regression_trainer.METRIC_DIRECTIONS.get(PRIMARY_METRIC, True)
            elif TASK_TYPE == "clustering":
                higher_is_better = clustering_trainer.METRIC_DIRECTIONS.get(PRIMARY_METRIC, True)
        except AttributeError:
             logger.warning(f"Could not find METRIC_DIRECTIONS in trainer. Defaulting to 'maximize'.")

        best_score_overall = -float('inf') if higher_is_better else float('inf')

        for model_alias, results in all_model_results.items():
            if results.get("model") and results.get("metrics"):
                current_score = results["metrics"].get(PRIMARY_METRIC)
                if current_score is not None and np.isfinite(current_score):
                    if (higher_is_better and current_score > best_score_overall) or \
                       (not higher_is_better and current_score < best_score_overall):
                        best_score_overall = current_score
                        best_model_alias_overall = model_alias
        
        if best_model_alias_overall:
            mlflow.set_tag("best_model_overall_alias", best_model_alias_overall)
            mlflow.log_metric(f"best_overall_{PRIMARY_METRIC}", best_score_overall)
            
            logger.info("\n--- OVERALL WINNER ---")
            logger.info(f"Task: {TASK_TYPE.upper()}, Metric: {PRIMARY_METRIC}")
            logger.info(f"Best Model: {best_model_alias_overall}, Score: {best_score_overall:.4f}")

        logger.info(f"\n=== Final Model Leaderboard (Based on Primary Metric: {PRIMARY_METRIC}) ===")
        
        valid_results = []
        invalid_results = []
        for model_alias, result_data in all_model_results.items():
            if result_data.get("model") and result_data.get("metrics"):
                score = result_data["metrics"].get(PRIMARY_METRIC)
                if isinstance(score, (int, float)) and np.isfinite(score):
                    valid_results.append((model_alias, score))
                else:
                    invalid_results.append((model_alias, "N/A"))

        # Sort only the list with valid scores using pandas
        if valid_results:
            leaderboard_df = pd.DataFrame(valid_results, columns=['model_alias', 'score'])
            leaderboard_df.sort_values(by='score', ascending=(not higher_is_better), inplace=True)
            sorted_valid_results = list(leaderboard_df.itertuples(index=False, name=None))
        else:
            sorted_valid_results = []
        
        final_leaderboard = sorted_valid_results + invalid_results
        
        final_results_for_ui = []
        if not final_leaderboard:
            logger.info("  No models were successfully trained to display on the leaderboard.")
        else:
            for model_name, score in final_leaderboard:
                score_display = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                logger.info(f"  {model_name}: Test {PRIMARY_METRIC} = {score_display}")
                final_results_for_ui.append({
                    "model_alias": model_name,
                    "primary_metric": PRIMARY_METRIC,
                    "score": score if isinstance(score, (int, float)) else None,
                    "mlflow_run_id": all_model_results.get(model_name, {}).get("mlflow_run_id", "N/A")
                })
            
        try:
            if best_model_alias_overall:
                winner_data = {
                    "model_alias": "OVERALL_BEST_MODEL",
                    "original_alias": best_model_alias_overall,
                    "primary_metric": PRIMARY_METRIC,
                    "score": best_score_overall,
                    "mlflow_run_id": all_model_results.get(best_model_alias_overall, {}).get("mlflow_run_id", "N/A")
                }
                final_results_for_ui.insert(0, winner_data)
            
            with open(RESULTS_FILE_PATH, 'w') as f:
                json.dump(final_results_for_ui, f, indent=4)
            logger.info(f"Final results summary for UI saved to {RESULTS_FILE_PATH}")
        except Exception as e:
            logger.error(f"Failed to save final results for UI: {e}", exc_info=True)

    logger.info(f"\nTraining pipeline for {TASK_TYPE.upper()} (All Steps) complete.")

if __name__ == "__main__":
    main()