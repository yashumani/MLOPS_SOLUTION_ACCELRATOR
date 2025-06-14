# run_experiments.py
import subprocess
import sys
import os
import shutil
from pathlib import Path
import mlflow
import logging
import argparse

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExperimentOrchestrator")

# --- Define Preprocessing "Recipes" ---
PREPROCESSING_RECIPES = [
    {
        "name": "baseline_mean_imputation",
        "flags": ["--imputation_strategy", "mean"]
    },
    {
        "name": "advanced_knn_imputation_with_outliers",
        "flags": ["--imputation_strategy", "knn", "--handle_outliers", "--create_interactions"]
    },
]

BASE_ARTIFACTS_PATH = Path("artifacts")

def run_command(command):
    """Runs a command as a subprocess and logs its output."""
    logger.info(f"Executing command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
    
    process.wait()
    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="MLOps Master Experiment Orchestrator")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", type=str, required=True, help="Name of the target column.")
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression", "clustering"])
    parser.add_argument("--n_trials_optuna", type=int, default=10, help="Number of Optuna trials for HPO.")
    
    args = parser.parse_args()

    mlflow.set_experiment(f"Full_Pipeline_Comparison_{Path(args.input).stem}")

    with mlflow.start_run(run_name="Master_Experiment_Run") as parent_run:
        logger.info(f"Starting Master Experiment Run. MLflow Run ID: {parent_run.info.run_id}")
        mlflow.log_params({
            "input_dataset": args.input,
            "task_type": args.task_type,
            "target_column": args.target
        })

        for recipe in PREPROCESSING_RECIPES:
            recipe_name = recipe["name"]
            recipe_flags = recipe["flags"]
            
            recipe_artifacts_path = BASE_ARTIFACTS_PATH / recipe_name
            
            with mlflow.start_run(run_name=recipe_name, nested=True) as prep_run:
                logger.info(f"\n--- Starting Recipe: {recipe_name} ---")
                mlflow.set_tag("recipe_name", recipe_name)
                
                # --- FIX: Add the --eda and --custom_eda_pdf flags ---
                prep_command = [
                    sys.executable, "src/prep_pipeline.py",
                    "--input", args.input,
                    "--target", args.target,
                    "--artifacts_path", str(recipe_artifacts_path),
                    "--eda", # Generate the ydata-profiling report
                    "--custom_eda_pdf" # Generate the custom PDF report
                ] + recipe_flags
                
                if not run_command(prep_command):
                    logger.error(f"Preparation failed for recipe {recipe_name}. Skipping training.")
                    continue
                
                logger.info(f"Preparation successful for recipe {recipe_name}.")
                
                train_command = [
                    sys.executable, "src/train_pipeline.py",
                    "--task_type", args.task_type,
                    "--n_trials_optuna", str(args.n_trials_optuna),
                    "--artifacts_path", str(recipe_artifacts_path) 
                ]
                
                if not run_command(train_command):
                    logger.error(f"Training failed for recipe {recipe_name}.")
                    continue
                
                logger.info(f"Training successful for recipe {recipe_name}.")

    logger.info("All experiments completed.")

if __name__ == "__main__":
    main()