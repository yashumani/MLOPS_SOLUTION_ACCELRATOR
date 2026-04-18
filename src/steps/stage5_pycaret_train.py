import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import mlflow
import joblib
import os

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")

    df = pd.read_csv(args.dataset_in)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in dataset for PyCaret training")

    # Ensure MLflow model registry URI is set to a local file store to avoid unsupported azureml:// registry
    os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")

    # Train via PyCaret
    metrics = {}
    manifest = {"engine": "pycaret", "models": []}

    try:
        if task_type == "classification":
            from pycaret.classification import setup, compare_models, pull, save_model
            setup(data=df, target=target_col, session_id=42, verbose=False, log_experiment=False)
            best = compare_models()  # get best model
            leaderboard = pull()
        else:
            from pycaret.regression import setup, compare_models, pull, save_model
            setup(data=df, target=target_col, session_id=42, verbose=False, log_experiment=False)
            best = compare_models()
            leaderboard = pull()

        metrics["best_model"] = str(best)
        metrics["leaderboard"] = leaderboard.to_dict()
        manifest["best_model_name"] = str(best)
        manifest["leaderboard_columns"] = leaderboard.columns.tolist()
        manifest["rows"] = int(leaderboard.shape[0])
        # Save model to folder
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model"
        save_model(best, str(model_path))
    except Exception as e:
        metrics["error"] = str(e)
        manifest["error"] = str(e)
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".error").write_text(str(e))

    # Write outputs
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f)

    # Log metrics and artifacts to MLflow (all wrapped to prevent step failure)
    try:
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("target_column", target_col)
        # Truncate best_model to 500 char limit for MLflow params
        best_model_str = str(metrics.get("best_model", "unknown"))[:500]
        mlflow.log_param("best_model", best_model_str)
        mlflow.log_metric("dataset_rows", int(df.shape[0]))
        mlflow.log_metric("dataset_cols", int(df.shape[1]))
        mlflow.log_dict(metrics, "pycaret_baseline_metrics.json")
        mlflow.log_dict(manifest, "pycaret_baseline_manifest.json")
        if "error" in metrics:
            mlflow.log_param("pycaret_error", metrics["error"])
    except Exception as mlflow_err:
        print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")


if __name__ == "__main__":
    main()
