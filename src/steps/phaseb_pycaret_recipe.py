import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import mlflow
import joblib

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--recipe_name", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).resolve().parents[2]
    recipe_path = root / "configs" / "recipes" / args.recipe_name
    if not recipe_path.is_file():
        raise FileNotFoundError(
            f"Recipe not found at {recipe_path}. Recipes must come from uploaded code (configs/recipes) and not workspaceblobstore."
        )
    with open(recipe_path, "r") as f:
        recipe = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")

    df = pd.read_csv(args.dataset_in)

    # Validate target column (required for classification/regression, not clustering)
    if task_type != "clustering":
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing in dataset for PyCaret recipe training")
    # Ensure MLflow model registry URI is set to a local file store to avoid unsupported azureml:// registry
    os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")

    metrics = {}
    manifest = {"engine": "pycaret", "recipe": recipe.get("recipe_name"), "models": []}

    try:
        if task_type == "classification":
            from pycaret.classification import setup, compare_models, pull, save_model
            setup(data=df, target=target_col, session_id=42, verbose=False, log_experiment=False)
            best = compare_models()
            leaderboard = pull()
        elif task_type == "clustering":
            import numpy as np
            from pycaret.clustering import setup, create_model, pull, save_model
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            _numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[_numeric_cols] = df[_numeric_cols].astype(np.float64)
            setup(data=df, session_id=42, verbose=False, log_experiment=False)
            best = create_model("kmeans")
            leaderboard = pull()
            predictions = best.predict(df)
            sil = silhouette_score(df.select_dtypes(include=[np.number]).astype(np.float64), predictions)
            db = davies_bouldin_score(df.select_dtypes(include=[np.number]).astype(np.float64), predictions)
            metrics["silhouette_score"] = round(sil, 4)
            metrics["davies_bouldin_score"] = round(db, 4)
        else:
            from pycaret.regression import setup, compare_models, pull, save_model
            setup(data=df, target=target_col, session_id=42, verbose=False, log_experiment=False)
            best = compare_models()
            leaderboard = pull()

        metrics["best_model"] = str(best)
        metrics["leaderboard"] = leaderboard.to_dict()
        manifest["best_model_name"] = str(best)
        manifest["leaderboard"] = leaderboard.to_dict()
        manifest["leaderboard_columns"] = leaderboard.columns.tolist()
        manifest["rows"] = int(leaderboard.shape[0])
        # Store best_metric for cross-engine comparison in aggregate step
        if task_type == "classification":
            manifest["best_metric"] = float(leaderboard.iloc[0]["Accuracy"]) if "Accuracy" in leaderboard.columns else None
        elif task_type == "clustering":
            manifest["best_metric"] = metrics.get("silhouette_score")
            manifest["metric_name"] = "silhouette_score"
        else:
            manifest["best_metric"] = float(leaderboard.iloc[0]["R2"]) if "R2" in leaderboard.columns else None
        # Save model to folder
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model"
        save_model(best, str(model_path))
    except Exception as e:
        metrics["error"] = str(e)
        manifest["error"] = str(e)
        # Mark model dir with error sentinel
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".error").write_text(str(e))

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f)

    # Log metrics to MLflow for Azure ML Studio tracking (all wrapped to prevent step failure)
    try:
        mlflow.log_param("engine", "pycaret")
        mlflow.log_param("recipe", recipe.get("recipe_name", "unknown"))
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("target_column", target_col)
        # Truncate best_model to 500 char limit for MLflow params
        best_model_str = str(metrics.get("best_model", "unknown"))[:500]
        mlflow.log_param("best_model", best_model_str)
        mlflow.log_metric("dataset_rows", int(df.shape[0]))
        mlflow.log_metric("dataset_cols", int(df.shape[1]))
        mlflow.log_dict(metrics, "pycaret_recipe_metrics.json")
        mlflow.log_dict(manifest, "pycaret_recipe_manifest.json")
        if "error" in metrics:
            mlflow.log_param("pycaret_recipe_error", metrics["error"])
    except Exception as mlflow_err:
        print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")


if __name__ == "__main__":
    main()
