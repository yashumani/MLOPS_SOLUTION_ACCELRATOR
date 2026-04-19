import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mlflow

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
    import os
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    time_budget = cfg.get("phases", {}).get("phase_b_recipes", {}).get("flaml_config", {}).get("time_budget", 120)

    df = pd.read_csv(args.dataset_in)

    # FLAML does not support clustering — skip gracefully
    if task_type == "clustering":
        print("⏭️  FLAML does not support clustering task type; skipping recipe")
        metrics = {"status": "skipped", "reason": "FLAML does not support clustering"}
        manifest = {"engine": "flaml", "recipe": recipe.get("recipe_name"), "models": [], "status": "skipped", "reason": "FLAML does not support clustering"}
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f)
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f)
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".skipped").write_text("FLAML does not support clustering")
        return

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in dataset for FLAML recipe training")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Sanitize column names — LightGBM/FLAML cannot handle special JSON chars like [ ] { } ( ) " :
    import re
    X.columns = [re.sub(r'[^\w]', '_', c) for c in X.columns]

    # Ensure MLflow model registry URI is set to a local file store to avoid unsupported azureml:// registry
    os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")

    metrics = {}
    manifest = {"engine": "flaml", "recipe": recipe.get("recipe_name"), "models": []}

    try:
        from flaml import AutoML
        automl = AutoML()
        task = "classification" if task_type == "classification" else "regression"
        metric = "accuracy" if task == "classification" else "r2"
        automl.fit(X_train=X, y_train=y, task=task, metric=metric, time_budget=time_budget, log_file_name="flaml_recipe.log")
        best_estimator = automl.best_estimator
        best_config = automl.best_config
        # Convert FLAML loss to actual metric score for correct champion comparison
        if task == "classification":
            best_metric = 1.0 - automl.best_loss  # best_loss = 1 - accuracy
        else:
            best_metric = -automl.best_loss if automl.best_loss < 0 else automl.best_loss
        metrics["best_estimator"] = str(best_estimator)
        metrics["best_config"] = str(best_config)
        metrics["best_metric"] = best_metric
        manifest["best_estimator"] = str(best_estimator)
        manifest["best_config"] = str(best_config)
        manifest["best_metric"] = best_metric
        try:
            import joblib
            model_obj = automl.model.estimator
            model_dir = Path(args.model_out)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pkl"
            joblib.dump(model_obj, model_path)
        except Exception:
            model_dir = Path(args.model_out)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / ".empty").touch()
    except Exception as e:
        metrics["error"] = str(e)
        manifest["error"] = str(e)
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".error").write_text(str(e))

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f)

    # Log metrics to MLflow for Azure ML Studio tracking (all wrapped to prevent step failure)
    try:
        mlflow.log_param("engine", "flaml")
        mlflow.log_param("recipe", recipe.get("recipe_name", "unknown"))
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("target_column", target_col)
        if "best_metric" in metrics:
            mlflow.log_metric("best_metric", float(metrics["best_metric"]))
        if "error" not in metrics:
            mlflow.log_param("best_estimator", metrics.get("best_estimator", "unknown"))
        mlflow.log_metric("dataset_rows", int(X.shape[0]))
        mlflow.log_metric("dataset_cols", int(X.shape[1]))
        mlflow.log_dict(metrics, "flaml_recipe_metrics.json")
        mlflow.log_dict(manifest, "flaml_recipe_manifest.json")
        if "error" in metrics:
            mlflow.log_param("flaml_recipe_error", metrics["error"])
    except Exception as mlflow_err:
        print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")


if __name__ == "__main__":
    main()
