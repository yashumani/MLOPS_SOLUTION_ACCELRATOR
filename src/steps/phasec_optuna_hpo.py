import argparse
import json
import re
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import mlflow

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Setup MLflow registry URI BEFORE any mlflow.log_* calls
os.makedirs("/tmp/mlflow-registry", exist_ok=True)
mlflow.set_registry_uri("file:///tmp/mlflow-registry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--study_out", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    n_trials = cfg.get("phases", {}).get("phase_c_hpo", {}).get("n_trials", 50)
    test_size = cfg.get("stages", {}).get("stage4_feature_engineering", {}).get("train_test_splits", [0.8])[0]
    test_size = 1 - float(test_size)

    df = pd.read_csv(args.dataset_in)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in dataset for HPO")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Sanitize column names — XGBoost cannot handle special chars like [ ] { } ( ) in feature names
    X.columns = [re.sub(r'[^\w]', '_', c) for c in X.columns]

    # Encode string labels for classification (XGBoost requires numeric targets)
    _label_encoder = None
    if task_type == "classification" and y.dtype == object:
        _label_encoder = LabelEncoder()
        y = pd.Series(_label_encoder.fit_transform(y), index=y.index, name=y.name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task_type == "classification" else None)

    import optuna
    try:
        import xgboost as xgb
    except Exception:
        raise RuntimeError("XGBoost not available in environment for HPO")

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }
        if task_type == "classification":
            model = xgb.XGBClassifier(**params, objective="binary:logistic")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            return score
        else:
            model = xgb.XGBRegressor(**params, objective="reg:squarederror")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    # Train final model with best params
    import xgboost as xgb
    if task_type == "classification":
        final_model = xgb.XGBClassifier(**best_params, objective="binary:logistic")
    else:
        final_model = xgb.XGBRegressor(**best_params, objective="reg:squarederror")
    final_model.fit(X_train, y_train)

    # Save model and study
    try:
        import joblib
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        joblib.dump(final_model, model_path)
        study_dir = Path(args.study_out)
        study_dir.mkdir(parents=True, exist_ok=True)
        study_path = study_dir / "study.pkl"
        joblib.dump(study, study_path)
    except Exception as e:
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".error").write_text(str(e))
        study_dir = Path(args.study_out)
        study_dir.mkdir(parents=True, exist_ok=True)
        (study_dir / ".error").write_text(str(e))

    metrics = {"best_params": best_params, "best_score": best_value}
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f)

    # Log metrics to MLflow for Azure ML Studio tracking (wrap in try-except to make non-fatal)
    try:
        mlflow.log_param("optimizer", "optuna")
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("target_column", target_col)
        mlflow.log_param("best_params", str(best_params)[:500])
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metric("best_score", float(best_value))
        mlflow.log_metric("dataset_rows", int(df.shape[0]))
        mlflow.log_metric("dataset_cols", int(df.shape[1]))
        mlflow.log_dict(metrics, "optuna_hpo_metrics.json")
    except Exception as mlflow_err:
        print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")


if __name__ == "__main__":
    main()
