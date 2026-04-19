import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, target_col: str | None) -> pd.DataFrame:
    # Separate target if present
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    # One-hot encode categorical
    df = pd.get_dummies(df, drop_first=False)

    # Sanitize column names — LightGBM/XGBoost cannot handle special chars like [ ] { } ( ) " :
    import re
    df.columns = [re.sub(r'[^\w]', '_', c) for c in df.columns]

    # Standard scale numeric
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    # Reattach target if present
    if y is not None:
        df[target_col] = y.values
    return df


def generate_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "sample_cols": df.columns[:10].tolist(),
    }


def save_outputs(df: pd.DataFrame, report: dict, report_dir: str, dataset_out: str):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)
    out_path = Path(dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--dataset_out", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    target_col = cfg.get("dataset", {}).get("target_column")

    df = load_csv(args.dataset_in)
    df2 = preprocess(df, target_col)
    report = generate_report(df2)
    save_outputs(df2, report, args.report_dir, args.dataset_out)

    # Log preprocessing parameters and metrics to MLflow
    try:
        mlflow.log_param("categorical_encoding", "onehot")
        mlflow.log_param("scaling", "standard")
        mlflow.log_metric("rows", int(report["rows"]))
        mlflow.log_metric("cols", int(report["cols"]))
        try:
            mlflow.log_dict(report, "preprocessing_report.json")
        except Exception as artifact_err:
            print(f"⚠️  MLflow artifact logging failed (non-fatal): {artifact_err}")
    except Exception as e:
        print(f"⚠️  MLflow logging failed (stage3): {e}")


if __name__ == "__main__":
    main()
