import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import mlflow

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def feature_engineer(df: pd.DataFrame, target_col: str | None) -> pd.DataFrame:
    # Separate target
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df

    # Remove near-zero variance features
    selector = VarianceThreshold(threshold=1e-6)
    X_sel = selector.fit_transform(X)
    kept_cols = X.columns[selector.get_support(indices=True)].tolist()
    df_out = pd.DataFrame(X_sel, columns=kept_cols)

    if y is not None:
        df_out[target_col] = y.values
    return df_out, kept_cols


def generate_report(kept_cols: list) -> dict:
    return {
        "kept_feature_count": len(kept_cols),
        "kept_features_sample": kept_cols[:20],
    }


def save_outputs(df: pd.DataFrame, report: dict, report_dir: str, dataset_out: str):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / "feature_engineering_report.json", "w") as f:
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
    df2, kept = feature_engineer(df, target_col)
    report = generate_report(kept)
    save_outputs(df2, report, args.report_dir, args.dataset_out)

    # Log feature engineering params and metrics to MLflow
    try:
        mlflow.log_param("feature_selector", "VarianceThreshold")
        mlflow.log_param("variance_threshold", 1e-6)
        mlflow.log_metric("kept_feature_count", int(report["kept_feature_count"]))
        mlflow.log_metric("rows_after_fe", int(df2.shape[0]))
        mlflow.log_metric("cols_after_fe", int(df2.shape[1]))
        try:
            mlflow.log_dict(report, "feature_engineering_report.json")
        except Exception as artifact_err:
            print(f"⚠️  MLflow artifact logging failed (non-fatal): {artifact_err}")
    except Exception as e:
        print(f"⚠️  MLflow logging failed (stage4): {e}")


if __name__ == "__main__":
    main()
