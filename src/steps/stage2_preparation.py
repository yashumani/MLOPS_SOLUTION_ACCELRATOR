import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mlflow

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_validator import drop_high_cardinality


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prep_dataframe(df: pd.DataFrame):
    # Identify categorical and numeric columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Impute missing values: mean for numeric, mode for categorical
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())
    for c in cat_cols:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "UNKNOWN")

    # Drop extreme high-cardinality categorical features (>100 unique)
    df, dropped = drop_high_cardinality(df, cat_cols, max_unique=100)
    return df, dropped


def generate_report(df: pd.DataFrame, dropped: list[str]) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "na_total": int(df.isna().sum().sum()),
        "cat_cols": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "num_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "dropped_high_cardinality": dropped,
        "dropped_high_cardinality_count": len(dropped),
    }


def save_outputs(df: pd.DataFrame, report: dict, report_dir: str, dataset_out: str):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / "prep_report.json", "w") as f:
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

    df = load_csv(args.dataset_in)
    df2, dropped = prep_dataframe(df)
    report = generate_report(df2, dropped)
    save_outputs(df2, report, args.report_dir, args.dataset_out)

    # Log preparation parameters and metrics to MLflow
    try:
        mlflow.log_param("impute_numeric", "mean")
        mlflow.log_param("impute_categorical", "mode")
        mlflow.log_param("high_cardinality_threshold", 100)
        mlflow.log_metric("rows", int(report["rows"]))
        mlflow.log_metric("cols", int(report["cols"]))
        mlflow.log_metric("na_total", int(report["na_total"]))
        mlflow.log_metric("num_cols_count", len(report["num_cols"]))
        mlflow.log_metric("cat_cols_count", len(report["cat_cols"]))
        mlflow.log_metric("high_cardinality_dropped", int(report["dropped_high_cardinality_count"]))
        try:
            mlflow.log_dict(report, "prep_report.json")
        except Exception as artifact_err:
            print(f"⚠️  MLflow artifact logging failed (non-fatal): {artifact_err}")
    except Exception as e:
        print(f"⚠️  MLflow logging failed (stage2): {e}")


if __name__ == "__main__":
    main()
