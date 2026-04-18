import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
import mlflow


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config from local file path."""
    print(f"📖 Loading config from: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset_uri(cfg: Dict[str, Any]) -> str:
    """Build azureml:// datastore URI from config."""
    azure_cfg = cfg.get("azure_ml") or cfg.get("azureml") or {}
    ds_cfg = cfg.get("dataset", {}) or {}
    
    datastore_name = ds_cfg.get("datastore_name")
    blob_path = ds_cfg.get("blob_path")
    
    if not (azure_cfg and datastore_name and blob_path):
        raise ValueError(
            "Missing required config fields. Need: "
            "azure_ml.{subscription_id,resource_group,workspace_name}, "
            "dataset.{datastore_name,blob_path}"
        )
    
    return (
        f"azureml://subscriptions/{azure_cfg['subscription_id']}"
        f"/resourcegroups/{azure_cfg['resource_group']}"
        f"/workspaces/{azure_cfg['workspace_name']}"
        f"/datastores/{datastore_name}"
        f"/paths/{blob_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Data Ingestion (Read-only via SDK v2)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dataset_in", type=str, required=False, help="Ignored (legacy mount param)")
    parser.add_argument("--dataset_out", type=str, required=True, help="Output path for dataset CSV")
    parser.add_argument("--eda_dir", type=str, required=True, help="Output directory for EDA report")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STAGE 1: DATA INGESTION (Read-only via azureml:// URI)")
    print("=" * 80)
    
    # Load config
    cfg = load_config(args.config)
    
    # Build dataset URI and read (no mounting, no datastore creation)
    dataset_uri = build_dataset_uri(cfg)
    print(f"🔗 Reading dataset (read-only): {dataset_uri}")
    
    df = pd.read_csv(dataset_uri)
    print(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} cols")

    # Optional: enable async logging if environment variable is set
    try:
        if os.environ.get("MLFLOW_ENABLE_ASYNC_LOGGING", "").lower() in ("1", "true", "yes"):
            mlflow.enable_async_logging()
    except Exception as e:
        print(f"⚠️  Could not enable MLflow async logging: {e}")
    
    # Validate target column
    target = cfg.get("dataset", {}).get("target_column")
    if target:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Available: {df.columns.tolist()}")
        print(f"✅ Target column '{target}' found")

    # Log basic params/metrics to MLflow for Studio visibility
    try:
        task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type")
        if task_type:
            mlflow.log_param("task_type", str(task_type))
        if target:
            mlflow.log_param("target_column", str(target))
        mlflow.log_param("dataset_uri", dataset_uri)
        mlflow.log_metric("dataset_rows", int(df.shape[0]))
        mlflow.log_metric("dataset_cols", int(df.shape[1]))
        total_missing = int(df.isna().sum().sum())
        mlflow.log_metric("missing_total", total_missing)
    except Exception as e:
        print(f"⚠️  MLflow logging (basic) failed: {e}")
    
    # Save dataset (write to job outputs only)
    os.makedirs(Path(args.dataset_out).parent, exist_ok=True)
    df.to_csv(args.dataset_out, index=False)
    print(f"💾 Saved dataset to: {args.dataset_out}")
    
    # Generate EDA summary
    eda_dir = Path(args.eda_dir)
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    eda = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
    }
    
    with open(eda_dir / "eda_summary.json", "w") as f:
        json.dump(eda, f, indent=2)

    # Also log EDA summary as an MLflow artifact for convenience
    try:
        mlflow.log_dict(eda, "eda_summary.json")
    except Exception as e:
        print(f"⚠️  MLflow artifact logging failed: {e}")
    
    print(f"📊 EDA saved to: {eda_dir / 'eda_summary.json'}")
    print("=" * 80)
    print("✅ Stage 1 completed successfully (read-only, no writes to datastore)")
    print("=" * 80)


if __name__ == "__main__":
    main()
