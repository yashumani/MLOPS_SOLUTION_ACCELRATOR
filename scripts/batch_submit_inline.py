#!/usr/bin/env python3
"""Submit all V3 pipelines in a single Python session (reuses MLClient).

Much faster than the subprocess approach since we avoid re-importing/re-initializing per job.
"""
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from azure.ai.ml import MLClient, Input
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
)

# Add pipelines + scripts dir to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_builder import full_pipeline  # noqa: E402
from _azure_ctx import load_azure_context, MissingAzureContextError  # noqa: E402

try:
    _ctx = load_azure_context()
except MissingAzureContextError as _exc:
    print(f"❌ {_exc}", file=sys.stderr)
    sys.exit(2)
SUB = _ctx.subscription_id
RG = _ctx.resource_group
WS = _ctx.workspace_name
COMPUTE = _ctx.compute
DATASTORE = "mlops_blob"

# Configs to submit
CONFIGS = [
    # Classification (5)
    "config_classification_telecom_churn_azureml.yml",
    "config_classification_telco_churn_azureml.yml",
    "config_classification_credit_default_azureml.yml",
    "config_classification_titanic_azureml.yml",
    "config_classification_cardiac_arrest_azureml.yml",
    # Regression (5)
    "config_regression_college_azureml.yml",
    "config_regression_insurance_azureml.yml",
    "config_regression_house_sales_azureml.yml",
    "config_regression_length_of_stay_azureml.yml",
    "config_regression_medical_charges_azureml.yml",
]

# Baseline job (only for telecom_churn which already ran)
BASELINE_JOB = "jovial_animal_l93wygps2h"

def derive_names(config_path):
    stem = Path(config_path).stem
    normalized = stem.replace("config_", "").replace("_azureml", "").replace("_local", "")
    experiment = f"{normalized}_v3"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    display = f"{experiment}_{ts}_{uid}"
    return experiment, display


def main():
    print("=" * 70)
    print(f"BATCH SUBMISSION: {len(CONFIGS)} pipelines")
    print("=" * 70)

    # Initialize MLClient ONCE
    print("Initializing Azure ML client...")
    ml = MLClient(
        ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential()),
        SUB, RG, WS,
    )
    print("  MLClient ready\n")

    dataset_uri = (
        f"azureml://subscriptions/{SUB}"
        f"/resourcegroups/{RG}"
        f"/workspaces/{WS}"
        f"/datastores/{DATASTORE}/paths/"
    )

    results = []
    for i, cfg_name in enumerate(CONFIGS):
        cfg_path = ROOT / "configs" / cfg_name
        if not cfg_path.exists():
            print(f"[{i+1}/{len(CONFIGS)}] SKIP: {cfg_name} not found")
            results.append((cfg_name, None, False))
            continue

        experiment, display = derive_names(cfg_name)
        
        # Read config for tags
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        dataset_name = (cfg.get("dataset") or {}).get("name", "unknown")
        task_type = cfg.get("task_type", "unknown")

        print(f"[{i+1}/{len(CONFIGS)}] {cfg_name}")
        print(f"  experiment={experiment}  task={task_type}  dataset={dataset_name}")

        try:
            # Build pipeline
            kwargs = dict(
                config_name=cfg_name,
                dataset_folder=Input(path=dataset_uri, type="uri_folder"),
            )
            
            job = full_pipeline(**kwargs)
            job.settings.default_compute = COMPUTE
            job.settings.force_rerun = True
            job.experiment_name = experiment
            job.display_name = display
            job.tags = {
                "dataset": dataset_name,
                "task": task_type,
                "preset": cfg.get("preset", "production"),
                "pipeline_version": "v3",
                "batch_submit": "true",
            }

            t0 = time.time()
            submitted = ml.jobs.create_or_update(job)
            elapsed = time.time() - t0
            
            job_name = submitted.name
            print(f"  OK: {job_name} ({elapsed:.0f}s)")
            print(f"  URL: https://ml.azure.com/runs/{job_name}?wsid=/subscriptions/{SUB}/resourcegroups/{RG}/workspaces/{WS}")
            results.append((cfg_name, job_name, True))

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((cfg_name, None, False))

        # Brief pause between submissions
        if i < len(CONFIGS) - 1:
            time.sleep(3)

    # Summary
    print(f"\n{'='*70}")
    print("BATCH SUBMISSION SUMMARY")
    print(f"{'='*70}")
    ok = sum(1 for _, _, s in results if s)
    print(f"Submitted: {ok}/{len(CONFIGS)}\n")
    
    for cfg, job, success in results:
        status = f"OK  {job}" if success else "FAIL"
        print(f"  {cfg:<55} {status}")
    
    print(f"\nMonitor: https://ml.azure.com/experiments?wsid=/subscriptions/{SUB}/resourcegroups/{RG}/workspaces/{WS}")


if __name__ == "__main__":
    main()
