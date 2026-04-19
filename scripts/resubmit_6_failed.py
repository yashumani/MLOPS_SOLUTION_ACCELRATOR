#!/usr/bin/env python3
"""Resubmit ONLY the 6 failed pipeline jobs with fixes applied.
Uses single MLClient session for efficiency (avoids repeated NFS import delays).
"""
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines"))
from pipeline_builder import full_pipeline

SUB = "93044a08-5661-4f1b-b424-5eafe066a9d1"
RG = "mvpv1"
WS = "mlops-accelerator"
COMPUTE = "mlopsv2computecluster"
DATASTORE = "mlops_blob"

# ONLY the 6 failed jobs
CONFIGS = [
    "configs/config_regression_college_azureml.yml",
    "configs/config_regression_insurance_azureml.yml",
    "configs/config_regression_house_sales_azureml.yml",
    "configs/config_regression_medical_charges_azureml.yml",
    "configs/config_regression_length_of_stay_azureml.yml",
    "configs/config_classification_telco_churn_azureml.yml",
]

print("=" * 70)
print(f"RESUBMITTING {len(CONFIGS)} FAILED JOBS WITH FIXES")
print("=" * 70)

print("Initializing Azure ML client...")
ml = MLClient(DefaultAzureCredential(), SUB, RG, WS)
print("  MLClient ready\n")

dataset_uri = (
    f"azureml://subscriptions/{SUB}"
    f"/resourcegroups/{RG}"
    f"/workspaces/{WS}"
    f"/datastores/{DATASTORE}/paths/"
)

results = []
for i, cfg_rel in enumerate(CONFIGS):
    cfg_path = ROOT / cfg_rel
    if not cfg_path.exists():
        print(f"[{i+1}/{len(CONFIGS)}] SKIP: {cfg_rel} not found")
        results.append((cfg_rel, None, False))
        continue

    cfg_name = Path(cfg_rel).name
    stem = cfg_name.replace("config_", "").replace("_azureml.yml", "").replace("_local.yml", "")
    experiment = f"{stem}_v3"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    display = f"{experiment}_{ts}_{uid}"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    dataset_name = (cfg.get("dataset") or {}).get("name", "unknown")
    task_type = cfg.get("task_type", "unknown")

    print(f"\n[{i+1}/{len(CONFIGS)}] {cfg_name}")
    print(f"  experiment={experiment}  task={task_type}  dataset={dataset_name}")

    try:
        job = full_pipeline(
            config_name=cfg_name,
            dataset_folder=Input(path=dataset_uri, type="uri_folder"),
        )
        job.settings.default_compute = COMPUTE
        job.settings.force_rerun = True
        job.experiment_name = experiment
        job.display_name = display
        job.tags = {
            "dataset": dataset_name,
            "task": task_type,
            "pipeline_version": "v3",
            "resubmit": "fix_flaml_optuna_column_sanitize",
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

    if i < len(CONFIGS) - 1:
        time.sleep(2)

print(f"\n{'='*70}")
print("RESUBMISSION SUMMARY")
print(f"{'='*70}")
ok = sum(1 for _, _, s in results if s)
print(f"Submitted: {ok}/{len(CONFIGS)}\n")
for cfg, job, success in results:
    status = f"OK  {job}" if success else "FAIL"
    print(f"  {cfg:<55} {status}")
