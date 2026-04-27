#!/usr/bin/env python3
"""Batch submit all V3 pipeline configs to Azure ML.

Usage:
    python batch_submit_all.py [--force_rerun] [--dry_run]
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _azure_ctx import load_azure_context, MissingAzureContextError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
PIPELINES_DIR = ROOT / "pipelines"

# Azure ML connection — fail closed if env vars missing
try:
    _ctx = load_azure_context()
except MissingAzureContextError as _exc:
    print(f"❌ {_exc}", file=sys.stderr)
    sys.exit(2)
SUB = _ctx.subscription_id
RG = _ctx.resource_group
WS = _ctx.workspace_name
COMPUTE = _ctx.compute

# Baseline job for classification telecom churn (has drift_baseline output)
BASELINE_JOB = "jovial_animal_l93wygps2h"

# All configs to submit (order: classification first, then regression)
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


def submit_one(config_name: str, force_rerun: bool = True, baseline_job: str = None):
    """Submit a single pipeline job and return (config, job_name, success)."""
    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        print(f"  SKIP: {config_name} not found")
        return config_name, None, False

    cmd = [
        sys.executable, str(PIPELINES_DIR / "submit_pipeline.py"),
        "--config", str(config_path),
        "--subscription_id", SUB,
        "--resource_group", RG,
        "--workspace_name", WS,
        "--compute", COMPUTE,
    ]
    if force_rerun:
        cmd.append("--force_rerun")
    if baseline_job:
        cmd.extend(["--baseline_job", baseline_job])

    print(f"\n{'='*70}")
    print(f"  SUBMITTING: {config_name}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        output = result.stdout + result.stderr
        
        # Extract job name from output
        job_name = None
        for line in output.split("\n"):
            if "Submitted job:" in line:
                job_name = line.split("Submitted job:")[-1].strip()
            elif "Web View:" in line:
                print(f"  {line.strip()}")

        if job_name:
            print(f"  OK: {job_name}")
            return config_name, job_name, True
        else:
            print(f"  FAILED - no job name in output")
            # Print last 10 lines for debugging
            for line in output.strip().split("\n")[-10:]:
                print(f"    {line}")
            return config_name, None, False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 900s")
        return config_name, None, False
    except Exception as e:
        print(f"  ERROR: {e}")
        return config_name, None, False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_rerun", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Batch submitting {len(CONFIGS)} pipeline jobs")
    print(f"  Compute: {COMPUTE}")
    print(f"  Force rerun: {args.force_rerun}")
    
    if args.dry_run:
        for c in CONFIGS:
            exists = (CONFIGS_DIR / c).exists()
            print(f"  {'OK' if exists else 'MISSING'}: {c}")
        return

    results = []
    for i, config_name in enumerate(CONFIGS):
        # Only use baseline for the original telecom churn config
        baseline = BASELINE_JOB if config_name == "config_classification_telecom_churn_azureml.yml" else None
        
        config, job, ok = submit_one(config_name, args.force_rerun, baseline)
        results.append((config, job, ok))
        
        # Small delay between submissions to avoid throttling
        if i < len(CONFIGS) - 1:
            time.sleep(5)

    # Summary
    print(f"\n{'='*70}")
    print("BATCH SUBMISSION SUMMARY")
    print(f"{'='*70}")
    success = sum(1 for _, _, ok in results if ok)
    print(f"  Submitted: {success}/{len(CONFIGS)}")
    print()
    for config, job, ok in results:
        status = f"OK: {job}" if ok else "FAILED"
        print(f"  {config:<55} {status}")
    
    if success > 0:
        print(f"\n  Monitor at: https://ml.azure.com/experiments?wsid=/subscriptions/{SUB}/resourcegroups/{RG}/workspaces/{WS}")


if __name__ == "__main__":
    main()
