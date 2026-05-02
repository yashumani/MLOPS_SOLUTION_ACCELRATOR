# V3 Usage Guide

This guide describes how to submit and monitor the production V3 Azure ML pipeline. Local step execution is not a production validation path.

## Prerequisites

- Azure ML workspace access to subscription `93044a08-5661-4f1b-b424-5eafe066a9d1`.
- Resource group `mvpv1` and workspace `mlops-accelerator`.
- Compute target `mlopsv2computecluster`.
- A YAML config under `configs/` with Azure ML datastore input paths.
- The working tree should be clean or intentionally reviewed before production submission.

## Choose a Config

Production configs live under `configs/`. Examples:

| Task | Example config |
|---|---|
| Classification | `configs/config_classification_telecom_churn_azureml.yml` |
| Regression | `configs/config_regression_college_azureml.yml` |
| Clustering | `configs/config_clustering_online_retail_azureml.yml` |

Each config controls dataset URI, task type, target column where applicable, stage parameters, Phase A engines, Phase B recipes, Phase C HPO, and final evaluation behavior.

## Submit a Job

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster
```

Intentional resubmission:

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --force
```

`--force` bypasses duplicate-submission guards. Use it only when the existing active job is known and intentional.

## What Happens at Submission

1. K2 config validation runs before Azure work.
2. The submission lock prevents concurrent local submit processes.
3. Active Azure ML jobs in the same experiment are checked.
4. The selected recipes and engines are resolved from config.
5. Azure ML receives a component pipeline job.
6. The submitted job name is printed and saved under `~/.mlops/last_submitted_job.json`.

On NFS-mounted workspaces, `ml_client.jobs.create_or_update()` can take several minutes. Seeing no immediate job ID during that upload window is normal.

## Monitor a Job

```bash
az ml job show \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query status \
  --output tsv
```

List active jobs:

```bash
az ml job list \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "[?status=='Running' || status=='Queued' || status=='Starting'].{name:name,status:status,experiment:experiment_name}" \
  --output table
```

## Quality Gate Behavior

Final evaluation computes `quality_gate_passed` and writes the value to MLflow and the final report.

Defaults:

| Task | Threshold | Blocking |
|---|---:|---|
| Classification | `0.50` | Warn-only |
| Regression | `0.0` | Warn-only |
| Clustering | `0.0` | Warn-only |

To hard-block weak champions, set:

```yaml
registry:
  block_on_quality_fail: true
```

To override thresholds, set:

```yaml
registry:
  min_quality:
    classification: 0.60
    regression: 0.10
    clustering: 0.05
```

Do not raise production thresholds casually; recent failures were caused by a dirty working tree that accidentally turned the gate into strict blocking.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| K2 validation fails | Config schema problem. | Fix the YAML config before submitting. |
| Submission appears stuck | NFS snapshot upload. | Wait 10 to 12 minutes before assuming failure. |
| Active job guard blocks submission | Same experiment already has an active job. | Wait for it or use `--force` only if intentional. |
| `pathOnCompute` warning appears | Azure ML SDK warning. | Non-fatal if a job name is printed. |
| `quality_gate_passed=false` | Champion below threshold or invalid. | Inspect champion score, threshold, and `block_on_quality_fail`. |
| `No sibling holdout.csv` | Legacy dataset artifact. | Resubmit through current Stage 4 to produce holdout siblings. |

## Documentation

Use `docs/PRODUCTION_FREEZE_SUMMARY.md` for current freeze status and `docs/COMMIT_LEDGER_20260501.md` for commit history.