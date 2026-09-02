# Runbook: Resubmit Failed Jobs

Current as of: 2026-08-02

Use this when an Azure ML job fails and a corrected config/code snapshot needs a new submission.

## 1. Identify Failed Jobs

```bash
az ml job list \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "[?status=='Failed'].{name:name,display_name:display_name,experiment_name:experiment_name,status:status}" \
  -o table
```

For a specific parent job:

```bash
az ml job list \
  --parent-job-name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "[].{step:display_name,status:status,name:name}" \
  -o table
```

## 2. Diagnose The Failed Step

Use Azure ML Studio logs for child steps. Download outputs only after the job/step reaches a terminal state.

Check likely categories:

| Failed area | Common cause |
|---|---|
| K2 before submission | Config schema issue. |
| `s01` | Dataset URI/path/file discovery issue. |
| `s02` | Data preparation, immutable train/locked-test split, row identity, or content-fingerprint issue. |
| `s03` | Training-only encoding/scaling/imputation or task-type mismatch. |
| `s04` | Training-only feature selection, PCA, NaN, or column-name issue. |
| `s05a`/`s05b` | Engine-specific training failure, unsupported task branch, time budget. |
| `s06` | Variant recipe contract issue or engine-specific variant failure. |
| `s08`/`s09` | HPO search space or champion manifest mismatch. |
| `s10` | Selection-evidence comparability, split/execution identity, locked-test, or exact bundle loading mismatch. |
| `s12` | Registry, exact model lineage, or workspace MLflow connectivity issue. |
| `s13` | Drift evidence/baseline folder invalid, missing final report fields, or optional drift dependency issue. |
| `s14` | Drift report schema or policy artifact generation issue. |

## 3. Preflight Before Resubmission

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config <config_path> \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --dry_run
```

Dry-run must pass before a live resubmission.

## 4. Submit Again

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config <config_path> \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --experiment_name <experiment_name> \
  --display_name <clear_display_name>
```

Use `--force` only if the active-job guard blocks the submission and you have intentionally verified that duplicate submission is safe.

## 5. After Completion

Download and inspect:

- `final_report`.
- `registry_info`.
- `drift_report`.
- `drift_baseline`.
- `retrain_decision` and `decision_ledger_record` if the job includes `s14`.

## 6. Do Not

- Do not rerun individual step scripts locally as acceptance.
- Do not bypass `submit_pipeline.py`.
- Do not edit old decision ledger records to make a baseline look approved.
- Do not approve a candidate baseline/model without downloaded evidence.
