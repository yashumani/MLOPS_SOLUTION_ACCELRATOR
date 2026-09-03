# Submission Guide

Current as of: 2026-08-02

This guide describes the supported way to submit and monitor V3 Azure ML jobs. Local step execution is not a production validation path.

## Runtime

Use the V3 runtime when running Python submission tooling on the workspace host:

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py --help
```

A known wrong environment is `azureml_py38`; it can fail before graph creation because dependencies such as `pydash` are missing.

## Canonical Entrypoint

Use only:

```bash
pipelines/submit_pipeline.py
```

This script performs:

1. K2 config schema validation.
2. Duplicate local submission lock.
3. Active Azure ML job checks by experiment.
4. Recipe/variant selection.
5. Azure ML pipeline graph construction.
6. Optional `--dry_run` graph output.
7. Azure ML job submission.
8. Last submitted job record under `~/.mlops/last_submitted_job.json`.

## Dry-Run Preflight

Dry-run builds and prints the Azure ML pipeline job without submitting it.

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --experiment_name regression_college_preflight \
  --display_name regression_college_preflight_20260516 \
  --dry_run
```

Use dry-run after pipeline graph or component contract edits. Dry-run is not behavioral acceptance.

## Standard Submission

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster
```

Add `--wait --stop_compute` when you intentionally want the submit process to wait and stop compute after completion.

## Baseline-Chained Submission

Use a previous approved `drift_baseline` URI:

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --experiment_name regression_college_auto_retrain \
  --display_name auto_retrain_regression_college_20260516 \
  --drift_baseline_in azureml://subscriptions/93044a08-5661-4f1b-b424-5eafe066a9d1/resourcegroups/mvpv1/workspaces/mlops-accelerator/datastores/mlops_blob/paths/azureml/df8ab328-9394-48ce-9495-5008ad95d745/drift_baseline/
```

Expected `s13` result when the baseline is valid:

- `comparison_drift.available=true`.
- `baseline_status=loaded`.
- Evidently and concept drift sections are populated when reference data/metrics are available.

## Intentional Resubmission

`--force` bypasses the local lock and active-job guard. It is audited under `~/.mlops/locks/.force_submit_audit.jsonl`.

Use only when you know the current active job state and want a duplicate/resubmission.

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --force
```

## Monitor Parent Job

```bash
az ml job show \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "{name:name,status:status,display_name:display_name,experiment_name:experiment_name}" \
  -o json
```

## Monitor Child Steps

```bash
az ml job list \
  --parent-job-name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "[].{step:display_name,status:status,name:name}" \
  -o table
```

Expected active terminal graph includes `s13` and `s14`. A current full run should end with `s14` after the new graph is live-submitted.

## Download Outputs

Drift report:

```bash
az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/aml_job_outputs \
  --output-name drift_report
```

Drift baseline:

```bash
az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/aml_job_baseline \
  --output-name drift_baseline
```

Azure ML may print the reusable datastore URI in the download banner even when job metadata omits `outputs.drift_baseline.path`.

`s14` artifacts after live validation:

```bash
az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/aml_job_retrain \
  --output-name retrain_decision

az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/aml_job_retrain_record \
  --output-name decision_ledger_record
```

## Current Evidence

- Current-checkout classification, regression, and clustering SDK dry-runs passed. This is graph-construction proof only.
- Exact-source classification, regression, and clustering canaries at `6447648a` completed with downloaded pipeline artifacts and exact registered-model smoke tests.
- Twelve qualification scenarios remain. Do not submit them until the legacy daily schedules are contained and the two workspace-default artifact datastores pass their recovery canary.
- First-cycle, second-cycle, and `s14` jobs named elsewhere in the docs are historical May 2026 evidence for earlier revisions, not current-source or deployed proof.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| K2 validation fails | Config schema issue. | Fix config before Azure submission. |
| Submission pauses before job name | NFS/code snapshot delay. | Wait; 10-12 minutes can be normal. |
| Active job guard blocks submission | Same experiment has active job. | Wait or use audited `--force` only if intentional. |
| `ModuleNotFoundError: pydash` before graph creation | Wrong local Python environment. | Use `/anaconda/envs/mlops_pipeline_v2/bin/python`. |
| `ReadOnlyDisabledSubscription` | Azure subscription is disabled/read-only even if cached CLI state says enabled. | Historical recovery path: restore billing/subscription state and verify a fresh ARM GET reports `Enabled`. The 2026-09-03 release check shows this blocker closed. |
| Default output upload or artifact download reports signature mismatch | Stored credentials on `workspaceblobstore` or `workspaceartifactstore` are stale. | Obtain approval and follow `OPERATIONAL_RUNBOOKS/workspace-datastore-credential-recovery.md`; do not rotate storage keys. |
| `comparison_drift.available=false` | No prior baseline provided or baseline invalid. | Supply approved `--drift_baseline_in`. |
| `s14` absent from Studio | Job was submitted before `s14` graph change. | Submit a fresh run after graph update. |

## Acceptance Rule

Dry-run and static checks are preflight only. Pipeline runtime behavior is accepted only after exact-source Azure ML job completion and downloaded output verification. Registered-model loading and deployed-endpoint inference are separate proof gates.
