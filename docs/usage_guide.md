# V3 Usage Guide

Current as of: 2026-08-02

This guide is a concise entry point for day-to-day V3 use. For the complete command reference, see `SUBMISSION_GUIDE.md`.

## Prerequisites

- Azure ML workspace access.
- Subscription `93044a08-5661-4f1b-b424-5eafe066a9d1`.
- Resource group `mvpv1`.
- Workspace `mlops-accelerator`.
- Compute `mlopsv2computecluster`.
- Runtime `/anaconda/envs/mlops_pipeline_v2/bin/python`.
- A classification, regression, or clustering config under `configs/` with Azure ML datastore input paths and an exact `dataset.content_sha256` for production submission.

## Choose A Config

| Task | Example config |
|---|---|
| Classification | `configs/config_classification_telecom_churn_azureml.yml` |
| Regression | `configs/config_regression_college_azureml.yml` |
| Clustering | `configs/config_clustering_online_retail_azureml.yml` |

Each config controls dataset URI, task type, target column where applicable, stage parameters, Phase A engines, Phase B recipes, Phase C HPO, final evaluation behavior, and registry settings.

## Dry-Run A Job

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --dry_run
```

Dry-run verifies config validation and graph construction. It does not prove production behavior.

## Submit A Job

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster
```

Use `--force` only for intentional resubmissions after checking active jobs.

## Submit With Drift Baseline

Use this for a second-cycle drift comparison:

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --drift_baseline_in azureml://subscriptions/93044a08-5661-4f1b-b424-5eafe066a9d1/resourcegroups/mvpv1/workspaces/mlops-accelerator/datastores/mlops_blob/paths/azureml/df8ab328-9394-48ce-9495-5008ad95d745/drift_baseline/
```

Expected `s13` state for a valid baseline:

- `comparison_drift.available=true`.
- `baseline_status=loaded`.
- The separate `s14` artifact may recommend `observe_only`, `refresh_baseline`, or `candidate_retrain` depending on policy; `s13` itself emits evidence only.

## What Happens At Submission

1. K2 config validation runs before Azure work.
2. The local lock prevents concurrent submissions.
3. Active Azure ML jobs in the same experiment are checked.
4. Recipes and engines are resolved.
5. Azure ML receives the component pipeline job.
6. The submitted job name is printed and saved under `~/.mlops/last_submitted_job.json`.

NFS-mounted workspaces can spend several minutes packaging/uploading code before a job name appears.

## Monitor A Job

```bash
az ml job show \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query status \
  --output tsv
```

List child steps:

```bash
az ml job list \
  --parent-job-name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "[].{step:display_name,status:status,name:name}" \
  --output table
```

Current active graph should include `s14` for fresh submissions from the updated branch.

## Download Artifacts

```bash
az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_outputs \
  --output-name final_report

az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_outputs \
  --output-name drift_report
```

For fresh submissions, also download `retrain_decision` and `decision_ledger_record` to review the terminal `s14` decision artifacts.

## Quality Gate Behavior

Phase selection uses comparable training/CV evidence. Final evaluation records the frozen champion's one-time locked-test audit and quality-gate status in `final_report`; the locked-test score does not choose the champion.

Defaults:

| Task | Threshold | Blocking |
|---|---:|---|
| Classification | `0.50` | Warn-only |
| Regression | `0.0` | Warn-only |
| Clustering | `0.0` | Warn-only |

Set `registry.block_on_quality_fail: true` only when you intentionally want weak champions to block registration.

## Current Auto-Retrain Status

- Current-checkout classification, regression, and clustering SDK dry-runs passed, which proves graph construction only.
- The 2026-08-02 exact-source Azure canary was rejected before job creation by `ReadOnlyDisabledSubscription`; current Azure runtime acceptance remains blocked.
- May 2026 first-cycle, second-cycle, and three-task rotation jobs in `AUTO_RETRAIN_OPERATING_LEDGER.md` are historical Azure evidence for earlier revisions.
- No current-revision registered-model or deployed-inference claim follows from those historical jobs.

## More Detail

Use these documents next:

- `PIPELINE_STAGES.md` for stage behavior.
- `PIPELINE_IO_CONTRACTS.md` for artifact contracts.
- `SUBMISSION_GUIDE.md` for complete command examples.
- `AUTO_RETRAIN_OPERATING_LEDGER.md` for retrain operations.
