# Runbook: Auto-Retrain Operations

Current as of: 2026-08-02

## Purpose

Operate the safe auto-retrain cycle:

```text
completed run -> s13 drift report/baseline -> s14 decision -> controller dry-run/submit -> manual promotion review
```

## First-Cycle Run

Submit without a prior baseline:

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster
```

Expected drift state:

- `s13` emits `drift_report` and `drift_baseline`.
- `comparison_drift.available=false`.
- The separate `s14` decision often becomes `refresh_baseline`; `s13` does not decide or submit.

## Second-Cycle Baseline Comparison

Submit with an approved previous baseline:

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --drift_baseline_in azureml://subscriptions/93044a08-5661-4f1b-b424-5eafe066a9d1/resourcegroups/mvpv1/workspaces/mlops-accelerator/datastores/mlops_blob/paths/azureml/df8ab328-9394-48ce-9495-5008ad95d745/drift_baseline/
```

Expected drift state for a valid baseline:

- `comparison_drift.available=true`.
- `baseline_status=loaded`.
- Feature PSI and concept/Evidently checks are populated when source artifacts support them.

## Controller Dry-Run

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python scripts/run_auto_retrain_controller.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --mode dry_run \
  --skip-active-check
```

Expected behavior:

- Reads the decision ledger.
- Resolves the latest approved baseline URI for the config/task/dataset.
- Builds a canonical `submit_pipeline.py` command.
- Prints a pending record.
- Does not submit when `--mode dry_run` is used.

## Review s14 Decision Artifacts

For fresh submissions, download:

```bash
az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_retrain \
  --output-name retrain_decision

az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_retrain_record \
  --output-name decision_ledger_record
```

Review:

- `decision.outcome`.
- `decision.severity`.
- `decision.should_submit`.
- `decision.eligible_for_promotion`.
- `decision.reasons`.
- `comparison.baseline_status`.
- `source.drift_execution_id`.
- `planned_schedules_table.rows`, which should contain the three planned regression, classification, and clustering schedules and exactly one current row for the running task.

## Approval Rules

| Action | Required evidence |
|---|---|
| Approve baseline reuse | Downloaded drift baseline, valid comparison metadata, explicit manual approval. |
| Submit candidate retrain | Policy outcome `candidate_retrain` or operator override with documented reason. |
| Promote candidate model | Final evaluation evidence, registry evidence, manual approval. |
| Block action | Missing baseline, invalid report, active job guard, failed quality gate, or task-specific warning. |

## Azure State And Current Blockers

- May 2026 first-cycle, second-cycle, and `s14` rotation jobs are historical evidence for their exact earlier revisions.
- Their downloaded schedule tables describe planned schedules; they are not proof of current Azure schedule state.
- Exact-source classification, regression, and clustering canaries at `6447648a` completed with S13/S14 evidence and exact registered-model smoke tests.
- A live 2026-09-03 audit found all three legacy daily schedules enabled. Each runs a static S1-S13 graph and bypasses S14/controller policy. Disable, do not delete, these schedules after explicit owner approval.
- The two workspace-default artifact datastores have stale stored account-key credentials. Complete their bounded recovery canary before qualification resumes.
- Do not approve a historical baseline for current reuse without exact model/data/config/code identity review.

## Do Not

- Do not set `approved_for_future_baseline=true` just because a job completed.
- Do not let a running pipeline submit another pipeline recursively.
- Do not use the standalone `PipelineTrigger` or a direct schedule submission path to bypass the S14/controller decision boundary.
- Do not treat local pytest as proof of production auto-retrain behavior.
- Do not bypass duplicate-submission guards unless using audited `--force` intentionally.
