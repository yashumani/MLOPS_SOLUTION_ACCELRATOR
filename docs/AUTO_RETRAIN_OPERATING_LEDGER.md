# Auto-Retrain Operating Ledger

Current as of: 2026-08-02

This document is the docs-facing operating summary for V3 auto-retrain. The append-only machine ledger remains JSONL and should not be replaced by this Markdown page. A local JSONL file is not multi-replica durable storage; production operation requires a shared server-owned store and concurrency control.

## Production Posture

Auto-retrain is implemented as safe automation:

1. `s13` detects drift and emits baseline artifacts.
2. `s14` applies policy and emits a decision artifact.
3. An external controller resolves approved baselines and submits candidate retrain jobs through `pipelines/submit_pipeline.py`.
4. Human/manual promotion remains the release gate.

No active pipeline stage recursively submits another pipeline run.

## Current Components

| Component | Owner | Purpose |
|---|---|---|
| `src/steps/s13_drift_monitor.py` | Pipeline stage | Produces `drift_report` and `drift_baseline`. |
| `src/steps/s14_retrain_decision.py` | Pipeline stage | Produces `retrain_decision` and `decision_ledger_record`. |
| `src/orchestration/auto_retrain_policy.py` | Pure policy | Converts drift/final/registry evidence into decision outcomes. |
| `src/orchestration/auto_retrain_decision_ledger.py` | Ledger utility | Appends/loads JSONL decisions and resolves approved baselines. |
| `src/orchestration/auto_retrain_controller.py` | External controller | Resolves baseline and builds/submits canonical candidate retrain commands. |
| `src/orchestration/auto_retrain_schedule_catalog.py` | Schedule catalog | Defines the three planned regression/classification/clustering auto-retrain schedules used by `s14` artifacts. |
| `scripts/run_auto_retrain_controller.py` | CLI wrapper | Runs controller in dry-run or submit mode. |
| `api/services/auto_retrain_service.py` | API service | Exposes schedule, ledger, baseline approval, and controller dry-run planning helpers. |
| `ui/pages/4_Auto_Retrain.py` | Streamlit page | Operator surface for planned schedules, baseline approvals, controller plans, and ledger review. |

## Decision Outcomes

| Outcome | Meaning | Typical next action |
|---|---|---|
| `observe_only` | Drift is low or policy says no action. | Keep monitoring. |
| `refresh_baseline` | No valid prior baseline or baseline should be refreshed. | Review and approve candidate baseline only if appropriate. |
| `candidate_retrain` | Drift severity indicates a candidate retrain/evaluation should run. | Controller may submit a candidate through `submit_pipeline.py`. |
| `promote_candidate` | Candidate meets promotion policy. | Manual promotion review; auto-promotion remains disabled. |
| `blocked` | Missing evidence, active job guard, config blocker, or policy blocker. | Fix blocker before submission/promotion. |

Feature PSI retrain decisions use prior-baseline comparison evidence from `comparison_drift.feature_psi_scores`. The top-level `feature_psi_scores` emitted by `s13` is a same-dataset smoke test for detector sanity and should not trigger candidate retraining by itself.

## Baseline Approval Rule

Only ledger records with `approved_for_future_baseline=true` may be automatically resolved as future baselines.

A completed run can produce a candidate baseline without approving it. The May 2026 second-cycle regression record is a historical example.

## Baseline Approval SOP

Future-baseline approval is append-only. Do not edit historical JSONL records to approve a baseline after the fact.

Before a baseline can be approved for future automatic resolution, an operator or ML lead must confirm:

1. The Azure ML parent job reached a terminal successful state.
2. `drift_report`, `drift_baseline`, `retrain_decision`, and `decision_ledger_record` were downloaded and inspected.
3. The `s14` decision is not `blocked`.
4. Final model metrics and data-quality warnings were reviewed for the task type.
5. The candidate baseline URI is explicit and reusable as an `azureml://.../drift_baseline/` URI.
6. Manual model-promotion status is recorded separately from baseline approval.

Approval must be represented by appending a new record or status record with `approved_for_future_baseline=true`, `output_baseline_uri` set to the reviewed baseline URI, and `promotion_status` set to an approved status such as `baseline_approved`. Candidate retrain outputs remain `approved_for_future_baseline=false` until that review happens.

## Duplicate Submission Guard

The controller refuses a submit-mode request when the ledger already contains an unresolved `candidate_retrain` or `promote_candidate` record for the same config, task type, dataset, and input baseline URI with a pending/submitted/running promotion status.

Dry-run mode can still print the planned command for operator review. A forced submit must include an explicit force reason, and the appended decision record must preserve the force metadata for audit.

## Historical Azure Evidence - May 2026

The records below prove behavior of the exact earlier revisions and jobs named in the table. They do not prove the current dirty checkout, current live schedule state, registered-model usability, or deployed inference.

| Evidence | Status |
|---|---|
| First-cycle classification validation | Completed. Produced `drift_report` and `drift_baseline`. |
| First-cycle regression validation | Completed. Produced `drift_report` and `drift_baseline`. |
| First-cycle clustering validation | Completed. Produced `drift_report` and `drift_baseline`. |
| Baseline-chained regression proof | Completed as job `loyal_owl_0h0rz9krcn`. |
| Regression second-cycle comparison | `comparison_drift.available=true`, `baseline_status=loaded`, 10 PSI features. |
| Regression policy result | `candidate_retrain`, severity `severe`, stability score `36`, max PSI `0.17703`, recommended cadence `7` days. |
| `s14` graph integration | Canonical dry-run passed and emitted `s14` outputs. |
| `s14` live Azure validation | Completed as job `brave_feijoa_j25yz3qkhn`; parent job `Completed`, child steps include completed `s13` then completed `s14`, and parent outputs include `retrain_decision` plus `decision_ledger_record`. |
| Baseline lineage validation | Downloaded artifacts from `brave_feijoa_j25yz3qkhn` preserve the original `azureml://.../drift_baseline/` URI in `comparison_drift.baseline_input_uri`, `retrain_decision.comparison.input_baseline_uri`, and `decision_ledger_record.input_baseline_uri`; the Azure mount path is retained separately as `baseline_mount_path`. |
| Controller duplicate guard | Submit-mode controller run refused an existing duplicate for `loyal_owl_0h0rz9krcn` before Azure submission; seed JSONL ledger stayed at 5 records. |
| API validation | Local FastAPI returned HTTP 200 for job status, outputs, drift, and baseline capture for `brave_feijoa_j25yz3qkhn`; `/drift` reports comparison ready, baseline loaded, 10 features, and `candidate_retrain`. |
| UI validation | Local Streamlit Drift Monitor browser smoke test rendered the fresh job with API connected, 10 features, baseline comparison ready, baseline status loaded, `candidate_retrain`, PSI table, and CSV download. |
| Three-task rotation proof | Completed regression `heroic_pepper_pxnq07lm2s`, classification `sleepy_cheetah_wshcvqkwbs`, and clustering `good_nutmeg_7fm8xk8rgd` with clean display names prefixed `auto_retrain_rotation_`. |
| `s14` planned schedule table | Downloaded `retrain_decision` and `decision_ledger_record` for all three rotation jobs; each table has 3 rows, exactly one current schedule, current schedule names match the approved catalog, and ledger metadata matches the decision payload table. |

For the current revision, classification/regression/clustering SDK dry-runs passed, but the 2026-08-02 Azure canary was rejected before job creation by `ReadOnlyDisabledSubscription`. Exact-source Azure runtime acceptance remains blocked.

## Historical Three-Task Rotation Evidence

| Task type | Parent job | Display name | Current schedule row | Decision outcome | Severity | `s14` table check |
|---|---|---|---|---|---|---|
| Regression | `heroic_pepper_pxnq07lm2s` | `auto_retrain_rotation_regression_college_20260517` | `auto-retrain-regression-college-daily` | `candidate_retrain` | `severe` | 3 rows, one current row, ledger metadata matches payload |
| Classification | `sleepy_cheetah_wshcvqkwbs` | `auto_retrain_rotation_classification_telecom_churn_20260517` | `auto-retrain-classification-telecom-churn-daily` | `observe_only` | `none` | 3 rows, one current row, ledger metadata matches payload |
| Clustering | `good_nutmeg_7fm8xk8rgd` | `auto_retrain_rotation_clustering_online_retail_20260517` | `auto-retrain-clustering-online-retail-daily` | `observe_only` | `none` | 3 rows, one current row, ledger metadata matches payload |

The planned schedule table is embedded in the historical `s14` JSON outputs rather than introduced as a new component output. It describes intended schedules; operators must query Azure for current enabled/disabled schedule truth.

## Known Baseline URI Pattern

A reusable baseline URI looks like:

```text
azureml://subscriptions/<subscription>/resourcegroups/<resource-group>/workspaces/<workspace>/datastores/mlops_blob/paths/azureml/<run-output-id>/drift_baseline/
```

Azure ML parent job metadata can omit `outputs.drift_baseline.path`. If that happens, download the output and use the URI printed by the Azure CLI banner.

## Controller Dry-Run Flow

The Streamlit **Auto Retrain** page can build the same dry-run plan without leaving the UI. It resolves the latest approved baseline, renders the canonical `submit_pipeline.py` command, and shows the pending ledger record for operator review. It does not submit a job from the UI plan action.

The page also supports explicit baseline approval by appending a new JSONL record with `approved_for_future_baseline=true`. Operators can provide either a reusable `azureml://.../drift_baseline/` URI directly or a completed baseline job name when the API can inspect the job output path.

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

Expected dry-run behavior:

- Resolves the latest approved baseline from the ledger.
- Builds a canonical `submit_pipeline.py` command.
- Prints a pending decision record.
- Does not submit and does not append a record unless configured to do so.

## Manual Promotion Flow

1. Download `final_report`, `drift_report`, `registry_info`, `retrain_decision`, and `decision_ledger_record`.
2. Confirm candidate metrics beat the production/baseline comparison criteria.
3. Confirm drift severity and data quality support retraining.
4. Confirm no task-type-specific warnings invalidate the result.
5. Approve the new model registration manually.
6. Append a new ledger record or status update marking baseline approval if the candidate baseline should be reused.

## Ledger Maintenance

Rules:

- JSONL is append-only.
- Do not rewrite previous records.
- A manual approval should append a new record/status rather than editing history.
- Keep `input_baseline_uri` and `output_baseline_uri` explicit.
- Use `approved_for_future_baseline=false` until a human approves reuse.

## Next Required Validation

The fresh `s13 -> s14` Azure proof is complete for regression, classification, and clustering. Remaining release work is operational hardening:

- Decide whether to approve any candidate baseline for future reuse by appending an approval record; do not edit existing JSONL records.
- If a controller submit proof is still required, use a non-duplicate approved baseline or an explicit `--force-submit --force-reason` override approved by an operator.
- Re-run API/UI smoke checks after deployment, not only against the local development server.
- Keep all three schedules enabled and baseline-aware; refresh schedules only when pipeline inputs or component contracts change.
