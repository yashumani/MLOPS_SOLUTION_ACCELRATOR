# Production Handoff - V3 Pipeline

Current as of: 2026-05-01
Branch: `prod-hardening-20260425`
Commit: `fd63a2e3`
Repository: `SAVYMINDS/YS_MVP`

This handoff covers the Azure ML V3 production pipeline, its submission guardrails, quality gates, drift monitoring, and operator workflow. It supersedes older backend-only handoff and freeze-review documents now archived outside the pipeline folder.

## Operator Summary

V3 is Azure-only and component-based. Use `pipelines/submit_pipeline.py` for submissions and `pipelines/pipeline_builder.py` for orchestration. Do not create local orchestrators, local validation substitutes, or programmatic datastore writes.

The latest production code fix is `fd63a2e3`, which corrected the final evaluation behavior that caused the recent failed submissions. The six failed configs were resubmitted and accepted by Azure ML.

## Required Azure Context

| Field | Value |
|---|---|
| Subscription | `93044a08-5661-4f1b-b424-5eafe066a9d1` |
| Resource group | `mvpv1` |
| Workspace | `mlops-accelerator` |
| Compute | `mlopsv2computecluster` |

## Production Submission

Use explicit Azure context on every production submission:

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster
```

Use `--force` only when resubmitting intentionally. The script has a lock file and an active-job check to prevent accidental duplicate submissions.

## Current Resubmitted Jobs

| Config | Job name | Latest status |
|---|---|---|
| `configs/config_classification_telecom_churn_azureml.yml` | `affable_oven_bkt6xg3g6r` | Running |
| `configs/config_clustering_atp1d_azureml.yml` | `gray_iron_jgsq32f4f4` | Completed |
| `configs/config_clustering_churn_uplift_azureml.yml` | `blue_shark_dfkrdkvvy1` | Completed |
| `configs/config_clustering_credit_default_azureml.yml` | `olive_forest_9wnh6dgxkc` | Completed |
| `configs/config_clustering_online_retail_azureml.yml` | `busy_king_3gnb7tzhjl` | Running |
| `configs/config_clustering_online_retail_ii_azureml.yml` | `willing_reggae_gl3cmvdgy6` | Running |

## Gates and Controls

| Control | Owner | Behavior |
|---|---|---|
| K2 schema gate | `pipelines/submit_pipeline.py` | Hard fail before Azure submission if config validation fails. |
| Submission lock | `pipelines/submit_pipeline.py` | Blocks concurrent submission processes. |
| Active-job guard | `pipelines/submit_pipeline.py` | Prevents duplicate jobs in the same experiment unless `--force` is passed. |
| Azure-only validation | Operator process | Pipeline tests and validation must run through Azure ML submission jobs. |
| Holdout split | `src/steps/stage4_feature_engineering.py` | Writes `train.csv`, `holdout.csv`, and `holdout_manifest.json` siblings. |
| Holdout readers | s5, s9, s10 step scripts | Training/HPO use `train.csv`; final evaluation uses `holdout.csv` when present. |
| Quality gate | `src/steps/final_evaluation.py` | Warn-only by default; blocks only when `registry.block_on_quality_fail=true`. |
| Model registration flavor | `src/steps/s12_model_registration.py` | Logs native LightGBM, XGBoost, CatBoost, or sklearn flavor when possible. |
| Drift monitoring | `src/steps/s13_drift_monitor.py` | Computes drift artifacts and non-blocking alert dispatch. |

## Quality Gate Defaults

| Task | Threshold | Blocking default |
|---|---:|---|
| Classification | `0.50` | `false` |
| Regression | `0.0` | `false` |
| Clustering | `0.0` | `false` |

Configs can override thresholds through `registry.min_quality` and can opt into hard blocking with `registry.block_on_quality_fail: true`.

## Immutable Files

Do not modify these without explicit approval:

| File | Reason |
|---|---|
| `pipelines/submit_pipeline.py` | Canonical production entrypoint. |
| `pipelines/pipeline_builder.py` | Canonical `@dsl.pipeline` assembly. |
| `src/orchestration/config_schema.py` | K2 config contract. |
| `src/steps/stage5_pycaret_train.py` | Baseline PyCaret step. |
| `src/steps/stage5_flaml_train.py` | Baseline FLAML step. |
| `src/steps/aggregate_baseline.py` | Phase A aggregation. |
| `src/steps/final_evaluation.py` | Final evaluation and quality gate. |

`final_evaluation.py` was edited for the critical production fix with explicit approval because it was the failed step and the root cause was inside that file.

## Monitoring

Query a parent job status:

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

## Documentation Archive

Stale or development-facing docs were moved to:

`/home/azureuser/cloudfiles/code/Users/yashu.savyminds/archive/mlops-solution-accelerator-v3-docs-archive-20260501/`

Use the files in `docs/` as the current source of truth.