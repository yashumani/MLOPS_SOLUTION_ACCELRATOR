# Production Handoff - V3 Pipeline

Current as of: 2026-05-18 UTC
Branch: `feature/auto-retrain-20260515`
Freeze head commit: `b526b4fb` (previous production freeze reference)
Repository: `SAVYMINDS/YS_MVP`

This handoff covers the Azure ML V3 production pipeline, its submission guardrails, quality gates, drift monitoring, and operator workflow. It supersedes older backend-only handoff and freeze-review documents now archived outside the pipeline folder.

## Operator Summary

V3 is Azure-only and component-based. Use `pipelines/submit_pipeline.py` for submissions and `pipelines/pipeline_builder.py` for orchestration. Do not create local orchestrators, local validation substitutes, or programmatic datastore writes.

The latest auto-retrain release slice adds safe `s13 -> s14` decisioning evidence, an artifact-only `s14` planned schedule table, and Azure-validated rotation runs for regression, classification, and clustering. It does not add recursive pipeline submission or automatic model promotion.

## Auto-Retrain Release Update

| Area | Current state |
|---|---|
| Decision stages | `s13` emits drift/baseline evidence; `s14` emits `retrain_decision` and `decision_ledger_record`. |
| Planned schedules | `s14` embeds a 3-row planned schedule table for regression, classification, and clustering in existing JSON artifacts. |
| Rotation proof | Regression `heroic_pepper_pxnq07lm2s`, classification `sleepy_cheetah_wshcvqkwbs`, and clustering `good_nutmeg_7fm8xk8rgd` completed through `s13` and `s14`. |
| Schedule names | `auto-retrain-regression-college-daily`, `auto-retrain-classification-telecom-churn-daily`, `auto-retrain-clustering-online-retail-daily`. |
| Decision outcomes | Regression `candidate_retrain`/`severe`; classification `observe_only`/`none`; clustering `observe_only`/`none`. |
| Manual gates | Model promotion and future-baseline approval remain manual and append-only. |

Scope boundary: this release covers the safe auto-retrain decision and operations layer for V3. The latest handoff slice adds the three-task planned schedule table to `s14` decision artifacts and documents Azure evidence for the rotation jobs. Broader branch work includes `s13` drift baseline lineage, controller/ledger planning, API/UI drift display, scheduling support, and documentation refreshes.

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

## Historical Resubmission Context

The table below is retained as May 2026 production-fix history, not as the current auto-retrain validation state. Use `AUTO_RETRAIN_OPERATING_LEDGER.md` for current auto-retrain evidence.

| Config | Job name | Latest status |
|---|---|---|
| `configs/config_classification_telecom_churn_azureml.yml` | `affable_oven_bkt6xg3g6r` | Running |
| `configs/config_clustering_atp1d_azureml.yml` | `gray_iron_jgsq32f4f4` | Completed |
| `configs/config_clustering_churn_uplift_azureml.yml` | `blue_shark_dfkrdkvvy1` | Completed |
| `configs/config_clustering_credit_default_azureml.yml` | `olive_forest_9wnh6dgxkc` | Completed |
| `configs/config_clustering_online_retail_azureml.yml` | `busy_king_3gnb7tzhjl` | Running |
| `configs/config_clustering_online_retail_ii_azureml.yml` | `willing_reggae_gl3cmvdgy6` | Running |

## Review Package

| Review item | Evidence |
|---|---|
| Azure validation | Rotation jobs `heroic_pepper_pxnq07lm2s`, `sleepy_cheetah_wshcvqkwbs`, and `good_nutmeg_7fm8xk8rgd` completed through `s13` and `s14`. |
| Schedule table | Downloaded `retrain_decision` and `decision_ledger_record` artifacts contain matching 3-row planned schedule tables with one current row per task. |
| Schedule state | `auto-retrain-regression-college-daily`, `auto-retrain-classification-telecom-churn-daily`, and `auto-retrain-clustering-online-retail-daily` are enabled with provisioning state `Succeeded`. |
| Active-job guard | Latest post-rotation check returned no active Azure ML jobs. |
| Compute | `mlopsv2computecluster` is `Succeeded`, `Standard_D4s_v3`, min `0`, max `8`. |
| Manual gates | No JSONL future-baseline approval was appended; promotion and baseline reuse remain operator decisions. |

Known review notes:

- The branch contains a broad auto-retrain feature set, including controller, ledger, API/UI, `s13`, `s14`, schedules, and documentation work. The latest handoff slice is specifically the `s14` planned schedule table plus Azure rotation evidence.
- `pipelines/submit_pipeline.py` and `pipelines/pipeline_builder.py` are approval-sensitive production files. Their changes should be reviewed as part of the broader auto-retrain branch, not as incidental docs work.
- The local rotation output folders currently preserve the downloaded `s14` decision artifacts. Baseline approval still requires the full manual evidence checklist in `AUTO_RETRAIN_OPERATING_LEDGER.md`.

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
| Retrain decision | `src/steps/s14_retrain_decision.py` | Emits policy decisions and ledger-shaped artifacts; does not submit pipelines. |
| Manual promotion | Operator process | Keeps model promotion and future-baseline approval outside automatic pipeline execution. |

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