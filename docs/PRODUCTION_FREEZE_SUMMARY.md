# V3 Production Freeze Summary - 2026-05-01

## Current State

The V3 pipeline is the production path for this repository. The active branch is `prod-hardening-20260425`; the current freeze head is `b526b4fb` (`docs(freeze): production docs and remaining hardening`). The latest critical pipeline behavior fix remains `fd63a2e3` (`fix(critical): clustering eval crash + quality gate defaults`).

Azure ML context:

| Setting | Value |
|---|---|
| Subscription | `93044a08-5661-4f1b-b424-5eafe066a9d1` |
| Resource group | `mvpv1` |
| Workspace | `mlops-accelerator` |
| Compute | `mlopsv2computecluster` |
| Submission entrypoint | `pipelines/submit_pipeline.py` |
| Pipeline assembly | `pipelines/pipeline_builder.py` |

The latest six failed jobs were resubmitted after the critical fix commit. All six were accepted by Azure ML. At the latest verification checkpoint, three had completed and three were still running.

| Dataset/config family | Resubmitted job | Latest status |
|---|---|---|
| `classification_telecom_churn` | `affable_oven_bkt6xg3g6r` | Running |
| `clustering_atp1d` | `gray_iron_jgsq32f4f4` | Completed |
| `clustering_churn_uplift` | `blue_shark_dfkrdkvvy1` | Completed |
| `clustering_credit_default` | `olive_forest_9wnh6dgxkc` | Completed |
| `clustering_online_retail` | `busy_king_3gnb7tzhjl` | Running |
| `clustering_online_retail_ii` | `willing_reggae_gl3cmvdgy6` | Running |

## Freeze Gates

| Gate | Current production behavior |
|---|---|
| Azure-only execution | Pipeline validation must happen through Azure ML jobs. Do not run local pipeline steps as a substitute for production validation. |
| Single orchestrator | `pipelines/pipeline_builder.py` remains the only `@dsl.pipeline` assembly path. |
| K2 config schema | `pipelines/submit_pipeline.py` validates configs before any Azure submission. Missing or invalid schema validation is a hard fail. |
| Duplicate submission guard | Submission lock and active-job checks prevent accidental duplicates. `--force` is the explicit override for intentional resubmission. |
| Read-only datastores | Step scripts must read Azure ML datastore URIs and write only to job output paths. No datastore creation, upload, or credential plumbing in steps. |
| Recipe placement | Recipe files are read from uploaded code under `configs/recipes/`, not from `workspaceblobstore`. |
| MLflow URI compatibility | Steps that use MLflow convert Azure ML `azureml://` tracking URIs to `https://` before MLflow operations. |
| Holdout isolation | Stage 4 emits `train.csv`, `holdout.csv`, and `holdout_manifest.json` beside `dataset_out`. Training and HPO prefer `train.csv`; final evaluation prefers `holdout.csv`. |
| Quality gate | `final_evaluation.py` records `quality_gate_passed`, `quality_threshold`, and `block_on_quality_fail`. Blocking is opt-in. |
| Clustering evaluation safety | Clustering final evaluation now uses numeric-only float data and aligns to `model.feature_names_in_` before `predict()`. |
| Drift gate | `s13_drift_monitor` emits drift metrics and non-blocking alerts when configured thresholds are breached. |
| Model registration | `s12_model_registration.py` inspects the model class and uses the matching MLflow flavor when possible. |

## Quality Gate Defaults

The current defaults are intentionally warn-only unless a config explicitly asks to block registration.

| Task type | Default threshold | Metric intent |
|---|---:|---|
| `classification` | `0.50` | Balanced accuracy at or above random-guess baseline. |
| `regression` | `0.0` | R2 at or above mean-predictor baseline. |
| `clustering` | `0.0` | Silhouette above zero, meaning at least some cluster separation. |

Default behavior is equivalent to:

```yaml
registry:
  min_quality:
    classification: 0.50
    regression: 0.0
    clustering: 0.0
  block_on_quality_fail: false
```

`block_on_quality_fail` defaults to `false`. When omitted, a weak champion logs warnings and metrics but does not stop the pipeline. Set it to `true` only when product owners intentionally want final evaluation to exit with code `2` and block downstream registration.

## Warnings Operators Should Understand

| Warning | Meaning | Action |
|---|---|---|
| `No sibling holdout.csv - falling back to internal split` | The job is using a legacy artifact without the Stage 4 holdout sibling. | Treat the evaluation as less reliable; submit a fresh pipeline so Stage 4 emits the holdout sibling. |
| `T17 QUALITY GATE FAIL` | Champion is invalid or below the configured threshold. | Inspect `quality_gate_passed`, champion score, and threshold. It blocks only when `registry.block_on_quality_fail=true`. |
| `pathOnCompute is not a known attribute... ignored` | Azure ML SDK warning during submission output construction. | Non-fatal. The job can still submit successfully. |
| `Submitting pipeline to Azure ML (this may take several minutes on NFS)` | Code snapshot upload is in progress. | Wait; NFS-mounted workspaces can take roughly 10 to 12 minutes. |
| `azureml.git.dirty=True` | Azure ML uploaded a dirty working tree, including uncommitted changes. | Before production submissions, commit or intentionally stash unrelated changes. |

## Latest Critical Fixes

Commit `fd63a2e3` fixed the two issues responsible for the most recent failed submissions:

1. Clustering final evaluation no longer sends raw object/categorical columns into KMeans `predict()`. It selects numeric columns, casts to `float64`, and aligns to `feature_names_in_` where available.
2. The quality gate defaults were restored to production-safe values: thresholds `0.50 / 0.0 / 0.0` and `block_on_quality_fail=false`.

The same commit also preserved the holdout isolation work and model-flavor registration hardening across:

| File | Production effect |
|---|---|
| `src/steps/final_evaluation.py` | Quality gate defaults, clustering eval safety, sibling holdout evaluation, quality metrics. |
| `src/steps/stage4_feature_engineering.py` | Writes `train.csv`, `holdout.csv`, and holdout manifest siblings. |
| `src/steps/stage5_pycaret_train.py` | Uses sibling `train.csv` when present and config-driven random seed. |
| `src/steps/stage5_flaml_train.py` | Uses sibling `train.csv` when present and config-driven random seed. |
| `src/steps/phasec_optuna_hpo.py` | Uses sibling `train.csv` when present. |
| `src/steps/s12_model_registration.py` | Uses LightGBM, XGBoost, CatBoost, or sklearn MLflow flavor as appropriate. |
| `src/steps/s13_drift_monitor.py` | Drift monitor hardening from the working tree. |

## Documentation Placement

Current production docs live in `docs/`. Historical development documents were moved outside the pipeline folder to:

`/home/azureuser/cloudfiles/code/Users/yashu.savyminds/archive/mlops-solution-accelerator-v3-docs-archive-20260501/`

That archive contains older PR body text, session history, prior production reports, and stale incident forensics. Do not use those archived files as current operating guidance.

## Remaining Freeze Watch Items

| Item | Status |
|---|---|
| Six resubmitted jobs | 3 Completed, 3 Running at latest verification checkpoint. Monitor the remaining three until terminal status is `Completed`. |
| API/UI working-tree changes | Still separate from the committed Azure ML step-script fix. Review and commit separately before freezing the UI/API surface. |
| Documentation cleanup | In progress in this working tree; not committed until explicitly requested. |