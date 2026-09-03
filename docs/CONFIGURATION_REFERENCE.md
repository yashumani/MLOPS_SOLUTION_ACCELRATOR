# Configuration Reference

Current as of: 2026-08-02

V3 is config-driven. Dataset paths, task type, engines, recipes, quality gates, drift behavior, and retrain policy must come from config or explicit CLI parameters, not hardcoded stage logic.

## Main Config Files

Production configs live under `configs/`.

Examples:

| Task | Config |
|---|---|
| Classification | `configs/config_classification_telecom_churn_azureml.yml` |
| Regression | `configs/config_regression_college_azureml.yml` |
| Clustering | `configs/config_clustering_online_retail_azureml.yml` |

`pipelines/submit_pipeline.py` validates configs through `src/orchestration/config_schema.py` before Azure work starts. This is the K2 gate.

## Required Top-Level Concepts

| Section/key | Purpose |
|---|---|
| `task_type` | Must be `classification`, `regression`, or `clustering`; forecasting is not in the active contract. |
| `dataset` | Dataset name, datastore path, and target column where applicable. |
| `azure_ml` | Workspace, compute, environment, and Azure ML settings when present. |
| `stages` | Stage-specific preprocessing, feature engineering, and validation settings. |
| `phases` | Phase A baseline, Phase B variants, Phase C HPO, and final evaluation settings. |
| `registry` | Registration and quality-gate behavior. |
| `random_seed` | Reproducibility seed; default behavior commonly uses `42` when absent. |

## Dataset Section

Typical shape:

```yaml
dataset:
  name: college
  target_column: Grad.Rate
  azureml_uri: azureml://subscriptions/.../datastores/mlops_blob/paths/...
  content_sha256: <sha256-of-the-exact-dataset-content>
```

Rules:

- Classification and regression configs need a target column.
- Clustering configs normally do not require a target column.
- Dataset paths must be Azure ML datastore URIs for production submissions.
- Production configs must bind the exact dataset bytes with `dataset.content_sha256`; diagnostic configs may be explicitly non-production.
- Step scripts must not create datastores or upload directly to datastores.

## Phase A Baseline

Phase A runs baseline engines, usually PyCaret and FLAML.

Expected config concepts:

```yaml
phases:
  phase_a_baseline:
    engines: [pycaret, flaml]
    recipe: recipe_baseline.yml
```

Behavior:

- `s05a` runs PyCaret baseline training.
- `s05b` runs FLAML baseline training where supported.
- `s05z` selects the Phase A champion.
- All selectable engines use the same training/CV folds, metric contract, seed, and locked-test boundary.

## Phase B Variant Search

Phase B uses intelligent variant selection instead of blind full-grid execution.

Expected config concepts:

```yaml
phases:
  phase_b_recipes:
    max_recipes: 20
    engines: [pycaret, flaml]
    top_k_from_phase_a: 3
```

Submission behavior:

- `submit_pipeline.py` resolves recipe paths into `variants_list`.
- `s06` receives `variants_list`, `engine_list`, and `time_budget_per_variant`.
- Variant recipes must be local code-upload files under `configs/recipes/`, not workspaceblobstore recipe URIs.

## Phase C HPO

Phase C tunes the selected champion family.

Expected config concepts:

```yaml
phases:
  phase_c_hpo:
    optimizer: optuna
    n_trials: 25
    timeout: 1800
```

Behavior:

- `s08` runs Optuna HPO using the Phase B champion manifest when available.
- `s09` aggregates the optimized model as the Phase C champion.

## Champion Selection, Final Audit, And Registry

Phase A, Phase B, and Phase C candidates are ranked using comparable training/CV evidence. `s10` freezes the winner before reading the Stage 2 locked test, then evaluates that one champion exactly once. Locked-test metrics may control the quality gate but never select a phase, estimator, threshold, or hyperparameter.

Quality gate defaults are warn-only unless blocking is explicitly enabled.

```yaml
registry:
  block_on_quality_fail: false
  pass_aliases: [champion]
  min_quality:
    classification: 0.50
    regression: 0.0
    clustering: 0.0
```

Rules:

- Do not raise thresholds casually; this can block otherwise valid jobs.
- `s12` handles model registration and emits `registry_info`.
- `pass_aliases` records the aliases an operator may apply after review; `s12`
  never applies an alias or MLflow stage automatically.
- Registration skip/block reasons should be documented in `registry_info`.

## Drift Config

Standalone drift settings live in `configs/drift_config.yaml`.

Concepts:

| Setting area | Purpose |
|---|---|
| Feature drift thresholds | PSI/Evidently thresholds and alert behavior. |
| Prediction/concept/label drift | Standalone drift library settings. |
| Retrain policy | Thresholds consumed by `s14` to produce a decision artifact. |
| Controller mode | External-controller dry-run/submit behavior; `s13` and `s14` never submit. |
| Alert channels | Teams/ACS/email best-effort alert settings. |

Production posture:

- Drift alerts are non-blocking.
- `s13` writes `drift_report` and `drift_baseline` as evidence only.
- `s14` applies configured policy and writes decision artifacts without Azure side effects.
- External controller owns actual candidate submissions.
- Auto-promotion remains disabled/manual.

## Baseline Chaining Config/CLI

Baseline chaining is controlled at submission time with:

```bash
--drift_baseline_in azureml://.../drift_baseline/
```

`submit_pipeline.py` wraps this as an Azure ML `uri_folder` and passes it to `s13` as `baseline_in`.

Expected outcomes:

| Case | Expected report state |
|---|---|
| No `--drift_baseline_in` | `comparison_drift.available=false`; run captures fresh baseline. |
| Valid baseline folder supplied | `comparison_drift.available=true`; comparison drift runs. |
| Invalid/empty baseline folder | `baseline_status=invalid_or_empty`; comparison is skipped. |

## Recipe Files

Recipe YAMLs live under `configs/recipes/`.

Typical recipe dimensions:

- Imputation.
- Encoding.
- Scaling.
- Imbalance handling.
- Feature selection.
- Outlier handling.

Task isolation:

- Classification recipes may use class imbalance controls such as SMOTE.
- Regression recipes must not use classification-only imbalance methods.
- Clustering recipes should focus on feature scaling, selection, and dimensionality reduction.

## Environment Variables

Common operator variables:

| Variable | Purpose |
|---|---|
| `AZURE_SUBSCRIPTION_ID` | Azure subscription for scripts that read env context. |
| `AZURE_RESOURCE_GROUP` | Azure resource group. |
| `AZURE_WORKSPACE_NAME` | Azure ML workspace. |
| `AZURE_COMPUTE` | Azure ML compute target. |
| `MLOPS_STATE_DIR` | Optional submit lock/audit state directory. |
| `MLOPS_AUTO_RETRAIN_LEDGER` | Optional controller ledger path. |
| `TEAMS_WEBHOOK_URL` | Optional drift alert channel. |
| `ACS_CONNECTION_STRING` | Optional Azure Communication Services alert channel. |

Do not put secrets in repository docs or config files.

## Validation

Before submission:

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --dry_run
```

Behavioral acceptance still requires a real Azure ML job and downloaded output artifacts.
