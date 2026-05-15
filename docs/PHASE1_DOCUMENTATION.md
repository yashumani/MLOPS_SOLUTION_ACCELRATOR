# Phase 1 Documentation — MLOps Solution Accelerator V3

> **Scope**: End-to-end pipeline steps s00 through s13, covering data ingestion, preprocessing, baseline training, Phase B variant search, Phase C HPO, final evaluation, model registration, and drift detection.
>
> **Last updated**: 2026-01
> **Pipeline version**: V3 (production)
> **Codebase root**: `mlops-solution-accelerator-v3/`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Design Principles](#2-architecture--design-principles)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [Configuration Reference](#4-configuration-reference)
5. [Pipeline Submission & Monitoring](#5-pipeline-submission--monitoring)
6. [Preprocessing Pipeline (s00–s04)](#6-preprocessing-pipeline-s00--s04)
7. [Baseline Training Phase A (s05a/b/t/z)](#7-baseline-training-phase-a-s05abtz)
8. [Phase B — Variant Search (s06)](#8-phase-b--variant-search-s06)
9. [Phase C — HPO and Final Evaluation (s09/s11/s10)](#9-phase-c--hpo-and-final-evaluation-s09s11s10)
10. [Model Registration (s12)](#10-model-registration-s12)
11. [Drift Detection (s13)](#11-drift-detection-s13)
12. [Utilities & Observability](#12-utilities--observability)
13. [Common Pitfalls & Troubleshooting](#13-common-pitfalls--troubleshooting)
14. [Quick Reference Card](#14-quick-reference-card)

---

## 1. Project Overview

### 1.1 What Is the MLOps Solution Accelerator V3?

The MLOps Solution Accelerator V3 is a production-grade, **Azure ML–native** automated machine learning pipeline that takes a raw tabular dataset and delivers a registered, quality-gated model in a single end-to-end pipeline run. It replaces V2 for all active development and is the canonical production system.

The accelerator solves the following problems:

| Problem | Accelerator solution |
|---|---|
| Manual feature engineering is slow and error-prone | Automated preprocessing recipe search across 457 variant recipes |
| Choosing the right ML framework is guesswork | Dual-engine (PyCaret + FLAML) baseline with transparent per-model metrics |
| Hyperparameter tuning is expensive without warm-starting | Phase C Bayesian HPO (Optuna) seeded with the Phase B champion |
| Models degrade silently in production | Integrated PSI-based drift detection (s13) with retraining cadence recommendations |
| Final evaluation scores can be gamed by leaking holdout data | Strict train/holdout split at s04; holdout is never touched until s10 |
| Experiment tracking is fragmented | Native MLflow hierarchy with nested runs, manifests, and candidate ledger |

### 1.2 Supported Task Types

| Task type | Key | Description |
|---|---|---|
| Binary / multi-class classification | `classification` | Telecom churn, fraud detection, medical diagnosis |
| Supervised regression | `regression` | House prices, college ranking, demand forecasting |
| Unsupervised clustering | `clustering` | Customer segmentation, anomaly grouping |
| Time-series / forecasting | `timeseries` | Auto-detected via dataset profile; uses statsmodels suite |

### 1.3 Key Differentiators

- **Intelligent variant search**: 457 preprocessing recipe YAMLs scored against a live `DatasetProfile` — only the top-k relevant recipes are trained, not all 457.
- **AIM Tournament**: multi-criteria scoring system (`src/utils/aim_tournament.py`) that combines raw metric score, stability, generalisation, and compute efficiency into a weighted champion ranking.
- **Candidate Ledger**: `src/utils/candidate_ledger.py` — persistent per-run record of all models evaluated, enabling cross-run champion comparisons and auditability.
- **Three-phase architecture**: Phase A (baseline) → Phase B (variant search) → Phase C (HPO) with a holdout-based final evaluation separating each phase's champion.
- **Azure-only testing**: no local run paths, no local fallback. All jobs execute on `mlopsv2computecluster` via `submit_pipeline.py`.

---

## 2. Architecture & Design Principles

### 2.1 Core Design Principles

The following seven principles govern all changes to the V3 codebase. Any PR that violates one of these principles must be rejected.

#### Principle 1 — Azure-Only Testing
All pipeline testing must be performed by submitting Azure ML jobs via `pipelines/submit_pipeline.py`. Local script execution of step scripts is forbidden. There is no `--local` mode and no local fallback logic.

#### Principle 2 — Single Orchestration System
The pipeline uses one and only one orchestration mechanism: Azure ML component-based `@dsl.pipeline` assembled in `pipelines/pipeline_builder.py` and submitted by `pipelines/submit_pipeline.py`. Do not create alternative runners, fallback executors, or local orchestration wrappers.

#### Principle 3 — Immutable Orchestration Files
The following files may not be modified without explicit team approval:

| File | Role |
|---|---|
| `pipelines/submit_pipeline.py` | Canonical submission entrypoint |
| `pipelines/pipeline_builder.py` | `@dsl.pipeline` definition + dynamic assembly |
| `src/orchestration/config_schema.py` | Config validation schema |
| `src/steps/stage5_pycaret_train.py` | PyCaret baseline training |
| `src/steps/stage5_flaml_train.py` | FLAML baseline training |
| `src/steps/aggregate_baseline.py` | Phase A result aggregation |
| `src/steps/final_evaluation.py` | Final holdout evaluation |

#### Principle 4 — Stable Component I/O Contracts
Component YAMLs under `components/` define the inputs, outputs, and environment for each Azure ML component. The CLI arguments and I/O contract must not change without updating the corresponding component YAML. Step script internals may evolve freely.

#### Principle 5 — Config-Driven Execution
Dataset paths, task types, engine lists, recipe paths, hyperparameter search spaces, and quality gate thresholds must all come from YAML config files under `configs/`. No hardcoding of these values in step scripts.

#### Principle 6 — Read-Only Datastore Access
Step scripts have read-only access to Azure ML datastores. No step script may:
- Instantiate `BlobServiceClient`, `DefaultAzureCredential`, or `ClientSecretCredential` for datastore operations.
- Call `ml_client.datastores.get()` or `ml_client.data_assets.create_or_update()`.
- Reference recipe files from `workspaceblobstore` paths.

All data input arrives via `azureml://` datastore URIs mounted automatically by Azure ML. All outputs go to job output paths.

#### Principle 7 — Task-Type Isolation
When fixing a bug for one task type, preserve all other task-type branches. Never remove classification-specific code when fixing regression, and vice versa. The pattern is:

```python
if task_type == "classification":
    from pycaret.classification import setup, compare_models
elif task_type == "regression":
    from pycaret.regression import setup, compare_models
elif task_type == "clustering":
    from pycaret.clustering import setup, create_model
```

### 2.2 Pipeline DAG

The full pipeline DAG from s00 to s13:

| Step ID | Display Name | Script | Purpose |
|---|---|---|---|
| s00 | Data Validation | `stage0_data_validation.py` | Pre-ingestion schema/quality checks (**reserved v3.1, not wired**) |
| s01 | Ingestion | `stage1_ingestion.py` | Load from datastore, EDA, quality gates |
| s02 | Preparation | `stage2_preparation.py` | Imputation, high-cardinality drop, statistical tests |
| s03 | Preprocessing | `stage3_preprocessing.py` | Encoding, scaling, VIF check |
| s04 | Feature Engineering | `stage4_feature_engineering.py` | Feature selection, PCA, train/holdout split |
| s05a | Baseline PyCaret | `stage5_pycaret_train.py` | PyCaret `compare_models` on `train.csv` |
| s05b | Baseline FLAML | `stage5_flaml_train.py` | FLAML AutoML on `train.csv` |
| s05t | Baseline Timeseries | `stage5_timeseries_train.py` | statsmodels suite (timeseries datasets only) |
| s05z | Aggregate Baseline | `aggregate_baseline.py` | Merge Phase A results, elect Phase A champion |
| s06 | Phase B Variant Runner | `s06_phaseb_variant_runner.py` | Batch variant×engine training, nested MLflow runs |
| s09 | Phase C HPO | `phasec_optuna_hpo.py` | Optuna Bayesian HPO on Phase B champion algorithm |
| s11 | Aggregate Phase C | `aggregate_phasec.py` | Normalise HPO output into champion manifest |
| s10 | Final Evaluation | `final_evaluation.py` | Holdout evaluation of all three phase champions |
| s12 | Model Registration | `s12_model_registration.py` | Register final champion in Azure ML Registry |
| s13 | Drift Detection | `s13_drift_detection.py` (or inline) | PSI-based drift analysis against reference baseline |

**Aggregate steps** use the `z` suffix (alphabetically last in the Azure ML Studio job graph).
**Sub-steps** use letter suffixes: `a` = PyCaret, `b` = FLAML, `t` = Timeseries.

### 2.3 Data Flow

```
Azure ML Datastore (mlops_blob)
         │
         ▼ [azureml:// URI via azureml-fsspec]
       s01 Ingestion ──────────────────────────────► eda_report/
         │
         ▼ dataset_out (raw CSV)
       s02 Preparation ─────────────────────────────► prep_report/
         │
         ▼ dataset_out (clean CSV)
       s03 Preprocessing ──────────────────────────► prep3_report/
         │
         ▼ dataset_out (encoded+scaled CSV)
       s04 Feature Engineering ──────────────────────► fe_report/
         │   ├── train.csv          ◄── s05a, s05b, s05t, s06, s09
         │   └── holdout.csv        ◄── s10 ONLY
         ▼
       s05a PyCaret ─────────────────────────────────► pycaret champion
       s05b FLAML ───────────────────────────────────► flaml champion
       s05t Timeseries (optional) ──────────────────► ts champion
         │
         ▼
       s05z Aggregate Baseline ──────────────────────► Phase A champion
         │
         ▼
       s06 Phase B Variant Runner ───────────────────► Phase B champion
         │
         ▼
       s09 Phase C HPO ──────────────────────────────► Optimised model
         │
         ▼
       s11 Aggregate Phase C ────────────────────────► Phase C champion
         │
         ▼
       s10 Final Evaluation (uses holdout.csv) ──────► final_report.json
         │
         ▼
       s12 Model Registration ───────────────────────► Registry (Staging)
         │
         ▼
       s13 Drift Detection ──────────────────────────► drift_report.json
                                                         drift_baseline/
```

### 2.4 Azure ML Component Model

Each step is packaged as an Azure ML component:

- **Component YAML** (`components/<step>.yml`): declares `inputs`, `outputs`, `command`, and `environment`.
- **Environment**: `azureml:mlops-v3-unified:<version>` — a single unified Conda environment for all steps, defined in `environments/unified_conda.yml` and `environments/azureml_unified_env.yml`.
- **Pipeline assembly**: `pipeline_builder.py` loads all component YAMLs via `load_component()`, wires their inputs and outputs, and returns a `@dsl.pipeline`-decorated function.
- **MLflow URI fix** (required in all step scripts that use MLflow):

```python
import mlflow, os
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if mlflow_uri.startswith("azureml://"):
    mlflow.set_tracking_uri(mlflow_uri.replace("azureml://", "https://"))
```

This is necessary because Azure ML sets `MLFLOW_TRACKING_URI` to an `azureml://` scheme URI that the MLflow model registry API does not support.

---
## 3. Prerequisites & Environment Setup

This section describes every prerequisite required before submitting a pipeline job. Local script execution is **not** a supported testing path; all runs must go through Azure ML.

### 3.1 Python Environment

The pipeline submission script and all utility imports require **Python 3.9**.

The conda environment is defined in `config/azure_ml_environment.yml` (environment name `savvy-minds-mlops-env`). Install it on any machine that will run `submit_pipeline.py`:

```bash
# Create the environment from the definition file
conda env create -f config/azure_ml_environment.yml

# Activate before any submission or utility work
conda activate savvy-minds-mlops-env
```

> **Note**: `environments/unified_conda.yml` is the Azure ML **compute-side** environment definition used inside pipeline steps. It is **not** used for the local submission environment. Always use `config/azure_ml_environment.yml` locally.

### 3.2 Core Package Requirements

The following packages are installed by `config/azure_ml_environment.yml` plus `requirements.txt`:

| Category | Package | Minimum Version | Purpose |
|---|---|---|---|
| **Azure SDK** | `azure-ai-ml` | `>= 1.11.0` | Pipeline submission, MLClient |
| **Azure SDK** | `azure-identity` | `>= 1.14.0` | DefaultAzureCredential auth chain |
| **Azure SDK** | `azure-storage-blob` | `>= 12.17.0` | Datastore interaction |
| **Azure SDK** | `azureml-mlflow` | `>= 1.51.0` | MLflow ↔ Azure ML bridge |
| **ML** | `pycaret` | latest | Baseline + Phase B training |
| **ML** | `flaml` | latest | Baseline + Phase B training (not clustering) |
| **ML** | `scikit-learn` | `>= 1.0.0` | Feature engineering, metrics |
| **ML** | `xgboost` | `>= 1.5.0` | Gradient boosting engine |
| **ML** | `lightgbm` | `>= 3.3.0` | Gradient boosting engine |
| **ML** | `catboost` | `>= 1.0.0` | Gradient boosting engine |
| **ML** | `imbalanced-learn` | `>= 0.8.0` | SMOTE and resampling (classification only) |
| **ML** | `optuna` | latest | Phase C Bayesian HPO |
| **ML** | `mlflow` | `>= 1.28.0` | Experiment tracking |
| **Feature Eng.** | `boruta` | latest | Boruta feature selection |
| **Feature Eng.** | `category-encoders` | latest | Target encoding, binary encoding |
| **Feature Eng.** | `statsmodels` | latest | Time-series models (ARIMA, SARIMA, ETS) |
| **Data Quality** | `pandera` | latest | Schema validation |
| **Data Quality** | `great-expectations` | latest | Data contract checks |
| **EDA** | `sweetviz` | latest | Stage 1 HTML EDA report |
| **Monitoring** | `evidently` | `>= 0.4` | Drift detection (s13) |
| **Utilities** | `pyyaml` | `>= 6.0` | Config parsing |
| **Explainability** | `shap` | latest | Model explanations |

### 3.3 Azure CLI

The `az ml` CLI extension is required for job monitoring, log retrieval, and output download. Install it once:

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install the Azure ML extension
az extension add -n ml

# Log in
az login

# Set your default subscription
az account set --subscription 93044a08-5661-4f1b-b424-5eafe066a9d1
```

Verify the extension is current:

```bash
az extension show -n ml --query version -o tsv
```

### 3.4 Required Azure Resources

The following Azure resources must exist **before** submitting any pipeline job. None of these should be created by pipeline step scripts — they are pre-provisioned infrastructure.

| Resource | Value | Notes |
|---|---|---|
| **Subscription** | `93044a08-5661-4f1b-b424-5eafe066a9d1` | Your Azure subscription |
| **Resource group** | `mvpv1` | Contains the ML workspace |
| **Workspace** | `mlops-accelerator` | Azure ML workspace |
| **Compute target** | `mlopsv2computecluster` | Default compute cluster |
| **Datastore** | `mlops_blob` | READ-ONLY; holds all input datasets |

The datastore `mlops_blob` is accessed **read-only** by pipeline steps via Azure ML automatic mounting. Pipeline step scripts never write to datastores and never create credentials — all data outputs use job output paths only.

### 3.5 Authentication

The submission script authenticates via `DefaultAzureCredential`, which walks the following chain in order:

1. **ManagedIdentity** — used on Azure ML compute instances automatically
2. **AzureCLI** — uses the token from `az login` on developer machines
3. **Interactive browser** — fallback for interactive sessions

On NFS-mounted workspaces (such as the Azure ML Compute Instance):

- `az login` is typically not required because the managed identity is active.
- The full pipeline upload via `ml_client.jobs.create_or_update()` can take **10–12 minutes** due to NFS snapshot behaviour. This is expected and is not an error. A `"🚀 Submitting pipeline to Azure ML (this may take several minutes on NFS)..."` message is printed at submission time to indicate the upload is in progress.

### 3.6 Environment Variables (Optional)

Azure context can be provided via environment variables instead of (or in addition to) CLI flags. The resolution order is: **CLI flag > `azureml` config block > environment variable**.

```bash
export AZURE_SUBSCRIPTION_ID="93044a08-5661-4f1b-b424-5eafe066a9d1"
export AZURE_RESOURCE_GROUP="mvpv1"
export AZURE_WORKSPACE_NAME="mlops-accelerator"
export AZURE_COMPUTE="mlopsv2computecluster"
```

With these variables set, the `--subscription_id`, `--resource_group`, `--workspace_name`, and `--compute` CLI flags can be omitted.

### 3.7 Pre-Submission Checklist

Before every submission, verify:

- [ ] Conda environment `savvy-minds-mlops-env` is active
- [ ] Working tree is clean (or intentionally reviewed) — a dirty working tree is uploaded as-is
- [ ] Target config file exists under `configs/`
- [ ] `az account show` returns the correct subscription
- [ ] No active job for the same experiment is already running (or `--force` is intentional)
- [ ] `--dry_run` has been tested to confirm YAML validity without submitting

---

## 4. Configuration Reference

Every pipeline run is controlled by a single YAML config file. All configs live under `configs/` and follow the schema defined in `src/orchestration/config_schema.py`.

### 4.1 Minimal Required Config

A minimal config contains exactly three required sections plus `recipes`:

```yaml
experiment_name: telecom_churn_v3_azure
preset: production
task_type: classification

dataset:
  name: telecom_churn
  target_column: churn          # Required for classification and regression
  blob_path: telecom_churn.csv  # Relative path within the mlops_blob datastore
  datastore_name: mlops_blob

azureml:
  subscription_id: <AZURE_SUBSCRIPTION_ID>
  resource_group: <AZURE_RESOURCE_GROUP>
  workspace_name: <AZURE_WORKSPACE_NAME>
  compute_target: <AZURE_COMPUTE>
  environment_name_preprocessing: mlops-v3-preprocessing
  environment_name_training: mlops-v3-training

recipes:
  - file: recipes/baseline_recipe.yml
```

All other keys documented below are optional. Omitting them causes the pipeline to use built-in defaults.

### 4.2 Top-Level Keys

| Key | Type | Required | Valid values | Default | Description |
|---|---|---|---|---|---|
| `experiment_name` | string | ✅ | — | — | Azure ML experiment name. All runs for this dataset share this name. Auto-derived by `submit_pipeline.py` if omitted on the CLI. |
| `task_type` | string | ✅ | `classification`, `regression`, `clustering`, `forecasting` | — | Controls model universe, metric selection, and valid recipe filters. |
| `preset` | string | ✅ | `diagnostic`, `production` | — | `production` runs the full Phase A+B+C pipeline. `diagnostic` uses reduced trials. |
| `dataset` | mapping | ✅ | — | — | Dataset location and schema. See §4.3. |
| `azureml` | mapping | ✅ | — | — | Azure ML workspace context. See §4.4. |
| `recipes` | list | ✅ | — | — | Baseline recipe list. At minimum one file entry pointing to `recipes/baseline_recipe.yml`. |
| `holdout_fraction` | float | — | `0.01`–`0.5` | `0.2` | Fraction of data reserved for final holdout evaluation. |
| `random_seed` | integer | — | any | `42` | Global random seed for reproducibility across all steps. |
| `stage1` | mapping | — | — | see §4.5 | Stage 1 data ingestion controls. |
| `stage2` | mapping | — | — | see §4.6 | Stage 2 data preparation controls. |
| `stage3` | mapping | — | — | see §4.7 | Stage 3 preprocessing controls. |
| `stage4` | mapping | — | — | see §4.8 | Stage 4 feature engineering controls. |
| `phases` | mapping | — | — | see §4.9 | Phase B and Phase C configuration. |
| `registry` | mapping | — | — | see §4.10 | Quality gate and model registration settings. |

### 4.3 `dataset` Block

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | ✅ | — | Logical dataset name. Used in MLflow tags and output filenames. |
| `blob_path` | string | ✅ | — | Relative path of the CSV inside the datastore (e.g. `telecom_churn.csv`). |
| `datastore_name` | string | ✅ | `mlops_blob` | Azure ML datastore name. Must be a READ-ONLY datastore. |
| `target_column` | string | classification/regression only | — | Column containing labels/values to predict. **Must be omitted for clustering**. |
| `local_path` | string | — | — | Optional local CSV path for non-Azure testing (not a production path). |
| `delimiter` | string | — | `,` | CSV field delimiter (e.g. `;` for European CSVs). |
| `encoding` | string | — | `utf-8` | File encoding. Use `latin-1` for datasets with non-UTF-8 characters such as £. |

**Cross-field validation**: `target_column` is required when `task_type` is `classification` or `regression`, and is forbidden when `task_type` is `clustering`.

### 4.4 `azureml` Block

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `subscription_id` | string | ✅ | — | Azure subscription ID. Use the placeholder `<AZURE_SUBSCRIPTION_ID>` in committed files; supply the real value via CLI or env var. |
| `resource_group` | string | ✅ | — | Azure resource group containing the workspace. |
| `workspace_name` | string | ✅ | — | Azure ML workspace name. |
| `compute_target` | string | ✅ | — | Compute cluster name (e.g. `mlopsv2computecluster`). |
| `environment_name_preprocessing` | string | — | `mlops-v3-preprocessing` | Azure ML environment for data processing steps (s1–s4). |
| `environment_name_training` | string | — | `mlops-v3-training` | Azure ML environment for training steps (s5a, s5b, s06, s09, s10). |
| `environment` | string | — | `mlops-v3-unified:20` | Unified environment tag. Overrides both preprocessing and training env names if provided. |

> **Security note**: Never commit real subscription IDs, resource group names, or workspace names to source control. Use `<AZURE_SUBSCRIPTION_ID>` style placeholders in YAML files and supply real values via CLI flags or environment variables.

### 4.5 `stage1` Block — Data Ingestion Controls

Stage 1 loads the raw dataset, generates EDA, and auto-detects time-series characteristics.

| Key | Type | Default | Description |
|---|---|---|---|
| `min_rows` | integer | `100` | Pipeline fails fast if the loaded dataset has fewer rows. |
| `max_missing_pct` | integer | `50` | Maximum allowed percentage of missing values across the dataset before Stage 1 raises an error. |
| `generate_sweetviz` | boolean | `false` | Generate a Sweetviz HTML EDA report. Set to `true` during debugging; leave `false` in production to save memory. |
| `eda_sample_size` | integer | `10000` | Maximum rows passed to Sweetviz to avoid OOM on large datasets. |
| `classification_min_samples_per_class` | integer | `30` | Minimum samples per class for classification. Fewer triggers a warning. |

### 4.6 `stage2` Block — Data Preparation Controls

Stage 2 applies data cleaning based on Stage 1 recommendations.

| Key | Type | Default | Description |
|---|---|---|---|
| `imputation_strategy` | string | `"from_stage1"` | How to determine imputation. `"from_stage1"` uses EDA recommendations; `"none"` skips imputation. |
| `statistical_tests_enabled` | boolean | `true` | Run normality (Shapiro-Wilk), outlier (IQR), and correlation (VIF) tests. Results inform Stage 3 scaling choices. |

### 4.7 `stage3` Block — Preprocessing Controls

Stage 3 applies recipe-driven imputation, encoding, and scaling.

| Key | Type | Default | Description |
|---|---|---|---|
| `adaptive_scaling` | boolean | `true` | If `true`, selects `RobustScaler` for non-normal distributions (detected by Stage 2 normality tests) and `StandardScaler` otherwise. If `false`, always uses `StandardScaler`. |
| `multicollinearity_check` | boolean | `true` | Run VIF (Variance Inflation Factor) analysis. Features with VIF > 10 are flagged; high-VIF features may be dropped depending on recipe settings. |

### 4.8 `stage4` Block — Feature Engineering Controls

Stage 4 applies feature selection and dimensionality reduction.

| Key | Type | Default | Description |
|---|---|---|---|
| `selection_method` | string | `"boruta"` | Feature selection algorithm. Options: `boruta` (wrapper, classification/regression only), `mutual_info` (filter, all tasks), `variance` (filter, all tasks — recommended for clustering). |
| `apply_pca_threshold` | integer | `100` | Apply PCA only if the number of features after selection exceeds this threshold. Set lower for clustering (e.g. `50`). |
| `pca_variance_retained` | float | `0.95` | Fraction of variance to retain when PCA is applied. |
| `imbalance_detection` | boolean | `true` | Detect class imbalance and log the imbalance ratio. **Not applicable for clustering** — set to `false`. |

> **Task-type rules for Stage 4**: `boruta` requires a target column and is incompatible with clustering. Always use `selection_method: variance` for clustering configs.

### 4.9 `phases` Block — Phase B and Phase C Configuration

The `phases` block contains two sub-blocks: `phase_b_recipes` (legacy hardcoded selection) and `phase_b` (Phase 1 intelligent selection, activated via `--use_phase1`).

#### 4.9.1 `phases.phase_b_recipes` — Legacy Variant Selection

Used when `--use_phase1` is **not** passed. Controls tier-based recipe selection.

| Key | Type | Default | Description |
|---|---|---|---|
| `library` | string | `variant_search` | Recipe library subdirectory. Options: `variant_search`, `v1_generated`, `enterprise_lightning_fast`. |
| `tier` | string | `balanced_performance` | Variant tier. Options: `lightning_fast`, `quick_exploration`, `balanced_performance`, `high_performance`, `state-of-the-art`. |
| `max_recipes` | integer | `8` | Maximum number of recipe variants to select. Clustering jobs typically use `10`; classification/regression use `8`. |
| `runtime_budget_sec` | integer | `300` | Maximum estimated per-variant runtime filter. Variants estimated to exceed this budget are excluded from selection. |
| `random_selection` | boolean | `false` | `false` = deterministic alphabetical selection. `true` = random seeded selection. |

#### 4.9.2 `phases.phase_b` — Phase 1 Intelligent Selection

Used when `--use_phase1` is passed. Controls intelligent variant scoring and recommendation.

| Key | Type | Default | Description |
|---|---|---|---|
| `enable_profiling` | boolean | `true` | Profile the dataset to derive statistical characteristics before scoring variants. |
| `profiling_output_path` | string | `"outputs/dataset_profile.json"` | Local path for the dataset profile output. |
| `library_dir` | string | — | Absolute or relative path to the recipe directory for scoring (e.g. `configs/recipes/classification/variant_search`). |
| `max_variants` | integer | `20` | Maximum number of variants to select after scoring. |
| `selection_strategy` | string | `"scored"` | How variants are selected. Options: `scored` (intelligent, by relevance score), `alphabetical`, `random_seeded`. |
| `min_relevance_score` | float | `30.0` | Minimum relevance score (0–100) for a variant to be included. Variants below this threshold are dropped. |
| `diversity_boost` | boolean | `true` | Ensure the selected variants cover a diverse set of preprocessing strategies, rather than clustering around one top-scoring approach. |
| `runtime_budget_sec` | integer | `180` | Estimated runtime filter applied before final selection. |
| `time_budget_per_variant` | integer | `300` | Maximum FLAML time budget per variant in seconds. Adaptive: larger datasets may receive a higher budget. |
| `engines` | list of strings | `["pycaret", "flaml"]` | Training engines to run each variant with. Clustering configs must set `["pycaret"]` only — FLAML does not support clustering. |

##### `phases.phase_b.planner` — Adaptive Planner Mode

Activated by passing `--enable_planner` on the CLI (requires `--use_phase1`).

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `false` | Enable two-round adaptive planner. Override via `--enable_planner`. |
| `round1_max_variants` | integer | `40` | Round 1 proxy training pool size. Fast proxy metrics are computed for this many variants. |
| `round2_max_variants` | integer | `10` | Round 2 full training shortlist. Only the top variants from Round 1 proceed to full training. |
| `proxy_prune_threshold` | float | `0.50` | Minimum proxy metric score to pass Round 1 pruning. Variants below this threshold are discarded. |
| `cache_enabled` | boolean | `true` | Reuse cached preprocessing transforms across rounds to avoid redundant computation. |

#### 4.9.3 `phases.phase_c_hpo` — Hyperparameter Optimisation

| Key | Type | Default | Description |
|---|---|---|---|
| `n_trials` | integer | `50` | Number of Optuna trials. Production: `50`–`100`. Diagnostic: `10`–`20`. |
| `timeout_seconds` | integer | `3600` | Hard time cap for HPO (1 hour). Optuna will stop after this duration even if `n_trials` has not been reached. |

### 4.10 `registry` Block — Quality Gate

| Key | Type | Default | Description |
|---|---|---|---|
| `block_on_quality_fail` | boolean | `false` | If `true`, the final evaluation step exits with a non-zero code when the champion metric is below the threshold, blocking downstream steps. |
| `min_quality.classification` | float | `0.50` | Minimum `balanced_accuracy_score` for a classification champion to pass the quality gate. |
| `min_quality.regression` | float | `0.0` | Minimum R² for a regression champion. |
| `min_quality.clustering` | float | `0.0` | Minimum silhouette score for a clustering champion. |

Default quality gate behaviour is warn-only (all thresholds set to their minimums). To enforce hard blocking:

```yaml
registry:
  block_on_quality_fail: true
  min_quality:
    classification: 0.60
    regression: 0.10
    clustering: 0.05
```

### 4.11 Available Config Files

The following configs ship with the repository under `configs/`:

| File | Task | Dataset | Target Column |
|---|---|---|---|
| `config_classification_telecom_churn_azureml.yml` | Classification | `telecom_churn.csv` | `churn` |
| `config_classification_bank_marketing_azureml.yml` | Classification | `bank-additional-full.csv` | `y` |
| `config_regression_college_azureml.yml` | Regression | `College.csv` | `Grad.Rate` |
| `config_regression_college_local.yml` | Regression (local path) | `College.csv` | `Grad.Rate` |
| `config_clustering_online_retail_azureml.yml` | Clustering | `online_retail.csv` | *(none)* |
| `config_clustering_online_retail_local.yml` | Clustering (local path) | `online_retail.csv` | *(none)* |

### 4.12 Config Validation (K2 Gate)

Config validation runs **before any Azure work** using the JSON schema in `src/orchestration/config_schema.py`. A validation failure exits immediately with code `2` and a descriptive error. Use `--dry_run` to test config validity without submitting:

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --dry_run
```

Common validation errors:

| Error | Cause | Fix |
|---|---|---|
| `'target_column' is a required property` | Missing `target_column` for classification/regression | Add `target_column` to the `dataset` block |
| `'task_type' is not one of [...]` | Invalid task type string | Use one of `classification`, `regression`, `clustering`, `forecasting` |
| `'preset' is not one of [...]` | Invalid preset string | Use `production` or `diagnostic` |
| Config is not valid YAML | Syntax error (tabs, indentation) | Validate YAML structure with `python -c "import yaml; yaml.safe_load(open('config.yml'))"` |

---

## 5. Pipeline Submission & Monitoring

### 5.1 Standard Submission

The canonical submission command:

```bash
cd /home/azureuser/cloudfiles/code/Users/yashu.savyminds/mlops-solution-accelerator-v3

python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster
```

With environment variables pre-set, the same command shortens to:

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml
```

### 5.2 Complete CLI Reference

`pipelines/submit_pipeline.py` — all supported flags:

#### 5.2.1 Required Flags

| Flag | Type | Description |
|---|---|---|
| `--config` | string | Path to the YAML config file (e.g. `configs/config_classification_telecom_churn_azureml.yml`). This is the only truly required flag — all Azure context can come from the config or environment variables. |

#### 5.2.2 Azure Context Flags

All four flags below fall back to config `azureml` block values, then to environment variables (`AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_WORKSPACE_NAME`, `AZURE_COMPUTE`).

| Flag | Default | Description |
|---|---|---|
| `--subscription_id` | from config / `$AZURE_SUBSCRIPTION_ID` | Azure subscription ID. |
| `--resource_group` | from config / `$AZURE_RESOURCE_GROUP` | Azure resource group. |
| `--workspace_name` | from config / `$AZURE_WORKSPACE_NAME` | Azure ML workspace name. |
| `--compute` | from config / `$AZURE_COMPUTE` | Compute cluster name. |

#### 5.2.3 Naming Flags

| Flag | Default | Description |
|---|---|---|
| `--experiment_name` | auto-derived from config filename | Azure ML experiment name. All submissions using the same config share one experiment, making metric trends visible in Studio. |
| `--display_name` | auto-generated with timestamp | Unique per-run display name shown in the Studio job list (e.g. `telecom_churn_v3_azure_20260502_143512`). |

#### 5.2.4 Execution Control Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--wait` | boolean flag | off | Stream job logs to the terminal and block until the job completes (succeeded, failed, or cancelled). |
| `--stop_compute` | boolean flag | off | Stop the compute cluster after the job completes. **Requires `--wait`** — if `--wait` is not passed, this flag has no effect. |
| `--dry_run` | boolean flag | off | Build the full pipeline job object and print its YAML representation, but **do not submit** to Azure ML. Use to validate config and pipeline structure without incurring compute costs. |
| `--debug` | boolean flag | off | Enable verbose logging, print full dataset URIs and component parameters, and expand tracebacks for all caught exceptions. |

#### 5.2.5 Intelligent Variant Selection Flags

These flags activate the Phase 1 intelligent variant recommendation system. All are optional; omitting them falls back to legacy tier-based selection.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--use_phase1` | boolean flag | off | Enable Phase 1 intelligent variant runner. The `VariantRecommender` profiles the dataset and scores all available recipes, selecting the top-k most relevant variants. |
| `--enable_planner` | boolean flag | off | Enable the V3-Proposed adaptive two-round planner mode. **Requires `--use_phase1`**. Round 1 trains proxy models on a large pool; Round 2 trains full models on the pruned shortlist. |
| `--round1_max_variants` | integer | `40` | Maximum number of variants in the Round 1 proxy training pool. Only used when `--enable_planner` is active. |
| `--round2_max_variants` | integer | `10` | Maximum number of variants promoted to Round 2 full training. Only used when `--enable_planner` is active. |
| `--proxy_prune_threshold` | float | `0.50` | Variants whose Round 1 proxy metric falls below this threshold are pruned and do not enter Round 2. Only used when `--enable_planner` is active. |
| `--disable_cache` | boolean flag | off | Disable preprocessing cache. Cached transforms normally allow Round 2 variants to reuse Stage 3/4 outputs, avoiding redundant computation. Pass this flag to force full recomputation. |

#### 5.2.6 Advanced Filtering and Gating Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--bundles_dir` | string | `None` | Path to a `variant_bundles/<task>` directory. Enables AIM-Tournament bundle gating, which pre-selects variant bundles based on data signals (imbalance ratio, dataset size, feature count) before the VariantRecommender runs. |
| `--imputation_preset` | string | `None` | Filter the selected variant pool to only variants whose imputation method matches the specified preset. Choices: `auto`, `statistical`, `ml_based`, `removal`, `pandas_native`, `composite`, `sampling`, `advanced`. |
| `--drift_baseline_in` | string | `None` | URI of a previous Stage 13 drift baseline output folder. When provided, the final evaluation step computes dataset drift relative to this baseline. |
| `--env_version` | string | from `azureml_unified_env.yml` | Override the Azure ML environment tag. Format: `name:version` (e.g. `mlops-v3-unified:23`). |

#### 5.2.7 Safety Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--force` | boolean flag | off | Bypass both duplicate-submission guards (lock file and active-job check). **Every use is AUDITED** — an entry is appended to `~/.mlops/locks/.force_submit_audit.jsonl` with timestamp, config name, and experiment. Use only when an active job is known and intentional. |

### 5.3 Submission Flow (What Happens After `submit_pipeline.py`)

When `submit_pipeline.py` is invoked, the following sequence executes before any Azure work begins:

1. **K2 config validation** — The YAML config is parsed and validated against the JSON schema in `src/orchestration/config_schema.py`. If validation fails, the script exits with code `2` immediately without contacting Azure.

2. **Duplicate-submission guards** (skipped with `--force`):
   - A **lock file** at `~/.mlops/locks/.submit.lock` is acquired. A second concurrent local submit process will block until the lock is released.
   - An **active-job check** queries Azure ML for running or queued jobs under the same experiment name. If any are found, the submit is aborted with an informational message.

3. **Recipe selection** — The tier-based selector (or Phase 1 recommender if `--use_phase1`) resolves the comma-separated variant list to pass to the s06 step.

4. **Bundle gating** — If `--bundles_dir` is set, AIM-Tournament computes data signals and selects enabled variant bundles, further refining the variant list.

5. **Imputation preset filtering** — If `--imputation_preset` is set, the variant list is filtered to only variants matching the specified imputation family.

6. **Pipeline job construction** — `pipeline_builder.py` is called to construct the `@dsl.pipeline` object with all I/O wired, environment versions set, and all parameters propagated.

7. **Azure ML submission** — `MLClient.jobs.create_or_update()` uploads the pipeline job. On NFS-mounted workspaces this step takes **10–12 minutes**. The submitted job name and Studio URL are printed on success.

8. **State persistence** — The job name is written to `~/.mlops/last_submitted_job.json` for retrieval by monitoring scripts.

### 5.4 Submission Examples

#### Classification — Standard Production Run

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait
```

#### Regression

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait
```

#### Clustering

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_clustering_online_retail_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait
```

#### With Phase 1 Intelligent Selection

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --use_phase1 \
  --wait
```

#### With Adaptive Planner (Two-Round)

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --use_phase1 \
  --enable_planner \
  --round1_max_variants 40 \
  --round2_max_variants 10 \
  --proxy_prune_threshold 0.50 \
  --wait
```

#### Dry Run (Validate Without Submitting)

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --dry_run
```

#### Intentional Force Resubmit

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --force \
  --wait
```

`--force` is audited; see `~/.mlops/locks/.force_submit_audit.jsonl` for the audit trail.

#### Stop Compute After Completion (Cost Control)

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait \
  --stop_compute
```

`--stop_compute` has no effect without `--wait`.

### 5.5 Monitoring Jobs

#### Retrieve the Last Submitted Job Name

```bash
cat ~/.mlops/last_submitted_job.json
```

#### Check Job Status via CLI

```bash
az ml job show \
  --name <JOB_NAME> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query status \
  --output tsv
```

#### List Active Jobs

```bash
az ml job list \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "[?status=='Running' || status=='Queued' || status=='Starting'].{name:name,status:status,experiment:experiment_name}" \
  --output table
```

#### Stream Logs (After Submission Without `--wait`)

```bash
az ml job stream \
  --name <JOB_NAME> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator
```

#### List Child Steps

Each pipeline job spawns child step jobs. To list them:

```bash
az ml job list \
  --parent-job-name <PIPELINE_JOB_NAME> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "[].{name:name, display_name:display_name, status:status}" \
  --output table
```

#### Azure ML Studio (Browser)

The submitted job URL is printed by `submit_pipeline.py`. Navigate to:

```
https://ml.azure.com/runs/<JOB_NAME>?wsid=/subscriptions/93044a08-5661-4f1b-b424-5eafe066a9d1/resourcegroups/mvpv1/workspaces/mlops-accelerator
```

The Studio UI shows:
- Pipeline run graph with step status (green = succeeded, red = failed, yellow = running)
- Each step's `Outputs + logs` tab with `std_log.txt` and `70_driver_log.txt`
- MLflow-logged metrics and artefacts under the `Metrics` and `Outputs` tabs

### 5.6 Downloading Job Outputs

#### Download All Outputs

```bash
az ml job download \
  --name <JOB_NAME> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --all \
  --download-path /tmp/<task>_outputs/
```

#### Download a Specific Output

```bash
# Download the final evaluation report from s10
az ml job download \
  --name <JOB_NAME> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --output-name s10.final_report \
  --download-path /tmp/final_report/
```

Key outputs to inspect by priority:

| Path (relative to download root) | Content | When to inspect |
|---|---|---|
| `s10/final_report/final_report.json` | Overall champion, all-phase comparison, final metrics | Every run — the primary result |
| `s06/leaderboard_csv/leaderboard.csv` | Phase B variant×engine results ranked by primary metric | Phase B analysis |
| `s5a/best_model/model_breakdown.csv` | Per-model performance from PyCaret baseline | Phase A analysis |
| `s5b/best_model/model_breakdown.csv` | Per-model performance from FLAML baseline | Phase A analysis |
| `s1/eda_report/eda_report.json` | Dataset profile, time-series detection flag | Data quality investigation |
| `s5a/best_model/model.pkl` | Phase A PyCaret champion serialised model | Inference or local inspection |
| `s06/champion_model/model.pkl` | Phase B champion serialised model | Inference or local inspection |

### 5.7 Troubleshooting Submission Issues

| Symptom | Likely Cause | Action |
|---|---|---|
| `K2: config schema validation FAILED` | Config YAML fails JSON schema validation | Fix the indicated field. Use `--dry_run` to iterate. |
| Submission appears stuck at "Submitting pipeline…" | NFS snapshot upload in progress | Wait 10–12 minutes before assuming failure. This is normal on Azure ML Compute Instances. |
| `Active job guard blocks submission` | Same experiment already has a running job | Wait for it to finish, or use `--force` only if the active job is intentional and known. |
| `pathOnCompute` warning in output | Azure ML SDK advisory | Non-fatal warning. A job ID will still be printed. Ignore unless no job ID appears. |
| `quality_gate_passed=false` in `final_report.json` | Champion metric below quality gate threshold | Inspect champion score and threshold. Consider reducing `min_quality` or fixing data quality issues. |
| `No module named X` in step logs | Missing package in the Azure ML compute environment | Update `environments/unified_conda.yml` and bump the environment version. |
| `FileNotFoundError: recipe_name.yml` | Recipe path not found in the uploaded code directory | Verify the recipe exists under `configs/recipes/<task>/`. |
| `MLflow tracking URI unsupported` | `MLFLOW_TRACKING_URI` set to `azureml://` scheme | Confirm the MLflow tracking URI fix (`azureml://` → `https://`) is present in the failing step script. |

### 5.8 Duplicate Submission Guards

The submission script includes two independent guards to prevent accidental duplicate runs:

1. **Lock file** — `~/.mlops/locks/.submit.lock` is a filesystem lock acquired at process start. A second concurrent invocation of `submit_pipeline.py` blocks until the first releases the lock.

2. **Active job check** — Before submitting, the script queries Azure ML for any jobs in the same experiment that are in `Running`, `Queued`, or `Starting` state. If any are found, the submission is aborted with a message identifying the active job.

Both guards are bypassed by `--force`. **Every `--force` invocation is audited** to `~/.mlops/locks/.force_submit_audit.jsonl` with:
- Timestamp (UTC ISO 8601)
- Config file path
- Experiment name
- Display name

To review the force-submit audit trail:

```bash
cat ~/.mlops/locks/.force_submit_audit.jsonl
```

## 6. Preprocessing Pipeline (s00 – s04)

The preprocessing pipeline transforms raw data from the Azure ML datastore into a clean, engineered, split-ready dataset ready for model training. It consists of five sequential steps: Data Validation (s00), Ingestion (s01), Preparation (s02), Preprocessing (s03), and Feature Engineering (s04). Each step is an independent Azure ML component with a well-defined I/O contract. Only s01–s04 are wired into the active production pipeline; s00 is reserved for v3.1.

---

### 6.1 Stage 0 — Data Validation (s00)

> **⚠️ NOT WIRED INTO ACTIVE PIPELINE.** This component is reserved for v3.1. The current production pipeline begins at s01 (Ingestion). The notes below document the component for future integration.

#### 6.1.1 Purpose

Stage 0 provides a pre-ingestion guardrail that validates dataset schema, type integrity, missing-value thresholds, and task-specific constraints (class balance for classification, numeric feature count for clustering) *before* any expensive downstream processing begins. Critical failures exit with code `1` and abort the pipeline; warnings proceed.

#### 6.1.2 Component I/O Contract

**Component YAML**: `components/stage0_data_validation.yml`

| Direction | Name | Type | Default | Description |
|---|---|---|---|---|
| **Input** | `config_name` | string | — | Path to the YAML config file (e.g. `configs/config_classification_telecom_churn_azureml.yml`) |
| **Input** | `dataset_in` | uri_folder | — | Raw dataset folder from the datastore mount |
| **Input** | `expectations_suite` | string | `"default"` | Great Expectations suite name (reserved, not yet active) |
| **Output** | `validation_report` | uri_folder | — | Folder containing `validation_results.json` |
| **Output** | `validated_dataset` | uri_file | — | Pass-through CSV (copy of input if validation passes) |

> **Known YAML note**: the component lists `validated_dataset` as `uri_file` but the step implementation writes to a folder path. This mismatch is tracked for the v3.1 fix.

#### 6.1.3 CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--config_name` | ✅ | Path to YAML config |
| `--dataset_in` | ✅ | Input dataset folder path |
| `--validation_report` | ✅ | Output path for `validation_results.json` |
| `--validated_dataset` | ✅ | Output path for the pass-through CSV |

#### 6.1.4 Core Logic

- **Schema check** (`_validate_schema`): verifies all columns declared in `stage0.expected_columns` config key are present; reports missing columns as critical failures.
- **Null check** (`_validate_null_values`): any column exceeding `stage0.max_null_pct` (default 50 %) triggers a critical failure.
- **Target column check** (`_validate_target_column`): verifies target exists, is non-null, and has the expected type.
- **Data type check** (`_validate_data_types`): verifies columns match declared types (numeric / categorical / datetime).
- **Task-specific validations**:
  - *Classification*: counts samples per class; fewer than 10 samples in any class = **critical failure**.
  - *Regression*: IQR outlier check; >10 % outliers in target = **warning**.
  - *Clustering*: requires ≥ 2 numeric feature columns; missing = **critical failure**.
- All results serialised to `validation_results.json` with keys: `passed`, `critical_failures[]`, `warnings[]`, `stats{}`.

#### 6.1.5 Config Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `stage0.expected_columns` | list[str] | `[]` | Columns that must be present |
| `stage0.max_null_pct` | float | `50.0` | Max acceptable null % per column |
| `stage0.classification_min_samples_per_class` | int | `10` | Minimum class support threshold |

#### 6.1.6 MLflow Logging

| Metric / Param | Type | Description |
|---|---|---|
| `validation_success` | metric (0/1) | 1 = all checks passed, 0 = critical failure |
| `critical_failures` | metric | Count of critical failures found |
| `warnings` | metric | Count of warnings found |
| `n_rows` | metric | Row count at validation time |
| `n_columns` | metric | Column count at validation time |
| `null_percentage` | metric | Average null % across all columns |

#### 6.1.7 Exit Codes

| Exit Code | Meaning |
|---|---|
| `0` | All checks passed (or warnings only) |
| `1` | One or more critical failures — pipeline should abort |

#### 6.1.8 Developer Notes

- The `DataValidator` class in `src/utils/data_validator.py` is the shared implementation; the step script is a thin CLI wrapper.
- The component is present in the repo and can be manually prepended to the pipeline builder for ad-hoc validation runs.
- Do **not** remove or rename the `stage0_data_validation.yml` component — it is referenced in future sprint plans.

---

### 6.2 Stage 1 — Ingestion (s01)

#### 6.2.1 Purpose

Stage 1 is the pipeline's data entry point. It resolves the `azureml://` datastore URI declared in the config, loads the raw CSV into memory, runs a comprehensive Exploratory Data Analysis (EDA), and applies RED / YELLOW / GREEN data-quality gates before writing a clean copy downstream. It is the only step that reads from the datastore; all subsequent steps consume the `dataset_out` URI file it produces.

#### 6.2.2 Component I/O Contract

**Component YAML**: `components/stage1_ingestion.yml`

| Direction | Name | Type | Description |
|---|---|---|---|
| **Input** | `config_name` | string | Path to YAML config file |
| **Input** | `dataset_in` | uri_folder | Mounted datastore folder (e.g. the `mlops_blob` mount point) |
| **Output** | `eda_report` | uri_folder | EDA artefacts folder: `eda_report.json`, optional `eda_report.html` (sweetviz), `dataset_profile.json` |
| **Output** | `dataset_out` | uri_file | Raw loaded CSV passed to s02 |

#### 6.2.3 CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--config_name` | ✅ | Path to YAML config |
| `--dataset_in` | ✅ | Mounted uri_folder path |
| `--eda_report` | ✅ | Output folder for EDA artefacts |
| `--dataset_out` | ✅ | Output CSV path |

#### 6.2.4 Core Logic

- **URI construction** (`build_dataset_uri()`): reads `dataset.blob_path` and `dataset.datastore_name` from config; constructs `azureml://subscriptions/{sub}/resourcegroups/{rg}/workspaces/{ws}/datastores/{ds}/paths/{blob_path}`.
- **Datastore read**: uses `azureml-fsspec` to mount and stream the CSV. No programmatic credential creation; Azure ML handles the identity token.
- **Data-quality gates** (`validate_data_quality()`):
  - 🔴 **RED (hard stop)**: row count < `stage1.min_rows` (default 1,000); any column missing > `stage1.max_missing_pct` (default 50 %); classification with < `stage1.classification_min_samples_per_class` (default 30) samples per class.
  - 🟡 **YELLOW (warning)**: missing rate moderate; class imbalance ratio < 0.1; duplicate row count > 5 %.
  - 🟢 **GREEN**: data passes all checks.
- **EDA** (`perform_eda()`): column-type inventory, per-column statistics (mean, std, min, max, missing count/%), top-20 pairwise correlations, target distribution analysis (class counts for classification; mean/std/skewness for regression).
- **Optional sweetviz report**: generated only if `stage1.generate_sweetviz: true` (default `false`). Uses `stage1.eda_sample_size` (default 10,000) rows to cap runtime.
- **EDA visualisations**: missing-value heatmap, outlier boxplots, target distribution charts — saved to `outputs/` for Azure ML Studio artefact viewer.
- Logs 20–30 metrics to MLflow via `log_eda_to_mlflow()`.

#### 6.2.5 Config Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `dataset.blob_path` | string | — | Relative path inside the datastore (e.g. `telecom_churn.csv`) |
| `dataset.datastore_name` | string | `mlops_blob` | Azure ML datastore name |
| `dataset.target_column` | string | — | Target column name (not required for clustering) |
| `stage1.min_rows` | int | `1000` | Minimum row count — RED gate |
| `stage1.max_missing_pct` | float | `50.0` | Max missing % per column — RED gate |
| `stage1.classification_min_samples_per_class` | int | `30` | Min class support — RED gate (classification only) |
| `stage1.generate_sweetviz` | bool | `false` | Whether to produce sweetviz HTML report |
| `stage1.eda_sample_size` | int | `10000` | Max rows passed to sweetviz to cap runtime |

#### 6.2.6 MLflow Logging

| Metric / Param | Type | Description |
|---|---|---|
| `task_type` | param | From config |
| `target_column` | param | From config |
| `dataset_uri` | param | Full `azureml://` URI used to load data |
| `numeric_columns` (first 10) | param | Comma-separated list |
| `categorical_columns` (first 10) | param | Comma-separated list |
| `dataset_rows` | metric | Total row count after load |
| `dataset_cols` | metric | Column count |
| `numeric_cols` | metric | Count of numeric columns |
| `categorical_cols` | metric | Count of categorical columns |
| `datetime_cols` | metric | Count of datetime columns |
| `duplicate_rows` | metric | Duplicate row count |
| `duplicate_pct` | metric | Duplicate row % |
| `constant_columns_count` | metric | Columns with zero variance |
| `high_missing_cols_count` | metric | Columns with > 50 % missing |
| `high_cardinality_cols_count` | metric | Columns with > 100 unique values |
| `missing_total` | metric | Total missing cell count |
| `avg_missing_pct` | metric | Mean missing % across columns |
| `target_class_count` | metric | (classification) Number of classes |
| `target_majority_pct` | metric | (classification) Majority class % |
| `target_minority_pct` | metric | (classification) Minority class % |
| `target_is_imbalanced` | param | (classification) Boolean string |
| `target_mean` / `target_std` / `target_skewness` | metric | (regression) Target distribution stats |

#### 6.2.7 Failure Modes

| Condition | Behaviour |
|---|---|
| `azureml-fsspec` not installed | `FileNotFoundError` on URI open — pipeline fails at step start |
| Row count below `min_rows` | RED gate fires; step exits with error message; job marked Failed |
| Sweetviz import error | Non-fatal; EDA falls back to JSON-only report |
| Target column absent (regression / classification) | Logged as warning; step proceeds but downstream steps may fail |

#### 6.2.8 Developer Notes

- **Never** pass credentials into this step. The `azureml://` URI is resolved via the Azure ML managed identity of the compute cluster.
- The `eda_report` folder is passed as an **optional** input to s02, allowing s02 to re-use EDA recommendations (imputation strategy, normality flags) without re-computing them.
- Sweetviz is disabled by default because it adds ~90 s on 200K-row datasets. Enable only for exploratory runs.

---

### 6.3 Stage 2 — Preparation (s02)

#### 6.3.1 Purpose

Stage 2 applies data-cleaning operations recommended by the Stage 1 EDA: missing-value imputation, high-cardinality column dropping, and optionally a suite of statistical tests (normality, outliers, feature-target correlation) whose results are persisted to `prep_report.json` and consumed by Stage 3 adaptive scaling.

#### 6.3.2 Component I/O Contract

**Component YAML**: `components/stage2_preparation.yml`

| Direction | Name | Type | Description |
|---|---|---|---|
| **Input** | `config_name` | string | Path to YAML config file |
| **Input** | `dataset_in` | uri_file | Loaded CSV from s01 |
| **Input** | `eda_report` | uri_folder | (optional) EDA artefacts from s01 — drives imputation strategy selection |
| **Output** | `prep_report` | uri_folder | Folder containing `prep_report.json` (imputation choices, statistical test results) |
| **Output** | `dataset_out` | uri_file | Cleaned CSV passed to s03 |

#### 6.3.3 CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--config_name` | ✅ | Path to YAML config |
| `--dataset_in` | ✅ | Input CSV path |
| `--eda_report` | ❌ | Optional EDA folder path |
| `--prep_report` | ✅ | Output folder for `prep_report.json` |
| `--dataset_out` | ✅ | Output CSV path |

#### 6.3.4 Core Logic

- **Target imputation guard** (`prep_dataframe()`): rows with a missing target value are **dropped unconditionally** for supervised tasks (classification / regression). Clustering skips this step.
- **Numeric imputation**: strategy resolved in priority order: (1) `stage2.imputation_strategy` config key, (2) recommendation from `eda_report/eda_report.json`, (3) default = **median**.
  - Supported strategies: `median`, `knn` (KNNImputer, k = 5), `iterative` (IterativeImputer with Bayesian Ridge), `ffill`.
  - > **⚠️ MEAN IMPUTATION IS EXPLICITLY FORBIDDEN** — this is a stakeholder requirement. Any PR introducing `strategy="mean"` must be rejected.
- **Categorical imputation**: always `most_frequent` (mode), regardless of config.
- **High-cardinality drop** (`drop_high_cardinality()` from `src/utils/data_validator.py`): categorical columns with > `max_unique` (default 100) unique values are dropped. This prevents memory explosion in one-hot encoding downstream.
- **Statistical tests** (`perform_statistical_tests()`):
  - *Normality*: Shapiro-Wilk for n < 5,000; Kolmogorov-Smirnov for larger datasets. Result (normal / non-normal) written per column into `prep_report.json`.
  - *Outlier detection*: IQR method (Q1 − 1.5×IQR, Q3 + 1.5×IQR) per numeric column. Outlier prevalence written per column.
  - *Feature-target correlation* (regression only): Pearson r computed for each numeric feature vs. target. Top pairs written to `prep_report.json`.
  - Stage 3 reads these results to choose per-column scaling (e.g. RobustScaler for high-outlier columns, StandardScaler for normal-distributed ones).
- **EDA generation**: `stage2_correlation_heatmap.png`, `stage2_sweetviz_report.html` (if enabled), `stage2_top_correlations.csv` saved to `outputs/` for Studio artefact viewer.

#### 6.3.5 Config Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `stage2.imputation_strategy` | string | `"from_stage1"` | Numeric imputation: `median`, `knn`, `iterative`, `ffill`, or `"from_stage1"` to auto-select from EDA |
| `stage2.statistical_tests_enabled` | bool | `true` | Whether to run normality / outlier / correlation tests |
| `stage2.high_cardinality_threshold` | int | `100` | Max unique values before a categorical column is dropped |

#### 6.3.6 MLflow Logging

| Metric / Param | Type | Description |
|---|---|---|
| `impute_numeric` | param | Actual strategy used (resolved, not "from_stage1") |
| `impute_categorical` | param | Always `"most_frequent"` |
| `high_cardinality_threshold` | param | Value used for drop logic |
| `rows` | metric | Row count after cleaning |
| `cols` | metric | Column count after cleaning |
| `na_total` | metric | Remaining NA count after imputation |
| `num_cols_count` | metric | Numeric column count |
| `cat_cols_count` | metric | Categorical column count |
| `high_cardinality_dropped` | metric | Number of columns dropped for high cardinality |
| `prep_report.json` | artifact | Full preparation report (logged via `log_dict`) |

MLflow run name: `"s02_preparation"`. Tags: `pipeline=v3_mlops`, `phase=preprocessing`, `step=s02`.

#### 6.3.7 Failure Modes

| Condition | Behaviour |
|---|---|
| All rows have missing target | Step produces a 0-row dataset; downstream training steps fail |
| `eda_report` folder absent | Falls back to `stage2.imputation_strategy` config; no error |
| KNN / iterative imputation on large dataset (> 100K rows) | May be slow; consider `median` for large datasets |

#### 6.3.8 Developer Notes

- `prep_report.json` is the **handshake artefact** between s02 and s03. If s03's `prep_report` input is absent, it falls back to config-only scaling, losing the per-column normality / outlier information.
- The high-cardinality threshold (100) is a global default but can be tuned per-run via config. Lowering it (e.g. to 50) will drop more columns but speed up encoding.

---

### 6.4 Stage 3 — Preprocessing (s03)

#### 6.4.1 Purpose

Stage 3 applies the recipe-driven preprocessing pipeline: categorical encoding, numeric scaling, and (optionally) multicollinearity filtering. It reads a preprocessing recipe YAML that specifies the encoding and scaling strategies, applies them to the cleaned dataset from s02, and produces a fully numeric, model-ready DataFrame. **SMOTE / ADASYN imbalance handling is explicitly NOT applied here** — it is deferred to Stage 6 (Phase B variant runner) where it runs inside cross-validation folds to prevent target leakage.

#### 6.4.2 Component I/O Contract

**Component YAML**: `components/stage3_preprocessing.yml`

| Direction | Name | Type | Description |
|---|---|---|---|
| **Input** | `config_name` | string | Path to YAML config file |
| **Input** | `dataset_in` | uri_file | Cleaned CSV from s02 |
| **Input** | `prep_report` | uri_folder | (optional) Preparation report from s02 — provides per-column normality / outlier flags for adaptive scaling |
| **Input** | `recipe_name` | string | (optional) Override recipe YAML path; defaults to `configs/recipes/{task}/baseline_recipe.yml` |
| **Output** | `prep3_report` | uri_folder | Folder containing `prep3_report.json` (encoding map, scaling map, VIF results) |
| **Output** | `dataset_out` | uri_file | Fully encoded and scaled CSV passed to s04 |

#### 6.4.3 CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--config_name` | ✅ | Path to YAML config |
| `--dataset_in` | ✅ | Input CSV path |
| `--prep_report` | ❌ | Optional s02 report folder |
| `--recipe_name` | ❌ | Override recipe path |
| `--prep3_report` | ✅ | Output folder for `prep3_report.json` |
| `--dataset_out` | ✅ | Output CSV path |

#### 6.4.4 Core Logic — Encoding

Encoding is controlled by the `preprocessing.encoding.method` key in the recipe YAML:

| Method | Implementation | Notes |
|---|---|---|
| `onehot` | `pd.get_dummies(drop_first=True)` | Default. Protects against perfect multicollinearity (dummy variable trap). |
| `label` | `sklearn.preprocessing.LabelEncoder` per column | Assigns integer codes; suitable for tree models. |
| `target` / `catboost` | `category_encoders.TargetEncoder` | Cross-validated target mean; recommended for high-cardinality columns. |

After any encoding, **column names are sanitised** by removing characters `[ ] < > { } , : ' " \` and replacing spaces with `_`. This is required for LightGBM and XGBoost JSON serialisation compatibility.

#### 6.4.5 Core Logic — Scaling

Scaling is controlled by `preprocessing.scaling.method` in the recipe YAML and is applied **only to the original numeric columns** (never to binary indicator columns produced by one-hot encoding):

| Method | Implementation | When to use |
|---|---|---|
| `standard` | `StandardScaler` (z-score) | Normally distributed features |
| `robust` | `RobustScaler` (median + IQR) | Features with significant outliers |
| `quantile` | `QuantileTransformer` | Skewed distributions |
| `yeo_johnson` | `PowerTransformer(method="yeo-johnson")` | Skewed features with zeros or negatives |
| `adaptive` | Per-column: `robust` if outliers > 5 %, `standard` if normal, else `minmax` | Driven by `prep_report.json` normality + outlier flags from s02 |
| `none` | No scaling | Tree-based models that do not require normalisation |

#### 6.4.6 Core Logic — Multicollinearity (VIF)

When `stage3.multicollinearity_check: true` and the feature count is ≤ 50:
- Variance Inflation Factor (VIF) is computed for all numeric columns.
- Columns with VIF > 10 are flagged as highly collinear.
- Results logged to MLflow and saved in `prep3_report.json`.
- VIF-driven column **removal** is controlled by `stage3.vif_drop_threshold`; default is to flag only (no auto-drop) to avoid losing informative features.

#### 6.4.7 SMOTE Deferral — Critical Design Decision

```
❌  SMOTE is NOT applied in s03.
✅  SMOTE runs inside cross-validation folds in s06 (Phase B variant runner).
```

Applying SMOTE before the train/test split would leak synthetic samples into the validation folds, producing artificially inflated recall scores. The current design ensures that SMOTE is only ever applied to the training fold, never to validation or holdout data.

#### 6.4.8 Config Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `stage3.adaptive_scaling` | bool | `true` | If `true`, per-column adaptive scaling uses s02 normality/outlier flags |
| `stage3.multicollinearity_check` | bool | `true` | Enable VIF computation (skipped when > 50 features for performance) |
| `stage3.vif_drop_threshold` | float | `null` | VIF above which columns are auto-dropped (null = flag only) |

#### 6.4.9 MLflow Logging

| Metric / Param | Type | Description |
|---|---|---|
| `input_rows` / `input_cols` | metric | Dataset shape before encoding |
| `output_rows` / `output_cols` | metric | Dataset shape after encoding |
| `features_added` | metric | Net new columns from one-hot encoding |
| `categorical_encoding` | param | Encoding method used |
| `scaling_strategy` | param | Scaling method used |
| `normal_columns_count` | metric | Count of normally distributed numeric columns |
| `outlier_columns_count` | metric | Count of high-outlier numeric columns |
| `vif_max` | metric | Maximum VIF value observed |
| `vif_high_count` | metric | Count of columns with VIF > 10 |
| `stage3_correlation_heatmap.png` | artifact | Saved to `outputs/` (Studio artefact viewer) |
| `stage3_sweetviz_report.html` | artifact | Saved to `outputs/` (if sweetviz enabled) |

#### 6.4.10 Failure Modes

| Condition | Behaviour |
|---|---|
| Recipe YAML not found | Falls back to baseline recipe; warning logged |
| `category_encoders` not installed | TargetEncoder path fails; step exits with ImportError |
| One-hot encoding produces > 500 columns | Warning logged; consider switching to `label` or `target` encoding |

#### 6.4.11 Developer Notes

- The baseline recipe (`configs/recipes/{task}/baseline_recipe.yml`) uses `encoding: onehot`, `scaling: standard`, `imbalance_handling: none`, `feature_selection: none`. This is the Phase A default — the variant search in Phase B tests more aggressive combinations.
- When the `recipe_name` input is omitted (as in the default pipeline wiring), the step automatically loads the baseline recipe. Phase B passes per-variant recipe paths explicitly.
- Column name sanitisation runs unconditionally; this is safe even for already-clean column names.

---

### 6.5 Stage 4 — Feature Engineering (s04)

#### 6.5.1 Purpose

Stage 4 performs feature selection, optional PCA dimensionality reduction, imbalance detection, and — most critically — the **train / holdout split**. The holdout set produced here is written as a sibling file alongside `dataset_out` and is only ever read by Stage 10 (Final Evaluation). No training step (s05a, s05b, s05t, s06, s09) ever touches the holdout data.

#### 6.5.2 Component I/O Contract

**Component YAML**: `components/stage4_feature_engineering.yml`

| Direction | Name | Type | Description |
|---|---|---|---|
| **Input** | `config_name` | string | Path to YAML config file |
| **Input** | `dataset_in` | uri_file | Preprocessed CSV from s03 |
| **Input** | `recipe_name` | string | (optional) Recipe YAML path for feature selection overrides |
| **Output** | `fe_report` | uri_folder | Folder with `fe_report.json` (selection method, PCA variance, imbalance flags, split metadata) |
| **Output** | `dataset_out` | uri_file | Combined (train + holdout) CSV — used as the official "full dataset" output |

> **Critical note**: the `dataset_out` folder also contains **sibling files** `train.csv` and `holdout.csv` alongside the primary CSV, plus `holdout_manifest.json`. Steps s05a / s05b / s05t / s06 read `train.csv`; step s10 reads `holdout.csv`. The component YAML only declares `dataset_out` as a single `uri_file` — the sibling files are a convention, not a declared output.

#### 6.5.3 CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--config_name` | ✅ | Path to YAML config |
| `--dataset_in` | ✅ | Input CSV path |
| `--recipe_name` | ❌ | Override recipe path |
| `--fe_report` | ✅ | Output folder for `fe_report.json` |
| `--dataset_out` | ✅ | Output path — folder that receives the combined CSV + sibling split files |

#### 6.5.4 Core Logic — ID Column Detection

Before any feature selection, ID-like columns are detected and dropped:
- **Name pattern**: column name contains `_id`, `customer_id`, `transaction_id`, `name`, `nameorig`, `namedest` (case-insensitive).
- **Cardinality check**: columns with > 95 % unique values are treated as identifiers and dropped.
- Dropped column names are recorded in `fe_report.json`.

#### 6.5.5 Core Logic — Feature Selection

Residual NaN values from s03 (edge cases) are filled with column medians before selection. Feature selection method is controlled by `stage4.selection_method`:

| Method | Implementation | Notes |
|---|---|---|
| `boruta` | `BorutaPy(estimator=RandomForestClassifier/Regressor, n_estimators="auto", random_state=42)` | Subsamples to 50K rows for performance. Falls back to `mutual_info` if 0 features are confirmed. |
| `mutual_info` | `SelectKBest(mutual_info_classif / mutual_info_regression, k="all")` | Scores and ranks features; threshold set by `stage4.mutual_info_threshold` |
| `variance` | `VarianceThreshold(threshold=0.0)` | Drops zero-variance (constant) columns |
| `none` | Pass-through | All features forwarded as-is |

For clustering tasks, feature selection defaults to `variance` (Boruta and mutual_info require a target column).

#### 6.5.6 Core Logic — PCA

PCA is applied when `len(features) > stage4.apply_pca_threshold` (default 100):
- `PCA(n_components=stage4.pca_variance_retained)` where `pca_variance_retained` is the explained variance ratio to retain (default 0.95 = 95 %).
- PCA is run **after** feature selection, so it acts on the already-pruned feature set.
- PCA components are named `PC_1`, `PC_2`, … in the output DataFrame.
- A warning is logged if PCA reduces the dataset to fewer than 3 components.

#### 6.5.7 Core Logic — Imbalance Detection

`detect_imbalance()` computes the minority-to-majority class ratio (classification only):

| Ratio | Recommendation |
|---|---|
| < 0.2 | SMOTE recommended → written to `fe_report.json` as `"imbalance_level": "high"` |
| 0.2 – 0.3 | Class weights recommended → `"imbalance_level": "moderate"` |
| > 0.3 | Balanced → `"imbalance_level": "low"` |

This recommendation is read by Phase B's `VariantRecommender` to score SMOTE-based recipe variants higher.

#### 6.5.8 Core Logic — Train / Holdout Split

`save_outputs()` performs the split and writes four files into the `dataset_out` folder:

| File | Contents | Consumer |
|---|---|---|
| `dataset.csv` (primary URI file) | Full dataset (train + holdout combined) | Declared `dataset_out` output |
| `train.csv` | Training portion (80 % by default) | s05a, s05b, s05t, s06, s09 |
| `holdout.csv` | Holdout portion (20 % by default) | s10 (Final Evaluation) **only** |
| `holdout_manifest.json` | Split metadata: fraction, seed, strategy, class distribution | s10 audit trail |

- **Stratified split** (classification): `train_test_split(..., stratify=y)` — falls back to random split if any class has fewer than 2 samples.
- **Random split** (regression / clustering): `train_test_split(..., stratify=None)`.
- `holdout_fraction` defaults to `0.2`; `random_seed` defaults to `42` — both configurable.

> **The holdout set is the gold standard for final model evaluation. It must never be touched by any training or hyperparameter tuning step. Violating this causes inflated evaluation scores and invalidates the entire experiment.**

#### 6.5.9 Config Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `stage4.selection_method` | string | `"boruta"` | Feature selection: `boruta`, `mutual_info`, `variance`, `none` |
| `stage4.apply_pca_threshold` | int | `100` | Feature count above which PCA is applied |
| `stage4.pca_variance_retained` | float | `0.95` | Explained variance ratio to retain |
| `stage4.imbalance_detection` | bool | `true` | Whether to run imbalance ratio check |
| `stage4.holdout_fraction` | float | `0.20` | Fraction of data reserved for final evaluation |
| `stage4.random_seed` | int | `42` | Random seed for reproducible splits |
| `stage4.mutual_info_threshold` | float | `0.0` | Minimum mutual info score to retain a feature (when `selection_method: mutual_info`) |

#### 6.5.10 MLflow Logging

MLflow run name: `"s04_feature_engineering"`.

| Metric / Param | Type | Description |
|---|---|---|
| `input_features` | metric | Feature count entering s04 |
| `output_features` | metric | Feature count after selection |
| `features_dropped` | metric | Features removed by selection |
| `selection_method` | param | Which selection method was used |
| `pca_applied` | param | `"true"` / `"false"` |
| `pca_components` | metric | Number of PCA components retained (if applied) |
| `pca_variance_explained` | metric | Cumulative variance explained by retained components |
| `imbalance_ratio` | metric | Minority / majority class ratio (classification) |
| `imbalance_level` | param | `"high"` / `"moderate"` / `"low"` / `"na"` |
| `holdout_rows` | metric | Row count in holdout set |
| `train_rows` | metric | Row count in training set |
| `holdout_fraction` | metric | Actual holdout fraction used |
| EDA outputs | artifact | Saved to `outputs/` (Studio artefact viewer) |

#### 6.5.11 Failure Modes

| Condition | Behaviour |
|---|---|
| Boruta selects 0 features | Automatic fallback to `mutual_info`; warning logged |
| PCA produces < 3 components | Warning logged; step proceeds with available components |
| Stratified split fails (< 2 samples per class) | Falls back to random split; warning logged |
| `train.csv` < 100 rows after split | Warning logged; downstream training may produce unreliable models |

#### 6.5.12 Developer Notes

- The sibling-file convention (`train.csv` / `holdout.csv` in the `dataset_out` folder) is an **internal pipeline contract**, not declared in the component YAML. If you change the output directory structure, update s05a, s05b, s05t, s06, s09, and s10 to match.
- Boruta can be slow on high-dimensional datasets (> 500 features). Set `stage4.selection_method: variance` or `none` for speed-oriented debugging runs.
- The `holdout_manifest.json` records `split_strategy`, `holdout_fraction`, `random_seed`, and per-class counts. This is the audit trail for final evaluation reproducibility.

---

## 8. Phase B — Variant Search (s06)

### 8.1 Purpose and Philosophy

Phase B answers the question: *"Given my dataset's actual characteristics, which preprocessing pipeline maximises model quality?"*

Rather than blindly executing all 457 recipe YAMLs (which would take hours), Phase B uses a two-step approach:

1. **Profile-first selection** — the `VariantRecommender` scores every available recipe against a `DatasetProfile` and keeps the top-k high-relevance variants.
2. **Batch execution** — the selected variants are all trained inside **one Azure ML step** (`s06`) using nested MLflow runs, so the Studio job graph stays flat and readable.

### 8.2 Anti-Pattern vs. Recommended Pattern

```python
# ❌ ANTI-PATTERN: blind iteration over all recipes
variants = glob("configs/recipes/classification/variant_search/*.yml")  # 180+ files
for v in variants:
    train(v)  # Wastes compute; irrelevant recipes dilute the leaderboard

# ✅ CORRECT: profile-first intelligent selection
profiler = DatasetProfiler()
profile  = profiler.profile_dataset(df, target_column)

recommender = VariantRecommender(profile=profile, all_variants=all_variants)
selected    = recommender.select_top_variants(max_variants=20, min_score_threshold=30.0)
# → Returns (variant, score, reasoning) tuples, sorted descending
```

### 8.3 Component I/O Contract

**Component YAML**: `components/s06_phaseb_variant_runner.yml` (version 7)

| Direction | Name | Type | Default | Description |
|---|---|---|---|---|
| **Input** | `config_name` | string | — | Path to YAML config file |
| **Input** | `variants_list` | string | — | Comma-separated recipe paths (e.g. `"configs/recipes/classification/variant_search/v1.yml,v2.yml"`) |
| **Input** | `engine_list` | string | `"pycaret,flaml"` | Comma-separated engines; clustering jobs pass `"pycaret"` only |
| **Input** | `dataset_in` | uri_file | — | Preprocessed dataset from s4 |
| **Input** | `time_budget_per_variant` | integer | `300` | FLAML time cap per variant (seconds) |
| **Input** | `flaml_min_budget` | integer | `120` | Floor for FLAML budget (prevents sub-threshold runs) |
| **Input** | `planner_enabled` | boolean | `false` | Enable two-round planner mode |
| **Input** | `round1_max_variants` | integer | `40` | Planner: proxy training pool size |
| **Input** | `round2_max_variants` | integer | `10` | Planner: full training shortlist size |
| **Input** | `proxy_prune_threshold` | number | `0.50` | Planner: minimum proxy metric to advance |
| **Input** | `cache_enabled` | boolean | `true` | Reuse cached preprocessing transforms |
| **Output** | `leaderboard_csv` | uri_file | — | All variant×engine results ranked by primary metric |
| **Output** | `all_results_json` | uri_file | — | Detailed per-variant training data |
| **Output** | `champion_manifest` | uri_file | — | JSON manifest of the Phase B champion |
| **Output** | `champion_model` | uri_folder | — | Serialised champion model artefacts |

### 8.4 VariantRecommender Scoring

`src/utils/variant_recommender.py` — `score_variant_relevance(variant) → (float, List[str])`

Scores are computed against the live `DatasetProfile` produced by `DatasetProfiler`. Each preprocessing dimension contributes points up to a ceiling:

| Dimension | Max Points | Key Logic |
|---|---|---|
| Imputation | 25 | KNN/iterative rewarded for high missing-rate data; mean/median for low |
| Encoding | 20 | Target encoding rewarded for high-cardinality; OHE for low-cardinality |
| Scaling | 15 | Robust scaling rewarded when `outlier_prevalence > 0.1` |
| Imbalance handling | 25 | SMOTE rewarded for `imbalance_ratio > 3`; **−10 penalty** if data is balanced; SMOTE silently skipped for regression/clustering |
| Feature selection | 15 | Boruta rewarded for high multicollinearity; variance threshold for low-feature sets |

**Leakage risk modifiers** (applied after dimension scoring):

| Leakage risk level | Multiplier |
|---|---|
| `"high"` | × 0.5 |
| `"critical"` | × 0.2 |

**Bonus**: Any variant scoring ≥ 80 points receives an extra +10 (can push above 100; score is clamped at 100).

**Diversity boost** (`select_top_variants(diversity_boost=True)`): once top variants are ranked, a +5 bonus is applied to the first variant with each unique `(imputation, encoding, feature_selection, imbalance)` key. This prevents selecting 10 variants that differ only in scaling method.

### 8.5 Recipe Libraries and Tier Selection

`src/utils/recipe_selector.py` — `RecipeSelector` class

| Library | Path | Status |
|---|---|---|
| `variant_search` | `configs/recipes/{task_type}/variant_search/` | **PRODUCTION** (default) |
| `v1_generated` | `configs/recipes/{task_type}/v1_generated/` | **DEPRECATED** — do not use for new runs |
| `enterprise_lightning_fast` | `configs/recipes/classification/enterprise_lightning_fast/` | Legacy curated fast variants |

**Tier system** (ascending compute cost):

| Tier | Description |
|---|---|
| `lightning_fast` | Minimal preprocessing; ideal for quick smoke-tests |
| `quick_exploration` | Low-cost variants; good for large datasets |
| `balanced_performance` | Default production tier |
| `high_performance` | Heavier preprocessing (KNN imputation, Boruta selection) |
| `state-of-the-art` | Maximum coverage; use with generous time budgets |

The tier is set via `phases.phase_b_recipes.tier` in the config YAML and controls which recipe subset `select_recipes_for_tier()` returns.

### 8.6 Primary Metric by Task Type

`get_primary_metric(task_type)` in `s06_phaseb_variant_runner.py` determines leaderboard ranking:

| Task type | Primary metric | Lower-better? |
|---|---|---|
| `classification` | `"Balanced Accuracy"` | No |
| `regression` | `"R2"` | No |
| `clustering` | `"silhouette_score"` | No |

For **error metrics** (RMSE, MAE, Davies–Bouldin), `get_result_score()` negates the value before comparison so that champion selection is always *higher-is-better*.

### 8.7 Nested MLflow Run Structure

```
Pipeline Run
└── s06 (Phase B Variant Runner)           ← parent step run
    ├── variant_01ace0cb — pycaret          ← child run per (variant × engine)
    ├── variant_01ace0cb — flaml
    ├── variant_02b3f9d1 — pycaret
    ├── variant_02b3f9d1 — flaml
    └── ...
```

Each child run logs:
- `params.*` — variant recipe dimensions, engine, n_models
- `metrics.*` — primary_metric, all task-appropriate metrics
- `tags.phase`, `tags.variant_id`, `tags.engine`, `tags.task_type`
- `artifacts/` — variant-specific model pkl (Phase B champion only)

### 8.8 Resilience Mechanisms

| Mechanism | Class / Function | Description |
|---|---|---|
| **Checkpointing** | `CheckpointManager` | Saves progress after each variant; allows resume on transient job failures |
| **Deadline guard** | `deadline_guard(deadline, label)` | Hard-kills training loop before Azure ML job timeout evicts the compute |
| **NaN/Inf sentinel** | `validate_metrics(metrics)` | Replaces invalid float metrics with `0.0` to prevent JSON serialisation errors |
| **Atomic writes** | `atomic_write(path, content)` | Writes temp file then `os.replace()` to prevent partial JSON output |
| **Data fingerprint** | `compute_data_fingerprint(df)` | SHA-256 of first 1,000 rows; stored in manifest for reproducibility audits |
| **Code version** | `get_code_version()` | Git SHA (8 chars) or timestamp fallback when git unavailable |

### 8.9 ChampionManifest Contract

The `champion_manifest` output is a stable JSON contract consumed by s08 aggregate step:

```json
{
  "phase": "B",
  "variant_id": "variant_01ace0cb3ddd",
  "engine": "pycaret",
  "algorithm": "lightgbm",
  "primary_metric": "Balanced Accuracy",
  "primary_metric_value": 0.8123,
  "all_metrics": {"accuracy": 0.91, "auc": 0.97, "f1": 0.87},
  "dataset_fingerprint": {"rows": 243083, "hash": "a1b2c3..."},
  "code_version": "3f7a8b1c",
  "task_type": "classification",
  "training_time_sec": 182.4
}
```

### 8.10 Task-Type Isolation Rule (Critical)

SMOTE is classification-only. s06 enforces this silently:

```python
# Applied in apply_variant_preprocessing() before training
if task_type != "classification" and recipe.get("imbalance_handling", {}).get("method") == "smote":
    print("⚠️ WARNING: Skipping SMOTE for non-classification task")
    recipe["imbalance_handling"]["method"] = "none"
```

Do **not** remove the `if task_type == "classification"` branches from this file. Fixing regression bugs must never delete classification code paths.

---

## 9. Phase C — HPO and Final Evaluation (s09 / s11 / s10)

### 9.1 Phase C HPO (s09) — phasec_optuna_hpo.py

#### Purpose

Phase C takes the Phase B champion algorithm and searches for its optimal hyperparameters using Bayesian optimisation (Optuna). It produces a single tuned model that replaces the Phase B default.

#### Component I/O Contract

**Component YAML**: `components/phasec_optuna_hpo.yml` (version 9)

| Direction | Name | Type | Default | Description |
|---|---|---|---|---|
| **Input** | `config_name` | string | — | Path to YAML config file |
| **Input** | `dataset_in` | uri_file | — | Feature-engineered dataset from s4 |
| **Input** | `phaseb_manifest` | uri_file | *optional* | Champion manifest from s08; used to warm-start search space |
| **Output** | `hpo_metrics_json` | uri_file | — | Best trial params + metrics |
| **Output** | `hpo_study` | uri_folder | — | Full Optuna study artefacts (trial database) |
| **Output** | `optimized_model` | uri_folder | — | Best model serialised as `.pkl` |

**Environment**: `azureml:mlops-v3-unified:23`

#### Config Keys

```yaml
phases:
  phase_c_hpo:
    n_trials: 50           # Number of Bayesian trials (default: 50)
    timeout: 3600          # Hard timeout in seconds (optional; respected since K9 fix)

random_seed: 42            # Controls Optuna sampler seed
```

The `timeout` key was historically ignored. As of the K9 fix it is respected — always set it when submitting to avoid unbounded compute usage.

#### Holdout Leak Prevention

Phase C checks for a sibling `train.csv` next to the provided `dataset_in` file:

```python
_train_sibling = dataset_path.parent / "train.csv"
if _train_sibling.exists() and _train_sibling.stat().st_size > 0:
    df = pd.read_csv(_train_sibling, sep=delimiter)
    # → holdout (test.csv) never seen during HPO
```

If no `train.csv` sibling exists, the step falls back to the full dataset with a warning. Operators should ensure upstream steps (s4) write a `train.csv` split alongside the primary output file.

#### MLflow Tracking URI Fix (Applied in s09)

```python
# At the top of main(), BEFORE any mlflow call
_safe_disable_autolog()
# Do NOT convert azureml:// → https:// here —
# it would break the pipeline's MLflow hierarchy.
# Use mlflow.set_tracking_uri() only for local testing.
```

> **Note**: `_safe_disable_autolog()` disables sklearn/pycaret autologging to prevent it from re-registering the model via the registry API (which fails against `azureml://` URIs).

#### Clustering HPO

When `task_type == "clustering"`, Phase C switches to `sklearn.cluster.KMeans` + Optuna over `n_clusters` and `init` parameters. FLAML is not used. Silhouette score is the objective.

### 9.2 Phase C Aggregate (s11) — aggregate_phasec.py

Reads `hpo_metrics_json` and `optimized_model` from s09, saves a normalised aggregate report.

**CLI arguments**:

| Argument | Description |
|---|---|
| `--config` | YAML config path |
| `--hpo_metrics` / `--hpo_metrics_json` | Path to `hpo_metrics.json` from s09 |
| `--optimized_model` | Path to optimised model folder |
| `--report_out` | Output path for `phasec_aggregate_report.json` |
| `--champion_out` | Output folder for champion model copy |

**Output schema** (`phasec_aggregate_report.json`):

```json
{
  "phase": "C",
  "task_type": "classification",
  "selection": {
    "algorithm": "lightgbm",
    "score": 0.8351,
    "metric": "balanced_accuracy",
    "params": {"n_estimators": 412, "learning_rate": 0.048, "num_leaves": 63}
  },
  "trials_completed": 50,
  "study_duration_sec": 1823.5
}
```

### 9.3 Final Evaluation (s10) — final_evaluation.py

#### Purpose

s10 is the quality gate. It loads the champions from all three phases, evaluates each on a **held-out test set** using the task-appropriate metric, and elects the overall winner.

#### Component I/O Contract

**Component YAML**: `components/final_evaluation.yml` (version 9)

| Direction | Name | Type | Description |
|---|---|---|---|
| **Input** | `config_name` | string | YAML config path |
| **Input** | `dataset_in` | uri_file | Full feature-engineered dataset |
| **Input** | `baseline_champion` | uri_folder | s5z champion model folder |
| **Input** | `phaseb_champion` | uri_folder | s08 champion model folder |
| **Input** | `phasec_champion` | uri_folder | s11 champion model folder |
| **Output** | `final_report` | uri_file | `final_report.json` with full comparison table |
| **Output** | `final_champion_model` | uri_folder | Winning model artefacts |

#### Metrics Used per Task Type

| Task type | Primary metric | Supporting metrics |
|---|---|---|
| `classification` | `balanced_accuracy_score` | `accuracy_score`, AUC, F1, precision, recall |
| `regression` | `r2_score` | MAE, RMSE, MAPE |
| `clustering` | `silhouette_score` | `calinski_harabasz_score`, `davies_bouldin_score` |

**Why `balanced_accuracy_score`?** The telecom churn dataset has significant class imbalance (~14 % churn). Using raw accuracy would produce misleadingly high scores for a majority-class predictor. `balanced_accuracy_score` corrects for this by averaging recall per class.

#### Train/Test Split Strategy

For classification, the final holdout uses a **stratified** split to preserve the class ratio:

```python
# Conceptual — actual split inside evaluate_champion()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_seed
)
```

#### MLflow Metrics Aggregation

`collect_all_stage_metrics()` scans ALL MLflow runs under the current experiment and categorises them into five buckets for the final report:

| Bucket | What it contains |
|---|---|
| `preprocessing_stages` | s1–s4 step metrics |
| `baseline_models` | s5a (PyCaret) and s5b (FLAML) per-model runs |
| `phaseb_recipes` | All s06 variant×engine child runs |
| `phasec_hpo` | All s09 Optuna trial runs |
| `aggregates` | s5z, s08, s11 aggregate champion metrics |

#### Performance Visualisations Generated

| File | Contents |
|---|---|
| `baseline_models_comparison.png` | Bar chart: PyCaret vs FLAML baseline champions |
| `phaseb_recipes_comparison.png` | Top 10 Phase B variant scores |
| `phase_comparison.png` | Baseline vs Phase B vs Phase C champion scores |

These are saved to the `outputs/` job directory and visible in Azure ML Studio under the s10 step outputs.

#### AIM-Tournament Integration

`final_evaluation.py` optionally calls `run_aim_tournament` from `src/utils/aim_tournament.py` when `--bundles_dir` is provided. This re-scores final candidates against curated bundle criteria before electing the champion. See `docs/AIM_TOURNAMENT.md` for the full scoring methodology.

---

## 10. Model Registration (s12)

### 10.1 Purpose

s12 registers the overall champion model produced by s10 into the MLflow Model Registry under a stable, environment-independent name. This makes the model addressable by downstream serving infrastructure without hard-coding job-specific run IDs.

### 10.2 Component I/O Contract

**Component YAML**: `components/s12_model_registration.yml` (version 1)

| Direction | Name | Type | Description |
|---|---|---|---|
| **Input** | `champion_manifest` | uri_file | `champion_manifest.json` from s10 |
| **Input** | `champion_model` | uri_folder | Champion model folder from s10 |
| **Input** | `config_name` | string | YAML config path |
| **Output** | `registry_info` | uri_file | `registry_info.json` with registration metadata |

### 10.3 Model Name Convention

The registered model name is derived automatically:

```python
model_name = f"{dataset}_{task_type}_mlops"
# e.g. "telecom_churn_classification_mlops"
#      "college_regression_mlops"
#      "online_retail_clustering_mlops"
```

Override via config:

```yaml
registry:
  model_name: "my_custom_model_name"
```

### 10.4 MLflow Flavor Detection

`ModelRegistry._detect_model_flavor(model_path)` inspects the saved model artefact and routes to the correct MLflow logging flavour to prevent mis-logging native tree models via the generic sklearn flavour (which strips native feature importance artefacts):

| Model family | MLflow flavour used |
|---|---|
| LightGBM | `mlflow.lightgbm` |
| XGBoost | `mlflow.xgboost` |
| CatBoost | `mlflow.catboost` |
| All others | `mlflow.sklearn` |

### 10.5 MLflow Tracking URI Fix (Critical)

Azure ML sets `MLFLOW_TRACKING_URI` to an `azureml://` scheme URI. The model registry API requires `https://`. s12 applies the conversion inside `ModelRegistry.__init__`:

```python
# src/steps/s12_model_registration.py — ModelRegistry.__init__
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if mlflow_uri.startswith("azureml://"):
    https_uri = mlflow_uri.replace("azureml://", "https://")
    mlflow.set_tracking_uri(https_uri)
    logger.info("🔗 MLflow tracking URI converted to HTTPS for registry access")
```

If an MLflow run is not already active when `register_champion_model()` is called (can happen after URI conversion tears down the existing run context), s12 starts an explicit run:

```python
if mlflow.active_run() is None:
    mlflow.start_run(run_name="s12_model_registration")
```

### 10.6 Stage Transition

After logging and registering the model, s12 transitions it to the **"Staging"** stage:

```python
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Staging"
)
```

This signals that the model has passed the full MLOps pipeline quality gate and is ready for review before promotion to "Production". Manual promotion to "Production" must be performed by an authorised operator via Azure ML Studio or the MLflow UI.

### 10.7 registry_info.json Output Schema

```json
{
  "model_name": "telecom_churn_classification_mlops",
  "version": "3",
  "stage": "Staging",
  "algorithm": "lightgbm",
  "task_type": "classification",
  "metrics": {
    "balanced_accuracy": 0.8351,
    "accuracy": 0.9142,
    "auc": 0.9721
  },
  "dataset": "telecom_churn",
  "config": "config_classification_telecom_churn_azureml.yml",
  "registered_at": "2026-01-15T14:32:07Z"
}
```

This file is the canonical handoff artefact consumed by deployment and serving pipelines. It does **not** contain model weights; it contains only the registry address (name + version) needed to pull the model.

### 10.8 Model Metadata Tags

s12 adds the following tags to every registered model version via `_add_model_metadata()`:

| Tag key | Value |
|---|---|
| `task_type` | e.g. `"classification"` |
| `dataset` | e.g. `"telecom_churn"` |
| `algorithm` | e.g. `"lightgbm"` |
| `mlops_pipeline` | `"v3"` |
| `registration_config` | Config file name |
| `primary_metric` | Metric name used for champion selection |
| `primary_metric_value` | Float value of that metric on holdout |

### 10.9 Operational Notes

- **Do not run s12 in isolation** against a workspaceblobstore model path. The `champion_model` input must be a job output path provided by the pipeline wiring in `pipeline_builder.py`.
- **Re-registration is idempotent** — each successful run increments the version number. To avoid stale versions, only trigger s12 after a full pipeline run that produced a higher-scoring champion.
- **Registry access requires workspace permissions** — the compute cluster's managed identity must have the `AzureML Data Scientist` role (or equivalent) scoped to the workspace. See `scripts/fix_compute_permissions.sh` for the grant command.

---

## 11. Drift Detection (s13)

### 11.1 Purpose

Stage 13 (s13) performs statistical drift analysis on production data by comparing feature distributions against a reference baseline captured at the time of model registration. It is the final step in the pipeline and produces a `drift_report.json` that feeds retraining cadence decisions.

### 11.2 Component I/O Contract

**Script**: `src/steps/s13_drift_detection.py`
**Component YAML**: `components/s13_drift_detection.yml` (or equivalent)

| Direction | Name | Type | Description |
|---|---|---|---|
| **Input** | `config_name` | string | YAML config path |
| **Input** | `dataset_in` | uri_file | Current production data sample |
| **Input** | `drift_baseline_in` | uri_folder | (optional) Reference baseline from a previous run |
| **Output** | `drift_report` | uri_file | `drift_report.json` with full drift assessment |
| **Output** | `drift_baseline_out` | uri_folder | Updated baseline for chaining to next run |

### 11.3 PSI Algorithm

**Population Stability Index (PSI)** is the primary drift metric. It compares the frequency distribution of a feature in the reference (training) vs. current (production) windows.

**Formula**:
$$\text{PSI} = \sum_{i=1}^{N} \left( A_i - E_i \right) \cdot \ln\left(\frac{A_i}{E_i}\right)$$

where $A_i$ = actual (current) proportion in bin $i$, $E_i$ = expected (reference) proportion.

**PSI thresholds**:

| PSI range | Status | Interpretation |
|---|---|---|
| < 0.10 | 🟢 GREEN | No significant drift |
| 0.10 – 0.25 | 🟡 YELLOW | Moderate drift — investigate |
| > 0.25 | 🔴 RED | Significant drift — retrain |

#### Numeric PSI (`compute_psi_numeric`)

- **Binning**: equal-width bins (default 10); edge bins clipped to the reference range.
- **Smoothing**: `EPS = 1e-6` added to every bin proportion to prevent `log(0)`.

#### Categorical PSI (`compute_psi_categorical`)

- Union of categories from reference and current data.
- Missing categories in current data: proportion = `EPS`.
- **Pseudo-count**: added to each category to prevent zero denominators.

#### Feature-level PSI (`compute_feature_psi`, `classify_feature_drift`)

- `compute_feature_psi(reference_df, current_df, feature_col)` → PSI float
- `classify_feature_drift(psi_value)` → `"GREEN"` / `"YELLOW"` / `"RED"`

### 11.4 Detector Self-Validation

On startup, the drift detector runs two built-in checks:

1. **Smoke test**: compare a DataFrame against itself — PSI must be ≈ 0 (tolerance 1e-9).
2. **Drift-injection test**: shift every numeric feature by 2× its standard deviation — PSI must exceed the RED threshold (0.25).

If either check fails, the step logs a warning and continues (it does not abort the pipeline).

### 11.5 Evidently Comparison (Dual API Compatibility)

s13 optionally uses `evidently` for a richer drift report. It maintains dual API compatibility:

- **Evidently ≤ 0.4.x**: `DataDriftTab`, `ColumnDriftMetric`, `DataDriftPreset`
- **Evidently ≥ 0.5+**: `DataDriftTable`, `ColumnDriftMetric` (renamed), `DataDriftTestPreset`

Four drift types are reported via Evidently when installed:

| Drift type | Description |
|---|---|
| Feature drift | Per-column distribution shift |
| Prediction drift | Shift in model output distribution |
| Concept drift | Change in feature-target relationship |
| Label drift | Shift in target variable distribution |

### 11.6 Stability Score

A composite **Stability Score** (0–100) is computed from five components:

| Component | Weight | Metric |
|---|---|---|
| PSI | 40% | Mean PSI across all features (lower = more stable) |
| Dataset size | 20% | Current / reference row count ratio |
| Feature complexity | 20% | Feature count stability |
| Class balance | 10% | Classification only — class ratio drift |
| Feature volatility | 10% | Fraction of RED-drift features |

$$\text{Stability Score} = 100 - \left(40 \cdot \text{psi\_score} + 20 \cdot \text{size\_score} + 20 \cdot \text{complexity\_score} + 10 \cdot \text{balance\_score} + 10 \cdot \text{volatility\_score}\right)$$

### 11.7 Retraining Cadence Recommendations

| Stability Score | Cadence |
|---|---|
| ≥ 80 | Quarterly |
| ≥ 60 | Monthly |
| ≥ 40 | Bi-weekly |
| < 40 | Weekly |

### 11.8 Output: `drift_report.json`

```json
{
  "pipeline_run_id": "...",
  "task_type": "classification",
  "reference_rows": 200000,
  "current_rows": 50000,
  "features_analysed": 18,
  "feature_drift_summary": {
    "GREEN": 12, "YELLOW": 4, "RED": 2
  },
  "per_feature_psi": {
    "tenure": 0.042,
    "monthly_charges": 0.187,
    "total_charges": 0.310
  },
  "overall_drift_status": "RED",
  "stability_score": 53.2,
  "retraining_cadence": "bi-weekly",
  "evidently_report": { "...": "..." },
  "drift_checker_results": {
    "feature_drift": "RED",
    "prediction_drift": "YELLOW",
    "concept_drift": "GREEN",
    "label_drift": "YELLOW"
  },
  "generated_at": "2026-01-15T14:40:00Z"
}
```

### 11.9 Baseline Chaining

`drift_baseline_out` is the reference baseline for the **next** run:

```
Run N:  reference = initial training distribution  → drift_baseline_out_N
Run N+1: reference = drift_baseline_out_N           → drift_baseline_out_N+1
```

This allows gradual baseline refresh (concept drift adaptation) while still detecting sudden distribution shifts.

### 11.10 MLflow Logging

| Metric / Param | Type | Description |
|---|---|---|
| `stability_score` | metric | Overall stability score (0–100) |
| `overall_drift_status` | param | `"GREEN"` / `"YELLOW"` / `"RED"` |
| `retraining_cadence` | param | Cadence recommendation string |
| `features_drift_red` | metric | Count of RED-drift features |
| `features_drift_yellow` | metric | Count of YELLOW-drift features |
| `features_drift_green` | metric | Count of GREEN-drift features |
| `drift_report.json` | artifact | Full drift report |

### 11.11 `DriftChecker` Class

`DriftChecker` (`src/utils/drift_checker.py` or inline) exposes four check methods:

| Method | Description |
|---|---|
| `check_feature_drift(ref_df, cur_df)` | Per-feature PSI drift |
| `check_prediction_drift(ref_preds, cur_preds)` | Model output distribution shift |
| `check_concept_drift(ref_df, cur_df, target_col)` | Feature-target relationship change |
| `check_label_drift(ref_labels, cur_labels)` | Target class distribution shift |

---

## 12. Utilities & Observability

### 12.1 MLflow Run Hierarchy

The pipeline establishes a structured MLflow run hierarchy:

```
Pipeline Run (experiment: <experiment_name>)
├── s01_ingestion
├── s02_preparation
├── s03_preprocessing
├── s04_feature_engineering
├── s05a_pycaret_baseline
│   └── [per-model child runs: lightgbm, xgboost, rf, ...]
├── s05b_flaml_baseline
│   └── [per-model child runs]
├── s05t_timeseries (optional)
├── s05z_aggregate_baseline
├── s06_phaseb_variant_runner
│   └── [per variant×engine child runs]
├── s09_phasec_hpo
│   └── [per Optuna trial child runs]
├── s11_aggregate_phasec
├── s10_final_evaluation
├── s12_model_registration
└── s13_drift_detection
```

**Key functions in `src/utils/mlflow_helper.py`**:

| Function | Description |
|---|---|
| `setup_mlflow_tracking(config, step_name)` | Apply URI fix + set experiment + start named run |
| `log_dataframe_stats(df, prefix)` | Log shape, dtypes, missing counts as params/metrics |
| `log_model_artifacts(model, path, flavor)` | Detect model flavour and log via correct MLflow flavour |
| `end_run_safely()` | End active MLflow run without raising if none is active |

### 12.2 Candidate Ledger

`src/utils/candidate_ledger.py` — persistent per-run record of all models evaluated.

**Column families**:

| Family | Columns |
|---|---|
| IDENTITY | `run_id`, `experiment_name`, `pipeline_run_id`, `step_name` |
| INPUT | `task_type`, `dataset`, `config_file`, `recipe_id`, `engine` |
| OUTPUT | `algorithm`, `primary_metric`, `primary_metric_value`, `all_metrics_json` |
| SIGNAL | `phase`, `champion_flag`, `quality_gate_passed`, `stability_score` |
| PROVENANCE | `code_version`, `data_fingerprint`, `timestamp_utc` |
| TOURNAMENT | `aim_score`, `tournament_rank`, `bundle_gate_passed` |

**Primary metric by task type**:

| Task | Metric |
|---|---|
| classification | `balanced_accuracy` |
| regression | `r2` |
| clustering | `silhouette_score` |
| timeseries | `rmse` |

**Key functions**:

| Function | Description |
|---|---|
| `CandidateLedger.add_candidate(entry)` | Append a new candidate record |
| `CandidateLedger.get_phase_champion(phase)` | Return highest-scoring entry for a given phase |
| `CandidateLedger.save(path)` | Write ledger to `candidate_ledger.csv` |
| `CandidateLedger.load(path)` | Load existing ledger (append mode for incremental runs) |

**Output files**: `candidate_ledger.csv`, `champion_summary.json` (in `outputs/` job folder).

### 12.3 Stage Signals

`src/utils/stage_signals.py` — inter-step communication mechanism.

**`StageSignal` dataclass**:
```python
@dataclass
class StageSignal:
    step: str            # e.g. "s04"
    status: str          # "SUCCESS" | "WARNING" | "FAILED"
    metrics: dict        # Step-specific metrics dict
    payload: dict        # Arbitrary structured data (e.g. imputation choices)
    timestamp_utc: str   # ISO 8601
```

**Key functions**:

| Function | Description |
|---|---|
| `emit_signal(step, status, metrics, payload)` | Write `<step>_signal.json` to step output folder |
| `read_signal(folder, step)` | Read `<step>_signal.json`; returns `None` if absent |
| `wait_for_signal(folder, step, timeout_s)` | Poll for signal file (used in sequential step coordination) |

### 12.4 Stage Registry

`src/utils/stage_registry.py` — metadata registry for all pipeline stages.

**`STAGE_REGISTRY` structure** (key `s01` example):
```python
STAGE_REGISTRY = {
    "s01": {
        "name": "Ingestion",
        "script": "stage1_ingestion.py",
        "phase": "preprocessing",
        "optional": False,
        "depends_on": [],
        "outputs": ["dataset_out", "eda_report"]
    },
    ...
}
```

**Active stages**: s01, s02, s03, s04, s05a, s05b, s05t (optional), s05z, s06, s09, s11, s10, s12, s13.

**Key functions**:

| Function | Description |
|---|---|
| `get_stage_info(step_id)` | Return registry entry for a step |
| `get_active_stages()` | List all non-reserved stages |
| `print_stage_banner(step_id)` | Print formatted step header to stdout (visible in `70_driver_log.txt`) |

### 12.5 AIM Tournament

`src/utils/aim_tournament.py` — multi-criteria model scoring beyond raw metric value.

**Metric catalog and weights by task type**:

| Task | Metrics weighted | Notes |
|---|---|---|
| classification | balanced_accuracy (40%), auc (20%), f1 (15%), stability (15%), compute_efficiency (10%) | |
| regression | r2 (35%), rmse (25%), mae (20%), stability (10%), compute_efficiency (10%) | |
| clustering | silhouette (40%), calinski_harabasz (25%), davies_bouldin_neg (20%), stability (15%) | |

**`run_aim_tournament(candidates, task_type, bundles_dir)` function**:
- Input: list of `CandidateLedger` entries
- Applies metric weights
- Applies bundle gate penalties (if bundle criteria not met)
- Returns ranked `(candidate, aim_score)` tuples

### 12.6 Bundle Gating

`src/utils/bundle_gating.py` — gates which recipe bundles are eligible for training.

**Key classes**:

| Class | Description |
|---|---|
| `GatingRule` | Single eligibility rule (e.g. `min_rows >= 1000`, `task_type in ["classification"]`) |
| `BundleConfig` | Collection of `GatingRule` objects for a specific bundle |

**Key functions**:

| Function | Description |
|---|---|
| `gate_bundle(bundle_config, dataset_profile)` | Returns `(passed: bool, failures: list[str])` |
| `load_bundle_gates(bundles_dir)` | Load all `.yml` bundle gate files from directory |
| `filter_eligible_bundles(all_bundles, profile)` | Return only passing bundles |

### 12.7 Azure ML Metrics Logger

`src/utils/azureml_metrics_logger.py` — thin wrapper around the Azure ML SDK `Run` object.

**Four design principles**:
1. **Fail-silent**: metric logging errors are caught and logged to stdout but never abort the step.
2. **Type coercion**: all metric values are coerced to `float` before logging to prevent SDK type errors.
3. **Name sanitisation**: metric names are sanitised (spaces → `_`, special chars stripped) to comply with Azure ML naming rules.
4. **Batch logging**: `log_metrics_batch(metrics_dict)` logs all metrics in one call to reduce SDK overhead.

**Factory function**:
```python
logger = create_metrics_logger(config, step_name)
logger.log_metric("balanced_accuracy", 0.84)
logger.log_metrics_batch({"r2": 0.91, "rmse": 4.2})
logger.log_param("algorithm", "lightgbm")
```

---

## 13. Common Pitfalls & Troubleshooting

### 13.1 Pitfall Reference Table

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `FileNotFoundError` on `azureml://` URI | `azureml-fsspec` not installed in compute environment | Ensure `azureml-fsspec` is in `environments/unified_conda.yml`; bump environment version |
| 2 | `K2: config schema validation FAILED` | Config YAML fails JSON schema validation | Fix the flagged field; use `--dry_run` to iterate quickly |
| 3 | FLAML job killed mid-run | FLAML `time_budget` exceeds Azure ML job timeout | Add `+360s` deadline buffer; set `phases.phase_c_hpo.timeout` |
| 4 | ID columns appear in model feature importance | `nameOrig`, `nameDest`, `transactionID` not filtered | s04 drops these via name-pattern + 95% cardinality check; verify they aren't renamed upstream |
| 5 | Phase B always loses to baseline | Double-preprocessing: PyCaret re-encodes already-encoded data | Use `preprocess=False` in PyCaret setup when data comes from s03 |
| 6 | Near-zero recall on minority class | Using `accuracy_score` on imbalanced data | Use `balanced_accuracy_score`; use `stratify=y` in all splits |
| 7 | Submission blocked: active job guard | Same experiment already has a running job | Wait for it to finish, or use `--force` with audit trail awareness |
| 8 | Submission hangs for 10-12 minutes | NFS snapshot upload during `ml_client.jobs.create_or_update()` | Normal on Azure ML Compute Instances; wait it out |
| 9 | `Model registry functionality is unavailable; got unsupported URI 'azureml://...'` | MLflow model registry requires HTTPS, not `azureml://` | Apply the `azureml:// → https://` URI fix in the failing step |
| 10 | `FileNotFoundError: recipe_name.yml` | Recipe referenced from `workspaceblobstore` path | Move recipe to `configs/recipes/<task>/`; reference via local code path |
| 11 | Champion model not found in s12 | Looking in phase child runs instead of the parent run | Champion artifact is in the **parent** aggregate run, not child training runs |
| 12 | SMOTE applied to regression data | Recipe YAML has `imbalance_handling: smote` without task-type guard | s06 silently skips SMOTE for non-classification; verify `task_type != "classification"` guard is present |
| 13 | PyCaret dtype error | Object-type numeric column passed to PyCaret | Run `pd.to_numeric()` coercion in s03 before encoding |
| 14 | Phase C HPO produces no improvement | `top_k_from_phase_a` too low, Phase B champion is weak | Increase `top_k`; check Phase A baseline metrics; verify `train.csv` sibling file exists |

### 13.2 Troubleshooting Checklist

#### Submission Failures

```
□ Config YAML passes --dry_run validation?
□ Compute cluster exists and is running (not stopped)?
□ Subscription/resource_group/workspace params match the YAML?
□ No active job in the same experiment (or --force used)?
□ Lock file present from a crashed previous run? → rm ~/.mlops/locks/.submit.lock
```

#### Mid-Run Failures

```
□ Check 70_driver_log.txt in Azure ML Studio for the failing step
□ Check std_log.txt for the Python traceback
□ Verify the Azure ML environment has all required packages
□ Check for azureml-fsspec in the conda environment
□ Confirm MLflow URI fix is present in the failing step script
```

#### Phase B Worse Than Phase A

```
□ Is preprocess=False set in PyCaret setup (prevents double-encoding)?
□ Is train.csv sibling file present (from s04)?
□ Are recipe variant paths valid (configs/recipes/<task>/variant_search/)?
□ Is the primary metric consistent across Phase A and Phase B?
□ Was the same holdout split used? (check holdout_manifest.json random_seed)
```

#### Registration Failures

```
□ Is the MLflow tracking URI fix applied in s12?
□ Does the compute cluster identity have AzureML Data Scientist role?
□ Is mlflow.active_run() None when register_champion_model() is called?
□ Does champion_manifest.json exist at the expected path?
```

---

## 14. Quick Reference Card

### 14.1 Submission Commands

```bash
# Standard classification run
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait --stop_compute

# Dry run (validate config, no Azure submission)
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --dry_run

# Force resubmit (bypass guards)
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  ... --force

# Phase 1 intelligent variant search
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --enable_phase1_intelligent_selection \
  --phase1_max_variants 20 ...
```

### 14.2 Step Summary Table

| Step | Script | Key output | Notes |
|---|---|---|---|
| s00 | `stage0_data_validation.py` | `validation_results.json` | **Not wired in production** (v3.1) |
| s01 | `stage1_ingestion.py` | `dataset_out`, `eda_report.json` | Data entry point; reads from datastore |
| s02 | `stage2_preparation.py` | `dataset_out`, `prep_report.json` | Imputation; MEAN forbidden |
| s03 | `stage3_preprocessing.py` | `dataset_out`, `prep3_report.json` | Encoding + scaling; SMOTE deferred |
| s04 | `stage4_feature_engineering.py` | `train.csv`, `holdout.csv` | Gold-standard holdout split |
| s05a | `stage5_pycaret_train.py` | `model.pkl`, `model_breakdown.csv` | Phase A PyCaret baseline |
| s05b | `stage5_flaml_train.py` | `model.pkl`, `model_breakdown.csv` | Phase A FLAML baseline |
| s05t | `stage5_timeseries_train.py` | `model.pkl`, `forecast_breakdown.csv` | Optional; timeseries only |
| s05z | `aggregate_baseline.py` | `phase_a_champion_manifest.json` | Elects Phase A champion |
| s06 | `s06_phaseb_variant_runner.py` | `leaderboard.csv`, `champion_manifest` | Intelligent variant search |
| s09 | `phasec_optuna_hpo.py` | `optimized_model/` | Bayesian HPO on champion algorithm |
| s11 | `aggregate_phasec.py` | `phasec_aggregate_report.json` | Normalise HPO output |
| s10 | `final_evaluation.py` | `final_report.json` | Holdout evaluation of all champions |
| s12 | `s12_model_registration.py` | `registry_info.json` | Register in Azure ML Registry |
| s13 | `s13_drift_detection.py` | `drift_report.json`, `drift_baseline/` | PSI drift analysis |

### 14.3 Minimal Config Skeleton

```yaml
experiment_name: "my_classification_experiment"
task_type: classification

dataset:
  blob_path: "my_dataset.csv"
  datastore_name: mlops_blob
  target_column: churn

azureml:
  subscription_id: "your-subscription-id"
  resource_group: "your-rg"
  workspace_name: "your-workspace"
  compute_cluster: mlopsv2computecluster
  environment_name: mlops-v3-unified

phases:
  phase_a_baseline:
    engines: [pycaret, flaml]
  phase_b_recipes:
    enabled: true
    max_variants: 20
  phase_c_hpo:
    n_trials: 50
    timeout: 3600

registry:
  min_quality: 0.70
```

### 14.4 PSI Drift Thresholds

| PSI | Status | Action |
|---|---|---|
| < 0.10 | 🟢 GREEN | Monitor normally |
| 0.10 – 0.25 | 🟡 YELLOW | Investigate; consider retraining |
| > 0.25 | 🔴 RED | Retrain required |

### 14.5 Retraining Cadence

| Stability Score | Cadence |
|---|---|
| ≥ 80 | Quarterly |
| ≥ 60 | Monthly |
| ≥ 40 | Bi-weekly |
| < 40 | Weekly |

### 14.6 MLflow URI Fix Snippet

Apply at the start of `main()` in any step that uses MLflow:

```python
import mlflow, os

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if mlflow_uri.startswith("azureml://"):
    mlflow.set_tracking_uri(mlflow_uri.replace("azureml://", "https://"))
```

Required in: `stage5_pycaret_train.py`, `stage5_flaml_train.py`, `s06_phaseb_variant_runner.py`, `phasec_optuna_hpo.py`, `s12_model_registration.py`.

### 14.7 Key File / Directory Reference

| Path | Purpose |
|---|---|
| `pipelines/submit_pipeline.py` | Canonical submission entrypoint (**immutable**) |
| `pipelines/pipeline_builder.py` | `@dsl.pipeline` definition (**immutable**) |
| `src/orchestration/config_schema.py` | Config JSON schema (**immutable**) |
| `configs/` | Task config YAMLs (6 configs) |
| `configs/recipes/{task}/variant_search/` | 457 preprocessing recipe YAMLs |
| `configs/recipes/{task}/variant_bundles/` | Pre-selected variant bundles |
| `components/` | 18 Azure ML component YAMLs |
| `src/steps/` | 19 step scripts + `__init__.py` |
| `src/utils/` | 20 utility modules |
| `environments/unified_conda.yml` | Single Conda environment for all steps |
| `docs/` | Architecture and operational documentation |
| `scripts/extract_job_results.py` | Download and export job outputs |
| `scripts/monitor_pipeline.sh` | Monitor running pipeline jobs |

### 14.8 Supported Task Types Summary

| Task type | Config `task_type` | Engines | Primary metric |
|---|---|---|---|
| Binary/multi-class classification | `classification` | PyCaret + FLAML | `balanced_accuracy` |
| Supervised regression | `regression` | PyCaret + FLAML | `r2` |
| Unsupervised clustering | `clustering` | PyCaret only | `silhouette_score` |
| Time-series forecasting | `timeseries` | statsmodels | `rmse` |

---

*End of Phase 1 Documentation — MLOps Solution Accelerator V3*
