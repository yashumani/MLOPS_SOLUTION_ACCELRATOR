# CLAUDE.md — MLOps Solution Accelerator V3

## Repository Overview

This is an Azure ML component-based MLOps pipeline that automates end-to-end machine learning workflows for classification, regression, and clustering tasks. It uses `@dsl.pipeline` orchestration, config-driven execution, and nested MLflow tracking.

**Repository root:** `/home/azureuser/cloudfiles/code/Users/yashu.savyminds/mlops-solution-accelerator-v3/`
**GitHub:** `SAVYMINDS/YS_MVP`, branch `v3-production`

## Quick Reference

### Submit a Pipeline Job
```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait --stop_compute
```

### Available Configs
| Config | Task |
|--------|------|
| `config_classification_telecom_churn_azureml.yml` | Classification (Telecom Churn) |
| `config_classification_bank_marketing_azureml.yml` | Classification (Bank Marketing) |
| `config_regression_college_azureml.yml` | Regression (College) |
| `config_clustering_online_retail_azureml.yml` | Clustering (Online Retail) |

## Directory Structure

```
components/          # 18 Azure ML component YAMLs
configs/             # Task configs + 457 variant recipes
  recipes/           # 260 classification + 127 regression + 67 clustering + 3 shared
  variant_bundles/   # Pre-selected variant bundles
environments/        # Conda + Azure ML environment definitions
pipelines/           # Pipeline builder + submission entrypoint
  pipeline_builder.py    # @dsl.pipeline definition
  submit_pipeline.py     # Canonical submission entrypoint
src/
  orchestration/     # Config validation (config_schema.py)
  steps/             # 19 step scripts (s00-s12)
  utils/             # 20 utility modules
  variant_search/    # Variant search engine
scripts/             # Operational scripts
docs/                # Architecture + operational docs
tests/               # Validation tests
```

## Pipeline Steps (s00–s12)

| Step | Name | Purpose |
|------|------|---------|
| s00 | Data Validation | Validate dataset schema, types, quality |
| s01 | Ingestion | Load dataset from Azure ML datastore |
| s02 | Preparation | Clean, validate, initial transforms |
| s03 | Preprocessing | Encoding, scaling, imputation (recipe-driven) |
| s04 | Feature Engineering | Feature selection, dimensionality reduction |
| s05a | Baseline PyCaret | PyCaret compare_models |
| s05b | Baseline FLAML | FLAML AutoML baseline |
| s05t | Baseline Timeseries | Optional timeseries baseline |
| s05z | Aggregate Baseline | Merge baselines, select Phase A champion |
| s06 | Phase B Variant Runner | N variants × engines with nested MLflow |
| s07 | Pipeline Attribution | Phase 2 pipeline attribution analysis |
| s08 | Phase B Aggregation | Merge variant results |
| s09 | Phase C HPO | Optuna hyperparameter optimization |
| s10 | Phase C Aggregation | Merge HPO results |
| s11 | Final Evaluation | Holdout evaluation, champion selection |
| s12 | Model Registration | Register champion in Azure ML registry |

## Critical Rules

### DO NOT
- Run steps locally — all testing via Azure ML submission jobs
- Hardcode dataset paths, task types, or hyperparameters
- Create or write to Azure ML datastores programmatically
- Reference recipe files from workspaceblobstore
- Remove classification-specific code when fixing regression (task-type isolation)
- Modify immutable files without approval (see below)

### Immutable Files (require approval to modify)
- `pipelines/submit_pipeline.py`
- `pipelines/pipeline_builder.py`
- `src/orchestration/config_schema.py`
- `src/steps/stage5_pycaret_train.py`
- `src/steps/stage5_flaml_train.py`
- `src/steps/aggregate_baseline.py`
- `src/steps/final_evaluation.py`

### MLflow Tracking URI Fix
All step scripts using MLflow must convert `azureml://` URIs to `https://`:
```python
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if mlflow_uri.startswith("azureml://"):
    mlflow.set_tracking_uri(mlflow_uri.replace("azureml://", "https://"))
```

### Data Flow Pattern
```
Input (azureml:// datastore URI) → pd.read_csv() → Processing → outputs (job paths)
```

### Recipe References
Recipes are in the code directory, NOT workspaceblobstore:
```python
recipe_path = ROOT / "configs/recipes/recipe_name.yml"
```

## Azure ML Context
- **Subscription:** `93044a08-5661-4f1b-b424-5eafe066a9d1`
- **Resource Group:** `mvpv1`
- **Workspace:** `mlops-accelerator`
- **Compute:** `mlopsv2computecluster`
- **Environment:** `mlops-v3-unified`

## Architectural Decisions

### ADR-001: Phase C Receives s4 (Baseline) Data, Not Phase B Output (2026-03-29)

**Decision:** VERDICT B — Phase C on baseline data is intentional, with recipe replication.

**Context:** Phase B (s06) runs N variant recipes in parallel. There is no single "Phase B dataset" to forward. Phase C (s09/phasec_optuna_hpo.py) needs a consistent data snapshot for HPO.

**Rationale:**
- Blueprint Row 11 specifies "Engineered dataset from s04" as Phase C input.
- `phasec_optuna_hpo.py` already replicates the champion recipe transforms (L390-440) on s4 data before HPO, so the champion's preprocessing is faithfully reproduced.
- Forwarding one variant's output would break the N-variant architecture and create a hidden coupling.

**Implications:** If a recipe adds derived columns, those are recreated inside Phase C via recipe replication — not piped from s06.

---

## Common Pitfalls
1. Missing `azureml-fsspec` → datastore URIs fail silently
2. FLAML timeout → use time-aware CV with +360s deadline buffer
3. ID columns as features → drop before encoding
4. Double-preprocessing → use `preprocess=False` in PyCaret
5. Near-zero recall → use `balanced_accuracy_score` + `stratify=y`
6. NFS delays → `ml_client.jobs.create_or_update()` takes ~12 min
7. Duplicate submissions → use `--force` flag only when intentional
