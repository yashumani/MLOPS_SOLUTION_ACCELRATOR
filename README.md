# MLOps Solution Accelerator V3

End-to-end Azure ML pipeline for automated machine learning across **classification**, **regression**, and **clustering** tasks. The pipeline runs a three-phase tournament (Baseline → Variant Search → HPO) to find the best model, then evaluates it on a holdout set and registers the champion.

## Features

- **Three task types** — classification, regression, clustering with task-specific recipes
- **Multi-engine training** — PyCaret and FLAML baselines run in parallel
- **Intelligent variant search** — 457 recipes scored by data profiling; top variants selected automatically
- **Optuna HPO** — hyperparameter optimization on the Phase B champion
- **AIM tournament** — Automated Intelligent Model selection across all phases
- **Nested MLflow tracking** — parent pipeline run with per-step and per-model child runs
- **Config-driven** — YAML configs control dataset, stages, phases, and engines

## Repository Structure

```
mlops-solution-accelerator-v3/
├── components/           # 18 Azure ML component YAMLs (I/O contracts)
├── configs/              # Task configs + 457 variant recipe YAMLs
│   ├── config_*.yml      # Per-task pipeline configurations
│   ├── recipes/          # Variant recipes (classification/regression/clustering)
│   └── variant_bundles/  # Pre-selected variant bundles
├── environments/         # Conda env + Azure ML environment definitions
├── pipelines/            # Pipeline definition + submission entrypoint
│   ├── pipeline_builder.py   # @dsl.pipeline definition
│   └── submit_pipeline.py    # Canonical submission script
├── src/
│   ├── steps/            # 19 pipeline step scripts (s00–s12)
│   ├── utils/            # 20 utility modules
│   └── variant_search/   # Variant search engine
├── scripts/              # Operational scripts (extract results, monitor)
├── docs/                 # Architecture and operational docs
└── tests/                # Validation tests
```

## Pipeline Steps

| Step | Name | Purpose |
|------|------|---------|
| s00 | Data Validation | Schema, types, quality checks |
| s01 | Ingestion | Load from Azure ML datastore |
| s02 | Preparation | Clean, validate, initial transforms |
| s03 | Preprocessing | Encoding, scaling, imputation |
| s04 | Feature Engineering | Feature selection, dimensionality reduction |
| s05a/b/t | Baseline Training | PyCaret + FLAML + Timeseries baselines |
| s05z | Aggregate Baseline | Merge baselines, select Phase A champion |
| s06 | Phase B Variants | Run top variants × engines (nested MLflow) |
| s08 | Model Selection | Phase A vs Phase B champion comparison |
| s09 | Phase C HPO | Optuna optimization on champion |
| s10 | Final Evaluation | Holdout evaluation (balanced accuracy + stratified split) |
| s12 | Model Registration | Register champion in Azure ML registry |

## Setup

### Prerequisites

- Python 3.10
- Azure ML workspace with a compute cluster
- Azure CLI authenticated (`az login`)

### Environment

```bash
conda env create -f environments/unified_conda.yml
conda activate mlops-v3-unified
```

Or install pip dependencies only:

```bash
pip install -r requirements.txt
```

## Usage

### Submit a Pipeline

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id <SUBSCRIPTION_ID> \
  --resource_group <RESOURCE_GROUP> \
  --workspace_name <WORKSPACE_NAME> \
  --compute <COMPUTE_CLUSTER> \
  --wait --stop_compute
```

### Configuration

Each task has a YAML config under `configs/`. Key sections:

- **dataset** — Azure ML datastore URI and target column
- **azure_ml** — subscription, resource group, workspace, compute, environment names
- **stages** — parameters for data validation through feature engineering (s00–s04)
- **phases** — baseline engines, variant recipes, HPO settings, final evaluation metrics

### Monitor a Running Job

```bash
bash scripts/monitor_pipeline.sh <JOB_NAME>
```

### Extract Job Results

```bash
python scripts/extract_job_results.py --job-name <JOB_NAME>
```

## Architecture Docs

- [AIM Tournament](docs/AIM_TOURNAMENT.md) — scoring methodology
- [Candidate Ledger](docs/LEDGER.md) — ledger specification
- [Phase B Variant Runner](docs/PHASE_B_VARIANT_RUNNER_ARCHITECTURE.md) — variant runner design
- [Variant Search Guide](docs/VARIANT_SEARCH_GUIDE.md) — recipe search usage
