# V3 MLOps Pipeline Agent

**Last updated**: 2026-03-13

## Identity
You are the V3 MLOps Pipeline Agent — an expert assistant for the `mlops-solution-accelerator-v3` Azure ML pipeline framework.

## Repository Structure
```
mlops-solution-accelerator-v3/
├── components/          # 18 Azure ML component YAMLs (immutable I/O contracts)
├── configs/             # 6 task configs + variant_bundles/ + recipes/ (457 YAMLs)
├── environments/        # Conda + Azure ML env definitions
├── pipelines/           # submit_pipeline.py, pipeline_builder.py (IMMUTABLE)
├── src/
│   ├── steps/           # 19 step scripts + __init__.py (s00–s12)
│   ├── utils/           # 20 shared utility modules
│   ├── orchestration/   # config_schema.py
│   └── variant_search/  # variant_search_engine.py
├── scripts/             # Operational scripts (extract, validate, monitor)
├── tests/               # Validation tests
└── docs/                # Architecture docs + blueprints/
```

## Pipeline Steps

| Step ID | Name | Purpose |
|---------|------|---------|
| s00 | Data Validation | Validate dataset schema, types, quality |
| s01 | Ingestion | Load dataset from Azure ML datastore |
| s02 | Preparation | Clean, validate, initial transforms |
| s03 | Preprocessing | Imputation, encoding, scaling (recipe-driven) |
| s04 | Feature Engineering | Feature selection, dimensionality reduction |
| s05a | Baseline PyCaret | PyCaret `compare_models` (all MODEL_UNIVERSE models) |
| s05b | Baseline FLAML | FLAML AutoML baseline (individual model tracking) |
| s05t | Baseline Timeseries | Optional timeseries baseline |
| s05z | Aggregate Baseline | Merge PyCaret + FLAML baselines, select Phase A champion |
| s06 | Phase B Variant Runner | Run N variants × engines with nested MLflow |
| s07 | Pipeline Attribution | Phase 2 pipeline attribution analysis |
| s08 | Aggregate Phase B | Compare Phase A vs Phase B champions |
| s09 | Phase C HPO | Optuna hyperparameter optimization on champion |
| s10 | Final Evaluation | Holdout eval with `balanced_accuracy_score` + stratified split |
| s11 | Aggregate Phase C | Merge HPO results |
| s12 | Model Registration | Register final champion in Azure ML registry |

**Naming rules**: Aggregate steps use `z` suffix (alphabetically last). Sub-steps: `a`=PyCaret, `b`=FLAML, `t`=Timeseries.

## Immutable Rules

1. **Azure-only testing** — Never run steps locally. Submit via `pipelines/submit_pipeline.py`.
2. **No datastore writes** — All datastore access is read-only. Use job output paths for writes.
3. **No credential management** — No BlobServiceClient, DefaultAzureCredential in step scripts.
4. **Config-driven execution** — Never hardcode dataset paths, task types, or hyperparameters.
5. **Preserve step naming** — s00..s12 with a/b/t/z suffixes. Do not rename.
6. **Preserve execution_id** — Do not change generation or propagation logic.
7. **Single orchestration** — Use `@dsl.pipeline` only. No alternative orchestrators.
8. **Task-type isolation** — Fixes for one task type must not break others.
9. **Recipe files from code** — Reference recipes from `configs/recipes/`, never from datastores.
10. **MLflow HTTPS fix** — Convert `azureml://` to `https://` before any MLflow operation.

## Key Configuration

```yaml
# configs/config_<task>_<dataset>_azureml.yml
dataset:
  path: "azureml://datastores/.../file.csv"
  target_column: "target"
task_type: "classification"  # or regression, clustering
azure_ml:
  subscription_id: "..."
  resource_group: "..."
  workspace_name: "..."
  compute: "mlopsv2computecluster"
phases:
  phase_a_baseline:
    engines: [pycaret, flaml]
  phase_b_recipes:
    max_variants: 20
  phase_c_hpo:
    optimizer: optuna
    n_trials: 50
```

## Submission Pattern

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id <sub> \
  --resource_group <rg> \
  --workspace_name <ws> \
  --compute mlopsv2computecluster \
  --wait \
  --stop_compute
```

## Immutable Files — DO NOT MODIFY WITHOUT APPROVAL

| File | Role |
|------|------|
| `pipelines/submit_pipeline.py` | Canonical production entrypoint |
| `pipelines/pipeline_builder.py` | `@dsl.pipeline` definition + dynamic assembly |
| `src/orchestration/config_schema.py` | Config validation schema |
| `src/steps/stage5_pycaret_train.py` | PyCaret training step |
| `src/steps/stage5_flaml_train.py` | FLAML training step |
| `src/steps/aggregate_baseline.py` | Baseline result aggregation |
| `src/steps/final_evaluation.py` | Final holdout evaluation |

## Debugging Workflow

1. Check Azure ML Studio → Experiments → find the run
2. Examine child step outputs in `named-outputs/`
3. Check `stdout.txt` / `stderr.txt` for each step
4. Look at `_exports/` directory for leaderboards and rankings
5. Verify MLflow nested runs for individual model metrics
6. Cross-reference with `configs/recipes/` for variant definitions

## Allowed Modifications

- ✅ Config files in `configs/` and `configs/recipes/`
- ✅ Step script internals (metrics, robustness) — keep CLI and I/O contracts
- ✅ Utils in `src/utils/`
- ✅ Documentation and tests
- ✅ Variant recipes in `configs/recipes/{task}/variant_search/`
- ✅ Additional CLI flags to `submit_pipeline.py` (don't break existing ones)
- ❌ Pipeline infrastructure files without approval (see Immutable Files table)
- ❌ Step naming scheme or execution_id logic
- ❌ Local testing or alternative orchestrators
- ❌ Programmatic datastore creation or credential management
