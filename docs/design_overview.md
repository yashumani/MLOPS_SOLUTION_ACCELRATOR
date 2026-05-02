# V3 Design Overview

This document describes the current production architecture for the MLOps Solution Accelerator V3. V3 is an Azure ML component pipeline; it is not a local `src/main.py` workflow.

## Architecture Principles

| Principle | Production rule |
|---|---|
| Azure-only validation | Production validation happens through Azure ML jobs. |
| Single orchestrator | `pipelines/pipeline_builder.py` owns `@dsl.pipeline` assembly. |
| Config-driven behavior | Dataset paths, task type, engines, recipes, budgets, and metrics come from YAML config. |
| Read-only datastore access | Step scripts read datastore URIs and write only job outputs. |
| Stable component contracts | Component YAML inputs and outputs are stable unless explicitly approved. |
| Task isolation | Classification, regression, and clustering branches are kept separate. |

## Pipeline Flow

| Step | ID | Purpose |
|---|---|---|
| Data validation | `s00` | Validate schema, column types, target presence, and dataset quality. |
| Ingestion | `s01` | Read Azure ML datastore input into the pipeline. |
| Preparation | `s02` | Clean, deduplicate, impute, and normalize early dataset issues. |
| Preprocessing | `s03` | Apply recipe-driven encoding, scaling, imputation, and imbalance handling. |
| Feature engineering | `s04` | Select features, reduce dimensionality when configured, and emit train/holdout siblings. |
| Baseline PyCaret | `s05a` | Train baseline PyCaret models. |
| Baseline FLAML | `s05b` | Train baseline FLAML models where supported. |
| Baseline timeseries | `s05t` | Optional timeseries baseline. |
| Baseline aggregate | `s05z` | Merge baseline outputs and pick Phase A champion. |
| Phase B variants | `s06` | Run intelligent recipe variants with nested MLflow runs. |
| Model selection | `s08` | Compare Phase A and Phase B champions. |
| Phase C HPO | `s09` | Tune the selected champion with Optuna. |
| Final evaluation | `s10` | Evaluate champions on holdout, run quality gate, write manifest. |
| Model registration | `s12` | Register the final champion when allowed. |
| Drift monitor | `s13` | Produce drift baseline, drift metrics, and non-blocking alerts. |

## Data Flow

```text
Azure ML datastore URI
  -> s00/s01/s02/s03/s04
  -> feature-engineered dataset_out
  -> sibling train.csv + holdout.csv + holdout_manifest.json
  -> s05/s06/s09 train-only model search
  -> s10 holdout final evaluation
  -> s12 model registration
  -> s13 drift baseline and monitoring artifacts
```

Stage 4 preserves the original `dataset_out` for component compatibility and adds sibling files beside it. That design avoids component YAML churn while preventing training steps from seeing holdout rows.

## Variant Search

Phase B does not blindly run every recipe. The production design profiles the dataset, scores available recipes, and selects a bounded set of relevant variants. The current clustering submissions used 10 selected variants; the telecom churn resubmission used 2 selected variants with PyCaret and FLAML engines.

## Observability

MLflow is the primary observability surface. Each step logs parameters, metrics, and artifacts. Training and variant steps use nested runs; aggregate and final steps write champion manifests and summary metrics. Final evaluation records `quality_gate_passed`, `quality_threshold`, and `block_on_quality_fail`.

## Production Risks Managed

| Risk | Mitigation |
|---|---|
| Dirty working-tree uploads | Commit or stash unrelated changes before production submission. |
| Duplicate submissions | Lock file plus active-job guard in `submit_pipeline.py`. |
| Holdout leakage | Stage 4 sibling split; downstream train-only readers; s10 holdout reader. |
| Clustering prediction crash | Numeric-only final eval with `feature_names_in_` alignment. |
| Over-blocking weak-but-valid jobs | Warn-only quality gate by default; blocking is explicit config. |
| MLflow Azure URI incompatibility | Convert `azureml://` tracking URI to `https://` in MLflow-using steps. |