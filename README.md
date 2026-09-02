
# Savvy Minds MLOps Solution Accelerator V3

This repository contains a config-driven Azure ML component pipeline that searches for the best validated end-to-end machine-learning configuration. A candidate includes preprocessing, feature engineering, engine, algorithm, hyperparameters, split policy, metrics, and execution identity; the product does not optimize only the estimator.

The active product contract supports exactly three task types: `classification`, `regression`, and `clustering`. Forecasting/time-series training is not part of the active graph. Current training engines are PyCaret and FLAML, with FLAML skipped explicitly where a task is unsupported.

## Core Workflow

1. `s01` ingests the configured Azure ML datastore input.
2. `s02` creates the immutable training partition and locked final-test partition before learned transformations.
3. `s03` and `s04` fit learned preprocessing and feature transformations on training rows only.
4. Phase A (`s05a`, `s05b`, `s05z`), Phase B (`s06`), and Phase C (`s08`, `s09`) compare candidates using training/CV evidence.
5. `s10` freezes one CV-selected champion before reading the locked test partition, then evaluates that champion exactly once.
6. `s12` registers the exact self-contained model bundle when registration gates allow it.
7. `s13` emits drift evidence and a candidate baseline. `s14` applies retraining policy and emits decision artifacts. Only the external controller may submit another run through the canonical submitter.

All experiment tracking uses the Azure ML workspace-provided MLflow tracking URI and preserves execution, config, split, candidate, code, and environment identity.

## Canonical Entry Points

- Pipeline assembly: `pipelines/pipeline_builder.py`
- Submission and SDK graph preflight: `pipelines/submit_pipeline.py`
- Component contracts: `components/*.yml`
- Stage behavior: `src/steps/*.py`
- Production configs and recipes: `configs/`

Do not submit production jobs through an alternate script or execute stage scripts locally as production validation.

## Preflight

Use the supported Azure ML environment and explicit workspace context:

```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --dry_run
```

`--dry_run` proves config compilation and Azure ML SDK graph construction only. A completed Azure ML job with downloaded artifacts is pipeline-runtime proof. Registered-model and deployed-inference proof require separate checks; none of these proof levels is interchangeable with local tests.

## Documentation

- [Documentation index](docs/README.md)
- [Product requirements](docs/PROJECT_REQUIREMENTS.md)
- [Pipeline stages](docs/PIPELINE_STAGES.md)
- [Pipeline I/O contracts](docs/PIPELINE_IO_CONTRACTS.md)
- [Configuration reference](docs/CONFIGURATION_REFERENCE.md)
- [Submission guide](docs/SUBMISSION_GUIDE.md)
- [Drift and retraining control](docs/DRIFT_DETECTION.md)
