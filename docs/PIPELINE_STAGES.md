# Pipeline Stages Reference

Current as of: 2026-08-02
Repository branch: `codex_ys/mlops-pipeline-correctness`
Validation posture: local tests and SDK dry-runs are preflight; exact-source Azure ML jobs and downloaded artifacts are required for pipeline-runtime acceptance

This document is the canonical stage-by-stage guide for the active V3 Azure ML pipeline. It explains what every wired pipeline stage does, what it consumes, what it emits, and what operators should expect.

## Source Of Truth

The live implementation source of truth is:

- `pipelines/pipeline_builder.py` for the Azure ML `@dsl.pipeline` graph.
- `components/*.yml` for component input/output contracts.
- `src/steps/*.py` for stage behavior.
- `src/utils/stage_registry.py` for canonical stage names and order.

Do not use V2 documents for active V3 operation. V2 is historical only.

## Current DAG

```text
s01 -> s02 -> s03 -> s04 ------------------------------------\
       |-> s05a ->\                                             |
       |-> s05b ---> s05z ------------------------------------\ |
       |-> s06 -----------------> s08 -> s09 -----------------+-> s10 -> s12 -> s13 -> s14
       \-> locked test ---------------------------------------/
```

Candidate selection uses training/CV evidence from Phase A, B, and C. The locked-test edge from `s02` is consumed only by `s10`, after one champion has been frozen.

Active wired stages:

`S01`, `S02`, `S03`, `S04`, `S05a`, `S05b`, `S05z`, `S06`, `S08`, `S09`, `S10`, `S12`, `S13`, `S14`.

Reserved or inactive identifiers:

| ID | Status | Notes |
|---|---|---|
| `s00` | Reserved, not wired | `components/stage0_data_validation.yml` and `src/steps/stage0_data_validation.py` exist, but `pipeline_builder.py` does not load or call them. |
| `s05t` | Legacy, not wired | Forecasting component/script files may remain for history, but forecasting is outside the classification/regression/clustering product contract. |
| `s07` | Not active | Historical/blueprint gap. No active `pipeline_builder.py` node. |
| `s11` | Not active | Historical/blueprint gap. Phase C aggregation is `s09`; model registration is `s12`. |

## Pipeline Functions

### `full_pipeline()`

Location: `pipelines/pipeline_builder.py`

Builds the standard V3 production pipeline. It loads dataset and recipe choices, wires all active components, supports optional `drift_baseline_in`, and returns pipeline-level outputs including drift and retrain decision artifacts.

Use this path for normal production submissions.

### `full_pipeline_v2()`

Location: `pipelines/pipeline_builder.py`

Builds the V3 pipeline variant with planner controls such as `planner_enabled`, `round1_max_variants`, `round2_max_variants`, `proxy_prune_threshold`, and `cache_enabled`. It preserves the same terminal `s13 -> s14` monitoring and retrain decision flow.

Use this path only when planner mode is explicitly enabled by the submission script/config.

### `submit_pipeline.py`

Location: `pipelines/submit_pipeline.py`

Canonical submission entrypoint. It performs K2 config validation, duplicate-submission lock handling, active Azure job checks, recipe selection, job tagging, optional `--drift_baseline_in` wrapping, and Azure ML submission or `--dry_run` graph emission.

No alternate production submitter should bypass this script.

## Stage Details

### S00 - Data Validation, Reserved

Component: `components/stage0_data_validation.yml`
Script: `src/steps/stage0_data_validation.py`
Status: not wired in the active DAG.

Purpose if enabled later:

- Validate dataset schema and quality before ingestion.
- Emit a validation report and validated dataset artifact.

Current expectation:

- Operators should not expect an `s00` node in Azure ML Studio for current V3 jobs.
- Docs and runbooks should label `s00` as reserved, not failed or missing.

### S01 - Ingestion

Component: `components/stage1_ingestion.yml`
Script: `src/steps/stage1_ingestion.py`

Purpose:

- Read raw input from the Azure ML datastore folder provided by the config.
- Locate/load the dataset file.
- Generate initial EDA/quality artifacts.

Inputs:

- `config_name`: config YAML filename.
- `dataset_in`: Azure ML `uri_folder` dataset input.

Outputs:

- `dataset_out`: raw/loaded CSV as `uri_file`.
- `eda_report`: EDA report folder.

Expected behavior:

- Reads datastore inputs only; it must not create or mutate datastores.
- Downstream `s02` consumes `dataset_out`.

### S02 - Preparation

Component: `components/stage2_preparation.yml`
Script: `src/steps/stage2_preparation.py`

Purpose:

- Clean and normalize early dataset issues.
- Handle deduplication, missing values, high-cardinality cleanup, basic statistical checks, and preparation reporting.
- Create the canonical train/locked-test partition before learned transformations.

Inputs:

- `config_name`.
- `dataset_in` from `s01.dataset_out`.
- `eda_report` from `s01.eda_report`.

Outputs:

- `dataset_out`: prepared partition-labeled CSV used by `s03`/`s04` reporting paths.
- `train_out`: prepared training partition with the locked test excluded.
- `raw_train_out`: pre-transform training partition used by Phase A/B/C model bundles.
- `raw_holdout_out`: immutable pre-transform locked test with canonical row identity; consumed only by `s10`.
- `split_manifest_out`: split policy, seed, row counts, row-identity hashes, and dataset version evidence.
- `prep_report`: preparation report folder.

Expected behavior:

- Preserve task-type isolation for classification, regression, and clustering.
- Emit one immutable split before any learned encoding, scaling, feature selection, resampling, or HPO.
- No training, candidate selection, threshold tuning, or drift-baseline path may consume `raw_holdout_out`.

### S03 - Preprocessing

Component: `components/stage3_preprocessing.yml`
Script: `src/steps/stage3_preprocessing.py`

Purpose:

- Apply baseline recipe preprocessing while fitting learned transformations on training rows only.
- Handle encoding, scaling, imputation, VIF checks, and classification-only imbalance handling where configured.

Inputs:

- `config_name`.
- `dataset_in` from `s02.dataset_out`.
- `prep_report` from `s02.prep_report`.
- `recipe_name`, normally `recipe_baseline.yml`.

Outputs:

- `dataset_out`: preprocessed CSV.
- `prep3_report`: preprocessing report folder.

Expected behavior:

- SMOTE/ADASYN style behavior must remain classification-only.
- Preserve Stage 2 partition and row identities; fit encoders, target encoders, and scalers from training rows only.
- Variant-specific preprocessing is not performed here; `s06` owns variant recipes.

### S04 - Feature Engineering

Component: `components/stage4_feature_engineering.yml`
Script: `src/steps/stage4_feature_engineering.py`

Purpose:

- Apply feature selection and optional dimensionality reduction using training rows only.
- Preserve the immutable split created by `s02`.

Inputs:

- `config_name`.
- `dataset_in` from `s03.dataset_out`.
- `recipe_name`, normally `recipe_baseline.yml`.

Outputs:

- `dataset_out`: feature-engineered dataset artifact.
- `fe_report`: feature engineering report folder.
- `train_out`: feature-engineered training rows used by the drift-evidence path.
- `holdout_out`: transformed compatibility output; it is not the authoritative `s10` locked-test input.

Expected behavior:

- Feature selectors, imputation statistics, and PCA fit on training rows only.
- Phase A/B/C train from `s02.raw_train_out` and persist candidate-specific fitted transforms in each model bundle.
- `s10` reads the authoritative `s02.raw_holdout_out` only after champion selection is frozen.

### S05a - Baseline PyCaret

Component: `components/stage5_pycaret_train.yml`
Script: `src/steps/stage5_pycaret_train.py`

Purpose:

- Train baseline PyCaret models for the current task type.
- Use `compare_models` for classification/regression and task-specific behavior for clustering.

Inputs:

- `config_name`.
- `dataset_in` from `s02.raw_train_out`.
- `split_manifest` from `s02.split_manifest_out`.

Outputs:

- `metrics_json`.
- `manifest_json`.
- `best_model` folder.

Expected behavior:

- Use task-specific PyCaret imports.
- Select with the common training/CV evaluator and log exact candidate lineage with MLflow.
- Persist the fitted preprocessor and estimator together as a raw-input `ModelBundle`.
- Avoid model registry operations from this stage.

### S05b - Baseline FLAML

Component: `components/stage5_flaml_train.yml`
Script: `src/steps/stage5_flaml_train.py`

Purpose:

- Run FLAML AutoML baseline training where supported.
- Track candidate model metrics and artifacts.

Inputs:

- `config_name`.
- `dataset_in` from `s02.raw_train_out`.
- `split_manifest` from `s02.split_manifest_out`.

Outputs:

- `metrics_json`.
- `manifest_json`.
- `best_model` folder.

Expected behavior:

- Skip or degrade gracefully for task types that FLAML cannot support.
- Select with the same folds, metric, seed, and evaluator contract as PyCaret.
- Persist the fitted preprocessor and estimator together as a raw-input `ModelBundle`.
- Respect configured time budgets.

### S05z - Aggregate Baseline

Component: `components/aggregate_baseline.yml`
Script: `src/steps/aggregate_baseline.py`

Purpose:

- Merge eligible baseline results from PyCaret and FLAML.
- Select the Phase A baseline champion from comparable training/CV evidence.

Inputs:

- `pycaret_manifest`, `pycaret_model`.
- `flaml_manifest`, `flaml_model`.
- `config_name`.

Outputs:

- `aggregate_report`.
- `champion_model`.

Expected behavior:

- Handle skipped or missing optional engines without failing unrelated task types.
- Reject manifest, bundle, split, or lineage identity mismatches.

### S06 - Phase B Variant Runner

Component: `components/s06_phaseb_variant_runner.yml`
Script: `src/steps/s06_phaseb_variant_runner.py`

Purpose:

- Run intelligent preprocessing variant search.
- Execute selected recipe/engine combinations in one Azure ML step with nested MLflow runs.

Inputs:

- `config_name`.
- `candidate_catalog`: immutable selected candidate catalog from canonical submission.
- `execution_manifest`: immutable execution/config/code/environment identity.
- `split_manifest`: Stage 2 split contract.
- `variants_list`: legacy compatibility input; canonical submission uses the candidate catalog.
- `engine_list`: e.g. `pycaret,flaml`.
- `dataset_in` from `s02.raw_train_out`.
- `time_budget_per_variant`.
- Planner controls when using `full_pipeline_v2()`.

Outputs:

- `leaderboard_csv`.
- `all_results_json`.
- `champion_manifest`.
- `champion_model` folder.

Expected behavior:

- Uses the raw Stage 2 training partition so every variant fits its own preprocessing recipe within training/CV folds.
- Does not blindly run all recipes.
- Never consumes locked-test rows.

### S08 - Phase C HPO

Component: `components/phasec_optuna_hpo.yml`
Script: `src/steps/phasec_optuna_hpo.py`

Purpose:

- Tune the champion algorithm family with Optuna.
- Use the Phase B champion manifest when available.

Inputs:

- `config_name`.
- `dataset_in` from `s02.raw_train_out`.
- `execution_manifest` from `s06.execution_manifest_out`.
- `phaseb_manifest` from `s06.champion_manifest`.

Outputs:

- `hpo_metrics_json`.
- `hpo_study` where component contract provides it.
- `optimized_model` folder.

Expected behavior:

- Maintain task-specific search spaces.
- Tune the exact Phase B algorithm family and preprocessing contract using training/CV evidence only.
- Respect configured trial/time limits.
- Avoid breaking classification while fixing regression or clustering, and vice versa.

### S09 - Aggregate Phase C

Component: `components/aggregate_phasec.yml`
Script: `src/steps/aggregate_phasec.py`

Purpose:

- Normalize HPO outputs into the Phase C champion shape used by final evaluation.

Inputs:

- `config_name`.
- `hpo_metrics_json` from `s08`.
- `optimized_model` from `s08`.

Outputs:

- `aggregate_report`.
- `optimized_champion_model`.

Expected behavior:

- Usually aggregates one optimized model, not a large tournament.

### S10 - Final Evaluation

Component: `components/final_evaluation.yml`
Script: `src/steps/final_evaluation.py`

Purpose:

- Validate comparable Phase A, Phase B, and Phase C selection evidence and freeze one champion before reading the locked test.
- Evaluate that frozen champion exactly once on the Stage 2 locked test.
- Apply the quality gate.

Inputs:

- `config_name`.
- `dataset_in` from `s02.raw_train_out` for selection-contract validation.
- `holdout_in` from `s02.raw_holdout_out`.
- `split_manifest_in` from `s02.split_manifest_out`.
- `execution_manifest_in` from `s06.execution_manifest_out`.
- `baseline_champion` from `s05z`.
- `phaseb_champion` from `s06`.
- `phasec_champion` from `s09`.

Outputs:

- `final_report`.
- `final_champion_model`.

Expected behavior:

- Selects the champion only from comparable training/CV scores, never from locked-test metrics.
- Validates exact split, execution, candidate, and model-bundle identity before prediction.
- Applies the frozen bundle to locked-test rows exactly once and records `locked_test_used_for_selection=false`.
- Warn-only quality gates are default unless config explicitly blocks on failure.

### S12 - Model Registration

Component: `components/s12_model_registration.yml`
Script: `src/steps/s12_model_registration.py`

Purpose:

- Register the selected champion model when registration gates allow it.
- Emit model registry metadata for downstream monitoring and retrain decisions.

Inputs:

- `config_name`.
- `champion_manifest` from `s10.final_report`.
- `champion_model` from `s10.final_champion_model`.

Outputs:

- `registry_info`.

Expected behavior:

- Use the workspace-provided Azure ML MLflow tracking URI unchanged with `azureml-mlflow`.
- Package the `ModelBundle` runtime modules, signature, and raw input example so
  the exact registered version can load outside the repository checkout.
- Persist execution, source, recipe-catalog, dataset, and registration-run
  identities on the model version and in `registry_info`.
- Leave every registered version unassigned; stage and alias promotion is a
  separate manual operator action.
- Skip or report registration blockers instead of silently failing.

### S13 - Drift Monitor

Component: `components/s13_drift_monitor.yml`
Script: `src/steps/s13_drift_monitor.py`

Purpose:

- Produce drift metrics, baseline artifacts, stability score, retraining cadence evidence, and non-blocking alert signals.
- Compare against a previous drift baseline when `drift_baseline_in` is supplied at submission time.
- Emit evidence only; do not evaluate S14 policy or submit another pipeline.

Inputs:

- `config_name`.
- `dataset_in` from `s04.train_out`; the locked final test is excluded.
- `final_report` from `s10`.
- `registry_info` from `s12`.
- Optional `baseline_in` from a previous run's `drift_baseline`.

Outputs:

- `drift_report` JSON.
- `drift_baseline` folder containing `feature_baseline.json`, `reference_distributions.json`, and `reference_data.csv`.

Expected behavior:

- Without `baseline_in`, `comparison_drift.available=false` and the run captures a fresh baseline.
- With `baseline_in`, Evidently comparison drift and concept drift checks run.
- Recent Azure evidence: regression second-cycle job `loyal_owl_0h0rz9krcn` confirmed `comparison_drift.available=true` and `baseline_status=loaded`.

### S14 - Retrain Decision Gate

Component: `components/s14_retrain_decision.yml`
Script: `src/steps/s14_retrain_decision.py`

Purpose:

- Consume `s13` drift outputs and apply the shared auto-retrain policy.
- Emit operator-readable decision artifacts.
- Keep actual retrain submission outside the pipeline in the controller.

Inputs:

- `config_name`.
- `drift_report` from `s13`.
- `candidate_baseline` from `s13.drift_baseline`.
- Optional `final_report` from `s10`.
- Optional `registry_info` from `s12`.
- Optional `trigger`, default `pipeline_s14`.

Outputs:

- `retrain_decision` JSON.
- `decision_ledger_record` JSON.

Expected behavior:

- Possible outcomes: `observe_only`, `refresh_baseline`, `candidate_retrain`, `promote_candidate`, `blocked`.
- Does not submit another Azure ML pipeline run.
- Does not approve a new baseline automatically.
- Terminal step for the current full pipeline.

## Orchestration And Auto-Retrain Modules

| Module | Purpose | Documentation expectation |
|---|---|---|
| `src/orchestration/config_schema.py` | K2 config validation before Azure work. | Config errors must be fixed before submission. |
| `src/orchestration/auto_retrain_policy.py` | Pure decision policy for drift/final/registry inputs. | No Azure side effects. |
| `src/orchestration/auto_retrain_decision_ledger.py` | Append-only JSONL ledger helpers and approved baseline resolver. | Ledger approval is explicit/manual. |
| `src/orchestration/auto_retrain_controller.py` | External controller that resolves approved baselines and builds canonical submissions. | Uses `submit_pipeline.py`; does not replace it. |
| `scripts/run_auto_retrain_controller.py` | CLI wrapper for controller dry-run/submit modes. | Dry-run first, submit only after Azure evidence. |

## Verification Checklist

- Run `git diff --check` after docs/code edits.
- Local tests prove only local contracts.
- Dry-run graph builds prove config compilation and Azure ML SDK DAG construction only.
- Exact-source Azure ML component/full-pipeline jobs plus downloaded outputs prove Azure runtime behavior.
- Registered-model and deployed-inference checks are separate acceptance gates and must be reported separately.
