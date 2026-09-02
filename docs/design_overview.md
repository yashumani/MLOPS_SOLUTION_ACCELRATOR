# V3 Design Overview

Current as of: 2026-08-02

The MLOps Solution Accelerator V3 is an Azure ML component pipeline for classification, regression, and clustering. It selects the best validated end-to-end pipeline configuration, not only the best estimator. Forecasting is outside the active product contract. Local tests and SDK dry-runs are preflight only; they are not substitutes for exact-source Azure runtime evidence.

## Architecture Principles

| Principle | Production rule |
|---|---|
| Azure-only validation | Production behavior is validated by Azure ML jobs and downloaded output artifacts. |
| Single orchestrator | `pipelines/pipeline_builder.py` owns Azure ML `@dsl.pipeline` assembly. |
| Canonical submission | `pipelines/submit_pipeline.py` is the only production submission entrypoint. |
| Config-driven behavior | Dataset paths, task type, engines, recipes, budgets, metrics, registry settings, and drift behavior come from config/CLI. |
| Read-only datastore access | Steps read Azure ML datastore inputs and write job outputs only. |
| Stable component contracts | Component YAML inputs/outputs are treated as external contracts. |
| Task isolation | Classification, regression, and clustering branches must remain independent. |
| Safe auto-retrain | Drift detection, decisioning, candidate submission, and promotion are separated. |

## Active Pipeline Flow

| Step | ID | Purpose | Status |
|---|---|---|---|
| Ingestion | `s01` | Read Azure ML datastore input and produce EDA artifacts. | Active |
| Preparation | `s02` | Clean data and create immutable raw training/locked-test partitions plus split evidence. | Active |
| Preprocessing | `s03` | Fit baseline recipe encoding, scaling, and imputation on training rows only. | Active |
| Feature engineering | `s04` | Fit feature selection/dimensionality reduction on training rows and preserve the split. | Active |
| Baseline PyCaret | `s05a` | Train PyCaret baseline candidates. | Active |
| Baseline FLAML | `s05b` | Train FLAML baseline candidates where supported. | Active |
| Baseline aggregate | `s05z` | Merge baseline results and choose Phase A champion. | Active |
| Phase B variants | `s06` | Run selected recipe/engine variant search in one Azure ML step. | Active |
| Phase C HPO | `s08` | Run Optuna HPO on the selected champion family. | Active |
| Phase C aggregate | `s09` | Normalize HPO output into a Phase C champion shape. | Active |
| Final evaluation | `s10` | Freeze one champion from comparable training/CV evidence, then audit it once on locked test. | Active |
| Model registration | `s12` | Register or report skipped registration for the final champion. | Active |
| Drift monitor | `s13` | Emit drift evidence, candidate baseline, stability, cadence, and alerts without policy or submission side effects. | Active |
| Retrain decision | `s14` | Apply auto-retrain policy and emit decision artifacts. | Active, terminal |

Reserved/inactive identifiers:

| ID | Status | Explanation |
|---|---|---|
| `s00` | Reserved | Component and script exist, but current `pipeline_builder.py` does not wire it. |
| `s05t` | Legacy | Forecasting files may exist, but the stage is not wired and forecasting is outside product scope. |
| `s07` | Inactive | Historical/blueprint placeholder, not in the active DAG. |
| `s11` | Inactive | Historical aggregate naming; active Phase C aggregate is `s09`. |

## Current DAG

```text
Azure ML datastore URI
  -> s01 ingestion
  -> s02 immutable train/locked-test split
       |-> s03 preprocessing -> s04 feature engineering --------------------\
       |-> s05a PyCaret baseline ->\                                        |
       |-> s05b FLAML baseline ---> s05z ---------------------------------\ |
       |-> s06 Phase B -> s08 Phase C HPO -> s09 aggregate ---------------+-> s10 final audit
       \-> locked test ---------------------------------------------------/       -> s12 registration
                                                                                    -> s13 evidence
                                                                                    -> s14 decision
```

`pipeline_builder.py` returns pipeline outputs from several branches, but operationally `s14` is the terminal decision stage.

## Data Flow

```text
s01 dataset_out
  -> s02 raw_train_out + raw_holdout_out + split_manifest_out
       -> raw_train_out -> Phase A/B/C training and CV selection
       -> raw_holdout_out -> s10 only, after champion freeze
       -> split_manifest_out -> candidate and final-audit identity checks
  -> s03/s04 training-fitted preprocessing and feature reports
  -> s12 registry_info
  -> s13 drift_report + drift_baseline
  -> s14 retrain_decision + decision_ledger_record
```

Stage 2 owns the canonical split. Learned transformations fit on training rows only. Each selectable candidate persists its fitted transformations with the estimator, and the locked test is never used to choose a phase, model, threshold, or hyperparameter.

## Variant Search

Phase B does not blindly run all recipes. The canonical submission path compiles a bounded immutable candidate catalog and passes it to `s06`; the legacy `variants_list` input is not the source of truth. `s06` executes selected recipe/engine combinations against the training partition with nested MLflow runs and emits the Phase B leaderboard, selection evidence, and champion bundle.

## Auto-Retrain Architecture

Auto-retrain is intentionally split across components:

| Layer | Responsibility |
|---|---|
| `s13` drift monitor | Produce drift metrics, baselines, stability, cadence, and alert evidence only. |
| `s14` retrain decision | Convert evidence into an operator-readable decision and ledger-shaped artifact. |
| External controller | Resolve approved baselines and submit candidate retrain/evaluation jobs through `submit_pipeline.py`. |
| Human/operator | Approve model and baseline promotion. |

This prevents a running pipeline from recursively scheduling itself and keeps promotion manual.

## Observability

Primary surfaces:

- Azure ML Studio parent and child job graph.
- MLflow nested runs, metrics, and artifacts.
- JSON manifests and reports from each stage.
- `final_report` for champion and quality gate evidence.
- `registry_info` for registration status.
- `drift_report` and `drift_baseline` for monitoring.
- `retrain_decision` and `decision_ledger_record` for s14 review.

## Risks Managed

| Risk | Mitigation |
|---|---|
| Duplicate submissions | Submit lock plus active Azure job guard in `submit_pipeline.py`. |
| Holdout leakage | `s02` immutable split; training/CV-only selection; one frozen champion audited once by `s10`. |
| Task-type regression | Explicit task branches and task-specific recipes. |
| Datastore mutation | Read-only datastore rule; outputs go to job paths. |
| MLflow identity and connectivity | Steps use the workspace-provided `azureml://` tracking URI unchanged with `azureml-mlflow`. |
| Over-automation of retrain | `s14` emits decisions only; controller submits candidates; humans promote. |
| Stale baseline reuse | Decision ledger requires explicit `approved_for_future_baseline=true`. |

## Current Validation Notes

- Current-checkout local tests and classification/regression/clustering Azure ML SDK graph dry-runs are preflight evidence only.
- Workspace and compute inventory have current Azure read-plane evidence.
- The current exact-source Azure canary was rejected before job creation by `ReadOnlyDisabledSubscription`; therefore the current revision has no Azure pipeline-runtime acceptance.
- May 2026 drift and `s14` jobs in `AUTO_RETRAIN_OPERATING_LEDGER.md` are historical Azure evidence for earlier revisions, not current-source or deployed proof.
- Registered raw-input model and deployed-inference acceptance remain separate and unproven for the current revision.
