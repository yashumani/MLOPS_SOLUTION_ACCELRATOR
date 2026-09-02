# Project Requirements

Current as of: 2026-08-02

This document adapts the workspace requirement notes into the active V3 product contract. Older workspace notes mention forecasting, a 13-step `s00`-`s12` pipeline, and fixed recipe counts; those claims are not authoritative for the current repository. The active graph ends at `s14`; `s00`, `s05t`, `s07`, and `s11` are reserved, legacy, or inactive.

## Product Goal

Provide an Azure ML based MLOps accelerator that identifies the best validated end-to-end pipeline configuration, not only the best estimator. A candidate includes preprocessing, feature engineering, engine, algorithm, hyperparameters, split policy, metric contract, and immutable execution identity. The system prepares configured data, performs comparable multi-engine search, audits the frozen champion, registers the exact model bundle, monitors drift, and produces safe retraining decisions.

## Product Scope

- Supported task types are exactly `classification`, `regression`, and `clustering`.
- Forecasting and time-series training are outside the active product contract and active DAG.
- Current training engines are PyCaret and FLAML. Azure ML hosts execution; Azure AutoML is not an implemented engine unless a separately validated adapter is added.

## Non-Negotiable Guardrails

| Requirement | Current rule |
|---|---|
| Proof boundaries | Local tests, SDK graph dry-runs, Azure component/pipeline jobs, registered-model checks, and deployed inference are separate evidence levels. |
| Azure-only behavioral testing | Pipeline runtime behavior is accepted only through exact-source Azure ML submissions and downloaded outputs, not local step execution or SDK dry-runs. |
| Canonical entrypoint | Production submissions go through `pipelines/submit_pipeline.py`. |
| Single orchestration system | Azure ML component pipeline built with `@dsl.pipeline`. |
| Immutable orchestration with approval | Do not casually modify `pipeline_builder.py`, `submit_pipeline.py`, or core stable interfaces. |
| Config-driven operation | Dataset paths, task type, engines, recipes, and thresholds come from config/CLI. |
| Read-only datastores | Step scripts read datastore inputs and write job outputs only. |
| Locked-test isolation | Candidate selection uses training/CV evidence; one frozen champion is evaluated once on the Stage 2 locked test partition. |
| Retrain ownership | `s13` emits evidence, `s14` emits policy decisions, and only the external controller may submit through the canonical entrypoint. |
| Manual promotion | Auto-retrain may recommend/submit candidates, but promotion remains manual until explicitly changed. |

## Functional Requirements

### FR-01: Configurable Dataset Ingestion

The system shall load datasets from Azure ML datastore URIs configured in `configs/*.yml`.

Acceptance:

- `s01` loads the configured dataset.
- No stage creates or mutates Azure ML datastores.
- Ingestion emits `dataset_out` and `eda_report`.

### FR-02: Task-Type Isolated Data Preparation

The system shall support classification, regression, and clustering without breaking one task type while fixing another.

Acceptance:

- Task-specific preprocessing branches remain intact.
- Classification-only imbalance handling is skipped/warned for non-classification tasks.
- Clustering does not require a supervised target column.

### FR-03: Holdout-Safe Feature Engineering

The system shall create one immutable training/search partition and one locked final-test partition before any learned transformation or model training.

Acceptance:

- `s02` emits `raw_train_out`, `raw_holdout_out`, and `split_manifest_out` with canonical row identity.
- `s03` and `s04` fit learned transforms on training rows only and preserve the split assignment.
- Training, thresholding, HPO, and candidate selection do not consume locked-test rows.
- `s10` validates the split manifest, freezes one champion from comparable training/CV evidence, and evaluates that champion exactly once on `raw_holdout_out`.

### FR-04: Multi-Engine Baseline Search

The system shall run baseline model search with PyCaret and FLAML where supported.

Acceptance:

- `s05a` emits PyCaret metrics, manifest, and best model.
- `s05b` emits FLAML metrics, manifest, and best model or an explicit supported skip state.
- Phase A engines use the same fold, metric, seed, and split contract.
- `s05z` aggregates eligible bundles and selects a baseline champion from comparable CV evidence.

### FR-05: Intelligent Variant Search

The system shall run selected Phase B recipe variants without blindly testing the full recipe library.

Acceptance:

- `submit_pipeline.py` resolves selected recipes into `variants_list`.
- `s06` executes variant/engine combinations in one Azure ML step.
- `s06` emits leaderboard, all results, champion manifest, and champion model.

### FR-06: Phase C Hyperparameter Optimization

The system shall optimize the selected Phase B champion algorithm family with Optuna without changing candidate identity or preprocessing semantics.

Acceptance:

- `s08` uses the Stage 2 training partition, execution manifest, and Phase B champion context; it never reads the locked test partition.
- `s09` emits a Phase C aggregate report and optimized champion model.

### FR-07: Frozen-Champion Final Evaluation And Quality Gate

The system shall select one champion from comparable training/CV evidence before reading the locked test partition, then perform one final unbiased audit.

Acceptance:

- Phase A, B, and C selection evidence uses the configured primary metric and one comparable CV contract.
- `s10` freezes the champion before locked-test prediction and records `locked_test_used_for_selection=false`.
- Holdout metrics may pass, warn, or block registration according to quality policy, but may not choose a phase, estimator, threshold, or hyperparameter.
- `s10` emits `final_report` and `final_champion_model`.
- Quality gates are documented in the report.
- Warn-only behavior is explicit unless blocking is configured.

### FR-08: Model Registration

The system shall register or report skipped registration for the selected model.

Acceptance:

- `s12` emits `registry_info`.
- Registration status and model metadata are visible to downstream stages.
- MLflow Azure URI compatibility is handled before registry operations.

### FR-09: Drift Monitoring

The system shall monitor data/model drift and emit reusable evidence and baselines without making or submitting a retraining decision.

Acceptance:

- `s13` emits `drift_report` and `drift_baseline`.
- First-cycle runs capture a baseline and mark comparison drift unavailable.
- Baseline-chained runs load the previous baseline and set comparison drift available when valid.
- `s13` has no Azure submission side effect and does not evaluate the S14 policy.

### FR-10: Safe Auto-Retrain Decisioning

The system shall separate drift signal production from retrain decisioning.

Acceptance:

- `s13` emits drift evidence and a candidate baseline only.
- `s14` emits `retrain_decision` and `decision_ledger_record`.
- `s14` does not recursively submit a new pipeline run.
- The external controller evaluates `s14` output and submits only through `submit_pipeline.py` when policy permits.

### FR-11: Operator Visibility

The system shall provide artifacts and runbooks that explain what happened and what to do next.

Acceptance:

- Docs explain every active stage and artifact contract.
- Runbooks explain submission, monitoring, downloads, failed jobs, and auto-retrain operations.
- Stale docs are replaced or clearly marked historical.
- Every validation claim identifies whether it is local, SDK graph, Azure pipeline, registered-model, or deployed-inference evidence.

## Non-Functional Requirements

| Area | Requirement |
|---|---|
| Reliability | Pipeline stages should fail loudly for contract violations and degrade gracefully for optional engines. |
| Observability | MLflow metrics, manifests, reports, and Azure ML artifacts must be sufficient to debug decisions. |
| Reproducibility | Config, split, dataset, candidate, code, environment, MLflow run, and execution identities must be preserved in outputs. |
| Cost control | Compute should remain bounded by configured budgets and optional `--stop_compute`. |
| Security | No secrets in repo, no programmatic datastore creation, no direct datastore writes from steps. |
| Documentation | Docs must reflect current code and live validation status; do not claim unvalidated Azure behavior. |

## Current Validation Status

| Evidence level | Current status | What it proves |
|---|---|---|
| Local unit/contract tests | Passed on the current working tree | Python contracts and focused behavior only; no Azure runtime proof. |
| Azure ML SDK graph dry-runs | Classification, regression, and clustering graph builds passed | Config compilation, component loading, and DAG construction only. |
| Azure read plane | Workspace and compute inventory were readable | Resource existence and compute configuration only. |
| Exact-source Azure pipeline | Blocked before job creation by `ReadOnlyDisabledSubscription` | No current-revision pipeline-runtime acceptance exists. |
| Registered raw-input model | Not proven for the current revision | Requires an exact-source successful job plus bundle download/registration checks. |
| Deployed inference | Not proven | Requires a separately deployed endpoint and bounded inference verification. |

May 2026 drift and `s14` job records are historical Azure evidence for earlier revisions. They remain useful operating history but do not close current-source Azure, registered-model, or deployed-inference acceptance.

## Documentation Requirements

The docs folder must include:

- Stage-by-stage pipeline behavior.
- I/O and artifact contracts.
- Config reference.
- Submission and monitoring runbooks.
- Auto-retrain operating ledger.
- Current requirements.
- Clear distinction between active, reserved, and historical stages.
