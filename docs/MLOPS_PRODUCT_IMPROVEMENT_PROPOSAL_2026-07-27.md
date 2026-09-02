# MLOps Solution Accelerator Product Improvement Proposal

**Review date:** 2026-07-27  
**Review snapshot:** branch `prod_hardening_20260523`, commit `eb739444f9d439b75232e0342f5d5c52ab86986c`, including the current uncommitted workspace changes  
**Scope:** architecture and code review only. This proposal does not authorize pipeline behavior changes, deployment, or an Azure ML production run.

## Executive Decision

The repository is a substantial MLOps prototype with useful stage isolation, recipe-driven preprocessing, Azure ML component orchestration, MLflow instrumentation, quality gates, registration, drift monitoring, and retraining policy work. It is not yet product-correct as a generic optimizer for the best end-to-end machine-learning configuration.

The highest-value work is not adding more stages or abstractions. It is making the existing search and evaluation contract trustworthy:

1. Make data-driven variant reduction the canonical path, not an optional path that falls back to alphabetical selection.
2. Evaluate every engine and phase under one split, metric, and budget protocol.
3. Reserve a locked test partition for one final audit after pipeline selection.
4. Preserve algorithm and preprocessing identity through HPO, model packaging, registration, and inference.
5. Make the recipe/configuration catalog valid, deterministic, and testable before Azure compute is consumed.

Once those corrections are made, the existing stage-oriented design can support the intended product without a new orchestration framework.

## Product North Star

For a supported tabular dataset, the accelerator should produce:

- the best validated **pipeline configuration**, not only the best estimator;
- a reproducible description of preprocessing, feature engineering, algorithm, hyperparameters, engine, data split, metrics, and environment;
- transparent evidence showing which configurations were rejected, shortlisted, trained, and compared;
- a self-contained model bundle that accepts the documented raw input schema;
- a registered model only when the configured policy permits it;
- drift and retraining evidence that is linked to the same data, configuration, and model lineage.

The supported task types are:

- classification;
- regression;
- clustering.

Forecasting should not remain in the active product contract unless it is explicitly restored as a fourth supported task.

## Current Architecture

The current graph implements the following broad flow:

1. S01-S04 ingest, prepare, preprocess, and engineer baseline features.
2. S05 runs engine baselines.
3. S06 evaluates recipe variants and selects a Phase B champion.
4. S08/S09 performs Phase C hyperparameter optimization.
5. S10 compares baseline, Phase B, and Phase C candidates.
6. S12 registers the selected model.
7. S13 emits drift evidence.
8. S14 evaluates retraining policy.

This is the right general decomposition. The defects are primarily in the contracts between these stages, not in the number of stages.

## Review Findings

### P1. The canonical submission path is not a data-driven configuration search

`--use_phase1` is opt-in in [`pipelines/submit_pipeline.py`](../pipelines/submit_pipeline.py#L467). When Phase 1 is not selected, missing recipe settings default to two recipes from `variant_search` ([lines 626-636](../pipelines/submit_pipeline.py#L626)). The selector takes the first recipes in alphabetical order ([`src/utils/recipe_selector.py`, lines 200-204](../src/utils/recipe_selector.py#L200)).

Even the opt-in profiling path depends on a local submit-host copy of the dataset. When that file is absent, submission explicitly falls back to alphabetical selection ([`pipelines/submit_pipeline.py`, lines 755-768](../pipelines/submit_pipeline.py#L755)). This means an Azure-accessible dataset can still receive a non-data-driven shortlist.

**Product impact:** the default behavior cannot support the claim that the accelerator finds the best pipeline configuration for the supplied data.

**Required correction:** make Azure-data profiling, deterministic eligibility checks, and data-driven shortlisting part of one canonical submission flow.

### P1. Round 1 reports proxy failures but does not prune candidates

S06 records whether the proxy evaluation failed or errored, then enters the full engine loop for that variant without using the result as a gate ([`src/steps/s06_phaseb_variant_runner.py`, lines 2607-2628](../src/steps/s06_phaseb_variant_runner.py#L2607)).

**Product impact:** the advertised low-cost pretraining stage does not reliably eliminate poor variants before expensive training.

**Required correction:** define explicit Round 1 eligibility and ranking semantics, retain rejection evidence, and train only the bounded survivors.

### P1. The locked holdout is used to select the final phase champion

S10 compares baseline, Phase B, and Phase C metrics and chooses the maximum-scoring candidate ([`src/steps/final_evaluation.py`, lines 992-1017](../src/steps/final_evaluation.py#L992)). Those metrics are produced from the canonical holdout. This makes the holdout part of model selection rather than an independent final audit.

This is a separate issue from the recently corrected preprocessing leakage. Fitting transforms only on training data prevents transform leakage, but repeatedly selecting against the holdout still creates optimistic selection bias. Scikit-learn documents the need to separate model selection from final evaluation and describes nested cross-validation where data is limited: [cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html), [nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

**Required correction:** use training CV for estimator search, a validation protocol for pipeline/phase selection, and a locked test partition once for the final report. Support stratified, grouped, and time-aware split policies where applicable.

### P1. Engine scores are not comparable

The PyCaret classification baseline computes balanced accuracy and threshold metrics by predicting on the full dataframe used for setup ([`src/steps/stage5_pycaret_train.py`, lines 175-205](../src/steps/stage5_pycaret_train.py#L175)). The FLAML baseline creates and evaluates against an internal 80/20 split ([`src/steps/stage5_flaml_train.py`, lines 115-160](../src/steps/stage5_flaml_train.py#L115)). S06 also changes its FLAML evidence method according to remaining runtime.

**Product impact:** a phase or engine may win because it received an easier evaluation protocol, not because its configuration is better.

**Required correction:** engines should produce fitted candidate artifacts. A shared evaluator should score every candidate using the same folds or validation partition, metric definitions, positive-label handling, and sample weights.

### P1. Fallback behavior breaks model/configuration identity

When the S06 leaderboard champion and retained in-memory model disagree, the code describes this as a hard error but downgrades it and trains a safety-net XGBoost or clustering model ([`src/steps/s06_phaseb_variant_runner.py`, lines 3141-3211](../src/steps/s06_phaseb_variant_runner.py#L3141)). The original champion manifest can therefore describe a different algorithm from the persisted `model.pkl`.

Phase C has the same identity problem. An unknown or unavailable champion algorithm silently falls back to XGBoost ([`src/steps/phasec_optuna_hpo.py`, lines 579-616](../src/steps/phasec_optuna_hpo.py#L579), [lines 652-674](../src/steps/phasec_optuna_hpo.py#L652)), and failed HPO trains another fallback model ([lines 809-877](../src/steps/phasec_optuna_hpo.py#L809)).

Phase C also does not receive the fitted Phase B candidate pipeline. It reads Stage 4 data and partially reconstructs the winning recipe, currently covering encoding and scaling rather than the full imputation, outlier, imbalance, feature-selection, and other learned-transform contract ([`src/steps/phasec_optuna_hpo.py`, lines 561-623](../src/steps/phasec_optuna_hpo.py#L561)).

**Product impact:** the registered artifact may not be the model/configuration reported as the winner.

**Required correction:** never silently replace a candidate family. A missing artifact is a failed candidate. A fallback may be evaluated only as a separate candidate with its own ID, manifest, metrics, and policy status. Phase C must tune the exact complete fitted Phase B pipeline and algorithm family, or emit `not_tunable` and preserve the Phase B candidate. Partial reconstruction is not configuration optimization.

### P1. Registered models do not share one raw-input inference contract

S06 and baseline stages generally persist an estimator after external preprocessing. S10 copies `model.pkl` as the champion artifact ([`src/steps/final_evaluation.py`, lines 1273-1295](../src/steps/final_evaluation.py#L1273)), and S12 accepts only that canonical artifact ([`src/steps/s12_model_registration.py`, lines 768-772](../src/steps/s12_model_registration.py#L768)). Phase C is the exception because it can wrap preprocessing and estimator in a scikit-learn pipeline.

S12's documentation mentions a signature, but registration does not currently provide a model signature or input example. MLflow recommends logging a model signature and input example, and it can package a scikit-learn pipeline as one model: [MLflow model documentation](https://mlflow.org/docs/latest/ml/model/), [Azure ML model logging guidance](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-mlflow-models?view=azureml-api-2).

**Required correction:** introduce one small `ModelBundle` contract for all phases:

- fitted preprocessing and estimator;
- raw input schema and target/identifier exclusions;
- MLflow signature and representative input example;
- label mapping and decision threshold where applicable;
- configuration, algorithm, engine, data, split, code, and environment IDs;
- load-and-predict smoke test before registration.

This is a contract, not a new serving framework.

### P1. The quality-gate policy is internally inconsistent

S10 documents `block_on_quality_fail=False` as warn-only and records the configured value ([`src/steps/final_evaluation.py`, lines 1027-1047](../src/steps/final_evaluation.py#L1027)). S12 nevertheless skips registration whenever `quality_gate_passed` is false ([`src/steps/s12_model_registration.py`, lines 948-957](../src/steps/s12_model_registration.py#L948)).

**Required correction:** emit one policy decision such as `pass`, `warn`, or `block`; downstream registration must act on that decision rather than reinterpret the raw metric result.

### P1. The committed recipe catalog is not build-clean

At this review snapshot:

| Task | Recipes checked | Valid | Invalid |
|---|---:|---:|---:|
| Classification | 260 | 245 | 15 |
| Regression | 127 | 103 | 24 |
| Clustering | 67 | 66 | 1 |
| **Total** | **454** | **414** | **40** |

Validation currently happens inside S06 after Azure compute starts. The catalog also contains semantically duplicate preprocessing contracts, particularly in generated recipe sets.

**Product impact:** invalid or duplicate configurations consume selection capacity and Azure runtime, and the search space cannot be versioned as a trustworthy product asset.

**Required correction:** make catalog validation a local/CI compile step. Assign a canonical hash to each normalized configuration, reject invalid task/transform combinations, deduplicate equivalent contracts, and publish a capability matrix before submission.

### P1. Configuration completeness depends on hidden defaults

Ten of the sixteen `config_*_azureml.yml` files omit the stage, phase, seed, holdout, and registry sections used by the expanded clustering configurations. The schema still includes `forecasting` and declares only `pycaret` and `flaml` as engines ([`src/orchestration/config_schema.py`, lines 7-10](../src/orchestration/config_schema.py#L7), [lines 112-166](../src/orchestration/config_schema.py#L112)).

**Required correction:** define one versioned product configuration schema with required product-level fields, strict nested validation, explicit defaults visible in a compiled configuration artifact, and migration rules.

### P1. Submission and resubmission do not preserve one immutable execution contract

The submitter can apply hidden defaults and choose a legacy graph, while API resubmission reloads the currently mutable YAML rather than the exact resolved configuration used by the original job ([`api/services/pipeline_service.py`, line 1335](../api/services/pipeline_service.py#L1335)).

**Product impact:** a "resubmit" operation can run different recipes, defaults, environments, or data versions under the same user intent.

**Required correction:** every accepted submission should create an immutable execution manifest containing the compiled configuration hash, selected recipe IDs, engines, seed, data asset/version, code SHA, component/environment hashes, budget, and parent MLflow run ID. Resubmission should use that revision or explicitly create a new revision.

### P1. Final MLflow reporting can mix unrelated executions

Final evaluation scans recent experiment runs and groups them by mutable run names rather than consuming explicit child run IDs from the current pipeline ([`src/steps/final_evaluation.py`, lines 83-127](../src/steps/final_evaluation.py#L83)).

**Product impact:** an "all stages" report can incorporate historical runs from another submission, weakening auditability and winner lineage.

**Required correction:** pass the immutable execution manifest and exact parent/child MLflow run IDs into final evaluation. The API should expose those IDs with the output metrics.

### P1. The intended second engine is unresolved

The stated product intent is PyCaret plus Azure ML. The current training code implements PyCaret plus FLAML, while Azure ML acts as the orchestrator. Azure ML AutoML supports tabular classification and regression (and forecasting), but its documented task list does not include clustering: [Azure AutoML task configuration](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train?view=azureml-api-2).

This needs a product decision before implementation:

- **Option A:** PyCaret + FLAML are the two search engines; Azure ML is the execution and governance platform.
- **Option B:** PyCaret + Azure ML AutoML are the two engines for classification/regression, with an explicitly documented clustering engine policy.
- **Option C:** retain FLAML only as a bounded implementation detail behind an engine-neutral contract, then add Azure ML AutoML after parity tests.

The proposal recommends Option C if Azure ML AutoML is a required product capability. It preserves current value without claiming that Azure ML AutoML already exists in the code.

### P1. Drift and retraining still have competing submission paths

S13/S14 now support the correct ownership boundary: pipeline stages emit evidence and an external controller decides whether to submit. However, [`scripts/setup_drift_schedule.py`](../scripts/setup_drift_schedule.py#L92) independently constructs a full pipeline on every schedule tick, bypassing the policy controller and canonical submitter. The API schedule catalog is planned configuration rather than authoritative Azure schedule state, and the current decision ledger is a local JSONL file.

**Product impact:** policy can say "blocked" while a legacy schedule still starts training, and multiple API instances cannot share one durable decision history.

**Required correction:** keep one lifecycle: schedule controller, durable evidence/baseline decision, canonical submission only when allowed. Read actual Azure schedule state and move production ledgers to shared durable storage.

### P1. Search budgets and reproducibility are not hard contracts

S06 raises FLAML budgets to an internal floor and can add deadline buffer rather than treating the configured value as a ceiling ([`src/steps/s06_phaseb_variant_runner.py`, lines 1758-1761](../src/steps/s06_phaseb_variant_runner.py#L1758), [lines 2641-2658](../src/steps/s06_phaseb_variant_runner.py#L2641)). PyCaret timeout checks surround blocking calls instead of enforcing cancellation. Phase C creates Optuna studies without a configured seeded sampler, while several paths retain hard-coded seed values. Source packaging can also lose Git identity and fall back to timestamp-based code versions.

**Product impact:** candidate completion depends on execution order and runtime variance, and repeated runs may not reproduce the same funnel or champion. If too few candidates complete, selecting the only survivor is not evidence that it is the best configuration.

**Required correction:** propagate one configured seed, inject source SHA and immutable data/environment identities, enforce wall-clock ceilings with cancellable worker boundaries, record timeouts as censored evidence, and require a configured minimum comparable candidate set before declaring a champion.

### P1. The current API security model is private-beta only

API clients share one static API key, while Azure actions run under one process identity ([`api/core/security.py`, line 10](../api/core/security.py#L10), [`api/core/azure_ml.py`, line 15](../api/core/azure_ml.py#L15)). Approval records do not yet establish an authenticated actor and tenant/workspace authorization.

**Product impact:** the API cannot safely claim multi-user or multi-tenant isolation.

**Required correction:** label the current deployment single-operator/private-beta. Before broader access, use Entra ID/OIDC, role claims, actor-scoped audit records, and workspace authorization. This is a release boundary, not a blocker for search-correctness work.

### P2. MLflow tracking is fragmented and version compatibility is unbounded

Multiple stages rewrite the Azure-provided `azureml://` tracking URI to `https://`; the shared helper states that the Azure scheme is unsupported ([`src/utils/azureml_metrics_logger.py`, lines 8-12 and 47-48](../src/utils/azureml_metrics_logger.py#L8)). Current Azure ML guidance says the workspace tracking URI starts with `azureml://`, the `azureml-mlflow` plugin handles it, and Azure compute normally configures it automatically: [Azure MLflow tracking configuration](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking?view=azureml-api-2).

S06 also redirects the model registry to a local file store. Root dependencies leave `mlflow`, `pycaret`, `flaml`, and core numerical packages unpinned. Microsoft currently documents Azure ML compatibility through MLflow 2.16.x and recommends compatible version pins: [Azure ML model logging guidance](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-mlflow-models?view=azureml-api-2).

**Required correction:** use one MLflow context helper, preserve the Azure tracking URI, use parent/child run lineage consistently, prohibit a local registry in production mode, and lock tested environment versions and hashes.

### P2. UI ownership and submission status are ambiguous

React is presented as the Streamlit replacement while both remain active. The Streamlit "Live Logs" view reconstructs status text rather than displaying a real Azure log stream. Asynchronous API request state is held in a process-local dictionary, so restarts and multiple workers can lose polling state.

**Required correction:** make React the target product UI and keep Streamlit temporarily as a read-only admin fallback. Call reconstructed output an "Execution Timeline" until a bounded Azure log-tail endpoint exists. Persist submission tickets and idempotency keys before enabling multiple API workers.

### P2. Product and proof documentation is stale

The root README still presents Version 1, active documents disagree about S14 validation status, and the reusable MLOps skill packet contains old branches, paths, task scope, engine claims, and MLflow URI guidance.

**Required correction:** generate the stage/config/recipe inventory from source, keep proof levels explicit, and update reusable skills to distinguish intended product behavior from current implementation behavior.

## Target Minimal Architecture

The recommended architecture keeps the existing Azure ML component model and introduces stricter contracts between stages:

1. **Compile configuration:** validate one versioned user configuration and materialize all defaults.
2. **Freeze execution:** create the immutable execution manifest used by every downstream stage and resubmission.
3. **Resolve data:** resolve the Azure ML data asset or datastore input without requiring a submit-host copy.
4. **Profile safely:** compute a bounded profile on training-only data and record its versioned artifact.
5. **Compile catalog:** validate capabilities, remove duplicate normalized contracts, and reject invalid recipes before submission.
6. **Round 0 feasibility:** eliminate configurations that cannot apply to the profiled schema.
7. **Round 1 proxy:** rank candidates with a deterministic, cheap evaluator and prune to a configured budget.
8. **Round 2 tournament:** train surviving engine/configuration candidates under one evaluator and hard wall-clock budget.
9. **Same-family HPO:** tune only supported champion families; preserve the untuned candidate when HPO is unsupported or fails.
10. **Validation selection:** select the pipeline champion using the shared validation protocol.
11. **Locked test audit:** evaluate the selected champion once and never use that result to select another candidate.
12. **Package model:** create and smoke-test the self-contained `ModelBundle`.
13. **Register by policy:** apply the explicit `pass`, `warn`, or `block` decision.
14. **Monitor and decide:** emit drift evidence in S13 and let the external S14/controller policy own retraining submission.

## What Should Remain

The following existing choices are directionally correct and should be refined rather than replaced:

- Azure ML components and pipeline jobs as the orchestration boundary;
- YAML configuration and recipe assets as reviewable product inputs;
- task-specific metric handling;
- MLflow for experiment and model lineage;
- S13 evidence generation separated from external retraining control;
- explicit registration and quality-gate stages;
- focused local tests plus bounded Azure canaries.

## Deliberate Non-Goals

This proposal does not recommend:

- a new workflow orchestrator;
- a plugin framework for stages;
- a feature store;
- a second metadata database;
- automatic production deployment;
- a distributed search scheduler;
- a generalized forecasting architecture;
- more generated recipes before the existing catalog is valid and deduplicated.

Each of those would increase surface area without correcting the current selection and evidence contracts.

## Recommended Delivery Gates

These are approval gates for a later implementation plan, not an authorization to start implementation.

### Gate 0: Product Contract

- Decide the second-engine policy.
- Freeze supported tasks at classification, regression, and clustering.
- Publish configuration schema v1 and normalized candidate identity.
- Define split, metric, budget, and policy-decision contracts.

### Gate 1: Evaluation Correctness

- Introduce the shared evaluator.
- Remove holdout-based champion selection.
- Eliminate silent algorithm fallback and manifest/model mismatches.
- Make warn versus block behavior consistent.

### Gate 2: Search Correctness

- Make data-driven selection canonical.
- Run profiling against Azure-resolved data.
- Make Round 1 prune candidates.
- Validate and deduplicate the complete recipe catalog in CI.

### Gate 3: Model and MLflow Contract

- Package every phase as the same self-contained `ModelBundle`.
- Add signature, input example, raw-input smoke test, and immutable lineage IDs.
- Normalize Azure MLflow setup and lock dependency versions.

### Gate 4: API and UI Product Workflow

- Expose compiled configuration, candidate funnel, rejection reasons, budgets, evidence levels, and policy decisions.
- Keep API submission behind the canonical submitter and expose one authoritative job state.
- Do not add UI controls for settings that the backend cannot validate and enforce.

### Gate 5: Azure Evidence

- Run one bounded canary for each supported task on the exact reviewed commit and environment hashes.
- Verify raw-input bundle inference, MLflow lineage, registration policy, drift handoff, and retraining decision.
- Keep local tests, Azure job completion, registered artifact validation, and deployed inference evidence as separate proof levels.

## Acceptance Criteria

The later implementation should not be called complete until:

- 100% of committed production recipes pass schema and semantic validation;
- equivalent normalized configurations cannot occupy multiple shortlist slots;
- candidate selection is deterministic for the same data profile, seed, catalog, and configuration;
- Round 1 demonstrably reduces the candidate set before full training;
- every engine is scored by the same evaluator;
- the locked test partition is evaluated exactly once after selection;
- model artifact identity always matches the winning manifest;
- unsupported HPO preserves the Phase B candidate without algorithm substitution;
- configured wall-clock budgets are not silently increased;
- champion selection fails closed when too few comparable candidates complete;
- the same seed, execution manifest, data version, code SHA, and environments reproduce the candidate funnel;
- all registered models pass raw-input load-and-predict tests;
- MLflow links data, profile, candidate, split, code, environment, model, and policy decisions;
- classification, regression, and clustering each have exact-commit Azure canary evidence.

## Proof Matrix

| Proof level | Required evidence | What it does not prove |
|---|---|---|
| Local contract | schema checks, recipe compile, unit/integration tests | Azure execution |
| Azure component | component outputs and MLflow records for exact code/environment | full pipeline correctness |
| Azure pipeline | bounded three-task canaries and stage contracts | deployed serving |
| Registered model | exact version, tags, bundle smoke test | endpoint behavior |
| Deployed inference | raw-request tests, monitoring, rollback evidence | future data quality |

## Decisions Required Before Planning

1. Confirm whether "Azure ML as the second engine" means Azure ML AutoML, or whether Azure ML is the platform and FLAML remains the second engine.
2. Choose the clustering engine policy if Azure ML AutoML is required for the other task types.
3. Choose the default final-evaluation policy: train/validation/test or nested CV for small datasets.
4. Define the default search budget in both candidate count and hard wall-clock time.
5. Decide whether a quality-gate warning may register a model with a non-production stage, or whether all failed gates must block registration.
6. Confirm whether invalid generated recipes should be repaired, quarantined, or regenerated from a smaller canonical catalog.

## Proposed Next Artifact

After these decisions are approved, the next artifact should be a file-by-file implementation plan with:

- contract changes and migration order;
- test-first acceptance cases;
- exact component and API impacts;
- backward-compatibility decisions;
- bounded Azure validation jobs and stop conditions;
- rollback and evidence requirements.

No broad implementation should begin before Gate 0 decisions are recorded.
