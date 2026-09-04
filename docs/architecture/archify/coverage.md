# MLOps Solution Accelerator V3 — capability coverage

Inspected: 2026-09-04. Source revision: `282537e43b9693287bb9924a9896aab071f60a26`. Branch: `codex_ys/mlops-pipeline-correctness`. This guide covers the implemented product capabilities and supporting operator tools at that revision. It is not a deployment or release-readiness report.

Start with [the visual index](index.html). The five views are [system overview](overview.html), [pipeline data flow](pipeline.html), [candidate search](search.html), [operations and retraining](operations.html), and [release evidence](release.html).

## Reading the evidence

Code and component input/output wiring take precedence over historical diagrams and prose. Source links below are pinned to the inspected commit. A stage is an Azure ML component invocation. A recipe is a configuration of data transformations; a candidate also binds the task, engine, estimator, hyperparameters, split and execution identity. Phase A discovers baselines, Phase B searches recipes and estimators, and Phase C tunes the Phase B winner. CV means cross-validation; EDA means exploratory data analysis; HPO means hyperparameter optimization; PSI means population stability index.

Descriptions marked **implemented** mean the behavior exists in source. They do not assert that it ran successfully in Azure. Configured qualification scenarios are intended workloads; a scenario catalog is not execution evidence. Historical or unwired modules are listed separately. No application, model training, cloud job, email or hosted pipeline was run by this documentation task.

## Exact active graph: 14 component invocations

Both `full_pipeline()` and `full_pipeline_v2()` instantiate these 14 components. `full_pipeline_v2` is a function name inside the V3 product, not proof that the historical V2 product is active. See [component loading and both builders](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L56).

| Stage | Exact component YAML | Implemented role and important inputs |
|---|---|---|
| S01 | `components/stage1_ingestion.yml` | Load configured CSV through Azure ML datastore URI, with mounted-input fallback; hash content, inspect quality and produce EDA. |
| S02 | `components/stage2_preparation.yml` | Assign canonical train/locked-test partition before learned transformations; emit raw training data, raw locked test and split manifest, alongside preparation artifacts. |
| S03 | `components/stage3_preprocessing.yml` | Apply baseline recipe transformations to the preparation branch; emit preprocessing evidence. |
| S04 | `components/stage4_feature_engineering.yml` | Apply baseline feature engineering to S03 output; publish processed data and a training-only reference for S13. |
| S05a | `components/stage5_pycaret_train.yml` | Discover PyCaret baseline using S02 raw training data and execution/split manifests; emit metrics and raw-input model bundle. |
| S05b | `components/stage5_flaml_train.yml` | Discover FLAML baseline using the same raw training data and manifests. For clustering this invocation emits an explicit unsupported/skip result. |
| S05z | `components/aggregate_baseline.yml` | Aggregate Phase A evidence and preserve its eligible champion bundle. |
| S06 | `components/s06_phaseb_variant_runner.yml` | Search candidates on S02 raw training data using the immutable candidate catalog and execution/split manifests; emit champion, leaderboard, search evidence and selection-only quality decision. |
| S08 | `components/phasec_optuna_hpo.yml` | Tune the complete eligible Phase B recipe/algorithm using S02 raw training data and S06 manifests. |
| S09 | `components/aggregate_phasec.yml` | Aggregate Phase C metrics and preserve its optimized champion bundle. |
| S10 | `components/final_evaluation.yml` | Compare compatible Phase A/B/C selection evidence, freeze one champion, then evaluate its exact bundle once against S02 locked test. |
| S12 | `components/s12_model_registration.yml` | Validate final quality and bundle identity; register the exact champion and emit registry identity, without automatic promotion. |
| S13 | `components/s13_drift_monitor.yml` | Use S04 training-only reference, S10 report, S12 registry identity and optional approved baseline to produce drift evidence and candidate baseline. |
| S14 | `components/s14_retrain_decision.yml` | Consume S13 evidence plus S10/S12 identity to write retraining decision and ledger-record artifacts; no nested training submission. |

Exact invocation evidence: [S01–S06](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L321), [S08–S14](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L397), [FLAML clustering skip](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage5_flaml_train.py#L119).

### Critical data-flow distinction

S02 `raw_train_out` feeds S05a, S05b, S06, S08 and S10. S02 `raw_holdout_out` feeds S10. S02 `split_manifest_out` binds the row partition. S03 → S04 is a separate diagnostic/reference branch; S03/S04 transformed data does **not** feed model training in the active builders. S04 `train_out` feeds S13 and excludes the locked final holdout. Learned transformations for evaluated candidates live inside their fitted recipe pipelines. Evidence: [raw inputs](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L329), [final evaluation and drift wiring](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L414), [canonical split and raw extraction](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage2_preparation.py#L104).

## Capability inventory

### 1. Three supported task families — implemented

Classification predicts categories, regression predicts continuous values, and clustering identifies groups without a supervised target. The compiler accepts exactly these three tasks. PyCaret and FLAML support the supervised product paths; clustering is PyCaret-only. The common evaluator's default comparison metrics are balanced accuracy, R² and silhouette score respectively; compiled configuration also carries the metric contract. Clustering is not a hidden supervised task, and the compiler clears its target column.

Evidence: [task and engine contract](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/contracts.py#L19), [compiler engine restrictions](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/config_compiler.py#L293), [default metrics](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/common_evaluator.py#L96), [target handling](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/config_compiler.py#L986).

### 2. Config compilation and recipe catalog — implemented

YAML config controls dataset, workspace/compute bindings, task, partitioning, engines, recipes, budgets, quality and retraining settings. Compilation materializes defaults, rejects unknown/invalid fields and incompatible task/engine combinations, and creates a reproducible compiled contract. Recipe utilities catalog and select transformation variants; the submission path packages an immutable candidate catalog rather than relying on a diagram's fixed recipe count. The workbench can preview configuration, schema, stage plan and budgets.

Evidence: [compiler](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/config_compiler.py#L357), [recipe catalog](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/recipe_catalog.py#L1), [config API](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/routers/configs.py#L41).

### 3. Canonical submission and replay — implemented

`pipelines/submit_pipeline.py` is the canonical Azure ML submission CLI. It validates config and revision contracts, enforces duplicate-submit locking and active-job checks, packages source/execution artifacts, binds component environments, and submits or emits a dry-run graph. Exact replay preserves parent identity; a changed source/config/execution revision requires an explicit new-revision reason. Batch wrappers and API submission converge on this boundary. A dry run proves graph construction only.

Evidence: [CLI and options](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/submit_pipeline.py#L1026), [revision checks](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/submit_pipeline.py#L650), [batch boundary](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/_canonical_batch_submit.py#L1).

### 4. Immutable execution and data identity — implemented

Execution, candidate, split and quality contracts bind configuration, source upload content, dataset version/content, environment, seeds, selection evidence and model identity. The submission source hash covers the upload manifest, not just the Git commit label. S01 recomputes canonical dataframe content identity. S06 binds candidate records to the runtime split. This lets later evaluation, registration, replay, controller planning and qualification verify that evidence belongs together.

Evidence: [contracts](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/contracts.py#L107), [source manifest](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/submit_pipeline.py#L572), [execution manifest](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/submit_pipeline.py#L794), [runtime split binding](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/s06_phaseb_variant_runner.py#L1573).

### 5. CSV ingestion and EDA — implemented

S01 constructs the configured `azureml://` datastore URI, reads CSV with configurable delimiter/encoding and supports a mounted-path fallback. It checks optional content SHA-256, analyzes columns, target and data quality, produces reports/visualizations, and derives recipe recommendations. Time-series detection in EDA does not add forecasting to the supported training contract. Active pipeline stages read datastore inputs and write job outputs; acquisition and asset-registration utilities are separate operator tools.

Evidence: [ingestion execution](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage1_ingestion.py#L923), [EDA](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage1_ingestion.py#L488), [recipe recommendations](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage1_ingestion.py#L741).

### 6. Preparation and locked-test isolation — implemented

S02 assigns stable row identities and partitions before learned transformations, applies configured exclusions and train-derived schema filters consistently, and keeps raw training and locked-test artifacts. The split manifest binds row membership. Preparation also emits diagnostics and prepared data for the S03/S04 branch. The common evaluator intentionally has no locked-test input; final evaluation checks the held-out row identity before its audit.

Evidence: [partition construction](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage2_preparation.py#L104), [raw extraction](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage2_preparation.py#L147), [holdout validation](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/final_evaluation.py#L1023).

### 7. Fitted preprocessing and feature engineering — implemented

Candidate recipes configure supported imputation, categorical encoding, numeric scaling, feature selection and dimensionality reduction. The inference-safe fitted preprocessor owns learned state and transforms future raw rows consistently. Evaluation combines preprocessing, optional classification-only imbalance resampling and estimation within each fold; nonclassification tasks cannot use that resampling path. Separate S03/S04 scripts retain baseline diagnostic and drift-reference behavior.

Evidence: [supported fitted transformations](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/fitted_variant_preprocessor.py#L14), [fold-local pipeline and resampling](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/common_evaluator.py#L104), [S03/S04 wiring](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L329).

### 8. Phase A baseline discovery — implemented

PyCaret and FLAML discover estimators under their configured budgets. Eligible candidates are evaluated under the common selection contract and fitted into raw-input model bundles. The baseline aggregator compares their eligible evidence. Saved Phase A bundles are reloaded and checked for deterministic predictions on example raw rows. S05b produces explicit skipped artifacts for clustering.

Evidence: [PyCaret baseline](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage5_pycaret_train.py#L1), [FLAML baseline](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/stage5_flaml_train.py#L119), [bundle fitting and smoke behavior](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/phasea_model_bundle.py#L264).

### 9. Phase B adaptive search — implemented

One S06 component runs a bounded candidate search. It validates the catalog/execution binding, performs feasibility screening (Round 0), proxy evaluation (Round 1) and fuller evaluation of survivors (Round 2), with planner controls and preprocessing-cache support. The CLI caps Round 1 at 40 and Round 2 at 8; actual requested counts can be lower. Deadline and timeout handling produce explicit failure/censored evidence. Outputs include a leaderboard, all-results JSON, candidate ledger/signals, champion manifest/bundle, execution/split manifests and a selection-only quality decision. Safety-net review is represented where required by config; it is not a substitute for full selection evidence.

Evidence: [screening and proxy stages](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/s06_phaseb_variant_runner.py#L1075), [catalog binding](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/s06_phaseb_variant_runner.py#L1521), [caps](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/s06_phaseb_variant_runner.py#L2772), [deadline enforcement](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/s06_phaseb_variant_runner.py#L366).

### 10. Common evaluation and candidate accounting — implemented

Discovery engines do not define the final cross-engine comparison. The common evaluator refits supervised candidates on deterministic matching CV folds, using isolated fitted preprocessing and a common metric contract. It records completion, timeout/censoring, split fingerprints and run lineage. Clustering has task-specific evaluation rather than supervised target folds. Candidate ledger utilities write per-candidate artifacts and merge stage tables/summaries. Only finite eligible evidence may select a winner.

Evidence: [evaluation specification](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/common_evaluator.py#L1), [evaluation implementation](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/common_evaluator.py#L296), [candidate artifacts and ledger](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/candidate_ledger.py#L211).

### 11. Phase C optimization — implemented

Optuna tunes the Phase B champion's algorithm and complete recipe on raw training data, preserving candidate/execution lineage. The implementation uses seeded sampling, bounded final fitting, trial/cost reporting and an explicit unsupported-skip output. S09 aggregates the resulting evidence and bundle. HPO is not a reason to reuse the locked test during search.

Evidence: [recipe and sampler](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/phasec_optuna_hpo.py#L166), [unsupported result and cost report](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/phasec_optuna_hpo.py#L301), [raw-data input](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L397).

### 12. Final champion audit and quality policy — implemented

S10 validates lineage and selection comparability, chooses one champion from Phase A/B/C selection evidence, and binds that candidate to its source bundle before final testing. Its locked-test metric and configured threshold determine `pass`, `warn` or `block`. An invalid champion blocks; configured quality failure policy decides whether an otherwise valid below-threshold model warns or blocks. `warn` may permit registration, but protected champion/production aliases cannot be assigned to warning-quality models.

Evidence: [comparability and selection](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/final_evaluation.py#L573), [quality decisions](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/final_evaluation.py#L653), [bundle binding](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/final_evaluation.py#L1058), [alias restrictions](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/config_compiler.py#L994).

### 13. Raw-input model bundles and registry — implemented

`ModelBundle` preserves fitted preprocessing, estimator, optional label decoder, input schema/example, dependencies, selection metrics and lineage. Its hash/integrity checks tie serialized state to the evaluated artifact. Prediction accepts raw feature rows and applies the saved transformations. S12 validates quality/bundle identity, logs the exact bundle and signature through MLflow, and contains a run-bound Azure ML SDK registration fallback for missing Azure ML artifact repository support. The pipeline gives S12 delegated user identity. Registration records an exact model version; it does not create a serving endpoint or promote the model.

Evidence: [bundle contract](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/model_bundle.py#L260), [prediction and serialization](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/utils/model_bundle.py#L372), [exact MLflow bundle](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/s12_model_registration.py#L213), [registration path](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/s12_model_registration.py#L670), [SDK fallback](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/steps/s12_model_registration.py#L804), [delegated identity](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L426).

### 14. Drift evidence and baseline governance — implemented

S13 produces drift reports and a candidate baseline from training-only reference data. An optional prior baseline permits comparison; same-run self-check evidence must not be represented as cross-job drift validation. The API joins S13 evidence to identity-matched S14 policy evidence. Baseline capture discovers a URI; baseline approval separately verifies completed-job status, task/config/dataset/source/execution identity, metadata and reference data before recording approval. Approving a drift baseline does not promote a production model.

Evidence: [S13 wiring](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L435), [drift inspection](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/pipeline_service.py#L1385), [approval validation](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/auto_retrain_service.py#L283), [baseline capture](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/pipeline_service.py#L1798).

### 15. Retraining decisions and external controller — implemented

S14 writes artifacts only. Pure policy decides observation, baseline refresh, candidate submission and promotion eligibility. The external controller resolves an approved baseline and explicit S14 evidence, validates source identity/freshness, reserves durable ledger state and submits through the canonical CLI. Successful candidate submission records `manual_pending`; uncertain submission outcomes require reconciliation. The daemon is a separate process. API controller-plan requests are dry-run previews. Automatic candidate retraining and manual promotion are distinct operations.

Evidence: [policy](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/auto_retrain_policy.py#L1), [controller plan](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/auto_retrain_controller.py#L125), [automated controller](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/automated_retrain_controller.py#L90), [daemon](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/run_auto_retrain_daemon.py#L64).

### 16. React and Streamlit operator interfaces — implemented

React exposes Home, Submit, Focus, Configs, Auto Retrain, Drift, Reports, Logs, Notifications, Settings and Users through an authentication gate. Streamlit provides a second API client. Operators can configure submissions, inspect jobs/experiments/steps, cancel and resubmit, inspect outputs and reports, approve baselines and preview controller plans according to access. The frontends are alternative clients; their default local ports overlap unless configured separately.

Evidence: [React routes](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/react-ui/src/App.tsx#L16), [Streamlit](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/ui/app.py#L12), [Streamlit API client](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/ui/api_client.py#L14), [launcher](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/run_app.sh#L16).

### 17. API job operations and durable submission tracking — implemented

FastAPI offers synchronous/asynchronous submit, request polling, jobs/experiments, detailed status, cancellation, resubmission, metrics, reports and output listing/download/preview. Asynchronous submission uses a two-thread in-process executor plus durable request documents. Restart recovery looks for matching Azure job tags; unresolved pending requests require reconciliation instead of blind resubmission. Durable records do not make the executor a distributed job queue.

Evidence: [pipeline routes](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/routers/pipelines.py#L58), [submission worker and recovery](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/pipeline_service.py#L526), [request store](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/submission_request_store.py#L42).

### 18. Authentication, user roles and deployment profiles — implemented

Development/private API-key operation and multi-user delegated Microsoft Entra authentication are distinct profiles. The multi-user backend validates token signature, issuer, audience, tenant, calling client, delegated scope and server-owned user authorization. Viewer, operator and admin roles govern writes and user management. Nondevelopment startup checks constrain origins, durable paths, reload and config mutation; multi-user state requires local-disk SQLite. Browser user identity and backend Azure credentials serve different purposes.

Evidence: [API authorization](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/core/security.py#L43), [Entra validation](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/core/entra_auth.py#L84), [profile checks](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/core/config.py#L79), [Azure client credential chain](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/core/azure_ml.py#L16).

### 19. Durable operational state and audit — implemented

SQLite stores request documents, controller events, decisions, workspace binding, users and audit records with write-ahead logging, full synchronization and transactions. Supported legacy JSON/JSONL stores have explicit migration support. The API binds state to a workspace, and authenticated actions retain actor context. This is single-host local state, not an external database or shared multi-host controller lock service.

Evidence: [SQLite store](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/operational_state.py#L51), [ledger](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/src/orchestration/auto_retrain_decision_ledger.py#L116), [migration utility](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/migrate_operational_state.py#L1), [user access](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/user_access_service.py#L13).

### 20. Reporting, notifications and monitoring — implemented

The API combines baseline/search/HPO/final metrics, named outputs and previews. The report service writes Markdown, JSON and CSV packages and can send attachments through configured SMTP. Sending email is separate from report generation. React Logs reconstructs a job/step timeline; Streamlit falls back to status and an Azure ML Studio link when raw log text is unavailable. The API health field based on ML client construction is not a workspace connectivity probe.

Evidence: [artifacts and reports](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/pipeline_service.py#L967), [report generation](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/notification_service.py#L120), [SMTP](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/notification_service.py#L449), [timeline](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/react-ui/src/pages/Logs.tsx#L19), [health semantics](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/routers/health.py#L13).

### 21. Schedule inspection and operator maintenance — implemented tools

The API compares the planned schedule catalog to Azure schedule readback; missing/disabled/unverified rows have distinct meanings. The legacy setup script explicitly refuses static training schedule creation and explains the external-controller boundary. Supporting scripts monitor batches, inspect Azure environments/datastores, capture dependency locks, validate controller releases and support documented credential recovery. Tool presence does not establish deployed scheduling, healthy credentials or a running daemon.

Evidence: [schedule readback](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/api/services/auto_retrain_service.py#L170), [schedule compatibility script](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/setup_drift_schedule.py#L1), [batch monitor](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/monitor_batch.py#L1), [controller validation](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/validate_controller_release.py#L36), [environment lock](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/capture_azure_environment_lock.py#L21).

### 22. Qualification data acquisition and profiling — implemented tools

Separate utilities acquire/normalize UCI data, stage OpenML sources, compute content/schema fingerprints, inspect dataset shape/targets/privacy properties, generate industry configs and plan/register versioned Azure ML data assets. These are release preparation actions outside the ordinary pipeline's read-only datastore stage behavior. Their execution requires its own environment and authorization.

Evidence: [UCI acquisition](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/acquire_uci_release_matrix.py#L359), [OpenML staging](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/stage_openml_release_sources.py#L132), [profiling](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/profile_industry_matrix.py#L256), [data asset planning/registration](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/register_qualification_data_assets.py#L25).

### 23. Fifteen industry qualification scenarios — configured, runtime unverified here

The execution catalog contains five classification, five regression and five clustering scenarios, with pinned dataset content/schema hashes. Classification: healthcare heart disease, financial-services credit default, telecom customer churn, manufacturing machine failure, insurance policy interest. Regression: education final grade, real-estate unit price, insurance claim cost, energy heating load, aviation airfoil noise. Clustering: healthcare cohorts, financial risk segments, telecom behavior segments, retail transactions, education student segments. Some scenarios reuse a dataset under a different task contract; 15 scenarios is not a claim of 15 distinct datasets.

Evidence: [complete scenario catalog](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/configs/qualification/industry_matrix_execution_catalog.yml#L1).

### 24. Registered-model inference smoke and qualification verifier — implemented tools

A dedicated submitter binds an isolated Azure ML smoke job to a specific qualification model version and execution/source/dataset identity. Scoring downloads that registered model, validates manual/unassigned lifecycle and lineage, and predicts on its saved raw-input example. The fail-closed qualification verifier joins monitor records, data asset audits, execution/split manifests, S06 selection evidence, S10 final evaluation/bundle evidence, registry/drift/retrain outputs and smoke results. Registry write, model reload/inference, and deployed endpoint acceptance are separate proof levels.

Evidence: [smoke identity validation](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/submit_registered_model_smoke.py#L46), [registered scoring](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/registered_model_inference_smoke/score.py#L47), [qualification verifier](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/scripts/verify_qualification_evidence.py#L310).

### 25. CI, dependency audit and release evidence — source-defined gates

GitHub Actions defines a backend job that materializes pinned runtime dependencies, checks dependency consistency, runs vulnerability audits with documented exceptions and executes backend tests excluding integration/Azure/slow markers. The React job installs dependencies, runs lint/type checks, tests, build and Playwright checks. Azure runtime behavior, scenario qualification and registered-model smoke evidence remain additional gates. No latest hosted CI state or current Azure acceptance was retrieved for this architectural inspection.

Evidence: [backend workflow](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/.github/workflows/release-candidate-ci.yml#L21), [React workflow](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/.github/workflows/release-candidate-ci.yml#L72), [runtime dependency specification](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/config/mlops_v3_unified_environment/conda_v33.yml#L1).

## Named pipeline outputs

The builders expose these named artifacts so the API and external verifier can retrieve them without interpreting a UI screenshot. Evidence: [pipeline return contract](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L460).

| Category | Exact output names |
|---|---|
| Data and diagnostics | `eda_report`, `prep_report`, `prep3_report`, `fe_report`, `dataset_processed`, `dataset_train`, `dataset_holdout` |
| Phase A | `baseline_pycaret_metrics`, `baseline_flaml_metrics`, `baseline_aggregate_report`, `baseline_champion_model` |
| Phase B and identity | `phaseb_leaderboard`, `phaseb_all_results`, `phaseb_champion_manifest`, `phaseb_champion_model`, `execution_manifest`, `split_manifest`, `quality_decision` |
| Phase C | `phasec_aggregate_report`, `phasec_champion_model` |
| Final model and operations | `final_report`, `final_champion_model`, `registry_info`, `drift_report`, `drift_baseline`, `retrain_decision`, `decision_ledger_record` |

`dataset_train` is S04's transformed training-only diagnostic/reference output; it must not be confused with the S02 raw training input used by candidate fitting. `dataset_holdout` is S02's raw locked test. `quality_decision` is S06 selection-only evidence; S10's final quality decision is carried in `final_report`.

## Historical, unwired and excluded behavior

| Item | Correct interpretation |
|---|---|
| S00 validation component | Source and YAML exist, but neither active builder loads/calls it. Ingestion/preparation validation still exists in the active stages. |
| S05t / forecasting files | Legacy training artifacts; forecasting is outside the current three-task contract. EDA time-series detection does not activate forecasting training. |
| S07 and S11 | No active component invocations at these identifiers. Do not fill numbering gaps with invented stages. |
| `full_pipeline_v2` | Active alternate V3 builder with planner controls; historical V2 documents do not define its behavior. |
| Azure AutoML | Azure ML hosts execution. There is no supported Azure AutoML engine in the inspected task/engine contract. |
| Older recipe counts and blueprint diagrams | Historical planning evidence; they do not override the runtime candidate catalog or active component wiring. |
| `src/main.py` and generic data-ingestion helpers | Ancillary/legacy code is not the active Azure ML submission path; helper support for asset registration does not make ordinary pipeline stages mutate datastores. |
| Static training schedules | The compatibility setup utility refuses creation/update. The external controller owns approved retraining submission. |
| Model-serving endpoint / live deployment | Bundle prediction and isolated registered-model smoke are implemented. No automatic endpoint deployment is part of the active pipeline or inspected API routers. |
| Automatic model promotion | Candidate submission is implemented; successful controller submissions remain `manual_pending`, and registry outputs preserve manual lifecycle. |
| UI online-endpoint YAML | Commented reference material, not proof of an active UI host; the file itself warns against hosting Streamlit on Azure ML Online Endpoints. |

Evidence: [current product contract](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/docs/PROJECT_REQUIREMENTS.md#L1), [reserved stages](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/docs/PIPELINE_STAGES.md#L32), [actual loaded components](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/pipelines/pipeline_builder.py#L56), [UI deployment reference](https://github.com/SAVYMINDS/YS_MVP/blob/282537e43b9693287bb9924a9896aab071f60a26/ui/azureml-streamlit-deployment.yml#L1).

## Coverage and verification boundary

The overview covers actors, interfaces, orchestration, Azure execution, storage, reports and control. The pipeline view covers all 14 active invocations and the split/reference branches. The search view expands recipe selection, shared evaluation, HPO and final bundle/quality selection. The operations view covers authentication, durable submissions, drift, baseline approval and the external candidate controller. The release view distinguishes local CI, environment/data preparation, Azure qualification, registered inference and manual release authority. This guide supplies source detail that would make those diagrams unreadably dense.

Diagram validation proves the delivered architecture artifacts satisfy the Archify rendering contract. It does not prove the application is deployed, a pipeline is green, all qualification jobs passed, drift control is live, or an endpoint is serving. Those claims require fresh evidence from their respective systems.
