# MLOps Solution Accelerator Requirements Review and Blocker Impact

Date: 2026-08-01  
Review target: `codex_ys/mlops-pipeline-correctness` at `eb739444f9d439b75232e0342f5d5c52ab86986c` plus the current working tree  
Workspace: Azure ML `mlops-accelerator`, resource group `mvpv1`, subscription `93044a08-5661-4f1b-b424-5eafe066a9d1`  
Review mode: source, contract, test, graph-dry-run, and read-only Azure inspection. No ML training ran locally and no Azure job was submitted.

> **Historical review baseline:** many source-level findings below were remediated on 2026-08-02. Use [MLOPS_REMEDIATION_STATUS_AND_USER_ACTIONS_2026-08-02.md](MLOPS_REMEDIATION_STATUS_AND_USER_ACTIONS_2026-08-02.md) for current blocker state, final local evidence, and operator actions. Keep this document as the original finding and impact record.

## Executive Verdict

The repository is not ready for beta acceptance or production release. The canonical graph builds for classification, regression, and clustering, and the local contract suite passes, but confirmed P1 defects can bias the pretraining funnel, exclude Phase A from champion selection, accept the wrong holdout or model identity, ignore declared split/configuration policy, and make auto-retrain operations misleading or unusable.

There is no confirmed source-level P0 data-loss or arbitrary-code-execution defect. There is one P0 external acceptance gate: live ARM reports the subscription state as `Warned`, while cached CLI metadata says `Enabled`. The Azure read plane works, but the current write plane and exact-source canary execution are unproven.

Release status by proof plane:

| Proof plane | Result | Meaning |
|---|---|---|
| Local unit/contract tests | Pass: 358 tests, 6 warnings | Useful local evidence only |
| Canonical SDK graph dry-run | Pass: all 3 task canaries | Graph construction and SDK validation work |
| Recipe catalog compile | Partial | Large portions of production catalogs are quarantined |
| Azure subscription/read plane | Partial | Workspace, compute, and pinned environments are readable |
| Azure write plane | Unresolved | ARM state is `Warned`; no current write probe was performed |
| Exact-source Azure canaries | Missing | No current classification/regression/clustering completion proof |
| MLflow/registration/inference | Missing live proof | Local contracts do not prove workspace behavior |
| Release provenance | Blocked | 70 modified and 57 untracked entries, including this review; branch has no upstream |

## Severity and Impact Categories

- **P0 - Acceptance impossible:** an external or systemic gate prevents the required Azure proof or operation.
- **P1 - Release blocker:** can produce a wrong winner, false lineage, broken core workflow, duplicate cost, or materially misleading operator state.
- **P2 - Material reliability gap:** increases failure/cost/support risk but does not by itself corrupt every run.
- **P3 - Cleanup:** maintainability or clarity issue with bounded direct impact.

Blockers are categorized by what happens if they are not fixed:

- **Model-selection integrity:** the product can select the wrong pipeline configuration or overstate its quality.
- **Execution reliability:** a supported configuration or operator workflow can fail at runtime.
- **Configuration fidelity and cost:** the system can execute something different from what the user configured or spend more than expected.
- **Identity, reproducibility, and audit:** a result cannot be reproduced or tied to the exact data/code/environment/model that produced it.
- **Registration and promotion safety:** the wrong model can be labeled, registered, or promoted.
- **Drift and controller safety:** retraining can be missed, duplicated, or authorized from stale/incomplete evidence.
- **Operator truth and UI:** the UI can report a state different from the policy or Azure state.
- **Security/release boundary:** the current deployment cannot safely support multiple actors or tenants.
- **Validation/release evidence:** local success can be mistaken for Azure or production proof.

## Requirement Traceability

| Requirement | Status | Review result |
|---|---|---|
| Classification, regression, clustering only | Mostly matches | Contracts enforce the three tasks, but an unsupported forecasting node still runs/skips in every graph. |
| Find the best end-to-end pipeline configuration | Does not yet match | Round 1 leaks proxy validation data, Phase A is ineligible, minimum comparable candidates are not enforced, and catalogs are heavily quarantined. |
| Pretraining stages reduce candidates before expensive training | Partial | Round 1 now prunes, but preprocessing is fitted before the proxy split and feasibility runs after the catalog cap. |
| Learned transforms fit on training data only | Partial | Stage 2-4 and the common evaluator are train/fold-local; the Round 1 proxy is not. |
| Frozen champion followed by one locked-test audit | Mostly implemented, not fully bound | S10 freezes before prediction, but does not verify the holdout against `SplitManifest` or the selected manifest against the source bundle. |
| PyCaret plus second engine | Decision blocked | Code implements PyCaret plus FLAML. The stated requirement says PyCaret plus Azure ML; Azure ML platform and Azure AutoML engine are not equivalent. |
| Config-driven execution | Does not match | Several accepted fields are forced, ignored, or replaced, including validation, planner, funnel, proxy, environment, and drift controls. |
| Immutable execution and candidate identity | Partial | S06 validates it; downstream stages, resubmit, controller replay, data, and environment binding remain incomplete. |
| MLflow logs all important stages and exact lineage | Partial local implementation | Exact child-run handling exists, but current Azure hierarchy/artifact proof is absent. |
| Exact ModelBundle registration and safe promotion | Partial | Exact returned version and pass/warn/block logic are implemented; upstream bundle identity can still be relabeled and no live registered inference proof exists. |
| Drift evidence, policy decision, external submission | Architecturally partial | S13/S14 separation is present, but policy configuration, ledger durability, UI payloads, baseline identity, and displayed evidence are defective. |
| UI/API operational visibility | Does not match | Auto-retrain plan requests fail, drift detail can contradict S14, async requests are ephemeral, and schedule state is planned rather than live. |
| Azure-only heavy execution | Matches review procedure, not acceptance | No heavy local execution occurred; Azure canary proof is still missing. |

## What Is Working

The following improvements are present and should be preserved:

- Stage 2 assigns the train/holdout boundary before learned preparation and fits imputers/statistical decisions on training rows (`src/steps/stage2_preparation.py:198-311`).
- Stages 3 and 4 fit learned transforms on training rows while carrying the holdout partition (`src/steps/stage3_preprocessing.py:232-442`, `src/steps/stage4_feature_engineering.py:91-346`).
- Phase B Round 2 uses a fold-local preprocessing/estimator pipeline and a common evaluator (`src/steps/s06_phaseb_variant_runner.py:2370-2405`).
- S10 selects from CV/validation evidence before making one locked-test prediction (`src/steps/final_evaluation.py:1109-1168`).
- S12 rejects blocked quality decisions, registers the exact serialized bundle, uses the model version returned by MLflow, and does not promote warning models (`src/steps/s12_model_registration.py:595-698`, `src/steps/s12_model_registration.py:705-722`).
- The API submits through the canonical CLI rather than constructing the graph directly (`api/services/pipeline_service.py:338-483`).
- S13 emits drift evidence and S14 emits the policy decision; neither directly submits an Azure job (`src/steps/s13_drift_monitor.py`, `src/steps/s14_retrain_decision.py:1-6`).
- Ledger and decision request paths are now contained under a server-owned root (`api/services/auto_retrain_service.py:48-120`).

## Blocker Register

### 0. External Azure Acceptance Gate

#### AX-01 - Azure write state is unresolved

- **Severity/category:** P0 external acceptance gate; validation/release evidence.
- **Evidence:** cached `az account show` reports `Enabled`, but a live ARM subscription GET reports `Warned`. Workspace compute and environments are readable. The last known write attempt was rejected with `ReadOnlyDisabledSubscription`; this review did not repeat a resource-creating write.
- **Failure:** the canonical submitter may still be unable to create a job even though the read plane and existing resources are healthy.
- **Impact if unfixed:** no exact-source Azure canaries, mounted-data proof, MLflow hierarchy, model registration, inference smoke, or compute scale-down evidence can be completed.
- **Required correction/proof:** restore a normal ARM subscription state, then submit one bounded canary through the canonical guarded path before running the three-task acceptance set.

### A. Model-Selection Integrity

#### MS-01 - Round 1 preprocessing leaks proxy-validation rows

- **Severity/category:** P1; model-selection integrity.
- **Evidence:** S06 fits `FittedVariantPreprocessor` on the full training partition at `src/steps/s06_phaseb_variant_runner.py:3038-3048`, then performs the proxy train/test split at `src/steps/s06_phaseb_variant_runner.py:1182-1233`.
- **Failure:** imputation, encoding, scaling, and target-dependent feature selection can see rows later treated as proxy validation.
- **Impact if unfixed:** proxy scores are optimistic or unstable; good variants can be pruned and leaky variants can advance before full training. The product can miss the best configuration even though the locked final holdout remains isolated.
- **Required correction:** make Round 1 preprocessing part of a split-local pipeline, or perform deterministic folds with every learned transform fitted only on each proxy-training fold.
- **Required test:** alter only proxy-validation labels/outliers and prove fitted transforms and selected features do not change.

#### MS-02 - Declared group/time split policies are not implemented end to end

- **Severity/category:** P1; model-selection integrity and configuration fidelity.
- **Evidence:** schema accepts `group` and `time` at `src/orchestration/config_schema.py:317-351`. `ensure_holdout_partition` supports random/stratified/time but rejects group at `src/utils/holdout_partition.py:66-73`. `EvaluationSpec` has no group/time contract and always uses `StratifiedKFold` or shuffled `KFold` at `src/utils/common_evaluator.py:38-45` and `src/utils/common_evaluator.py:178-212`.
- **Failure:** a group configuration fails in Stage 2; a time configuration gets a chronological holdout but random CV; repeated entities or future observations can cross selection folds.
- **Impact if unfixed:** optimistic selection scores, temporal leakage, entity leakage, and runtime failure for a schema-valid configuration.
- **Required correction:** implement group-aware holdout/CV and forward-only temporal CV, pass the policy into every evaluator and HPO path, or reject unsupported strategies during compilation.

#### MS-03 - Phase A cannot become champion

- **Severity/category:** P1; model-selection integrity and requirement mismatch.
- **Evidence:** PyCaret and FLAML mark current baseline artifacts `raw_input_bundle_eligible=False` at `src/steps/stage5_pycaret_train.py:274-295` and `src/steps/stage5_flaml_train.py:198-215`. Aggregation excludes them at `src/steps/aggregate_baseline.py:110-168`.
- **Failure:** S05z emits no selectable Phase A champion even when the baseline is objectively best.
- **Impact if unfixed:** FR-04/FR-07 A/B/C comparison is false; if Phase B/C fail, a valid baseline cannot be registered; the selected pipeline may be worse than an excluded baseline.
- **Required correction:** train Phase A from raw training input with a persisted fold-local preprocessing graph, or explicitly redefine Phase A as diagnostics-only and remove it from champion claims.

#### MS-04 - One surviving candidate can be declared the best configuration

- **Severity/category:** P1; model-selection integrity and quality policy.
- **Evidence:** Phase B fails only when zero results survive (`src/steps/s06_phaseb_variant_runner.py:610-617`); S10 accepts one selectable candidate (`src/steps/final_evaluation.py:566-624`). `tests/test_locked_test_selection.py:130` explicitly permits a sole Phase B survivor.
- **Failure:** widespread timeout/failure can leave one weak candidate that is declared champion without meaningful comparison.
- **Impact if unfixed:** a completion accident becomes product evidence for “best configuration,” weakening quality and cost claims.
- **Required correction:** add a configured minimum comparable candidate count/coverage rule by task and engine, and emit `block` when it is not met.

#### MS-05 - Locked holdout and selected bundle identities are not closed

- **Severity/category:** P1; identity/audit and registration safety.
- **Evidence:** Stage 2 creates `test_ids_hash` in `src/steps/stage2_preparation.py:507-529`, but S10 has no split-manifest input (`components/final_evaluation.yml:7-35`) and checks only holdout row-ID presence/uniqueness (`src/steps/final_evaluation.py:1031-1044`). S10 loads selection evidence separately from the bundle, then rebuilds a bundle using the manifest candidate ID without checking the source bundle ID/hash (`src/steps/final_evaluation.py:390-497`, `src/steps/final_evaluation.py:1511-1557`).
- **Failure:** a stale/substituted holdout with unique IDs is accepted; bundle B can be evaluated and relabeled as candidate A.
- **Impact if unfixed:** the “locked test once” and exact-winner registration claims are unauditable; S12 validates the relabeled output rather than detecting the original mismatch.
- **Required correction:** wire `SplitManifest` and `ExecutionManifest` into S10, compare exact ordered holdout IDs/hash, and reject candidate ID/bundle ID/recipe hash mismatches before prediction.

#### MS-06 - Recipe behavior can differ from recipe identity

- **Severity/category:** P1; model-selection integrity and reproducibility.
- **Evidence:** `FittedVariantPreprocessor` has no explicit task type (`src/utils/fitted_variant_preprocessor.py:44-49`), chooses classification vs regression mutual information from target cardinality `<=30` (`:242-267`), converts explicit threshold `0.0` to `0.01` (`:211-216`), and silently keeps all features if no feature survives (`:278-285`).
- **Failure:** low-cardinality regression can use classification MI, high-cardinality classification can use regression MI, and the executed threshold/fallback differs from the hashed recipe.
- **Impact if unfixed:** candidate identity no longer describes actual computation; repeatability and fair comparison fail silently.
- **Required correction:** require explicit task semantics in the preprocessor, preserve zero values, and make empty-selection behavior explicit and identity-bearing.

#### MS-07 - The production search catalogs are not release-ready

- **Severity/category:** P1; search-space completeness.
- **Evidence:** deterministic compile found: classification 260 checked / 43 quarantined / 162 unique; regression 127 / 101 quarantined / 23 unique; clustering 67 / 51 quarantined / 11 unique. Quarantine is allowed at `src/utils/recipe_catalog.py:215-307`, and submission proceeds at `pipelines/submit_pipeline.py:863-895`.
- **Failure:** unsupported encodings, outlier methods, feature selectors, and incomplete metadata remove most regression and clustering recipes.
- **Impact if unfixed:** the product searches a narrow, task-skewed subset while claiming broad pipeline optimization. It may never consider the best configuration.
- **Required correction:** define the supported canonical transform matrix, repair/regenerate or retire incompatible production recipes, and make zero quarantined production recipes a release gate.

#### MS-08 - Feasibility runs after the Round 1 catalog cap

- **Severity/category:** P1; search funnel reliability.
- **Evidence:** S06 selects the bounded diverse shortlist at `src/steps/s06_phaseb_variant_runner.py:3006-3014`, then runs validation and Round 0 feasibility at `:3015-3066`.
- **Failure:** infeasible high-ranked recipes can occupy all bounded slots while feasible lower-ranked recipes are never checked.
- **Impact if unfixed:** the tournament can be empty or artificially weak even when the catalog contains viable candidates.
- **Required correction:** perform cheap schema/semantic/feasibility checks before the bounded data-aware shortlist, retaining rejection evidence.

#### MS-09 - Timeout evidence is not marked as censored

- **Severity/category:** P2; selection evidence and budget truth.
- **Evidence:** `CandidateEvidence.censored` defaults to false (`src/utils/common_evaluator.py:56-76`), and the hard-timeout return does not set it true (`src/utils/common_evaluator.py:503-537`). Process startup and termination grace also sit outside the configured join timeout.
- **Failure:** a timed-out trial is recorded as ordinary incomplete evidence and actual elapsed time can exceed the stated candidate budget.
- **Impact if unfixed:** tournament diagnostics and cost reports cannot distinguish censored trials from normal failures or enforce precise wall-clock claims.
- **Required correction:** set explicit censored/timeout status and record total wall time including worker startup/termination with a bounded tolerance.

### B. Configuration, Engine, and Immutable Execution

#### CE-01 - The second-engine requirement is unresolved

- **Severity/category:** P1 product decision; requirements fidelity.
- **Evidence:** contracts allow only PyCaret and FLAML (`src/orchestration/contracts.py:18-20`), while the stated product requirement names PyCaret and Azure ML. The proposal already records this unresolved decision at `docs/MLOPS_PRODUCT_IMPROVEMENT_PROPOSAL_2026-07-27.md:339-346`.
- **Failure:** if “Azure ML” means Azure AutoML, the required engine is absent. If it means the hosting platform, the requirement and UI terminology are wrong.
- **Impact if unfixed:** acceptance criteria, architecture, engine parity tests, and clustering policy remain ambiguous.
- **Required decision:** choose either “PyCaret + FLAML on Azure ML” or “PyCaret + Azure AutoML,” then update contracts, adapters, catalog capability matrix, UI, and tests consistently.

#### CE-02 - Accepted configuration fields are forced or ignored

- **Severity/category:** P1; configuration fidelity and cost.
- **Evidence:** compiler forces profiling and `profile_scored` selection (`src/orchestration/config_compiler.py:529-553`), forces planner enabled (`:580-585`), derives Phase A engines from Phase B (`:474-478`, `:782-787`), and compiles `validation_fraction` although Stage 2 emits `validation_count=0` (`src/steps/stage2_preparation.py:507-518`). Submit-side catalog selection ignores compiled `max_variants` (`pipelines/submit_pipeline.py:871-879`). Regression/clustering replace the configured proxy threshold with `-0.5`/`0.0` (`src/steps/s06_phaseb_variant_runner.py:1293-1346`). `planner_enabled` is not bound to its component input value (`components/s06_phaseb_variant_runner.yml:47-50`, `:112`).
- **Failure:** valid explicit configuration compiles successfully into different behavior.
- **Impact if unfixed:** operator intent, budget, candidate count, and evidence are unreliable; the UI can preview a configuration the runtime does not execute.
- **Required correction:** remove obsolete fields, honor them exactly, or reject incompatible values. The compiled artifact must be the only runtime policy source.

#### CE-03 - Data identity is mutable and can be `unversioned`

- **Severity/category:** P1; identity/reproducibility.
- **Evidence:** missing versions become `unversioned` at `src/orchestration/config_compiler.py:674-687`; candidate identity uses only `name@version:blob_path` at `pipelines/submit_pipeline.py:485-501`; the mounted input is the datastore root rather than a resolved immutable data asset (`pipelines/submit_pipeline.py:847-861`).
- **Failure:** replacing a blob at the same path can produce different training data under the same candidate/execution identity.
- **Impact if unfixed:** winners cannot be reproduced, audited, or safely resubmitted.
- **Required correction:** bind a versioned Azure ML data asset or immutable content digest plus exact resolved path and reject `unversioned` production submissions.

#### CE-04 - Environment identity can disagree with the graph

- **Severity/category:** P1; identity/reproducibility.
- **Evidence:** `--env_version` is accepted and hashed (`pipelines/submit_pipeline.py:738-739`, `:826-831`, `:468-501`), but components are fixed to `mlops-v3-unified:23`, and registration uses `mlops-v3-registration:2` (`components/s06_phaseb_variant_runner.yml:117`, `components/s12_model_registration.yml:36`).
- **Failure:** passing another environment label changes the manifest without changing what Azure executes; one environment hash cannot describe all nodes.
- **Impact if unfixed:** dependency lineage and registered-model provenance can be false even when the job succeeds.
- **Required correction:** derive manifest identity from every resolved node environment/version/digest, or render component environments from one validated immutable mapping.

#### CE-05 - Downstream stages do not consume the frozen execution manifest

- **Severity/category:** P1; identity and registration safety.
- **Evidence:** the manifest is wired to S06, but not independently into S08, S10, or S12 (`pipelines/pipeline_builder.py:157-184`, `:194-209`; equivalent V2 graph at `:357-421`).
- **Failure:** downstream stages infer identity from mutable artifacts/environment variables and cannot independently reject cross-execution substitution.
- **Impact if unfixed:** HPO, final evaluation, and registration lineage is not cryptographically closed.
- **Required correction:** make execution, split, candidate, and bundle identity required typed inputs through S08/S10/S12 and validate each boundary.

#### CE-06 - Resubmit and controller replay use current mutable configuration

- **Severity/category:** P1; reproducibility and drift-controller safety.
- **Evidence:** API resubmit extracts/infer a config name and calls current submission (`api/services/pipeline_service.py:1335-1352`). The controller validates config/task/dataset but not config hash/source SHA (`src/orchestration/auto_retrain_controller.py:238-251`) and submits the current config path (`:324-343`).
- **Failure:** editing YAML, recipes, data, source, or environments changes what a “resubmit” or approved S14 decision runs.
- **Impact if unfixed:** replay and policy approval no longer refer to the evaluated revision; audit and rollback claims fail.
- **Required correction:** archive and replay the compiled execution revision by ID, or explicitly create a new revision and require a new policy decision.

#### CE-07 - Legacy direct-submit scripts are broken and bypass ownership

- **Severity/category:** P1; execution reliability and governance.
- **Evidence:** `scripts/batch_submit_inline.py:107-127` and `scripts/resubmit_6_failed.py:88-104` call `full_pipeline` without required manifest/catalog inputs and call `jobs.create_or_update` directly. Required inputs are at `pipelines/pipeline_builder.py:83-94`.
- **Failure:** scripts fail graph construction today; a superficial repair would bypass compiler, source identity, duplicate guards, and force audit.
- **Impact if unfixed:** operators have broken workflows and a path to non-canonical, unaudited Azure submissions.
- **Required correction:** delete/archive them or make thin wrappers around `pipelines/submit_pipeline.py`.

#### CE-08 - Force-submit audit fails open

- **Severity/category:** P1; submission governance and duplicate cost.
- **Evidence:** `--force` bypasses both the local lock and active-job guard (`pipelines/submit_pipeline.py:1362-1415`). `_record_force_audit` catches an audit write failure, logs a warning, and allows submission to continue (`pipelines/submit_pipeline.py:329-344`).
- **Failure:** the strongest safety bypass can run with no durable audit record when the state directory is unavailable or read-only.
- **Impact if unfixed:** duplicate high-cost jobs can be intentionally or accidentally submitted without accountable evidence.
- **Required correction:** require a reason and fail closed unless a durable audit reservation is committed before bypassing guards.

### C. Drift, Controller, API, and UI

#### DC-01 - Both UIs send an invalid auto-retrain plan request

- **Severity/category:** P1; execution reliability and operator workflow.
- **Evidence:** API requires `decision_path` at `api/schemas/pipeline.py:292-308`. React omits it at `react-ui/src/pages/AutoRetrain.tsx:73-80`; Streamlit omits it at `ui/pages/4_Auto_Retrain.py:186-195`.
- **Failure:** every UI Build Plan request receives HTTP 422.
- **Impact if unfixed:** the guarded external-controller workflow is unusable from either supported UI.
- **Required correction:** expose selectable/downloaded S14 decision artifacts, include the bounded relative path, and generate clients/types from OpenAPI.

#### DC-02 - Drift UI/API can disagree with S14 policy

- **Severity/category:** P1; operator truth.
- **Evidence:** S14 evaluates `comparison_drift.feature_psi_scores` (`src/orchestration/auto_retrain_policy.py:78-102`), while the API feature table reads top-level S13 self-check PSI (`api/services/pipeline_service.py:1202-1247`). It also looks for obsolete embedded `auto_retrain_decision` instead of downloading the separate S14 output (`:1211-1214`, `:1302-1319`).
- **Failure:** S14 can authorize `candidate_retrain` while the UI shows no feature drift and an empty decision.
- **Impact if unfixed:** operators can approve, reject, or ignore retraining based on the wrong evidence.
- **Required correction:** download and join S13 and S14 outputs by decision/execution identity and expose comparison PSI separately from self-check PSI.

#### DC-03 - Drift policy configuration is not used by S14

- **Severity/category:** P1; configuration fidelity and controller safety.
- **Evidence:** `configs/drift_config.yaml:13-17` declares feature drift `0.15`; S14 policy defaults to `0.10` at `src/orchestration/auto_retrain_policy.py:23-35`, and no production call passes a loaded policy.
- **Failure:** PSI 0.12 can trigger retraining despite a configured threshold of 0.15.
- **Impact if unfixed:** policy decisions and Azure spend differ from operator configuration.
- **Required correction:** define one validated policy schema consumed by S13, S14, API, and UI; remove stale trigger settings from the old config.

#### DC-04 - Controller ledger is local, unlocked, and race-prone

- **Severity/category:** P1; duplicate cost and audit durability.
- **Evidence:** append is an unlocked local JSONL write (`src/orchestration/auto_retrain_decision_ledger.py:94-105`). Duplicate checking happens before submission (`src/orchestration/auto_retrain_controller.py:112-151`), but the record is appended after the subprocess returns (`scripts/run_auto_retrain_controller.py:180-197`).
- **Failure:** concurrent controllers can both pass the check; a crash after Azure submission can leave no record; container restart can lose local state.
- **Impact if unfixed:** duplicate expensive jobs, corrupt JSONL, lost baseline approvals, and unauditable policy decisions.
- **Required correction:** use shared durable storage with atomic reservation/idempotency, compare-and-set status transitions, and crash reconciliation against Azure job state.

#### DC-05 - Baseline approval and comparison identity are too weak

- **Severity/category:** P1; drift correctness and cross-dataset safety.
- **Evidence:** API accepts any syntactically valid `azureml://` baseline URI for a selected config without resolving ownership/content identity (`api/services/auto_retrain_service.py:230-283`). S13 marks comparison available from metadata before proving reference data exists (`src/steps/s13_drift_monitor.py:546-585`).
- **Failure:** a cross-dataset/stale/incomplete baseline can be approved; policy can evaluate partial evidence as if comparison were available.
- **Impact if unfixed:** missed drift, false retraining, and contaminated future baseline decisions.
- **Required correction:** validate workspace, producing job, config hash, task, dataset, model, baseline schema/hash, and required reference data before approval.

#### DC-06 - Async submission state is ephemeral and React does not poll it

- **Severity/category:** P1; operator truth and duplicate cost.
- **Evidence:** async request records are process-local (`api/services/pipeline_service.py:491-544`). React defaults to async and stops after receiving the request at `react-ui/src/pages/Submit.tsx:40-49`.
- **Failure:** UI remains pending; API restarts lose tickets; users retry because they cannot see terminal submission state.
- **Impact if unfixed:** duplicate jobs, uncertain ownership, and poor incident recovery.
- **Required correction:** persist idempotent submission requests, poll to canonical job identity/terminal state, and reconcile after restart.

#### DC-07 - Planned schedules are presented as live Azure state

- **Severity/category:** P2; operator truth.
- **Evidence:** API returns a hard-coded planned catalog (`api/services/auto_retrain_service.py:164-175`); React renders `enabled_expected` as Enabled (`react-ui/src/pages/AutoRetrain.tsx:108-123`).
- **Failure:** an absent, disabled, or failed Azure schedule can appear enabled.
- **Impact if unfixed:** operators assume monitoring/retraining automation exists when it does not.
- **Required correction:** reconcile actual Azure schedule state and display source, freshness, last run, and `unverified` when unavailable.

#### DC-08 - Authentication is a private-beta boundary

- **Severity/category:** P1 for multi-user/tenant release; security boundary.
- **Evidence:** all operational users share one API key (`api/core/security.py:10-24`) and Azure process identity; approvals record only `approved_via=api` (`api/services/auto_retrain_service.py:259-283`).
- **Failure:** actor roles, tenant/workspace authorization, and accountable approval identity are absent.
- **Impact if unfixed:** the API cannot safely support multiple users or tenants and cannot provide defensible approval audit.
- **Required correction:** keep deployment labeled single-operator/private-beta until Entra ID/OIDC, actor roles, workspace authorization, and actor-bound audit are implemented.

### D. Release and Validation Evidence

#### RV-01 - The reviewed source is not a reproducible release revision

- **Severity/category:** P1; release provenance.
- **Evidence:** branch `codex_ys/mlops-pipeline-correctness` has no upstream and currently has 70 modified plus 57 untracked entries, including this review document.
- **Failure:** no commit checkout reproduces the reviewed system; CI, rollback, and exact-SHA Azure proof cannot bind to this state.
- **Impact if unfixed:** a successful run cannot be promoted as a reviewed release and fixes can be lost or mixed with unrelated changes.
- **Required correction:** inventory intentional changes, commit a review revision, establish upstream/PR/CI, and tie source package digest to that revision without discarding existing work.

#### RV-02 - Passing tests overstate coverage

- **Severity/category:** P1 validation gate.
- **Evidence:** root `test_azure_ml_setup.py` catches failures and returns booleans rather than asserting (`:34-66`) and is excluded by `pytest.ini:2`. Engine parity uses the same sklearn pipeline twice with only a different engine label (`tests/test_common_evaluator.py:33-72`). Current tests explicitly accept one selectable phase and do not exercise real Azure adapters, UI plan payloads, all catalogs, or registered raw-input inference.
- **Failure:** authentication/data-plane/adapter/product defects coexist with a green suite.
- **Impact if unfixed:** local green is mistaken for Azure readiness and regressions reach expensive canaries late.
- **Required correction:** add assertion-based opt-in Azure probes and end-to-end contract tests for the blockers in this document; keep proof levels separately reported.

#### RV-03 - Documentation contains conflicting product truth

- **Severity/category:** P2; operator and engineering truth.
- **Evidence:** `docs/PROJECT_REQUIREMENTS.md:84-90` still says S10 selects by evaluating all phases on holdout, conflicting with the newer locked-test-once design (`docs/MLOPS_PRODUCT_IMPROVEMENT_PROPOSAL_2026-07-27.md:75-81`). Other docs still describe `azureml://` to HTTPS MLflow URI rewriting and inconsistent S14 readiness.
- **Failure:** implementers can reintroduce selection leakage or tracking failures by following stale guidance.
- **Impact if unfixed:** review criteria, runbooks, and UI claims diverge from source.
- **Required correction:** designate one authoritative versioned product contract and generate stage/proof inventories from source.

#### RV-04 - Unsupported forecasting remains in the supported graph

- **Severity/category:** P2; cost and failure surface.
- **Evidence:** task contracts support only three tasks (`src/orchestration/contracts.py:18-20`), but both pipeline builders instantiate S05t (`pipelines/pipeline_builder.py:134-138`, `:338-342`).
- **Failure:** every supported run schedules a component whose only valid action is to skip.
- **Impact if unfixed:** unnecessary startup cost, environment pull, logs, and another point of failure; UI stage truth differs from graph truth.
- **Required correction:** remove S05t from the three-task product graph or create a separately versioned forecasting product.

## Azure and Validation Evidence Collected

### Local evidence

- `pytest tests -q -p no:cacheprovider`: **358 passed, 6 warnings** in 107.17 seconds.
- Canonical `submit_pipeline.py --dry_run` with SDK validation: **classification pass, regression pass, clustering pass**.
- All dry-runs targeted `mlopsv2computecluster`; no training executed.
- Catalog compile counts are recorded in MS-07.

### Live Azure read-plane evidence

- Cached `az account show`: `Enabled`.
- Live ARM subscription GET: `Warned`.
- `mlopsv2computecluster`: `Succeeded`, `Standard_D4s_v3`, min 0, max 8.
- Compute instance `mlopspipelinev2`: `Succeeded`, `Standard_E4ds_v4`.
- Environment `mlops-v3-unified:23`: `Succeeded` (latest workspace version is 27).
- Registration environment `mlops-v3-registration:2`: `Succeeded`.
- SDK job listing timed out after 64 seconds, so no current job history was accepted as evidence.

These read checks prove resource visibility, not write permission or pipeline correctness. No Azure write was attempted in this review.

## Required Remediation Order

1. **Resolve the product contract:** decide Azure AutoML versus FLAML, choose supported split policies, and define minimum comparable candidates.
2. **Clear the Azure acceptance gate:** restore subscription write state and prove a bounded job can be created without bypassing guards.
3. **Fix selection correctness:** MS-01 through MS-06, especially Round 1 leakage, split policy, Phase A eligibility, minimum candidates, and identity closure.
4. **Make the search space truthful:** repair catalogs, move feasibility before caps, and make every accepted control enforceable.
5. **Close immutable execution:** bind data, all environments, manifests, resubmit, controller replay, S10, and S12.
6. **Repair controller/UI operations:** valid decision payloads, joined S13/S14 truth, durable idempotent ledger, baseline validation, async polling, and live schedule state.
7. **Create a reviewable release revision:** preserve current work, commit intentional changes, run CI, and tag the exact source package digest.
8. **Run Azure acceptance on the cluster:** bounded classification, regression, and clustering canaries; do not run full training locally.

## Exit Criteria

The pipeline should not be called ready until all of the following are true:

- Round 1 and Round 2 learned transforms are fold/split-local.
- Group/time policies are either implemented end to end or rejected before submission.
- Phase A, B, and C use the same raw-input bundle and comparison contract, or Phase A is explicitly removed from champion claims.
- A configured minimum comparable candidate set is enforced.
- Holdout, candidate, bundle, execution, data, code, and environment identities are validated through registration.
- Production recipe catalogs have zero quarantined entries and deterministic deduplication.
- All accepted configuration values are honored or rejected; none are silently rewritten.
- Auto-retrain UI requests succeed, S13/S14 state is joined truthfully, and the ledger is durable/idempotent.
- The reviewed revision is committed, has CI evidence, and can be reproduced from source control.
- Live ARM/write-plane access is healthy.
- Classification, regression, and clustering exact-revision Azure jobs complete with job IDs and output artifacts.
- MLflow parent/child lineage, exact registered version, warning/no-promotion behavior, and raw-input registered-model prediction are proven in the workspace.
- Compute returns to min nodes after each bounded canary.

## Final Assessment

The current checkout contains substantial correctness improvements, and its canonical graph is locally buildable. The remaining blockers are not cosmetic: they affect which configuration wins, whether the winner identity is truthful, whether configured policy is executed, and whether operators can safely drive or understand retraining. The next implementation should start with the selection and identity blockers, not UI polish or broad refactoring.
