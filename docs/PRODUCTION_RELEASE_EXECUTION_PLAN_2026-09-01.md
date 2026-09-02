# MLOps Solution Accelerator Production Release Execution Plan

Date: 2026-09-01

## Goal

Bring MLOps Solution Accelerator V3 to verified production-release readiness for
classification, regression, and clustering. Production readiness requires a
traceable source revision, green CI, Azure ML pipeline evidence, production
security and state controls, and a 15-scenario industry matrix consisting of
five different industries for each supported task type.

The plan stops at the production deployment and model-promotion approval gate.
It does not authorize endpoint deployment, traffic changes, or promotion of a
registered model to production.

## Product Contract

- Supported tasks: classification, regression, and clustering.
- Training engines: PyCaret and FLAML on Azure ML.
- Clustering uses PyCaret because FLAML does not implement the clustering
  contract.
- Azure ML is the execution, tracking, registry, and governance platform. Azure
  AutoML is not a V3 training engine unless a separately tested adapter is added.
- The product selects the best validated end-to-end configuration, not merely
  the highest-scoring estimator.

## Execution Boundary

- Local: source editing, Git, static/configuration checks, Azure submission, and
  monitoring only.
- Azure compute instance or CI: dependency, API/UI, integration, and broad test
  execution.
- Azure compute cluster: dataset profiling, preprocessing, variant search,
  training, HPO, final evaluation, drift components, and model smoke tests.
- Existing workspace data is preferred. External datasets are used only when a
  matrix slot has no suitable dataset with defensible provenance and licensing.

## Industry Acceptance Matrix

The following matrix is provisional until schema, target, quality, license,
privacy, and checksum checks complete on Azure. Each accepted scenario receives
an immutable config and dataset identity.

### Classification

| Industry | Candidate dataset | Intended outcome |
| --- | --- | --- |
| Healthcare | Cardiac arrest or Diabetes 130-US | Clinical outcome classification |
| Financial services | Credit default or credit-card fraud | Default or fraud classification |
| Telecommunications | Telco churn | Customer churn classification |
| Transportation | Titanic passenger safety or airline delay | Safety or delay classification |
| Insurance | Workers Compensation | Claim outcome classification |

### Regression

| Industry | Candidate dataset | Intended outcome |
| --- | --- | --- |
| Education | College | Graduation-rate regression |
| Real estate | House sales | Sale-price regression |
| Insurance | Insurance charges | Claim or charge regression |
| Healthcare | Length of stay | Stay-duration regression |
| Aviation | Airlines delay | Delay-duration regression |

### Clustering

| Industry | Candidate dataset | Intended outcome |
| --- | --- | --- |
| Healthcare | Kidney disease or Diabetes 130-US | Patient cohort discovery |
| Financial services | Credit default | Customer risk segmentation |
| Telecommunications | Churn uplift | Customer behavior segmentation |
| Retail | Online retail | Customer or transaction segmentation |
| Education | College | Institution segmentation |

Workers Compensation, house sales, and any dataset whose source metadata is
incomplete remain candidates, not approved test data. If one is unsuitable, use
a reputable public dataset from an official government, UCI, or clearly licensed
OpenML source. Record source URL, license, retrieval date, original checksum,
uploaded blob path, and Azure ML data asset version.

## Release Phases

### Phase 0: Baseline And Scope Freeze

1. Preserve the current feature branch and inventory every modified, deleted,
   and untracked path.
2. Confirm the source-of-truth requirements and engine contract.
3. Define the release rubric and classify every open issue as release-blocking
   or documented non-blocking risk.
4. Record the Azure subscription, workspace, compute, identity, and datastore
   access paths.

Exit gate: one agreed release contract and a complete release-candidate file
inventory.

### Phase 1: Data Qualification

1. Inventory workspace datastores, registered data assets, and blob candidates.
2. Profile candidate schemas on Azure; do not download full datasets locally.
3. Verify task suitability, target semantics, row/feature counts, missingness,
   class balance or target distribution, sensitive fields, and license.
4. Assign exactly five distinct industries to each task type.
5. Generate SHA-256 fingerprints and immutable Azure data asset versions.
6. Acquire and upload external public data only for uncovered matrix slots.

Exit gate: 15 approved data records and 15 immutable execution configs.

### Phase 2: Release Candidate And CI

1. Review the large recipe retirement set and prove every deletion is intended.
2. Split the current work into traceable commits without disturbing unrelated
   user changes.
3. Pin production dependencies and Azure ML environments.
4. Establish CI for unit, contract, security, config-catalog, API, and React
   checks.
5. Push the feature branch and tie all evidence to the exact source SHA.

Exit gate: a reproducible release candidate with green CI.

### Phase 3: Production Blocker Remediation

1. Bound or repair Azure active-job enumeration without bypassing duplicate-job
   protection.
2. Populate production dataset fingerprints and enforce immutable execution
   revisions.
3. Complete production authentication and authorization; shared API keys are
   insufficient for externally exposed or multi-user operation.
4. Replace process-local mutable request/retrain state with durable,
   transactional state where deployment topology requires it.
5. Recheck evaluator parity, locked-test isolation, candidate identity, quality
   policy, exact model registration, drift-policy ownership, API semantics, and
   UI truthfulness.

Exit gate: no open correctness, security, identity, or state blocker.

### Phase 4: Three-Task Diagnostic Gate

1. Submit one bounded classification canary through
   `pipelines/submit_pipeline.py`.
2. Submit regression only after classification reaches a terminal accepted
   state, then submit clustering.
3. Capture parent and child Azure job IDs, execution/config/data/code/environment
   identities, MLflow run IDs, failed-step evidence, and cluster shutdown.
4. Correct shared failures before expanding to the industry matrix.

Exit gate: one accepted end-to-end Azure run for every task type.

### Phase 5: Fifteen-Scenario Matrix

1. Run all 15 bounded end-to-end scenarios on the compute cluster.
2. Keep configured time, variant, HPO, and concurrency limits immutable.
3. Require preprocessing, candidate search, training, final locked-test audit,
   MLflow lineage, exact registration, and raw-input inference evidence for each
   scenario.
4. Rerun only failed immutable revisions. Never replace failed evidence by
   silently changing a config under the same identity.

Exit gate: 15 accepted scenario records, five per task type and five distinct
industries within each task type.

### Phase 6: Final Production Review

1. Verify CI, Azure runs, MLflow lineage, model bundles, registration, drift,
   controller, API/UI, security, operations, and documentation against one SHA.
2. Reconcile README, production handoff, API docs, drift guide, configuration
   reference, and operational runbooks.
3. Produce a blocker register, residual-risk register, rollback plan, and exact
   release evidence index.
4. Present the release candidate for explicit deployment and model-promotion
   approval.

Exit gate: `prod-ready` or `prod-ready with exceptions` under the release-review
rubric. A local pass, a submitted job, or a registered model alone is not enough.

## Required Evidence Per Scenario

- industry, task type, target or clustering objective, source, and license
- Azure data asset name/version, blob URI, content hash, and schema fingerprint
- config hash, execution ID, candidate IDs, seed, budgets, and recipe IDs
- source SHA plus component and environment identities
- Azure parent/child job IDs and terminal statuses
- MLflow parent/child run IDs and metrics produced by the shared evaluator
- locked-test fingerprint and proof it was not used for selection
- exact registered model name/version and model-bundle manifest
- raw-input inference smoke-test result
- drift evidence/policy result where applicable
- failure classification, correction commit, and rerun identity when applicable

## Cost And Safety Controls

- Run one task canary at a time before the 15-scenario matrix.
- Do not increase configured budgets silently.
- Do not use `--force` without a recorded resubmission reason.
- Do not bypass the canonical submitter or active-job guard.
- Do not place full datasets or training outputs on the local machine.
- Stop cluster compute after bounded runs when supported.
- Do not deploy endpoints, change traffic, or promote models during validation.

## Working Schedule

- Baseline, data matrix, and release inventory: 1-2 business days.
- Release-candidate cleanup, CI, and blocker remediation: 3-6 business days.
- Three diagnostic canaries and focused correction loop: 2-4 business days.
- Fifteen bounded industry scenarios: 4-8 business days, depending on queue and
  failure rate.
- Final security/release evidence and handoff: 1-2 business days.

The earliest credible production-readiness window is 10-15 business days if the
three diagnostic canaries pass without a structural failure. Material auth,
state, environment, or cross-task defects can extend the window to 3-4 weeks.
Status is reported by passed and blocked gates, not as a percentage.
