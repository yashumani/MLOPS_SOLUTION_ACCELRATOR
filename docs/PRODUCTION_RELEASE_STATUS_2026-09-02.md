# Production Release Status

Date: 2026-09-02  
Branch: `codex_ys/mlops-pipeline-correctness`  
Starting baseline: `eb739444f9d439b75232e0342f5d5c52ab86986c`  
Azure workspace: `mvpv1/mlops-accelerator`  
Azure compute: `mlopsv2computecluster`

## Release Decision

Status: **not production-release ready**.

The subscription, workspace, compute cluster, qualification datastore, licensed
test data, exact 15-scenario matrix, and unified Azure ML runtime are available.
The accepted runtime is now adopted by every active pipeline component that
uses the unified environment.

The remaining release gates are a pushed exact source SHA with green backend
and React CI, owner-approved repair of the two workspace artifact datastores, a
production API topology decision, accepted Azure diagnostics and 15-scenario
runs, post-training operational evidence, and final production approval.

Current authorization covers development commits and non-production Azure ML
qualification. It does not authorize a production endpoint, traffic change,
model alias or stage promotion, shared datastore credential mutation, or other
production deployment.

## Requirement Alignment

| Requirement | Implementation | Current evidence |
| --- | --- | --- |
| Task types | Classification, regression, and clustering | Exactly five industries per task; Azure diagnostics pending |
| Training engines | PyCaret and FLAML for supervised tasks; PyCaret for clustering | Contract tests pass; Azure comparison proof pending |
| Variant funnel | Stages 1-4 prepare data and narrow candidates before baseline, Phase B, HPO, and champion selection | Source and catalog tests pass; full Azure proof pending |
| Holdout isolation | Learned transforms fit training rows and apply to locked holdout rows | Slice 2 tests pass; Azure proof pending |
| Canonical submission | API and scripts route through governed recipe resolution, active-job guard, lock, force audit, and immutable candidate wiring | Slice 3 tests pass; exact-SHA CI pending |
| Exact model version | Registration binds the version returned by `log_model` | Focused tests pass; concurrent Azure proof pending |
| Drift and retraining | Stage 13 emits evidence; Stage 14 decides; external controller owns submission | Slice 4 tests pass; scheduled Azure proof pending |
| MLflow lineage | Dataset, execution, environment, stage, model, and registration identities are represented | Environment tracking passed; end-to-end lineage pending |
| UI and API | Configuration, submission, status, drift, retraining, and notifications are represented | React and API source checks pass; deployment acceptance pending |

Worktree and dirty-source Azure results are development evidence. Final release
evidence must reference the same pushed commit SHA in CI and Azure ML.

## Completed Evidence

### Azure and Data

| Gate | Evidence | State |
| --- | --- | --- |
| Azure access | Subscription enabled; workspace and compute reachable | Passed |
| Matrix cardinality | Five classification, five regression, five clustering scenarios | Passed |
| Data suitability | Profile job `blue_stone_mfcj9dx8p5` | 15 schema passes, zero failures |
| Data governance | Source, license, source ID, checksums, exclusions, and privacy disposition | Approved for non-production qualification |
| Immutable data | Versioned source paths plus 11 Azure ML data assets at `20260902.1` | Passed |
| Execution configs | 15 schema-v2 configs and one execution catalog | Passed; all target `mlops-v3-unified:32` |

### Unified Runtime

| Evidence | Value |
| --- | --- |
| Accepted environment | `mlops-v3-unified:32` |
| Smoke job | `cool_pumpkin_wknxj9y9l2` - `Completed` |
| Built image digest | `sha256:5adebdd6ab3ba64ad1a9b8828f18a0b5af4b4d0cad300cee18ebd1bc51c4aa33` |
| Runtime checks | Clean `pip check`; 16 required imports; Evidently pass; Azure MLflow pass |
| MLflow alignment | `mlflow==2.14.3`, `mlflow-skinny==2.14.3` |
| Lock job | `purple_sun_qmnsfrxrnr` - `Completed` |
| Lock inventory | 218 pip packages; 244 conda packages |
| Freeze SHA-256 | `3f58c5ea0bc83052a5d977f6ad037c37e6f481b3c6427a2de6699c18d7f6e69e` |

The downloaded freeze hash matches the Azure evidence. Its Conda-generated
`pip @ file:///home/conda/...` entry is nonportable, so the raw freeze remains
evidence while `conda_v32.yml` is the installable pinned definition. The
`pkg_resources` warning comes from AzureML dataprep; Setuptools 80.9.0 is a
documented compatibility pin and a future AzureML dependency-upgrade item.

Active component identity is exact: 13 components use
`azureml:mlops-v3-unified:32`; model registration remains isolated on
`azureml:mlops-v3-registration:2`.

### Source Validation

| Slice | Local commit | Validation |
| --- | --- | --- |
| Product and execution contracts | `7d30979e` | 48 tests passed |
| Holdout-safe evaluation and model identity | `52a88df3` | 169 tests passed |
| Canonical pipeline submission | `3ebee28b` | 58 tests passed |
| Drift, retraining, API, and UI | `e79416b0` | 126 backend tests; React typecheck, 5 tests, and build passed |
| Recipe catalog and 15-scenario data matrix | `95b89185` | 34 tests passed |
| Environments, CI, runbooks, and evidence | This release slice | Final local validation pending commit |

`pytest --collect-only -q` currently discovers 466 tests. Heavy model training
has not run locally. Local work is limited to contracts, unit tests, schema
compilation, React checks, and source-control verification.

## Qualification Matrix

| Task | Industry | Scenario | Target or excluded outcome |
| --- | --- | --- | --- |
| Classification | Healthcare | Heart disease prediction | `heart_disease` |
| Classification | Financial services | Credit default prediction | `y` |
| Classification | Telecommunications | Customer churn prediction | `Churn` |
| Classification | Manufacturing | Machine failure prediction | `Machine failure` |
| Classification | Insurance | Policy interest prediction | `CARAVAN` |
| Regression | Education | Final grade prediction | `G3`; exclude `G1`, `G2` |
| Regression | Real estate | Unit-area sale price | `house_price_unit_area` |
| Regression | Insurance | Claim-cost prediction | `UltimateIncurredClaimCost` |
| Regression | Energy | Building heating load | `Y1`; exclude `Y2` |
| Regression | Aviation | Airfoil noise prediction | `scaled_sound_pressure` |
| Clustering | Healthcare | Heart cohorts | Exclude `heart_disease` |
| Clustering | Financial services | Payment-behavior segments | Exclude `y` and source ID |
| Clustering | Telecommunications | Service-behavior segments | Exclude `Churn` |
| Clustering | Retail | Transaction segments | Exclude transaction and customer identifiers |
| Clustering | Education | Student-support segments | Exclude `G1`, `G2`, `G3` |

All scenarios are approved only for non-production qualification. Configured
exclusions must remain enforced, and only aggregate metrics and lineage may
leave the Azure ML workspace.

## Active Blockers By Impact

### 1. Release Source and CI Traceability

Category: change control and reproducibility.  
Release impact: critical and active.

The remediation is organized into reviewable local feature-branch commits and
the repository now contains backend and React CI. The branch still has no
upstream, and GitHub `git ls-remote` did not return during this session.

Impact if unresolved: Azure results cannot be tied to independently reviewable
source, and later changes can invalidate an apparently successful run.

Exit evidence: push this feature branch, obtain green CI for its exact SHA, and
run Azure jobs whose source metadata carries that same clean SHA.

### 2. Workspace Artifact Datastore Credentials

Category: platform reliability and evidence integrity.  
Release impact: critical, active, and requires explicit owner approval.

`workspaceblobstore` and `workspaceartifactstore` contain stale stored account
keys for `mlopsaccelerat7263606092`. `mlops_blob` works and is the explicit
pipeline output datastore, but that does not repair all MLflow and job-artifact
transport paths.

Impact if unresolved: a component can compute successfully while model files,
logs, MLflow artifacts, or release evidence remain inaccessible. Such a run
cannot be accepted or reliably recovered.

Owner action: approve the bounded shared non-production datastore refresh, then
execute `docs/OPERATIONAL_RUNBOOKS/workspace-datastore-credential-recovery.md`.
Acceptance requires the write, upload, and download canary to pass.

### 3. Production API Topology, Authentication, and State

Category: production security and availability.  
Release impact: conditional critical blocker; owner decision required.

The current API uses a shared `X-API-Key`. Submission request state and the
auto-retrain ledger are filesystem-backed with process and file locks.

Impact if deployed publicly or with multiple replicas: users cannot be
individually authorized or audited, replicas can disagree about submissions,
and restart or storage loss can orphan decisions.

Exit decision: document an internal single-instance exception, or implement
Microsoft Entra/OIDC authorization and transactional shared state for the
selected production topology.

### 4. End-to-End Model and Operations Evidence

Category: product acceptance.  
Release impact: critical and active; dependent on blockers 1-3.

No exact-SHA post-remediation classification, regression, or clustering
diagnostic is accepted yet. None of the 15 final qualification pipelines has
been accepted against the release candidate.

Impact if unresolved: variant elimination, engine comparison, HPO identity,
locked holdout behavior, MLflow lineage, exact model registration, raw-input
inference, drift policy, retraining control, and rollback remain unproved on the
actual platform.

Exit evidence: three sequential task diagnostics, all 15 immutable scenarios,
focused reruns only for failed immutable revisions, and indexed operational
artifacts for every accepted run.

### 5. Production Deployment and Promotion Approval

Category: governance.  
Release impact: intentional final gate.

Production endpoint deployment, traffic movement, and model promotion remain
stopped until explicit final human approval after every technical gate passes.

## Blockers No Longer Active

- Azure subscription, workspace, and compute access are working.
- The exact five-classification, five-regression, five-clustering scope is set.
- All 15 datasets have suitable schemas, provenance, licenses, content hashes,
  schema hashes, immutable paths, and non-production privacy dispositions.
- Unified environment 32 passed image build, runtime smoke, MLflow, Evidently,
  package consistency, and exact lock capture on the Azure compute cluster.
- All active unified-runtime components and 15 qualification configs use v32.
- The recipe catalog has zero quarantined production recipes after 188
  intentional incompatible or superseded recipe retirements.
- Holdout leakage, canonical-submission bypass, ledger path escape, drift-policy
  bypass, and latest-model-version race have code and focused-test fixes.

The five original code findings remain release-open until the exact-SHA Azure
acceptance runs pass.

## Critical Path

1. Commit the environment, CI, runbook, and evidence slice; restore GitHub
   connectivity; push the feature branch; and obtain green exact-SHA CI.
2. Obtain explicit approval for the shared datastore refresh and pass its
   bounded write/upload/download canary.
3. Resolve the production API topology decision.
4. Run one sequential diagnostic for classification, regression, and
   clustering on `mlopsv2computecluster`.
5. Run all 15 industry scenarios and rerun only failed immutable revisions.
6. Verify MLflow lineage, exact model versions, raw-input inference,
   monitoring, retraining, rollback, and operator runbooks.
7. Perform the final exact-SHA security and release review, then request
   production deployment and model-promotion approval.

## Earliest Credible ETA

After GitHub connectivity, datastore-refresh approval, and the API topology
decision are available, the earliest credible path is 4-7 business days:

- Push, CI, datastore canary, and three task diagnostics: 1-2 days.
- Fifteen Azure qualification runs plus focused reruns: 2-3 days.
- Operational evidence, security review, and release decision: 1-2 days.

A shared pipeline defect discovered by the three diagnostics, or a decision to
implement multi-replica identity and durable state before release, extends that
window. Progress is reported by passed and blocked gates, not percentages.
