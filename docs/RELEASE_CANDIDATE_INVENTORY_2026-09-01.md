# Release Candidate Inventory

Date: 2026-09-01
Branch: `codex_ys/mlops-pipeline-correctness`
Baseline HEAD: `eb739444f9d439b75232e0342f5d5c52ab86986c`

## Purpose

This inventory turns the current broad remediation worktree into reviewable
release slices. It is not release evidence by itself. Azure jobs, CI, MLflow,
registration, inference, and deployment evidence must reference the eventual
committed source SHA.

## Reconciliation Outcome

The accumulated remediation work is now separated into reviewed local commits:

| Slice | Commit |
| --- | --- |
| Product and execution contracts | `7d30979e` |
| Holdout-safe evaluation and model identity | `52a88df3` |
| Canonical pipeline submission | `3ebee28b` |
| Drift, retraining, API, and operator UIs | `e79416b0` |
| Recipe catalog and 15-scenario matrix | `95b89185` |
| Environments, CI, runbooks, and evidence | This release slice |

The branch must still be pushed and pass CI before these local commits become
independent release evidence.

## Pre-Reconciliation Snapshot

The following counts describe the 2026-09-01 worktree used to define the
release slices. They are retained as historical scope evidence, not current
working-tree status.

| State | Count | Release interpretation |
| --- | ---: | --- |
| Tracked modified | 86 | Remediation across pipeline, API, UI, drift, config, and tests |
| Tracked deleted | 188 | Intentional incompatible or superseded recipe retirement |
| Untracked files | 129 | New implementation, tests, docs, React UI, and environments |

Tracked modified files by top-level area:

| Area | Count |
| --- | ---: |
| API | 9 |
| Components | 13 |
| Configs and recipes | 16 |
| Documentation | 6 |
| Pipelines | 2 |
| Scripts | 3 |
| Source | 23 |
| Tests | 3 |
| Streamlit UI | 6 |
| Repository metadata/root files | 5 |

Untracked files by top-level area:

| Area | Count |
| --- | ---: |
| API | 3 |
| Components | 1 |
| Registration environment | 3 |
| Configs | 3 |
| Documentation and runbooks | 16 |
| React UI | 46 |
| Scripts | 3 |
| Source | 16 |
| Tests | 36 |
| Streamlit UI | 2 |

## Recipe Retirement Disposition

All 188 tracked deletions are under `configs/recipes`:

| Task | Deleted | Remaining checked | Remaining valid | Semantic unique | Duplicates removed at compile | Quarantined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification | 41 | 219 | 219 | 164 | 42 | 0 |
| Regression | 98 | 29 | 29 | 26 | 2 | 0 |
| Clustering | 49 | 18 | 18 | 13 | 4 | 0 |

Release disposition: accept the deletion set as one explicit catalog-cleanup
slice. The previous review required incompatible generated recipes to be
repaired or retired and zero production quarantine. The current catalog meets
the zero-quarantine gate, and deterministic selection removes remaining
semantic duplicates from shortlist capacity.

Evidence:

```text
classification: checked=219 valid=219 unique=164 duplicates=42 quarantined=0
regression: checked=29 valid=29 unique=26 duplicates=2 quarantined=0
clustering: checked=18 valid=18 unique=13 duplicates=4 quarantined=0
10 passed in 5.70s
```

## Submission Guard Evidence

The canonical duplicate-submission guard now uses the experiment-scoped Azure
ML Run History query with a server-side non-terminal status filter, bounded page
size, transport timeouts, continuation-token protection, and fail-closed retry
behavior. A missing experiment is treated as an empty first-run history only
when Azure identifies the requested experiment and workspace exactly.

Local focused validation:

```text
6 passed in 16.28s
```

Live read-only validation against `mvpv1/mlops-accelerator`:

```text
canary_cardiac_arrest_v3_azure: active=0 details=[]
canary_college_regression_v3_azure: active=0 details=[]
canary_college_clustering_v3_azure: active=0 details=[]
classification_telecom_churn_auto_retrain: active=0 details=[]
```

Disposition: the functional active-job enumeration blocker is resolved in the
current worktree. It remains part of the release gate until the change is
committed, pushed, and verified by CI for the exact release SHA.

## Release Slices

The work was reviewed and committed in dependency order. Files were staged by
explicit release scope rather than broad repository staging.

### Slice 1: Product And Execution Contracts

Scope:

- `docs/PROJECT_REQUIREMENTS.md`
- `src/orchestration/contracts.py`
- `src/orchestration/config_schema.py`
- `src/orchestration/config_compiler.py`
- `src/orchestration/execution_identity.py`
- schema, execution-identity, replay, and config tests

Exit evidence: strict three-task/two-engine contract, immutable compiled config,
source/data/environment identities, and replay semantics pass in CI.

### Slice 2: Data Isolation, Evaluation, And Model Identity

Scope:

- holdout partitioning and data identity
- fitted variant preprocessing
- shared evaluator and candidate comparison
- Phase A/Phase B bundles and Phase C same-family HPO
- final evaluation and exact registration
- associated component definitions and tests

Exit evidence: fold-local transforms, no selection on locked test, comparable
metrics, identity-preserving HPO, exact model version binding, and raw-input
bundle inference pass in CI and Azure.

### Slice 3: Canonical Pipeline And Submission Governance

Scope:

- `pipelines/pipeline_builder.py`
- `pipelines/submit_pipeline.py`
- canonical batch wrappers
- component input/output wiring
- duplicate-job guard, force audit, and submission tests

Exit evidence: all task graphs compile from one submitter, immutable execution
manifests are archived, and active-job checks complete within a bounded time.

### Slice 4: Drift, Retraining, API, And Operator Surfaces

Scope:

- S13 evidence, S14 decision, external controller, schedule catalog, and ledger
- FastAPI schemas/services/routers and durable submission state
- notification service
- Streamlit and React operator flows
- security, API, UI, and drift tests

Exit evidence: policy controls submission, request paths are server-owned,
retries are idempotent, auth matches deployment topology, UI state is truthful,
and mutable state is durable where more than one process can serve traffic.

### Slice 5: Recipe Catalog And Production Configs

Scope:

- 188 recipe retirements
- retained recipe repairs
- deterministic catalog compiler and selector
- 15 industry configs and production dataset fingerprints
- catalog and config tests

Exit evidence: zero quarantined production recipes, deterministic bounded
selection, five distinct industries per task type, and immutable data hashes.

### Slice 6: Environments, CI, Documentation, And Release Evidence

Scope:

- locked Azure ML environments
- CI workflow and repository checks
- README, API/configuration guides, production handoff, drift guide, runbooks
- release evidence index and rollback instructions

Exit evidence: CI is green for the exact source SHA, documentation agrees with
the implementation, and every Azure scenario is indexed by immutable identity.

## Release Blocker Rubric

| Blocker | Impact if unresolved | Required exit evidence |
| --- | --- | --- |
| No upstream/exact release SHA | Azure results cannot be tied to reviewable source | Feature branch pushed; CI and Azure manifests reference one SHA |
| Active-job enumeration not yet tied to release SHA | The worktree fix could be absent from the reviewed build even though live validation passes | Commit and push the guarded implementation; CI repeats the six focused tests for the exact release SHA |
| Production data hashes absent | Production submissions correctly fail closed | All accepted production configs contain verified SHA-256 values |
| Dataset provenance/RBAC incomplete | Data cannot be reproduced or least-privilege access cannot be proved | Data catalog, license, immutable asset version, and Entra data-plane role evidence |
| No post-remediation Azure runs | Local behavior may differ from Azure components/environments | Three diagnostics followed by 15 accepted industry scenarios |
| MLflow/registration/inference unproved | Reported winner may not be the registered runnable model | Scoped run lineage, exact version, and raw-input smoke test for each required scenario |
| Shared API key/local mutable state | Multi-user or multi-replica operation is unsafe and non-durable | Entra/OIDC authorization and transactional durable state for the selected topology |
| No final security/release review | Residual blockers can be mislabeled as non-blocking | Exact-SHA release review with no open release-blocking finding |

## Immediate Next Actions

1. Commit this final environment, CI, runbook, and evidence slice; restore
   GitHub connectivity; push the feature branch; and obtain green exact-SHA CI.
2. Obtain explicit owner approval for the shared non-production datastore
   credential refresh and pass its bounded canary.
3. Resolve the production API topology decision, then run three task
   diagnostics followed by the exact 15-scenario Azure qualification matrix.
