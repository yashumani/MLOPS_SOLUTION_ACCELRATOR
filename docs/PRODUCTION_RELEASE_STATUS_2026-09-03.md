# Production Release Status

Date: 2026-09-03
Branch: `codex_ys/mlops-pipeline-correctness`
Azure workspace: `mvpv1/mlops-accelerator`
Azure compute: `mlopsv2computecluster`
Azure environment: `azureml:mlops-v3-unified:33`

## Release Decision

Status: **not production-release ready**.

The source contracts and Release Candidate CI pass at code release candidate
`e82f42d6`. The unified Azure runtime, immutable 15-scenario catalog, and one
complete Azure canary per task type also pass. The remaining release path is
blocked by two owner-approved workspace actions, 12 unexecuted industry
scenarios, the API/retraining deployment decisions, and final production
approval.

No model training or dataset-scale processing was run locally. Local validation
is limited to unit/contract tests, graph compilation, UI checks, and source
review. Azure ML compute owns model execution.

## Verified Gates

| Gate | Evidence | State |
| --- | --- | --- |
| Source publication and CI | Fork code candidate `e82f42d6`; GitHub run `33779003942` | Backend and React jobs passed |
| Current source validation | Non-Azure CI-equivalent backend suite: 527 passed; focused API/security/state suite: 73 passed | Passed locally and in hosted CI |
| React UI | Five tests, TypeScript lint, and production build | Passed locally and in hosted CI |
| Azure access | Subscription, workspace, compute, datastore, jobs, and model registry readable | Passed |
| Runtime | `mlops-v3-unified:33` on `mlopsv2computecluster` | Passed |
| Qualification catalog | Five industries each for classification, regression, and clustering | Passed |
| Exact-head graph preflight | All 15 scenarios pass canonical `--dry_run` at `e82f42d6`; submitted count is zero | Passed |
| Classification canary | `clever_parsnip_bkxp5z6gl6`; model version 4; smoke `jolly_honey_b8p6z98b67` | Passed |
| Regression canary | `goofy_planet_yz78rj7tqz`; model version 3; smoke `goofy_toe_wp29x5h9vy` | Passed |
| Clustering canary | `joyful_pumpkin_f4cm3x626m`; model version 1; smoke `jovial_mangos_bchzbq8qj1` | Passed |
| S14/controller refusal | All three `should_submit=false` decisions refused before submission or ledger mutation | Passed |
| Full industry matrix | Three accepted; 12 not executed | Incomplete |
| Production endpoint/promotion | No approval and no production action | Intentionally stopped |

A read-only refresh after CI confirmed all three schedules remain enabled. It
also confirmed `workspaceblobstore` and `workspaceartifactstore` retain their
June 16, 2025 AccountKey records; no repair was applied.

## Active Blockers By Impact

### Legacy Retraining Schedules

Impact: **release-blocking correctness, cost, and audit risk**.

The following schedules were enabled and ran at 02:00 UTC on 2026-09-03:

- `auto-retrain-clustering-online-retail-daily`
- `auto-retrain-regression-college-daily`
- `auto-retrain-classification-telecom-churn-daily`

Each executes a static S1-S13 graph and omits S14 and the external controller.
Their S12 registration attempts were skipped while parent jobs still reported
`Completed`; the regression S13 artifact also contains contradictory policy and
legacy-trigger fields.

If unresolved, the workspace continues unconditional daily training, consumes
capacity, and produces misleading release evidence. Required owner action:
approve disabling, not deleting, all three schedules and choose either
observe-only/manual retraining or a deployed compliant external controller.

### Workspace-Default Artifact Datastores

Impact: **release-blocking artifact integrity and recoverability risk**.

`workspaceblobstore` and `workspaceartifactstore` hold stale account-key
credentials. Probe `verify-workspace-datastores-6447648a-20260903142255`
failed default output upload and independent SDK artifact download with a
signature mismatch.

If unresolved, successful compute can still lose logs, artifacts, or release
evidence. Required owner action: approve refreshing the stored current key on
only these two datastores, without rotating the storage key, then run
`OPERATIONAL_RUNBOOKS/workspace-datastore-credential-recovery.md`.

### Remaining Industry Qualification

Impact: **release-blocking product coverage risk**.

Twelve scenarios remain: four classification, four regression, and four
clustering. Their data assets, hashes, exclusions, configs, and canonical graph
preflights pass, but they have not executed.

If unresolved, all three task types have representative proof but the required
five-industry breadth per task remains unverified. After schedule containment
and datastore repair, execute the six two-parent waves through
`scripts/batch_submit_all.py` and require exact-version registered-model smoke
evidence for every accepted scenario.

### Production API And Retraining Topology

Impact: **conditional production security and availability risk**.

The implemented release profile supports one private operator, one API process,
one controller writer, a strong shared key, explicit HTTPS origins, disabled
config mutation, and absolute durable state paths. Startup fails closed for
unsafe private settings and for the unimplemented `multi_user` profile. The
React UI no longer accepts a statically embedded API key.

Required owner decision: approve the constrained private topology or require
Microsoft Entra/OIDC, actor authorization, and transactional shared state for a
public or multi-replica deployment. Code safeguards are not deployment proof.

### Production Deployment And Model Promotion

Impact: **intentional governance gate**.

No production endpoint, traffic change, model alias, stage transition, or model
promotion is authorized. Request separate approval only after every technical
gate above passes.

## Critical Path

1. Disable the three legacy schedules after owner approval and record live
   disabled state.
2. Approve the retraining mode for this release.
3. Refresh the two stored datastore credentials after owner approval and pass
   the bounded write/upload/download canary.
4. Execute and verify the remaining 12 scenarios on Azure ML compute.
5. Consolidate MLflow lineage, holdout, registration, raw-input inference,
   drift, retraining, and rollback evidence.
6. Resolve API topology, obtain green exact-commit CI, and perform the final
   release review.
7. Request separate production deployment and model-promotion approval.

Conditional ETA after the three owner decisions and two Azure approvals are
available: 6-8 clean Azure hours for the remaining matrix plus up to one
business day for evidence consolidation and release review. Failures requiring
focused immutable reruns extend that estimate.

## Evidence

The release evidence bundle is stored outside the source tree at:

`snapshots/mlops-v33-final-preflight/e82f42d6/`

The exact-head directory contains the 15 canonical dry-run logs and summary.
The accepted Azure canary, data-asset, schedule, datastore, controller, queue,
and owner-action evidence remains indexed under the qualified runtime evidence
directory `snapshots/mlops-v33-final-preflight/6447648a/`.
