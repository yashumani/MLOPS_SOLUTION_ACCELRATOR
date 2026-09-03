# Production Release Status

Date: 2026-09-03
Branch: `codex_ys/mlops-pipeline-correctness`
Azure workspace: `mvpv1/mlops-accelerator`
Azure compute: `mlopsv2computecluster`
Azure environment: `azureml:mlops-v3-unified:33`

## Release Decision

Status: **not production-release ready**.

The source contracts and Release Candidate CI pass at code candidate
`5499f5056a1c7cf29d597957dafe1ec32ed8d4ff`. The unified Azure runtime,
immutable 15-scenario catalog, and one complete historical Azure canary per task
type also pass. This is not exact-candidate Azure qualification: those three
canaries executed at `6447648a`, not the current source revision. The remaining
release path requires two approved workspace actions, 12 never-executed industry
scenarios, final-candidate qualification, API/retraining deployment decisions,
and separate production approval.

No model training or dataset-scale processing was run locally. Local validation
is limited to unit/contract tests, graph compilation, UI checks, and source
review. Azure ML compute owns model execution.

## Verified Gates

| Gate | Evidence | State |
| --- | --- | --- |
| Source publication and CI | Fork code candidate `5499f505`; GitHub run `33792464847`, completed 2026-09-03 at 18:49 UTC | Backend and React jobs passed |
| Current source validation | Non-Azure backend suite: 574 passed, 6 warnings; focused evidence verifier suite: 22 passed | Passed locally; current Backend contracts CI also passed |
| React UI | Five tests, TypeScript lint, and production build | Passed locally and in hosted CI |
| Qualification batch monitoring | Canonical submission JSON, fail-closed status/timeout semantics, structured evidence, and read-only smoke against three accepted Azure parents | Passed |
| Qualification artifact verification | Nonempty artifact contracts, locked-test isolation, MLflow/data/model lineage, exact registered-version smoke, and full-matrix source identity | Three historical scenarios accepted; incomplete full matrix correctly refused |
| Qualification execution release gate | Live schedule state, canary identity/status, default-artifact download, named probe download, marker integrity, and freshness are required before submission | Passed; live negative smoke refused with zero submissions |
| Azure access | Fresh subscription state `Enabled`; schedule and datastore reads passed; earlier job/model reads succeeded | Subscription blocker resolved; this is not a datastore write/read canary |
| Runtime | `mlops-v3-unified:33` on `mlopsv2computecluster` | Passed |
| Qualification catalog | Five industries each for classification, regression, and clustering | Passed |
| Prior-code graph preflight | All 15 scenarios pass canonical `--dry_run` at `ed5edb06`; submitted count is zero | Passed at that revision; repeat after final source freeze |
| Historical classification canary | `clever_parsnip_bkxp5z6gl6`; model version 4; smoke `jolly_honey_b8p6z98b67`; source `6447648a` | Passed |
| Historical regression canary | `goofy_planet_yz78rj7tqz`; model version 3; smoke `goofy_toe_wp29x5h9vy`; source `6447648a` | Passed |
| Historical clustering canary | `joyful_pumpkin_f4cm3x626m`; model version 1; smoke `jovial_mangos_bchzbq8qj1`; source `6447648a` | Passed |
| S14/controller refusal | All three `should_submit=false` decisions refused before submission or ledger mutation | Passed |
| Full industry matrix | Three historical scenarios accepted; 12 never executed; no full frozen-candidate acceptance | Incomplete |
| Production endpoint/promotion | No approval and no production action | Intentionally stopped |

A read-only refresh and the exact-code live release-gate smoke confirmed all
three schedules remain enabled. The smoke exited `2`, recorded gate state
`blocked`, and submitted zero jobs. The refresh also confirmed
`workspaceblobstore` and `workspaceartifactstore` retain their June 16, 2025
AccountKey records; no repair was applied. The latest read-only refresh was at
18:46-18:48 UTC on 2026-09-03. Metadata age alone does not establish current
credential validity; the last functional datastore probe remains failed and was
not rerun during this refresh.

Git publication is no longer blocked. The existing stored GitHub credential was
used with a command-scoped credential-helper override; no credential or global
Git configuration was changed. The push of `5499f505` and its exact-commit CI
both succeeded.

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

The most recent functional check of `workspaceblobstore` and
`workspaceartifactstore` failed with account-key authentication errors. Probe
`verify-workspace-datastores-6447648a-20260903142255`
failed default output upload and independent SDK artifact download with a
signature mismatch.

If unresolved, successful compute can still lose logs, artifacts, or release
evidence. Required owner action: approve refreshing the stored current key on
only these two datastores, without rotating the storage key, then run
`OPERATIONAL_RUNBOOKS/workspace-datastore-credential-recovery.md`.

### Remaining Industry Qualification

Impact: **release-blocking product coverage risk**.

Twelve scenarios have never executed: four classification, four regression, and
four clustering. Their data assets, hashes, exclusions, configs, and prior-code
canonical graph preflights pass, but execution remains unverified.

If unresolved, all three task types have representative proof but the required
five-industry breadth per task remains unverified. After schedule containment
and datastore repair, execute the six two-parent waves through
`scripts/batch_submit_all.py` and require exact-version registered-model smoke
evidence for every accepted scenario. The runner now refuses `--execute`
unless all three schedules are disabled and the named fresh datastore canary
proves both workspace-default download paths.

Before the matrix starts, freeze the candidate Git commit and uploaded source
hash. `scripts/verify_qualification_evidence.py --require-complete-matrix`
requires all 15 scenarios to match that declared candidate, with five distinct
industries per task type. The three earlier canaries are useful regression
evidence, not automatic acceptance of a newer candidate. Under the current
gate, they must also be rerun at the final frozen candidate. Do not mix revision
identities or weaken the gate to count historical runs as current execution.

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
2. Approve the retraining mode and API topology for this release before freezing
   the source candidate.
3. Refresh the two stored datastore credentials after owner approval and pass
   the bounded write/upload/download canary.
4. Resolve release-scope code changes, freeze the candidate, and repeat all 15
   graph preflights. Execute the 12 new scenarios and requalify the three
   historical scenarios on Azure ML compute at that exact candidate.
5. Consolidate MLflow lineage, holdout, registration, raw-input inference,
   drift, retraining, and rollback evidence.
6. Verify the chosen API topology, obtain green exact-commit CI, and perform the
   final release review.
7. Request separate production deployment and model-promotion approval.

The previous estimate of 6-8 clean Azure hours covered only the 12 new
scenarios, plus up to one business day for evidence consolidation and review.
It excludes owner waiting time, workspace repair, final-candidate reruns of the
three historical scenarios, and any requested public/multi-user API work. It
is therefore not a complete release ETA. Rebaseline after the owner decisions,
datastore canary, and frozen-candidate first wave establish the actual scope
and runtime; failures requiring immutable reruns extend the schedule.

## Evidence

The release evidence bundle is stored outside the source tree at:

`snapshots/mlops-v33-final-preflight/5499f505/`

The current code checkpoint contains `accepted-three-report.json` (exit 0,
`state=passed`, three accepted, `release_matrix_accepted=false`) and
`complete-matrix-negative-report.json` (exit 1, `state=failed`, missing complete
coverage and a declared release candidate). These checks only read collected
JSON evidence; they do not execute training or resubmit jobs.

The previous `ed5edb06` directory contains the 15 canonical dry-run logs and
summary, plus the live zero-submission release-gate refusal. Qualification-monitor
smoke evidence for the three accepted parents remains under the previous
exact-head checkpoint.
The accepted Azure canary, data-asset, schedule, datastore, controller, queue,
and owner-action evidence remains indexed under the qualified runtime evidence
directory `snapshots/mlops-v33-final-preflight/6447648a/`.

Verifier usage and evidence contracts are documented in
`OPERATIONAL_RUNBOOKS/qualification_evidence.md`. Exact-code CI is recorded at
https://github.com/yashumani/MLOPS_SOLUTION_ACCELRATOR/actions/runs/33792464847.
