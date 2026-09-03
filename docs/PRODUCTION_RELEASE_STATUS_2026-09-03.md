# Production Release Status

Date: 2026-09-03
Branch: `codex_ys/mlops-pipeline-correctness`
Azure workspace: `mvpv1/mlops-accelerator`
Azure compute: `mlopsv2computecluster`
Azure environment: `azureml:mlops-v3-unified:33`

## Release Decision

Status: **not production-release ready**.

The previously published source contracts and Release Candidate CI pass at code candidate
`5499f5056a1c7cf29d597957dafe1ec32ed8d4ff`. The unified Azure runtime,
immutable 15-scenario catalog, and one complete historical Azure canary per task
type also pass. This is not exact-candidate Azure qualification: those three
canaries executed at `6447648a`, not the current source revision. The remaining
release path requires finishing the approved automated controller and multi-user
API, 12 never-executed industry scenarios, final-candidate qualification of all
15 scenarios, live deployment acceptance, and separate production approval.

Both approved workspace repairs are verified complete. The subsequent SQLite
initialization fix passes thread/process concurrency checks and the full local
backend suite (634 passed). The owner selected their verified Entra account as
the sole initial admin, with website-based user management. That code, browser
sign-in, and the admin page now pass local checks. Hosted CI and live deployment
acceptance for the new changes must be recorded separately; the earlier green
CI does not cover them.

No model training or dataset-scale processing was run locally. Local validation
is limited to unit/contract tests, graph compilation, UI checks, and source
review. Azure ML compute owns model execution.

## Verified Gates

| Gate | Evidence | State |
| --- | --- | --- |
| Source publication and CI | Fork code candidate `5499f505`; GitHub run `33792464847`, completed 2026-09-03 at 18:49 UTC | Backend and React jobs passed |
| Published-source validation | Non-Azure backend suite: 574 passed, 6 warnings; focused evidence verifier suite: 22 passed | Passed at the published candidate, not the new uncommitted changes |
| New API/identity/controller/state changes | Full marker-filtered backend suite: 634 passed, six warnings; focused security/admin/state suite: 60 passed | Passed locally, including the prior concurrency failure; no live user sign-in proof |
| New React UI | Seven unit tests, three browser tests, TypeScript lint, and production build | Passed locally; browser identity provider/API mocked, not live Entra proof |
| Qualification batch monitoring | Canonical submission JSON, fail-closed status/timeout semantics, structured evidence, and read-only smoke against three accepted Azure parents | Passed |
| Qualification artifact verification | Nonempty artifact contracts, locked-test isolation, MLflow/data/model lineage, exact registered-version smoke, and full-matrix source identity | Three historical scenarios accepted; incomplete full matrix correctly refused |
| Qualification execution release gate | All three schedules disabled; canary Completed; nine default artifacts and named probe downloaded and verified at 20:08 UTC | Passed after approved recovery |
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

The owner approved disabling the three legacy schedules and refreshing only
the two default datastore credentials. Both actions were executed and verified.
No schedules were deleted, no storage key was rotated, and `mlops_blob` was not
changed. Canary `verify-workspace-datastores-70e5c60f-approved-20260903` completed
on `mlopsv2computecluster`. The canonical live verifier passed at
`2026-09-03T20:08:04Z`, including nine default artifacts and the named probe.
The probe marker SHA-256 is
`547d704a8c1b380fafef75f66dcd38765cbe56b43ecf11c003e6f75280d9cf1a`.
The old failed canary is historical, not the current recovery status.

Git publication is no longer blocked. The existing stored GitHub credential was
used with a command-scoped credential-helper override; no credential or global
Git configuration was changed. The push of `5499f505` and its exact-commit CI
both succeeded.

## Closed Workspace Blockers

### Legacy Retraining Schedules

Status: **closed by approved containment**. All three return
`is_enabled=false`, provisioning `Succeeded`.

The following schedules were enabled and ran at 02:00 UTC on 2026-09-03:

- `auto-retrain-clustering-online-retail-daily`
- `auto-retrain-regression-college-daily`
- `auto-retrain-classification-telecom-churn-daily`

Each executes a static S1-S13 graph and omits S14 and the external controller.
Their S12 registration attempts were skipped while parent jobs still reported
`Completed`; the regression S13 artifact also contains contradictory policy and
legacy-trigger fields.

Disabling them prevents further unconditional training from these schedules.
The owner selected an automated external controller as the replacement; it is
under development and has not been deployed or enabled.

### Workspace-Default Artifact Datastores

Status: **closed by approved credential refresh and functional canary**.

The previous functional check of `workspaceblobstore` and
`workspaceartifactstore` failed with account-key authentication errors. Probe
`verify-workspace-datastores-6447648a-20260903142255`
failed default output upload and independent SDK artifact download with a
signature mismatch.

The approved recovery replaced only the stored current credentials and passed
the new upload/download probe. No further owner repair is required now. Repeat
the bounded canary if its evidence is older than 24 hours before qualification.

## Closed Code Blocker

### Concurrent Operational State

The earlier `sqlite3.OperationalError: database is locked` during WAL
initialization is fixed with bounded busy-only initialization retry. Unexpected
SQLite errors still fail closed. The previously failing concurrent-reservation
test, a synchronized four-process cold-start test, and the full 634-test backend
suite pass. This is local code proof; production local-disk placement and
persistence still need deployment verification.

## Active Blockers By Impact

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

The owner chose multi-user access through an explicit allowlist and automated
external retraining. Development now contains Entra delegated-access-token
validation, tenant/object-ID allowlisting, viewer/operator authorization,
durable request audit, trusted actor propagation to submissions/approvals,
transactional state migration, and bounded S14 discovery/submission code.

The controller requires fresh matching S14 evidence, uses the canonical
submitter, reserves the decision transactionally, and blocks automatic retry
when submission outcome is uncertain. Candidate promotion remains manual.
These are implementation claims, not deployed acceptance. Shared-state tests
now pass, but the daemon's live discovery adapter is not yet verified. React
Entra sign-in and the admin-only Users page are implemented and browser-tested
with mocked identity/API responses.

Initial-user scope is resolved: only `yashu.savyminds@gmail.com`, object ID
`b03e4295-9fce-4b3b-b6ba-e7e750e639ef`, starts as admin. The checked bootstrap
file contains no other users. Admins can later add existing Entra users and
assign roles from the website, without modifying Azure RBAC or directory roles.

Owner/deployment inputs still needed: Entra API and SPA app registration IDs
and the intended HTTPS UI redirect URL. Directory
app-registration changes require a separate explicit scope. Proposed state
topology is one API/controller host with persistent local disk, not Azure Files,
NFS, or multiple hosts. Confirm the host/disk before deployment. No live access
configuration or directory permissions have been changed. See
`OPERATIONAL_RUNBOOKS/admin-user-management.md` for the exact bootstrap and
server setup.

### Production Deployment And Model Promotion

Impact: **intentional governance gate**.

No production endpoint, traffic change, model alias, stage transition, or model
promotion is authorized. Request separate approval only after every technical
gate above passes.

## Critical Path

1. Record exact-commit hosted CI for the passing local implementation.
2. Validate the automated controller's live read-only discovery adapter and
   supply real app/host settings. The owner-only bootstrap is already defined.
3. Pass full local contract checks and exact-commit hosted CI, then validate
   multi-user login and a bounded controller canary on the approved server.
4. Freeze the candidate and repeat all 15
   graph preflights. Execute the 12 new scenarios and requalify the three
   historical scenarios on Azure ML compute at that exact candidate.
5. Consolidate MLflow lineage, holdout, registration, raw-input inference,
   drift, retraining, and rollback evidence.
6. Verify the chosen API topology, obtain green exact-commit CI, and perform the
   final release review.
7. Request separate production deployment and model-promotion approval.

The previous estimate of 6-8 clean Azure hours covered only the 12 new
scenarios, plus up to one business day for evidence consolidation and review.
It excludes owner waiting time, final-candidate reruns of the
three historical scenarios, and any requested public/multi-user API work. It
is therefore not a complete release ETA. Workspace recovery and scope decisions
are now complete, but API/controller implementation and live acceptance are
not. Rebaseline after server acceptance and the frozen-candidate first wave establish the actual scope
and runtime; failures requiring immutable reruns extend the schedule.

## Evidence

The release evidence bundle is stored outside the source tree at:

`snapshots/mlops-v33-final-preflight/70e5c60f/platform-recovery/`

This contains `approved-scope.json`, `verified-recovery.json`, and the actual
downloaded canary artifacts. The earlier code qualification evidence is at:

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
