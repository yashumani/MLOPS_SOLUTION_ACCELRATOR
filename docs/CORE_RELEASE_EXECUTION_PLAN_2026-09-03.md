# Core Release Execution Plan

## Goal And Authority

Complete automated retraining, full industry qualification, and final release
acceptance for the MLOps Solution Accelerator. The optional website and Entra
website deployment are deferred and are not prerequisites for these gates.

The owner authorized this work on 2026-09-03. Local activity is restricted to
source inspection, edits, version control, and transfers. Do not execute local
Python, pytest, npm, pipeline steps, graph validation, controllers, training,
data processing, model inference, or evidence-verification scripts. Execute
those activities on Azure. Hosted CI is supporting remote evidence, not a
replacement for Azure execution.

Use the canonical repository and existing feature branch:

- Branch: `codex_ys/mlops-pipeline-correctness`.
- Starting commit: `dc969f129b619ead6e7a8f651929592db29718b8`.
- Starting CI: `33808576794`, Backend contracts and React checks passed.
- Subscription: `93044a08-5661-4f1b-b424-5eafe066a9d1`.
- Workspace: `mvpv1/mlops-accelerator`, tenant
  `3bc05bc3-19d1-4d30-89c5-134f4b278b11`.
- Remote orchestration: existing compute instance `mlopspipelinev2`.
- Model execution: existing cluster `mlopsv2computecluster`.
- Starting pinned environment: `mlops-v3-unified:33`.

Do not create another repository copy. The existing Azure notebook checkout
is dirty on `prod_hardening_20260523`; leave it untouched. Upload immutable
Azure job code assets from the canonical feature branch and execute job-mounted
source, not a new Git checkout. Preserve unrelated remote changes. Do not alter
production traffic, model aliases/stages, or promote models. Those actions need
separate approval. Never bypass active-job, source-identity, policy, or ledger
guards to make a test pass.

## Progress

Planning refresh: canonical branch and clean worktree were rechecked at
`6190c5e07b7fe2f405c14e744598b5277c7d2978`. Azure and hosted-CI results below
are recorded terminal evidence from the preceding execution, not fresh cloud
checks in this planning turn. No release gate is complete.

| Milestone | State | Exit evidence |
| --- | --- | --- |
| Recover scope and current source | Complete | Clean starting feature branch; exact-commit hosted CI already passed |
| Establish Azure-only execution session | Complete | `mlopspipelinev2` Running; managed-identity workspace/cluster reads succeeded; all three legacy schedules disabled |
| Controller bootstrap correction and remote preflight | Tests passed; live access blocked | Corrected Azure job `controller-preflight-dc969f12-20260904b`: 56 tests passed, 0 failures/errors/skips; managed-identity workspace read denied |
| Publish controller corrections and regression tests | Published; hosted CI passed | Commit `6190c5e07b7fe2f405c14e744598b5277c7d2978`, CI `33823964566`: 652 backend tests passed, 1 Windows-only test skipped; 7 UI unit and 3 browser tests passed |
| Transfer published source for fresh Azure validation | Complete for candidate `6190c5e0` | Upload and metadata confirmed; full round-trip download SHA-256 equals `c9ba2a49775f7871518acef99c32977c78c3fcd87ff3c99b336d0154db4e8766` |
| Automated retraining acceptance | Pending | Live discovery, submission/refusal, replay/concurrency, restart and recovery evidence |
| Final-source qualification | Pending | 15 accepted scenarios at one Git and uploaded source identity |
| Final release acceptance | Pending | Complete evidence report, operational recovery proof, exact-commit green CI, no unresolved release-blocking findings |

### Next Execution Order

| Order | Work and owner | Completion check and dependency |
| --- | --- | --- |
| 1 | Complete for candidate `6190c5e0`: agent diagnosed key-retrieval transport failures and verified the uploaded archive by downloading it and comparing SHA-256. | Keep the verified blob immutable and recheck its checksum inside the Azure job. No key rotation or permission change was needed. |
| 2 | Owner: approve workspace-only Reader for the cluster identity and approve supervised controller hosting with idle shutdown disabled. Agent: apply only approved changes and verify them. | Fresh managed-identity workspace read succeeds; persistent storage and service lifecycle are verified. These are two separate shared-resource approval boundaries. |
| 3 | Agent: run the expanded Azure preflight and controller acceptance. | All 59 focused tests in the published candidate pass on Azure, then real discovery, one permitted submission, refusal/replay/concurrency, terminal candidate evidence and restart/recovery pass. Test count alone is not acceptance. |
| 4 | Agent: finish any resulting corrections, obtain green exact-commit CI and freeze source/config/data/environment identities. | One final Git commit and uploaded source checksum are recorded. Do not mix the earlier archive or CI results into proof for changed source. |
| 5 | Agent: qualify all 15 catalog scenarios on Azure in bounded waves. | Start with one scenario per task, at most two active parent pipelines; then complete the remaining 12. Require terminal success and exact registered-model raw-input inference for every scenario. |
| 6 | Agent: consolidate and verify final release evidence. | Complete-matrix verifier exits 0 with `release_matrix_accepted: true`; controller acceptance, lineage, isolation, registration, drift and recovery evidence all pass; no unresolved release-blocking findings. |

The first checkpoint is prerequisite resolution, not another full training run.
Do not spend cluster time repeating a known permission failure. Recheck live
datastore/schedule gates before qualification; historical health is not a
substitute for the existing freshness checks. Record durations from the first
three final-source scenarios to calculate the remaining Azure runtime and ETA.
Approval wait time, queue time and corrective reruns remain explicit schedule
dependencies; no calendar completion date is supported yet.

### Latest Published Candidate And Verified Transfer

- Published source: `6190c5e07b7fe2f405c14e744598b5277c7d2978` on the existing
  `yashumani` feature branch; no merge or production deployment.
- Hosted [CI run 33823964566](https://github.com/yashumani/MLOPS_SOLUTION_ACCELRATOR/actions/runs/33823964566)
  completed successfully for that exact commit. Backend output:
  `652 passed, 1 skipped, 5 warnings in 33.55s`. The skip is the Windows-only
  liveness probe on Linux. Runtime dependency audit passed with 7 documented
  ignored findings; this is not a zero-exception audit. UI checks also passed.
- New archive: `snapshots/controller-preflight-6190c5e0.zip` in the outer
  workspace, 1,340,279 bytes, SHA-256
  `c9ba2a49775f7871518acef99c32977c78c3fcd87ff3c99b336d0154db4e8766`.
- Intended blob: `mlops_blob/qualification/code/controller-preflight-6190c5e0.zip`.
  Initial upload failed with `NoAuthenticationInformation` at
  `2026-09-04T01:06:42.4353668Z`, request ID
  `2acee484-f01e-0022-0b09-3c7fc5000000`. A bounded retry subsequently
  succeeded at `2026-09-04T01:21:37+00:00`, ETag `0x8DF0A22DD71B572`.
  Blob properties confirmed 1,340,279 bytes and the expected source metadata.
- Full content verification subsequently passed: the downloaded archive's
  SHA-256 exactly matched the source hash above. Terminal output:
  `{"Verified":true,"SHA256":"c9ba2a49775f7871518acef99c32977c78c3fcd87ff3c99b336d0154db4e8766","CredentialPersisted":false}`.
  This closes transfer verification, not Azure execution of the candidate.
- Explicit CLI key retrieval failed with `ConnectionResetError(10054)`.
  Using the same Azure login with IPv4 `curl.exe` to the documented ARM
  `listKeys` API succeeded; the existing key was passed only through a
  process-scoped environment variable to the verification download. Tokens
  and keys were never printed or saved; the environment was restored afterward.
  No security setting, key rotation, identity substitution, role or host change
  was required. General CLI transport reliability is not claimed repaired.
  [Microsoft List Keys API](https://learn.microsoft.com/en-us/rest/api/storagerp/storage-accounts/list-keys?view=rest-storagerp-2025-06-01).
- Azure CLI 2.84.0 can catch automatic key-retrieval errors and continue with
  only a warning, hidden by `--only-show-errors`. This is a supported
  explanation for the missing-authentication symptom, not proof of the exact
  historical request failure. Fail closed if credentials cannot be retrieved;
  do not send an unauthenticated fallback request.
  [Versioned CLI implementation](https://github.com/Azure/azure-cli/blob/azure-cli-2.84.0/src/azure-cli/azure/cli/command_modules/storage/_validators.py#L154-L171).
- Three new launcher tests passed in hosted CI but have not run on Azure.
  The older 56-test Azure result does not cover these or prove the new archive.
- `configs/jobs/validate_controller_archive.yml` still describes historical
  job b and archive a. Before another submission, review a new unique job name,
  current archive URI, expected checksum and source tags together. Do not
  submit that file unchanged or overwrite a previous job.

Remote setup evidence: subscription `Enabled`, identity
`systemAssignedIdentity`, cluster provisioning `Succeeded`, autoscale 0-8.
The notebook SDK environment warned that `mlflow==3.1.1` and
`mlflow-skinny==2.22.1` differ. It was used for read-only control-plane checks,
not qualification. Do not repair that shared environment blindly or count it
as the pinned runtime; use the immutable Azure environment for actual tests.

### Recorded Azure Checkpoint: Corrected Launcher

The multiline launch failure is resolved. The readable
`scripts/bootstrap_controller_archive.py` is encoded into a single-line job
command to preserve indentation. Its source SHA-256 is
`ab325a64927f8d2256c9c92cceefd335703126f6a01cbbf26ecc297200b20005`.
Checksum and Python syntax were checked on the Azure instance before submission.
The cluster then verified the existing archive checksum, safely extracted 724
entries, and ran the focused controller/state tests.

```text
controller-preflight-dc969f12-20260904b: Failed
56 passed in 3.60s
JUnit: tests=56, failures=0, errors=0, skipped=0
checks.remote_contract_tests: passed
AuthorizationFailed: Microsoft.MachineLearningServices/workspaces/read
```

The overall job failed because its compute managed identity lacks workspace
read permission, not because a controller test failed. Live schedule/datastore
gates and Run History discovery were not reached. This is not evidence of a
subscription outage, and it does not contradict the separate compute-instance
identity's successful submissions and artifact downloads.

- Test execution: `2026-09-04T00:47:59.703716+00:00`, JUnit duration 3.599 seconds.
- Preflight finished: `2026-09-04T00:48:04.907149+00:00`.
- Runtime source SHA-256: `91b9d60305e3fbb5d54b4a5649430dced083f470657324f988b7299bbb392b39`.
- Cluster packages: azure-ai-ml 1.34.1, azure-identity 1.25.3, mlflow and
  mlflow-skinny 3.15.0, numpy 1.26.4, scikit-learn 1.4.2.
- Remote evidence: `/tmp/controller-preflight-b-progress-6lx12jbj`, including
  named `evidence` output with `controller-tests.xml`, `controller-tests.log`,
  `controller_preflight.json`, and `package_identity.json`. Durable originals
  remain attached to Azure job `controller-preflight-dc969f12-20260904b`.
- No local application/tests executed. No role assignment, host lifecycle
  change, daemon activation, training submission or model promotion occurred.
- This evidence was collected before source publication. The archive is an
  unfrozen prerelease snapshot, not the final 15-scenario release source.
- Publication preparation adds three launcher regression tests: payload/source
  identity and syntax, refusal outside Azure, and refusal before extraction on
  checksum mismatch. They are included in the next Azure preflight's required
  test set. They were not in job `controller-preflight-dc969f12-20260904b` and
  are not included in its 56-test result. Hosted CI provides separate supporting
  evidence; the next Azure preflight must use a newly packaged current source.

### Owner Action: Cluster Read Access

ARM independently confirmed that the denied principal is the system-assigned
identity of `mlopsv2computecluster`, not the workspace or compute-instance
identity:

- Principal/object ID: `b1d3be3f-0a4d-462c-be37-b809e5cda716`.
- Client/application ID from the denial: `fe4ee26c-4e05-4b52-890f-6c1b91a0cea6`.
- Tenant: `3bc05bc3-19d1-4d30-89c5-134f4b278b11`.
- Workspace-scope role listing with inherited roles returned `[]`.

Proposed fix for the observed read denial: an administrator can assign the
built-in **Reader** role to this identity at this workspace only, through
Azure Portal > `mlops-accelerator` > Access control (IAM), or Azure Cloud Shell:

```bash
az role assignment create \
  --subscription 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --assignee-object-id b1d3be3f-0a4d-462c-be37-b809e5cda716 \
  --assignee-principal-type ServicePrincipal \
  --role Reader \
  --scope /subscriptions/93044a08-5661-4f1b-b424-5eafe066a9d1/resourceGroups/mvpv1/providers/Microsoft.MachineLearningServices/workspaces/mlops-accelerator
```

This command is a proposal and has **not** been executed. Explicit owner
approval is required before the agent changes shared access. Reader grants
workspace asset visibility; Microsoft also documents datastore credential
visibility for workspace readers. It does not authorize asset creation,
training submission or model promotion. Do not grant subscription-wide access,
Owner, or Contributor to address this read-only preflight failure.
[Microsoft workspace role documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles?view=azureml-api-2).

After access is granted, use a fresh Azure process/token and a new uniquely
named preflight job, then verify every remaining live check. Reader addresses
the observed ARM read denial; additional artifact or Run History authorization
has not yet been tested and must not be assumed. A separately approved option
is to place live controller checks on the already-authorized compute instance;
no credential substitution or changed execution design was attempted here.

### First Attempt: Historical Launcher Failure

The controller can now initialize its own explicitly workspace-bound state
without deploying the optional API. New tests cover repeated and competing
initialization, wrong-workspace rejection, nonempty legacy state, path
containment, and Azure access failure. At the first attempt these edits were
uncommitted and unvalidated. The first Azure job failed before reaching them.
The later job b and published-candidate evidence above supersede that status;
neither is the final source freeze.

- Azure preflight job: `controller-preflight-dc969f12-20260904a`.
- Execution: one node on `mlopsv2computecluster`, environment
  `mlops-v3-unified:33`, managed identity, 1800-second execution limit.
- Input: `azureml://datastores/mlops_blob/paths/qualification/code/controller-preflight-dc969f12-20260904a.zip`.
- Archive SHA-256: `05b4ae9e8d5580f31f47756bf52248040e70e811e2a69e0d21946a07a95fefe1`.
- Archive size: 1,295,260 bytes. Upload confirmed
  `2026-09-04T00:17:33+00:00`, ETag `0x8DF0A19EAB9083E`.
- Upload uses existing account-key access. No key rotation, permission change,
  secret output, or datastore configuration change was performed. OAuth upload
  lacked the storage data role; the local ML CLI extension was unavailable.
- Submission ran from the Azure instance SDK using managed identity, not the
  dirty remote checkout. The cluster verifies the archive checksum before
  executing its contents. No project code or tests were executed locally.
- Named `evidence` output includes contract-test logs/JUnit, runtime/source
  identities, live discovery results, and `controller_preflight.json` when the
  validation entry point executes. This failed attempt produced none of these
  files. The report is designed not to claim controller or matrix acceptance.

Terminal output from the remote managed-identity monitor:

```text
{"poll": 5, "name": "controller-preflight-dc969f12-20260904a", "status": "Failed"}
```

Azure `artifacts/user_logs/std_log.txt`:

```text
  File "<stdin>", line 16
    raise RuntimeError("Source archive exceeds the reviewed size bounds")
    ^
IndentationError: expected an indented block after 'if' statement on line 15
```

The execution wrapper recorded exit code 1 at
`2026-09-04T00:32:08.540970Z`. Logs were downloaded on the Azure instance to
`/tmp/controller-preflight-evidence-ak6pcyvf`. No local tests ran. The
subscription accepted the job and the cluster launched Python; neither the
controller tests nor live discovery can be counted as passed.

### Active Blockers And Impact

| Category | Blocker and impact if unresolved | Owner and next action |
| --- | --- | --- |
| Permissions | Cluster managed identity cannot perform workspace read. The 56 focused tests pass, but live datastore/schedule checks and completed-job discovery cannot execute under that identity. | Owner: approve the workspace-scoped Reader proposal above, or perform it in Azure. Agent: revalidate with fresh credentials and a new bounded job; no broad role escalation. |
| Service availability | Instance is `Stopped` with idle shutdown `PT15M`; an interactive daemon cannot establish continuous availability. | Owner: approve the host lifecycle and supervised service change, including ongoing Azure compute cost. Agent: then verify persistent local disk, supervision, restart and recovery. |
| Release qualification | The final code identity is not frozen and the full 15-scenario evidence set does not exist for it. Earlier passes cannot establish this release's acceptance. | Agent: complete controller acceptance, freeze source with green CI, run all 15 scenarios and final evidence verification on Azure. |

The first attempt stopped after the launch blocker. On continuation, that
launcher was corrected and job `controller-preflight-dc969f12-20260904b` proved
the focused tests pass, then stopped at the managed-identity authorization
boundary. A later turn published the corrections and obtained green hosted
CI, then resolved the transfer blocker with verified uploaded bytes. Fresh
checks still return no cluster role assignments and a stopped instance with
`PT15M` idle shutdown. No role assignment, daemon deployment or model promotion
has been performed. Both shared-resource approvals have remained unanswered
across at least three goal turns. Automatic pursuit is blocked on those owner
decisions, not marked complete. Resume the same goal and verified artifact
after approval; do not create another working copy or repeat a known-denied
preflight merely to produce activity.

The compute instance has `idleTimeBeforeShutdown: PT15M`. Continuous service
acceptance cannot rely on an interactive terminal staying connected. Before
deploying the daemon, obtain approval for its host lifecycle, verify persistent
local storage and supervision, and test restart/recovery. No idle-shutdown or
shared service setting has been changed. This is an operational dependency,
not a reason to run the controller locally. Explicit approval to disable idle
shutdown and install an automatically restarted service was requested during
this attempt and has not yet been received.

## 1. Automated Retraining

1. Inspect the remote checkout, active jobs, compute, credentials, pinned
   environment, schedules, and datastore health before submitting anything.
   The previous datastore canary is historical until its freshness and outputs
   pass the existing live release gate; repeat it remotely if required.
2. Validate the actual Run History discovery adapter against Azure: completed
   parent filtering, timestamp ordering, pagination, experiment scope, and
   bounded scan behavior. An empty result does not prove the positive path.
3. Remove the controller's dependency on starting the optional API for state
   initialization. Keep explicit tenant/workspace binding, identity validation,
   server-owned paths, transactional writes, and legacy migration protections.
4. Use a reviewed explicit watch manifest and managed identity. Persist SQLite
   state on the Azure instance's persistent local filesystem, not the mounted
   workspace SMB/NFS share or temporary disk. Verify filesystem placement,
   permissions, identity access, and backup/restore before continuous operation.
5. Run focused controller/state tests on Azure. These tests are supporting
   evidence; mocks are never reported as live policy or submission proof.
6. Produce genuine source pipeline evidence and an explicitly approved baseline
   scoped to qualification. Verify S13 emits drift evidence only and S14 emits
   the policy decision only. Do not edit an S14 artifact to manufacture consent.
7. Verify a permitted fresh decision produces exactly one candidate through
   `pipelines/submit_pipeline.py`, with matching source decision, configuration,
   execution, baseline, and model identities. Observe that candidate to terminal
   status and validate its artifacts.
8. Verify blocked, stale, mismatched, missing, and replayed decisions cannot
   create a job. Test concurrent workers and restart persistence. Simulate an
   ambiguous submit response only in isolated qualification state and prove it
   becomes `reconciliation_required`, with no automatic resubmission.
9. Verify the remote service lifecycle, audit output, restart and recovery
   behavior. Keep legacy unconditional schedules disabled. Configure only
   reviewed targets; model promotion remains manual.

Acceptance requires both permitted and refused live paths, exact job/decision
identities, persistent state, replay protection after restart, a tested recovery
procedure, and a supervised remote controller. A running process alone is not
acceptance. Missing shared identity permissions or destructive recovery actions
are approval boundaries, not reasons to use personal credentials silently.

## 2. Full Industry Qualification

The canonical catalog is
`configs/qualification/industry_matrix_execution_catalog.yml`. Reuse its
versioned datasets and reviewed configurations unless remote evidence proves
they are unsuitable. Any replacement needs provenance, content/schema hashes,
task suitability, and renewed configuration validation.

| Task | Required industries |
| --- | --- |
| Classification | Healthcare, financial services, telecommunications, manufacturing, insurance |
| Regression | Education, real estate, insurance, energy, aviation |
| Clustering | Healthcare, financial services, telecommunications, retail, education |

1. Finish necessary corrections and remote regression checks, obtain green
   exact-commit CI, and freeze one final Git commit and uploaded source SHA-256.
2. Audit all 15 Azure data assets and run all canonical graph preflights remotely
   using that source. Record config, data, environment, and candidate identities.
3. Recheck the live release gate: disabled legacy schedules, fresh completed
   datastore canary, successful default-artifact and named-probe downloads.
4. Use `scripts/batch_submit_all.py` and the canonical submitter. Start with one
   representative scenario per task, in waves of at most two pipeline parents.
   Then run the remaining scenarios in bounded waves. Inspect cluster capacity
   before each wave; preserve configured budgets and submission guards.
5. Monitor through `scripts/monitor_batch.py`. Require terminal success, not
   merely submission acceptance. Record failed child steps and exact immutable
   identities. Do not advance a failed wave as passed.
6. For every parent, download required outputs on Azure, run the exact registered
   model's raw-input smoke through `scripts/submit_registered_model_smoke.py`,
   and validate evidence remotely after each wave.
7. Verify the actual configured engine/phase execution, configuration pruning,
   candidate selection, and learned-transform isolation. Current engines are
   PyCaret and FLAML hosted on Azure ML; do not label this native Azure AutoML
   support without an implemented and validated adapter.

All 15 scenarios must be rerun at the final source, including the three
historical passes. If runtime source changes after freezing, results from the
old source cannot be combined with the new source for acceptance: refreeze and
obtain all 15 at the new required identity. Store evidence outside the frozen
source tree so evidence collection does not change the uploaded source hash.

## 3. Final Release Acceptance

Collect one indexed evidence bundle containing:

- Final source, upload, environment, configuration, and data identities.
- Exact-commit hosted CI output and Azure-side contract test results.
- Every parent/child job, selected candidate, rejection/pruning reason, and
  execution-scoped MLflow parent/child linkage.
- Training-only fit/selection evidence and exactly one locked-test evaluation
  of the frozen champion. Quality policy must have truthful pass/warn/block
  semantics and agree with registration.
- Exact numeric model versions, self-contained preprocessing/model bundles,
  and successful raw-input inference for every scenario.
- S13 drift evidence, S14 policy decisions, controller audit, replay/refusal
  evidence, and approved baseline lineage.
- Backup/restore, restart, failed-submission reconciliation, service recovery,
  and rollback procedures with remote test outputs.

Run `scripts/verify_qualification_evidence.py --require-complete-matrix` on
Azure against the complete manifest. Require exit code 0 and
`release_matrix_accepted: true`; partial reports are not release acceptance.
Perform a final review of source and evidence for unresolved correctness,
security, data-integrity, and operational failures. Document warnings and
limitations without concealing them behind a successful Azure parent status.

The three requested gates are complete only after controller acceptance, the
full matrix, and final review all pass. Actual production deployment/traffic
and model promotion remain separate decisions. Deferred website work is listed
as optional follow-up, not an undisclosed requirement or a completed feature.

## Tracking And Estimates

The agent owns implementation, remote tests, job monitoring, corrections, and
evidence collection. Request owner input only for genuine new authority or
resource-access boundaries. Track exact passed, failed, pending, and blocked
checks; never report percentage completion.

Report a measured ETA after the first final-source wave, using observed Azure
queue time, runtime, required reruns, and remaining scenarios. Do not reuse the
older 12-scenario estimate as an estimate for this larger scope.

Maintain the durable handoff with source identity, commands, job handles,
checkpoint evidence, outstanding failures, and the next bounded action.
Continuation resumes existing jobs and state; it must not duplicate submissions.

Stop and refresh the handoff on a new unplanned blocker, 3 consecutive failed
attempts, the same command run 5 times without new information, 10 polls, or
approximately 150 tool calls without a verifiable checkpoint. Do not switch
approaches or bypass a gate after reaching a stop condition. Report the exact
failure, impact, owner and next bounded action. The goal stays incomplete
until all three release gates have terminal evidence; production deployment
and model promotion still require separate approval.
