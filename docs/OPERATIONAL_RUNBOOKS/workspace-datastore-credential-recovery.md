# Workspace Datastore Credential Recovery

## Scope

Use this runbook only for the non-production Azure ML workspace
`mvpv1/mlops-accelerator`. It refreshes stored account-key credentials for the
shared `workspaceblobstore` and `workspaceartifactstore` datastores. It does not
rotate the storage account key and does not modify `mlops_blob`.

This is a shared infrastructure change. Obtain explicit human approval before
running the `--apply` command. Credentials being available is not approval.

## Current Failure

Both shared datastores point to storage account `mlopsaccelerat7263606092` and
use `AccountKeyConfiguration`. Output mount, output upload, and SDK artifact
download fail with `AuthenticationFailed` and a signature mismatch. The
`mlops_blob` datastore on the same account reads and writes successfully.

Impact: a multi-stage Azure ML pipeline can lose component outputs, MLflow
artifacts, model artifacts, and downloadable release evidence even when user
code succeeds.

## Preconditions

- Azure CLI is signed in to subscription
  `93044a08-5661-4f1b-b424-5eafe066a9d1`.
- The operator can list storage-account keys and update Azure ML datastores.
- Explicit approval has been recorded for this shared workspace change.
- Use the repository's existing `.venv-review` Python environment; the Azure
  CLI `ml` extension is not required.

## Dry Run

From the repository root:

```powershell
$python = '..\..\.venv-review\Scripts\python.exe'
& $python scripts\refresh_workspace_datastore_credentials.py
```

Expected: both datastore account/container pairs are validated and the script
reports that no credentials changed.

## Apply Approved Refresh

Do not print or paste the key into logs or chat.

```powershell
$key = az storage account keys list `
  --subscription 93044a08-5661-4f1b-b424-5eafe066a9d1 `
  --resource-group mvpv1 `
  --account-name mlopsaccelerat7263606092 `
  --query "[0].value" `
  --output tsv

if ([string]::IsNullOrWhiteSpace($key)) { throw 'Storage account key was not returned' }
$env:AZURE_STORAGE_ACCOUNT_KEY = $key
try {
  & $python scripts\refresh_workspace_datastore_credentials.py `
    --apply `
    --confirm-shared-workspace-change
} finally {
  Remove-Item Env:AZURE_STORAGE_ACCOUNT_KEY -ErrorAction SilentlyContinue
  $key = $null
}
```

The script allowlists both datastore names, verifies their expected storage
account and containers, and never prints the key.

## Validate Before Pipeline Submission

Submit `configs/jobs/verify_workspace_artifact_datastores.yml` with the Azure ML
SDK. Accept the repair only when all three checks pass:

1. Job status is `Completed`.
2. The `probe` output uploads through the default `workspaceblobstore`.
3. `ml_client.jobs.download(...)` downloads the run artifacts through
   `workspaceartifactstore`, and `workspace_datastore_probe.json` is present.

Do not submit classification, regression, or clustering diagnostics until all
three checks pass. A successful `mlops_blob` write does not validate either
workspace-default datastore.

The canonical qualification runner enforces this boundary live. After the
approved schedule containment and a successful fresh canary, pass that exact
job name on every qualification wave:

```powershell
& $python scripts\batch_submit_all.py `
  --scenario '<qualification-scenario-id>' `
  --execute `
  --datastore-canary-job '<completed-canary-job-name>' `
  --result-json '<wave-submission-evidence.json>'
```

Before the first submission, the runner re-reads all three legacy schedules
and requires each to be disabled with provisioning state `Succeeded`. It then
requires the named canary to be `Completed`, downloads its default artifacts
through `workspaceartifactstore`, downloads the named `probe` output through
`workspaceblobstore`, validates `workspace_datastore_probe.json`, and rejects a
probe older than 24 hours. Any missing, stale, unreadable, or nonconforming
evidence exits before a qualification job is submitted.

## Rollback

If the bounded canary still fails, do not rotate keys or alter role assignments
as an unplanned workaround. Preserve the failed run ID and error, confirm which
storage key the workspace should use, and repeat the approved refresh once with
that key. Escalate identity-based datastore migration as a separate reviewed
infrastructure change.
