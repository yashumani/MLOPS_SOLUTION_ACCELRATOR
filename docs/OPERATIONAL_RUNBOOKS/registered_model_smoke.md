# Registered Model Inference Smoke

Use this check after a governed qualification pipeline reaches `Completed`.
It downloads the parent's `registry_info`, requires an exact numeric model
version, and submits a small Azure ML command job whose uploaded code contains
only the smoke scorer. The repository's `src/utils` package is intentionally
absent, so a model that omitted its runtime modules fails during deserialization.

## Preconditions

- The parent is tagged as a qualification scenario and is `Completed`.
- The feature-branch worktree is clean.
- Azure CLI authentication can access the workspace.
- These variables identify the non-production qualification workspace:

```powershell
$env:AZURE_SUBSCRIPTION_ID = '<subscription-id>'
$env:AZURE_RESOURCE_GROUP = '<resource-group>'
$env:AZURE_WORKSPACE_NAME = '<workspace-name>'
$env:AZURE_COMPUTE = '<compute-name>'
```

## Submit

```powershell
python scripts/submit_registered_model_smoke.py `
  --parent-job <completed-parent-job> `
  --output-datastore mlops_blob
```

Add `--wait` when the caller should block until Azure reports a terminal state.
The output datastore defaults to `mlops_blob`; the submitter binds every run to
a unique path under `qualification/registered-model-smoke`. This avoids the
workspace default datastore and records the exact evidence URI in the local
submission record.
The default evidence record is written under
`$MLOPS_STATE_DIR/registered_model_smokes`, or `~/.mlops` when that variable is
not set.

## Acceptance

The smoke job must reach `Completed`, and its
`registered_model_inference_smoke.json` output must report:

- `status: passed`;
- the exact model name and numeric version from `registry_info`;
- matching execution, source-code, and dataset SHA-256 identities;
- an MLflow signature and saved raw input example;
- matching input and prediction row counts;
- `current_stage: None`, no protected alias, and manual promotion tags.

This proves registered-model loading and raw-input prediction. It does not prove
a deployed endpoint, production traffic, or production model promotion.
