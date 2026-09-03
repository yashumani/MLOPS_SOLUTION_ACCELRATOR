# Runbook: Verify Qualification Evidence

Current as of: 2026-09-03

Use `scripts/verify_qualification_evidence.py` after a qualification wave is
terminal and each registered model has passed the raw-input inference smoke.
The command validates downloaded evidence only. It does not submit jobs,
change models, or query Azure.

## Evidence Manifest

Create one JSON manifest that binds the terminal monitor summaries, the live
data-asset audit, and every scenario's downloaded outputs:

```json
{
  "schema_version": "1.0",
  "monitor_summaries": [
    "wave-01/monitor-summary.json",
    "wave-02/monitor-summary.json"
  ],
  "data_asset_audit": "azure-data-asset-audit.json",
  "release_candidate": {
    "git_commit": "<40-character-git-sha>",
    "runtime_source_sha256": "<64-character-upload-source-sha>"
  },
  "scenarios": [
    {
      "scenario_id": "classification-healthcare-heart-disease",
      "parent_job": "<azure-parent-job-name>",
      "pipeline_evidence_dir": "classification-healthcare-heart-disease/pipeline",
      "registered_model_smoke_submission": "classification-healthcare-heart-disease/smoke-submission.json",
      "registered_model_smoke_evidence": "classification-healthcare-heart-disease/smoke-evidence"
    }
  ]
}
```

Relative paths resolve from the evidence manifest directory. Absolute paths
are also accepted for an operator-owned evidence workspace.

Each `pipeline_evidence_dir` must contain exactly one file under each of these
downloaded `named-outputs` directories:

- `execution_manifest`
- `split_manifest`
- `quality_decision`
- `final_report`
- `registry_info`
- `drift_report`
- `retrain_decision`
- `decision_ledger_record`

The Stage 6 `quality_decision` is intentionally a selection-only block. Stage
10 owns the single locked-test evaluation and emits the final `pass`, `warn`,
or `block` decision. A valid release scenario must have a Stage 10 `pass` or
`warn` decision that permits exact-version registration.

## Verify A Partial Wave

```powershell
python scripts\verify_qualification_evidence.py `
  --manifest '<qualification-evidence-manifest.json>' `
  --output-json '<qualification-evidence-report.json>'
```

Exit code `0` means every scenario included in that manifest passed its
artifact contract. It does not mean the 15-scenario release matrix is complete.
The report will keep `release_matrix_accepted` set to `false`.

## Verify The Release Matrix

```powershell
python scripts\verify_qualification_evidence.py `
  --manifest '<qualification-evidence-manifest.json>' `
  --output-json '<qualification-evidence-report.json>' `
  --require-complete-matrix
```

Final mode fails unless all of the following are true:

- The manifest contains exactly the 15 scenarios in the canonical catalog.
- Classification, regression, and clustering each contain five scenarios and
  five distinct industries.
- Every parent job is `Completed` in a passing canonical monitor summary.
- The live data-asset audit covers every catalog scenario and matches both its
  content and schema hashes.
- Split, execution, final evaluation, model bundle, MLflow, registry, drift,
  S14, and ledger identities agree.
- Selection explicitly excludes the locked test and the frozen champion is
  evaluated on it exactly once.
- Each exact numeric registered model version passes raw-input inference and
  retains its lineage tags.
- No scenario shows a model alias, lifecycle-stage transition, or promotion.
- Every scenario matches the declared release Git commit and immutable upload
  source hash.

Only exit code `0` together with `release_matrix_accepted: true` is matrix-level
artifact acceptance. This remains separate from datastore health, schedule
containment, API topology, CI, production deployment, and model-promotion
approval.

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Requested evidence scope passed. |
| `1` | Evidence was readable but one or more acceptance contracts failed. |
| `2` | The manifest, catalog, or required input could not be interpreted. |
