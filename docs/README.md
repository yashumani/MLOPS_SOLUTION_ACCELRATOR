# V3 Operational Documentation

Current as of: 2026-08-02
Repository: `SAVYMINDS/YS_MVP`
Branch: `codex_ys/mlops-pipeline-correctness`

This folder is the operational documentation set for the MLOps Solution Accelerator V3. It explains the current Azure ML pipeline, active stage contracts, expected artifacts, operations, and validation posture without treating historical V2 or early V3 claims as current proof.

Important placement note: `docs/` is excluded from Azure ML code uploads by `.amlignore`, so documentation updates do not change the code snapshot submitted to Azure ML jobs.

## Start Here

| Need | Document |
|---|---|
| Understand the active pipeline architecture | `design_overview.md` |
| Understand every pipeline stage and function | `PIPELINE_STAGES.md` |
| Understand component inputs, outputs, and artifacts | `PIPELINE_IO_CONTRACTS.md` |
| Configure datasets, phases, recipes, registry, and drift | `CONFIGURATION_REFERENCE.md` |
| Submit, dry-run, monitor, and download Azure ML jobs | `SUBMISSION_GUIDE.md` |
| Operate drift and auto-retrain safely | `AUTO_RETRAIN_OPERATING_LEDGER.md` and `OPERATIONAL_RUNBOOKS/auto_retrain.md` |
| See current requirements and acceptance criteria | `PROJECT_REQUIREMENTS.md` |

## Current Canonical References

| Document | Purpose |
|---|---|
| `PIPELINE_STAGES.md` | Stage-by-stage guide for `s01` through terminal `s14`, with `s00`, `s07`, and `s11` marked inactive/reserved. |
| `PIPELINE_IO_CONTRACTS.md` | Pipeline-level inputs/outputs, stage artifacts, drift baseline contract, and retrain decision ledger shape. |
| `CONFIGURATION_REFERENCE.md` | Config sections, task isolation, recipe rules, Phase A/B/C settings, registry gates, and drift baseline chaining. |
| `SUBMISSION_GUIDE.md` | Canonical `submit_pipeline.py` usage, dry-run, baseline-chained submissions, monitoring, downloads, and troubleshooting. |
| `PROJECT_REQUIREMENTS.md` | Current functional and non-functional requirements adapted from workspace notes and corrected for the active `s14` graph. |
| `AUTO_RETRAIN_OPERATING_LEDGER.md` | Safe auto-retrain architecture, current Azure evidence, ledger rules, controller flow, and next validation. |
| `OPERATIONAL_RUNBOOKS/` | Operator runbooks for monitoring jobs, resubmitting failed runs, and auto-retrain operations. |

## Supporting Production Docs

| Document | Purpose |
|---|---|
| `DRIFT_DETECTION.md` | Detailed drift monitoring design and s13/s14 split. |
| `FASTAPI_INTEGRATION.md` | API service contract for UI and automation layer. |
| `DEPENDENCIES.md` | Dependency reproducibility workflow and SBOM notes. |
| `PRODUCTION_FREEZE_SUMMARY.md` | Historical freeze status from 2026-05-01. Check current stage docs before using as operational truth. |
| `COMMIT_LEDGER_20260501.md` | Commit-by-commit ledger for the production branch relative to `origin/main` at the freeze point. |
| `PRODUCTION_HANDOFF.md` | Handoff notes from the earlier freeze. Use `SUBMISSION_GUIDE.md` for current submit commands. |
| `PHASE1_DOCUMENTATION.md` | Historical January 2026 snapshot. It contains superseded stage/task details and is not an active contract. |
| `azure_devops_integration.md` | Azure DevOps integration notes; not the canonical Azure ML submission path. |
| `MLOPS-v3-blueprint.CSV` | Pipeline blueprint source material. |
| `MLOps Pipeline Blueprint - DataSet Selection-sources.csv` | Dataset source blueprint. |
| `blueprints/` | Production and post-production blueprint CSVs. |

## Current Active Pipeline Status

The active product supports only classification, regression, and clustering. Candidate selection uses comparable training/CV evidence; `s10` evaluates one frozen champion once on the locked Stage 2 test partition. `s13` emits evidence, `s14` emits policy decisions, and only the external controller may submit another run through `pipelines/submit_pipeline.py`.

Active wired stages:

```text
s01, s02, s03, s04, s05a, s05b, s05z, s06, s08, s09, s10, s12, s13, s14
```

Reserved/inactive identifiers:

| ID | Status |
|---|---|
| `s00` | Component/script exist, not wired in current pipeline graph. |
| `s05t` | Legacy forecasting files may exist, but the stage is not wired and forecasting is outside product scope. |
| `s07` | Not active in current `pipeline_builder.py`. |
| `s11` | Not active in current `pipeline_builder.py`. |

Current terminal step: `s14` retrain decision gate.

## Current Validation Status

| Proof level | Current status |
|---|---|
| Local unit/contract tests | Current-checkout preflight only; not Azure proof. |
| Azure ML SDK dry-runs | Classification, regression, and clustering graph construction passed; no job execution. |
| Azure read plane | Workspace and compute inventory were readable. |
| Exact-source Azure pipeline | Blocked before job creation by `ReadOnlyDisabledSubscription`; no current-revision Azure runtime proof. |
| Registered model | No current-revision raw-input registration proof. |
| Deployed inference | No current-revision endpoint proof. |

Named May 2026 drift and rotation jobs in the operating ledger are historical Azure evidence for earlier revisions. They must not be presented as proof for the current dirty checkout or any deployment.

## Documentation Rules

- Prefer current code and Azure outputs over old narrative docs.
- Do not copy V2 docs into active docs unless they are explicitly marked historical.
- Update `PIPELINE_STAGES.md` and `PIPELINE_IO_CONTRACTS.md` whenever component wiring or outputs change.
- Update `SUBMISSION_GUIDE.md` whenever CLI flags or monitoring procedures change.
- Update `AUTO_RETRAIN_OPERATING_LEDGER.md` whenever auto-retrain validation state changes.

## Archived Docs

Historical documents are archived at:

```text
/home/azureuser/cloudfiles/code/Users/yashu.savyminds/archive/mlops-solution-accelerator-v3-docs-archive-20260501/
```

Keep future development notes in an external archive or a clearly marked draft folder, not mixed into production docs.
