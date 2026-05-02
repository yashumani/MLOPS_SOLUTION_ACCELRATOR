# V3 Production Documentation

Current as of: 2026-05-01  
Repository: `SAVYMINDS/YS_MVP`  
Branch: `prod-hardening-20260425`  
Freeze head commit: `b526b4fb`

This folder is the production-facing documentation set for the V3 MLOps Solution Accelerator. Development notes, PR drafts, historical incident writeups, and session logs have been moved out of this folder so operators see only current guidance.

Important placement note: `docs/` is excluded from Azure ML code uploads by `.amlignore`, so documentation updates do not change the code snapshot submitted to Azure ML jobs.

## Current Docs

| Document | Purpose |
|---|---|
| `PRODUCTION_FREEZE_SUMMARY.md` | Holistic freeze status, current gates, thresholds, warnings, and latest job resubmissions. |
| `COMMIT_LEDGER_20260501.md` | Commit-by-commit ledger for this production branch relative to `origin/main`. |
| `PRODUCTION_HANDOFF.md` | Operator handoff for submitting, monitoring, and governing V3 jobs. |
| `design_overview.md` | Current Azure ML component pipeline architecture. |
| `usage_guide.md` | Azure-only usage guide for production submissions and monitoring. |
| `DRIFT_DETECTION.md` | Drift monitoring design, thresholds, artifacts, and alert behavior. |
| `FASTAPI_INTEGRATION.md` | API service contract for the UI and automation layer. |
| `DEPENDENCIES.md` | Dependency reproducibility workflow and SBOM notes. |
| `azure_devops_integration.md` | Azure DevOps integration notes. |
| `MLOPS-v3-blueprint.CSV` | Pipeline blueprint. |
| `MLOps Pipeline Blueprint - DataSet Selection-sources.csv` | Dataset source blueprint. |
| `blueprints/` | Production and post-production blueprint CSVs. |

## Archived Docs

Historical documents were moved to:

`/home/azureuser/cloudfiles/code/Users/yashu.savyminds/archive/mlops-solution-accelerator-v3-docs-archive-20260501/`

Archived content includes PR draft text, session history, older branch reviews, superseded hardening reports, and stale clustering failure forensics. Keep future development notes in an external archive or a clearly marked draft folder, not in this production docs folder.