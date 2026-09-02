# Operational Runbooks

Current as of: 2026-08-02

These runbooks are operator-focused. They require all submissions to go through `pipelines/submit_pipeline.py`; they do not imply that the current checkout is Azure-validated. Local tests, SDK dry-runs, exact-source Azure jobs, registered models, and deployed inference must be reported as separate proof levels.

## Runbook Index

| Runbook | Purpose |
|---|---|
| `monitor_jobs.md` | Check parent job status, child steps, and download artifacts. |
| `resubmit_failed_jobs.md` | Diagnose and resubmit failed configs safely. |
| `auto_retrain.md` | Operate baseline chaining, s13/s14 outputs, and controller dry-runs. |

## Workspace Defaults

| Setting | Value |
|---|---|
| Subscription | `93044a08-5661-4f1b-b424-5eafe066a9d1` |
| Resource group | `mvpv1` |
| Azure ML workspace | `mlops-accelerator` |
| Compute | `mlopsv2computecluster` |
| Runtime | `/anaconda/envs/mlops_pipeline_v2/bin/python` |

Use explicit parameters in commands. Do not rely on hidden local defaults.
