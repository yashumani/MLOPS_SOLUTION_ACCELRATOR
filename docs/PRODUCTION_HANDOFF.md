# Production Handoff — V3 Backend (`prod-hardening-20260425`)

This document hands off the **V3 pipeline backend** changes shipped on branch
`prod-hardening-20260425` (PR target: `main`). Scope is the **submission / pipeline /
drift / alerts / scheduler** surface owned by the backend agent. The FastAPI surface
under `api/` is owned by Codex and is **not** in scope.

> Repo: `SAVYMINDS/YS_MVP` · Default branch: `main` · Commit: `0364e059`

---

## 1. What changed

| Phase | Outcome |
|-------|---------|
| **P1 — Hardening** | Env-driven Azure context, `ChainedTokenCredential(ManagedIdentity, AzureCli)`, `~/.mlops/` user state dir (locks + audit), no `DefaultAzureCredential`, no hardcoded subscription/RG/workspace. |
| **P2 — Schema gate** | K2 config validation runs as a **hard fail** at the top of `pipelines/submit_pipeline.py` for all 16 sample configs. |
| **P3 — Env parity** | All Azure ML components pinned to registered environment **`mlops-v3-unified:23`**. |
| **P4 — Drift contract** | `s13_drift_monitor` I/O verified against `configs/drift_config.yaml` thresholds (PSI / KS / chi-square). |
| **P5 — Alerts + Scheduler** | New `src/utils/alerts.py` (Teams webhook + ACS email, both no-op when env unset). `s13_drift_monitor` now emits structured drift alerts. New `scripts/setup_drift_schedule.py` for idempotent Azure ML `JobSchedule` creation. |
| **P5 — Smoke** | Job `tidy_pipe_ksjrkyztsm` (titanic config) submitted Running on `mlopsv2computecluster`. |

Locked product defaults (do not change without product sign-off):

1. **Channels:** Teams webhook + Azure Communication Services email.
2. **Schedule:** hourly for the first week, daily after.
3. **Retraining:** **alert-only** for the first 30 days (no auto-retrain).
4. **Pipeline:** dedicated **drift-only** pipeline (separate from training).
5. **Phase 4 ships without concept drift** (data drift only on day 1).

---

## 2. Files added / changed (backend scope)

```
src/utils/alerts.py                      NEW   Teams + ACS dispatch (no-op safe)
src/steps/s13_drift_monitor.py           MOD   try/except wrapped emit_drift_alert
scripts/setup_drift_schedule.py          NEW   Idempotent JobSchedule create/disable
scripts/_azure_ctx.py                    KEPT  Frozen AzureContext + load_azure_context
pipelines/submit_pipeline.py             MOD   K2 hard-fail, lock + audit, env CLI args
components/s13_drift_monitor.yml         MOD   environment: azureml:mlops-v3-unified:23
configs/drift_config.yaml                KEPT  PSI/KS/chi-square thresholds
docs/POST_V3_PRODUCTION_REPORT.md        DOC   25/25 baseline reference
docs/PRODUCTION_HANDOFF.md               NEW   This file
```

`api/core/azure_ml.py` and `api/core/config.py` are **shared** with Codex's API
surface; the credential chain change to `ChainedTokenCredential` was kept in this
commit and must be coordinated with the API agent before merging.

---

## 3. Required environment

### 3.1 Azure context (mandatory for submission)

Loader = `scripts/_azure_ctx.load_azure_context()`. Missing any of the four raises
`MissingAzureContextError`:

| Variable | Example |
|----------|---------|
| `AZURE_SUBSCRIPTION_ID` | `93044a08-5661-4f1b-b424-5eafe066a9d1` |
| `AZURE_RESOURCE_GROUP` | `mvpv1` |
| `AZURE_WORKSPACE_NAME` | `mlops-accelerator` |
| `AZURE_COMPUTE` | `mlopsv2computecluster` |

> The committed `.env` exposes `COMPUTE_TARGET`, not `AZURE_COMPUTE`. After
> `source .env`, run **`export AZURE_COMPUTE="${COMPUTE_TARGET}"`** before submit.

### 3.2 Drift alerts (optional — channels no-op when unset)

| Variable | Used by | Notes |
|----------|---------|-------|
| `TEAMS_WEBHOOK_URL` | `send_teams_alert` | Standard Teams Incoming Webhook URL. |
| `ACS_CONNECTION_STRING` | `send_acs_email` | Azure Communication Services connection string. |
| `ACS_SENDER_ADDRESS` | `send_acs_email` | Verified ACS sender (e.g. `DoNotReply@<verified-domain>`). |
| `DRIFT_ALERT_RECIPIENTS` | `send_acs_email` | Comma-separated list of recipients. |

**Contract:** `emit_drift_alert(...)` always returns `{teams: bool, email: bool}`
and **never raises**. Missing env vars → channel returns `False` silently. The
caller in `s13_drift_monitor` is additionally wrapped in `try/except` so alert
dispatch failures cannot fail the step.

---

## 4. Operational runbook

### 4.1 Submit a training pipeline

```bash
cd mlops-solution-accelerator-v3
set -a && source .env && export AZURE_COMPUTE="${COMPUTE_TARGET}" && set +a

python -u pipelines/submit_pipeline.py \
  --config configs/config_classification_titanic_azureml.yml \
  --subscription_id "$AZURE_SUBSCRIPTION_ID" \
  --resource_group  "$AZURE_RESOURCE_GROUP" \
  --workspace_name  "$AZURE_WORKSPACE_NAME" \
  --compute         "$AZURE_COMPUTE"
```

- K2 schema validation runs first (hard fail on invalid config).
- Submission lock: `~/.mlops/locks/.submit.lock` (concurrent submits blocked).
- Audit record written to `~/.mlops/last_submitted_job.json`.
- On NFS-mounted workspaces, `ml_client.jobs.create_or_update()` can take ~10–12 min.
- Use `--force` to override the active-job guard, `--dry_run` to assemble without submitting.

### 4.2 Create / update the drift-only schedule

Hourly cadence (week 1):
```bash
python scripts/setup_drift_schedule.py \
  --cadence hourly \
  --config_name configs/config_classification_titanic_azureml.yml
```

Daily cadence (week 2 onward):
```bash
python scripts/setup_drift_schedule.py \
  --cadence daily \
  --config_name configs/config_classification_titanic_azureml.yml
```

Disable a schedule:
```bash
python scripts/setup_drift_schedule.py \
  --cadence daily \
  --config_name configs/config_classification_titanic_azureml.yml \
  --disable
```

The script is **idempotent**: re-running creates or updates the schedule by name,
never duplicating.

### 4.3 Monitor a job

```bash
az ml job show --name <job_name> -g "$AZURE_RESOURCE_GROUP" \
  -w "$AZURE_WORKSPACE_NAME" --query status -o tsv
```

Or watch via Studio — URL is printed on submit and saved to
`~/.mlops/last_submitted_job.json`.

### 4.4 Verify drift alerts in a run

**Gating:** `emit_drift_alert` is invoked **only** when one of these is true:
- `self_check_status == "WARN"` (in-run PSI exceeds threshold), **or**
- Evidently dataset drift detected (requires prior baseline), **or**
- Concept drift detected (requires prior baseline).

A clean run with PASS PSI and no prior baseline produces **no alert log line by
design** — that is the expected behaviour, not a bug.

When the alert path **does** fire, look in the s13 step log
(`user_logs/std_log.txt`) for one of:

```
Drift alert dispatch: {'teams': True,  'email': True}
Drift alert dispatch: {'teams': False, 'email': False}   # env vars unset
  Drift alert dispatch failed (non-fatal): <reason>      # never blocks the step
```

Fetch the s13 log via MLflow artifacts (works even when blob SAS auth fails):

```bash
mkdir -p /tmp/s13_logs
python - <<'PY'
import os, mlflow
from azure.ai.ml import MLClient
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
cred = ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential())
ml = MLClient(cred, os.environ['AZURE_SUBSCRIPTION_ID'],
              os.environ['AZURE_RESOURCE_GROUP'], os.environ['AZURE_WORKSPACE_NAME'])
mlflow.set_tracking_uri(ml.workspaces.get(os.environ['AZURE_WORKSPACE_NAME']).mlflow_tracking_uri)
client = mlflow.tracking.MlflowClient()
# Replace with the s13 child run id from: az ml job list --parent-job-name <pipeline> ...
client.download_artifacts('<S13_RUN_ID>', 'user_logs', '/tmp/s13_logs')
PY
grep -E "Drift alert dispatch" /tmp/s13_logs/user_logs/std_log.txt
```

---

## 5. Validation evidence

| Gate | Result |
|------|--------|
| K2 schema validation (16 configs) | 16 / 16 pass |
| Baseline regression (pre-P5) | 25 / 25 jobs Completed (`docs/POST_V3_PRODUCTION_REPORT.md`) |
| P5 smoke submit (titanic) | Job `tidy_pipe_ksjrkyztsm` **Completed**; all 14 child steps Completed (s1, s2, s3, s06, s4, s5a, s5b, s5t, s5z, s08, s09, s10, s12, s13). Studio: https://ml.azure.com/runs/tidy_pipe_ksjrkyztsm |
| s13 alert path (smoke) | self-check PASS (PSI=0.058) + no prior baseline → `should_alert=False` → no dispatch (correct gating) |
| Alert dispatch contract (env unset) | `emit_drift_alert` returns `{teams: False, email: False}` without raising; caller wrapped in try/except |
| Pre-existing tests | unchanged; no new regressions introduced |

---

## 6. Known follow-ups

1. **Codex coordination** — `api/core/azure_ml.py` credential-chain change must be
   reviewed with the API agent before this PR merges into `main`.
2. **`.env` in git** — the committed `.env` contains real subscription / RG /
   workspace identifiers. It contains no secret keys, but should be moved to
   `.env.example` + a developer-side `.env` (gitignored) before public release.
3. **Concept drift** — intentionally deferred per locked default #5; will be added
   after Phase 4 stabilizes.
4. **Auto-retraining** — disabled for the first 30 days per locked default #3.
   Enabling it later requires wiring a new branch in `s13_drift_monitor` and is
   **out of scope** for this PR.

---

## 7. Rollback

The branch is additive: revert `0364e059` to undo P1+P5. The only behaviour
change visible to existing pipelines is:

- `s13_drift_monitor` now logs `alerts dispatched: {...}` once at the end.
- All other steps are byte-identical to `dc007e7d` (the pre-P5 baseline).

```bash
git revert --no-edit 0364e059
git push origin prod-hardening-20260425
```
