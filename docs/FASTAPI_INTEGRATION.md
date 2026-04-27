# FastAPI Integration — V3 Pipeline

> **Service:** `uvicorn api.main:app` on `:8000`
> **Branch:** `FAST-API-v1`
> **Auth:** Header `X-API-Key: <redacted>` (configured with the `API_KEY` environment variable)
> **OpenAPI:** `GET /openapi.json` · interactive docs at `/docs`

---

## 1. Purpose

The FastAPI layer wraps the synchronous Azure ML SDK so the React/Streamlit UI and external automation can:
- enumerate experiments and jobs,
- fetch metrics, drift, outputs, logs, and Studio URLs,
- submit and resubmit pipeline runs,
all from a stable HTTP contract instead of importing `azure.ai.ml` directly.

---

## 2. Architecture

```
┌────────┐   X-API-Key    ┌──────────────────┐  asyncio.run_in_executor   ┌──────────────────┐
│  UI /  │ ──────────────▶│   FastAPI (api/) │ ──────────────────────────▶│ Azure ML SDK     │
│ caller │                │   uvicorn :8000  │                            │ (sync MLClient)  │
└────────┘                └──────────────────┘                            └──────────────────┘
                                  │
                                  ├── ThreadPoolExecutor(4)  → parallel report downloads
                                  └── BackgroundWarmer task  → /experiments cache (TTL 120 s)
```

- Sync SDK calls run in the default executor — keeps the event loop responsive.
- Long-running fan-outs (e.g. job-output downloads) use a 4-worker `ThreadPoolExecutor`.
- The experiments cache is a thread-safe in-memory dict guarded by `threading.Lock`.

---

## 3. Authentication

All `/api/v1/pipelines/*` routes require the header:

```
X-API-Key: <redacted>
```

`/api/v1/health` is unauthenticated for readiness probes. The legacy path `/health` returns 404 — always use `/api/v1/health`.

---

## 4. API Surface (17 routes)

| Method | Path | Notes |
|--------|------|-------|
| GET    | `/api/v1/health` | unauthenticated |
| GET    | `/api/v1/pipelines/experiments` | cached; supports `max_results_per_experiment`, `force_refresh` |
| POST   | `/api/v1/pipelines/experiments/refresh` | 202; fires off a background refresh |
| GET    | `/api/v1/pipelines/jobs/{job_name}` | parent job summary |
| GET    | `/api/v1/pipelines/jobs/{job_name}/children` | step-level breakdown |
| GET    | `/api/v1/pipelines/jobs/{job_name}/metrics` | per-phase model metrics |
| GET    | `/api/v1/pipelines/jobs/{job_name}/drift` | parses `drift_report` (see §6) |
| GET    | `/api/v1/pipelines/jobs/{job_name}/outputs` | lists named outputs |
| GET    | `/api/v1/pipelines/jobs/{job_name}/outputs/{name}/content` | inline JSON / CSV / text preview |
| GET    | `/api/v1/pipelines/jobs/{job_name}/logs` | aggregated step logs |
| GET    | `/api/v1/pipelines/jobs/{job_name}/studio` | Azure ML Studio deep link |
| POST   | `/api/v1/pipelines/submit` | submit new pipeline |
| POST   | `/api/v1/pipelines/jobs/{job_name}/resubmit` | clone-and-submit |
| GET    | `/api/v1/pipelines/configs` | list available config YAMLs |
| GET    | `/api/v1/pipelines/configs/{name}` | fetch single config |
| GET    | `/api/v1/pipelines/datasets` | list registered datasets |
| GET    | `/api/v1/pipelines/compute` | list compute targets |

> The exact, generated list is always at `GET /openapi.json`.

---

## 5. Phase 1 — Experiments Cache Preload

### Why
Cold list of experiments+jobs is **~58–63 s** (272 KB payload, 103 experiments / 594 jobs at time of writing). Blocking the UI on every page load was unacceptable.

### Configuration ([api/core/config.py](../api/core/config.py))

| Field | Default | Description |
|-------|---------|-------------|
| `experiment_cache_enabled` | `True` | Master switch |
| `experiment_cache_preload_count` | `20` | `max_results_per_experiment` used by warmer |
| `experiment_cache_ttl_seconds` | `120` | How often warmer refreshes |

### Behavior
1. `lifespan()` ([api/main.py](../api/main.py)) creates one `MLClient` and spawns `_experiments_warm_loop` as `asyncio.create_task(...)`.
2. The loop runs `refresh_experiments_cache(preload_count)` in the executor, then `await asyncio.sleep(ttl)`.
3. `GET /experiments` returns the snapshot when `max_results_per_experiment` matches AND `force_refresh=false`; else triggers a live fetch.
4. Response headers: `X-Cache: HIT|MISS`, `X-Cache-Age` (seconds), `X-Cache-FetchedAt` (ISO Z).
5. `POST /experiments/refresh` returns `202 Accepted` with the previous cache metadata while a refresh runs in the background.

### Verified performance

| Scenario | Latency |
|----------|---------|
| Cache HIT | **0.006 s** |
| Cache MISS / force_refresh | **62.7 s** |
| Cache age (after 65 s sleep) | **63 s** |

---

## 6. Drift Endpoint — `GET /jobs/{job_name}/drift`

Parses the producer-side artifact `drift_report` written by [`src/steps/s13_drift_monitor.py`](../src/steps/s13_drift_monitor.py) (lines 498–530).

### Request

```
GET /api/v1/pipelines/jobs/happy_owl_sfmkgs2jrd/drift
X-API-Key: <redacted>
```

### Response (`DriftResponse`)

```jsonc
{
  "job_name": "happy_owl_sfmkgs2jrd",
  "overall_drift_detected": false,
  "stability_score": 65.0,
  "drift_type": "self_check",     // "comparison" | "self_check" | "psi" | null
  "drifted_columns": [],
  "features": [
    {"feature": "age",              "psi": 0.027702, "drift_detected": false, "severity": "none"},
    {"feature": "bmi",              "psi": 0.025250, "drift_detected": false, "severity": "none"},
    {"feature": "region_northeast", "psi": 0.007653, "drift_detected": false, "severity": "none"},
    "..."
  ],
  "evidently_report_path": null,
  "studio_url": "https://ml.azure.com/runs/happy_owl_sfmkgs2jrd?wsid=..."
}
```

### Severity thresholds (PSI)

| Range | Severity |
|-------|----------|
| `≥ 0.25` | `severe` |
| `≥ 0.10` | `moderate` |
| else | `none` |

### Drift-type resolution precedence

1. If `comparison_drift.available` is true → `comparison`.
2. Else if `self_check` present → `self_check` (overall = `status == "WARN"`).
3. Else → `psi` (overall = any feature ≥ 0.10).

---

## 7. Known Issues / Where We Are Stuck

| # | Item | Severity | Notes |
|---|------|----------|-------|
| 1 | Warmer log line `"warmed experiments cache: ..."` not visible in `/tmp/api.log` | low (cosmetic) | The `api.services.pipeline_service` logger has no handler attached and uvicorn does not configure the root logger. The cache *is* populated — verified via `X-Cache: HIT` and `X-Cache-Age` matching the warmer cycle. |
| 2 | Streamlit UI on `:8501` not in the API restart script | medium | Manual relaunch required after `pkill -f uvicorn`. |
| 3 | Drift download glob bug (historical) | resolved | `ml_client.jobs.download(output_name="drift_report")` writes a file literally named `drift_report` (no `.json`). Original parser globbed `*.json` and matched zero files. Fixed by also matching extensionless files with parseable JSON content. Cautionary precedent for any new endpoint that downloads named outputs. |
| 4 | No automated test against a running API | medium | Verification is currently manual `curl` against live workspace. A pytest fixture against a dev mock is the next reasonable step. |

---

## 8. Restart Recipe

```bash
cd /path/to/mlops-solution-accelerator-v3
pkill -f "uvicorn api.main"
sleep 2
nohup uvicorn api.main:app --host 127.0.0.1 --port 8000 > /tmp/api.log 2>&1 &
sleep 8 && tail -5 /tmp/api.log     # expect "Application startup complete."
# warmer first cycle completes ~60–70 s after startup
```

To verify cache is populating:

```bash
sleep 70
curl -sD- -H 'X-API-Key: <key>' \
  'http://localhost:8000/api/v1/pipelines/experiments?max_results_per_experiment=20' \
  -o /dev/null | grep -i x-cache
# Expect: x-cache: HIT, x-cache-age: ~63
```
