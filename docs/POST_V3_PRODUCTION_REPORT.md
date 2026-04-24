# Post-V3 Production Launch — Engineering Status Report

> **Audience:** Engineering, Product, Stakeholders
> **Branch:** `FAST-API-v1`
> **Workspace:** `mlops-accelerator` (RG `mvpv1`, sub `93044a08-...`)
> **Reporting window:** Items completed since V3 production cutover.
> **Last verified runtime:** `happy_owl_sfmkgs2jrd` (regression_insurance_v3).

---

## 1. Executive Summary

Two production-impacting workstreams have advanced since V3 launched:

| # | Workstream | Status | User-visible impact |
|---|------------|--------|---------------------|
| 1 | **FastAPI integration layer** (`api/`) over the existing Azure ML pipeline | **Operational** on `localhost:8000` | Programmatic + UI access to 17 endpoints; sub-10 ms experiment list |
| 2 | **Drift Detection (s13)** producer → API consumer wiring | **Operational end-to-end** | `GET /jobs/{job}/drift` now returns PSI per feature, stability score, drift type |

Both items had at least one regression caught and fixed during verification (documented under §5 *Verification Results*).

---

## 2. Phase 1 — Experiments Cache Preload (FastAPI)

### Why
Cold `GET /api/v1/pipelines/experiments` was taking **~58–63 s** because the Azure ML SDK enumerates every experiment and lists jobs serially. This blocked the UI on every page load.

### What was built
- Background **warmer task** spawned on FastAPI startup (`api/main.py` lifespan).
- Thread-safe in-memory cache populated every `experiment_cache_ttl_seconds` (default **120 s**).
- `GET /experiments` returns from cache when `max_results_per_experiment` matches; otherwise falls back to a live fetch.
- `POST /experiments/refresh` (202 Accepted) forces a refresh out-of-band.
- Response headers expose cache state: `X-Cache: HIT|MISS`, `X-Cache-Age`, `X-Cache-FetchedAt`.

### Files modified
| File | Change |
|------|--------|
| [api/core/config.py](../api/core/config.py) | Added `experiment_cache_enabled`, `experiment_cache_preload_count`, `experiment_cache_ttl_seconds` |
| [api/services/pipeline_service.py](../api/services/pipeline_service.py) | Added `_experiments_cache`, `_experiments_lock`, `refresh_experiments_cache()`, `get_cached_experiments()` |
| [api/main.py](../api/main.py) | Added `_experiments_warm_loop()` coroutine + lifespan registration |
| [api/routers/pipelines.py](../api/routers/pipelines.py) | Cache-aware `/experiments` GET + `/experiments/refresh` POST |

### Expected vs Actual

| Metric | Expected | Actual (verified) |
|--------|----------|-------------------|
| Cache HIT latency | < 50 ms | **6.3 ms** |
| Cache MISS / cold fetch | 50–70 s | **62.7 s** |
| Cache age after TTL+1 sleep | ~60–125 s | **63 s** |
| Payload size (max=20) | ~270 KB | **272,691 B** |

> Verified via `curl -D-` against live API (PID 285639 at time of test).

---

## 3. Phase 2 — Drift Detection End-to-End

### Pipeline producer (`src/steps/s13_drift_monitor.py`) — read-only

s13 is the terminal step in the V3 DAG. On every pipeline run it:

1. Loads the s04 feature-engineered dataset.
2. Runs a self-check PSI on an 80/20 split (expected ≈ 0 — validates the detector).
3. Computes per-feature PSI, max PSI, drifted-feature list, status (`PASS|WARN`).
4. Computes a stability score 0–100 + recommended retraining cadence.
5. *(Optional)* Compares against a previous-run baseline via Evidently / concept drift.
6. Writes:
   - `drift_report` (JSON dict — see schema below)
   - `drift_baseline` (folder for chaining into future runs)

#### Producer JSON schema (the contract the API consumes)

```jsonc
{
  "execution_id": "s13_<dataset>_<epoch>",
  "config_name": "config_regression_insurance_azureml.yml",
  "task_type": "regression",
  "dataset_name": "insurance",
  "n_rows": 1338,
  "n_features": 11,
  "target_column": "charges",
  "detector": "psi",
  "self_check": {
    "method": "train_test_split_80_20_seeded",
    "overall_psi": 0.012,
    "max_feature_psi": 0.027,
    "max_feature_name": "age",
    "drifted_features": [{"feature": "...", "psi": ..., "severity": "..."}],
    "n_drifted": 0,
    "status": "PASS"
  },
  "feature_psi_scores": {"age": 0.027702, "bmi": 0.02525, ...},
  "stability_assessment": {
    "stability_score": 65.0,
    "components": {...},
    "recommended_cadence": "monthly",
    "recommended_days": 30,
    "rationale": "..."
  },
  "champion_info": {...},
  "comparison_drift": {"available": false},
  "warnings": [],
  "runtime_seconds": 12.4
}
```

### API consumer (`api/services/pipeline_service.py::get_job_drift`)

Surfaces the s13 report through `GET /api/v1/pipelines/jobs/{job_name}/drift`.

#### Response schema (`DriftResponse`)

| Field | Type | Source in s13 report |
|-------|------|----------------------|
| `job_name` | `str` | request parameter |
| `overall_drift_detected` | `bool` | `comparison_drift.evidently.dataset_drift` → `self_check.status == WARN` → any feature PSI ≥ 0.10 |
| `stability_score` | `float \| null` | `stability_assessment.stability_score` |
| `drift_type` | `"comparison" \| "self_check" \| "psi" \| null` | resolved by precedence above |
| `drifted_columns` | `list[str]` | `evidently.drifted_columns` → `self_check.drifted_features` → derived from PSI ≥ 0.10 |
| `features` | `list[DriftResultItem]` | `feature_psi_scores` mapped to `(feature, psi, drift_detected, severity)` |
| `evidently_report_path` | `str \| null` | `report.evidently_report_path` |
| `studio_url` | `str` | constructed |

Severity thresholds (PSI): `≥ 0.25` severe · `≥ 0.10` moderate · else none.

### Bugs found & fixed during verification

| # | Symptom | Root cause | Fix location |
|---|---------|------------|--------------|
| 1 | Drift response keys wrong (`psi_scores`, `drifted_columns` as objects) | Parser was written against an older schema | Rewritten parser ([api/services/pipeline_service.py](../api/services/pipeline_service.py) lines ~727–905) — supports nested-dict per-feature values + 3-level fallback chain |
| 2 | All completed jobs returned `features: []` even after fix #1 | `ml_client.jobs.download(output_name="drift_report")` produces a file literally named `drift_report` (no extension). The parser globbed `rglob("*.json")` → matched zero files | Now globs `*.json` first, then any extension-less file with parseable JSON content ([api/services/pipeline_service.py](../api/services/pipeline_service.py) ~line 770–786) |

### Expected vs Actual (regression_insurance_v3 / `happy_owl_sfmkgs2jrd`)

| Field | Expected | Before fix | After fix (verified) |
|-------|----------|-----------|----------------------|
| `features` count | 11 (one per feature) | 0 | **11** |
| `stability_score` | numeric | `null` | **65.0** |
| `drift_type` | `self_check` (no baseline supplied) | `null` | **`self_check`** |
| Top feature | `age` (highest PSI) | n/a | **`age`, PSI 0.0277, severity `none`** |

---

## 4. Drift Detection Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ s04 feat_eng│───▶│             │    │ baseline_in │ (prior run, optional)
└─────────────┘    │  s13 drift  │◀───┘             │
                   │   monitor   │
┌─────────────┐    │             │───▶ drift_report (JSON)
│ s11 final   │───▶│             │───▶ drift_baseline (folder, for next run)
└─────────────┘    └─────────────┘
                           │
                           ▼
                  ┌──────────────────────────────────┐
                  │ FastAPI /jobs/{j}/drift consumer │
                  │ → DriftResponse JSON              │
                  └──────────────────────────────────┘
```

s13 is **terminal** in the DAG — nothing depends on it. Its `drift_baseline` folder feeds the *next* pipeline run as `baseline_in`, enabling Evidently/concept drift comparisons over time.

---

## 5. Verification Results (raw)

```bash
# Cache HIT (warmed)
$ curl -D- '/api/v1/pipelines/experiments?max_results_per_experiment=20'
HTTP/1.1 200 OK
x-cache: HIT
x-cache-age: 63
TIME:0.006314

# Cache MISS (force_refresh)
$ curl -D- '/api/v1/pipelines/experiments?...&force_refresh=true'
HTTP/1.1 200 OK
x-cache: MISS
TIME:62.682957

# Drift endpoint (after fix)
$ curl '/api/v1/pipelines/jobs/happy_owl_sfmkgs2jrd/drift'
{
  "job_name": "happy_owl_sfmkgs2jrd",
  "overall_drift_detected": false,
  "stability_score": 65.0,
  "drift_type": "self_check",
  "features": [11 entries; top: age PSI=0.0277, bmi PSI=0.0253, ...]
}
```

---

## 6. Known Issues & Where We Are Stuck

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Warmer log line `"warmed experiments cache..."` not visible in stdout | **Cosmetic** | The `api.services.pipeline_service` logger has no handler; uvicorn does not configure root. Cache *is* populated (proven by `X-Cache: HIT` + `X-Cache-Age` matching warmer cycle). Fix: add `logging.getLogger("api").addHandler(...)` or use uvicorn's logger. |
| 2 | Streamlit UI on `:8501` not in restart script | **Operational gap** | Manual relaunch needed after API restarts. |
| 3 | `comparison_drift` always `{"available": false}` for current jobs | **Expected** | No prior baseline yet supplied via `baseline_in`. Will populate once chained-run baselines exist. |
| 4 | Drift download glob bug (extensionless file) | **FIXED this round** | Documented as cautionary: Azure ML's `jobs.download()` for a `uri_file` output preserves the file's original name (no `.json` suffix even though content is JSON). |

---

## 7. Files Touched This Round (post-launch)

| File | Type | Purpose |
|------|------|---------|
| [api/core/config.py](../api/core/config.py) | NEW fields | Cache configuration |
| [api/services/pipeline_service.py](../api/services/pipeline_service.py) | ENHANCEMENT | Cache + drift parser + drift download glob fix |
| [api/main.py](../api/main.py) | NEW lifespan task | Background warmer |
| [api/routers/pipelines.py](../api/routers/pipelines.py) | ENHANCEMENT | Cache-aware `/experiments`, new `/experiments/refresh` |
| [docs/POST_V3_PRODUCTION_REPORT.md](POST_V3_PRODUCTION_REPORT.md) | NEW | This document |
| [docs/FASTAPI_INTEGRATION.md](FASTAPI_INTEGRATION.md) | NEW | API surface + auth + cache architecture |
| [docs/DRIFT_DETECTION.md](DRIFT_DETECTION.md) | UPDATED | Added §15 *API Integration* |
| [docs/blueprints/V3_PostProduction_Enhancements_Blueprint.csv](blueprints/V3_PostProduction_Enhancements_Blueprint.csv) | NEW | Stakeholder change-log blueprint |

> **Not modified** (immutable per repo policy): `pipelines/submit_pipeline.py`, `pipelines/pipeline_builder.py`, `src/steps/s13_drift_monitor.py`, `src/orchestration/config_schema.py`, training-stage scripts, `docs/blueprints/V3_Production_Blueprint.csv`, `V3_Variant_Configurations.csv`.

---

## 8. Next Steps (recommended)

1. Wire warmer log line through uvicorn's logger so it surfaces in `/tmp/api.log`.
2. Add a `scripts/restart_services.sh` covering both API (`:8000`) and Streamlit (`:8501`).
3. Submit a chained pipeline run with `baseline_in` pointed at `happy_owl_sfmkgs2jrd`'s `drift_baseline` to populate the `comparison_drift` block end-to-end.
4. Add a `/api/v1/pipelines/drift/summary` aggregate endpoint listing PSI status across the most recent N completed jobs.
