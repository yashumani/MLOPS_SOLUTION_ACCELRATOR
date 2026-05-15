# 4 · Drift Monitor

**File:** [`ui/pages/5_Drift_Monitor.py`](../pages/5_Drift_Monitor.py)
**Icon:** 📉
**Single sentence:** Per-feature Population Stability Index (PSI) report for any completed job that ran the optional drift step.

## Who is this for?

MLOps engineers monitoring production retraining jobs and data scientists
investigating "why did my model degrade this week?".

## Pre-requisite

Drift analysis is the **optional** `s13_drift_monitor` step. It is only
present in a job's outputs when the job was submitted with a
`baseline_job` set (see the **Submit Pipeline** page). If you pick a job
that doesn't have a drift report, this page tells you exactly that and
points you back to the Submit page.

## What you see

1. **Job picker** — completed jobs only.
2. **Extract button** — first click downloads `drift_report` from Azure ML
   and parses the PSI table; subsequent visits use the cache.
3. **Summary KPIs** — total features · no drift · moderate drift · severe drift.
4. **Stability strip** — overall stability score (0–100), drift type,
   overall drift detected (YES / no).
5. **"How to read PSI" expander** — built-in legend with the standard
   PSI thresholds and recommended actions.
6. **PSI bar chart expander** — every feature, colour-coded green / orange /
   red, with reference lines at 0.10 and 0.25.
7. **Feature table expander** — searchable + severity-filterable + sorted
   by PSI descending.
8. **Top-4 most-drifted gauges expander** — circular gauges for the worst
   four features (collapsed by default to keep the page light).
9. **Download** — full drift report as CSV.

## PSI thresholds

| PSI range | Severity | Recommended action |
|-----------|----------|--------------------|
| `< 0.10`  | 🟢 Negligible | None — distribution is stable |
| `0.10 – 0.25` | 🟠 Moderate | Investigate; monitor closely |
| `> 0.25`  | 🔴 Severe | Retrain — distribution has materially changed |

The **Stability score** is `100 × (1 − mean(PSI))` clipped to `[0, 100]`.

## Backend endpoints called

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/pipelines/experiments` | Job picker tree |
| `GET /api/v1/pipelines/jobs/{job_name}/drift` | PSI report (cached, status-aware TTL) |

## Common workflows

- **"Did anything drift in last night's retraining run?"** → pick the job
  → Extract → glance at the KPIs and stability score.
- **"Which feature is the worst offender?"** → Feature table expander →
  sort by PSI descending (default).
- **"Find all drifted categorical features"** → Feature table → search
  `category_` (or your prefix).

## What it does NOT do

- It does not compare two jobs side-by-side (roadmap).
- It does not plot PSI over time (no time-series storage yet).
- It does not auto-create alerts (roadmap).
