# Savyminds MLOps V3 — UI Documentation

This folder is the **single source of truth** for what every Streamlit page
in `ui/` does, who it is for, and how it maps to the V3 Azure ML pipeline
(stages `s00..s12`, Phase A / B / C, MLflow nested runs).

It is meant to be read like a product spec: open one page guide per tab and
you should be able to demo the application end-to-end.

## What is this app?

The app is a **two-service** product:

| Service | Port | Public URL on this compute |
|---------|------|----------------------------|
| FastAPI backend | `8000` | <https://mlopspipelinev2-8000.eastus2.instances.azureml.ms/docs> |
| Streamlit UI    | `8501` | <https://mlopspipelinev2-8501.eastus2.instances.azureml.ms/> |

The **Streamlit UI** is what end users open in a browser. It calls the
**FastAPI backend** over HTTP, which in turn talks to **Azure ML** to
submit, monitor, and inspect pipeline jobs. The user never installs an
Azure ML SDK — everything is mediated by the API.

## End-user personas

| Persona | What they want | Pages they live in |
|---------|----------------|--------------------|
| **ML engineer** running experiments | Submit jobs, watch progress, compare champions, debug failures | Submit · Focus · Live Logs |
| **Data scientist** tuning recipes | Edit configs, duplicate variants, compare leaderboards | Configs · Focus · Submit |
| **MLOps / SRE** monitoring health | Spot failures, tail logs, check drift on production retraining jobs | Home · Live Logs · Drift Monitor |
| **Stakeholder** reviewing results | One-click champion summary, exportable leaderboard, drift report | Focus · Drift Monitor |

## The 6 pages (information architecture)

Each tab has **one job to do**. If you find yourself building feature X and
can't decide which tab it belongs in, ask: *"Which user job am I serving?"*
and look up the table below.

| # | Tab | Single-sentence purpose | Doc |
|---|-----|------------------------|-----|
| 0 | **Home** | Filterable inventory of every Azure ML job + quick navigation. | [00-home.md](00-home.md) |
| 1 | **Submit Pipeline** | Launch one new Azure ML pipeline job from a config YAML. | [01-submit.md](01-submit.md) |
| 2 | **Focus** | Single-job cockpit — overview, leaderboard, outputs, drift, logs. | [02-focus.md](02-focus.md) |
| 3 | **Configs** | Browse, edit, duplicate, delete the YAML files that define jobs. | [03-configs.md](03-configs.md) |
| 4 | **Drift Monitor** | PSI-based feature drift report for any completed job (optional `s13`). | [04-drift.md](04-drift.md) |
| 5 | **Live Logs** | Step-level log tail with severity filter, search, download. | [05-live-logs.md](05-live-logs.md) |
| 6 | **UI Action Items** | Release-risk ordered backlog for fixing the UI before production. | [06-ui-action-items.md](06-ui-action-items.md) |

## How the UI maps to the V3 pipeline

The V3 pipeline runs as a single Azure ML `@dsl.pipeline` with these stages:

```
s00  Data validation              ← shown in Focus → Overview (timeline, "Data" group)
s01  Ingestion                    ← shown in Focus → Overview
s02  Preparation
s03  Preprocessing
s04  Feature engineering
s05a PyCaret baseline             ─┐
s05b FLAML baseline                │  Phase A — Baseline
s05t Time-series baseline          │  shown in Focus → Leaderboard ("Phase A" expander)
s05z Baseline aggregate           ─┘
s06  Variant runner                ← Phase B — Leaderboard "Phase B" expander
s08  Optuna HPO                   ─┐
s09  Phase C aggregate             │  Phase C — Leaderboard "Phase C" expander
s10  Final evaluation              │  Final
s12  Model registration           ─┘
s13  Drift monitor (optional)      ← only present when baseline_job is set on submit
```

- The **step timeline** on Focus → Overview groups these into collapsible
  phase sections (Data / Phase A / Phase B / Phase C / Final / Register /
  Monitor).
- The **leaderboard** on Focus → Leaderboard groups model rows by the same
  phase keys and surfaces the global champion in a dedicated card.
- The **drift report** is only available when a job was submitted with a
  `baseline_job` — this is documented inline on both the Submit and Drift
  Monitor pages.

## Component reuse map

| Component (`ui/components/*.py`) | Used by |
|----------------------------------|---------|
| `sidebar.py`                     | Every page (connection status, settings) |
| `theme.py`                       | Every page (CSS injection, page header) |
| `job_picker.py`                  | Focus, Drift Monitor, Live Logs |
| `step_timeline.py`               | Focus → Overview |
| `metrics_table.py`               | Focus → Leaderboard |
| `config_summary_card.py`         | Submit, Configs |
| `config_viewer.py`               | Configs |
| `drift_gauge.py` / `drift_heatmap.py` | Drift Monitor (gauges expander), helper |
| `log_stream.py`                  | Focus → Logs, Live Logs |
| `status_badge.py`                | Home, Focus, Live Logs |
| `file_browser.py`                | Focus → Outputs (local outputs/) |

## Page-by-page guides

Open the file for the page you care about:

- [`00-home.md`](00-home.md) — Dashboard
- [`01-submit.md`](01-submit.md) — Submit Pipeline
- [`02-focus.md`](02-focus.md) — Focus
- [`03-configs.md`](03-configs.md) — Configs
- [`04-drift.md`](04-drift.md) — Drift Monitor
- [`05-live-logs.md`](05-live-logs.md) — Live Logs
- [`06-ui-action-items.md`](06-ui-action-items.md) — UI fix action items

## Running the UI locally

See the deployment guide: [`../README_AZUREML_DEPLOY.md`](../README_AZUREML_DEPLOY.md).

Minimum `.env`:

```
API_KEY=<shared-secret>
API_BASE_URL=http://localhost:8000
UI_BASE_URL=http://localhost:8501
AZURE_SUBSCRIPTION_ID=<sub>
AZURE_RESOURCE_GROUP=<rg>
AZURE_WORKSPACE_NAME=<ws>
```

Start both services:

```bash
bash scripts/run_app.sh
```
