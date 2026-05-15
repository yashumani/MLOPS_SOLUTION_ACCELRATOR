# 2 · Focus — Single-job cockpit

**File:** [`ui/pages/2_Focus.py`](../pages/2_Focus.py)
**Icon:** 🎯
**Single sentence:** Everything about *one* job in one place — overview, leaderboard, outputs, drift, logs.

## Who is this for?

Anyone investigating a specific run. After Submit succeeds, the user is
deposited here with the new job pinned. From the Home jobs table, the
"Focus" button does the same.

## State model

The "focused job" lives in `st.session_state["focused_job"]`. When unset,
the page shows a job picker; once set, every tab targets that one job.

A 🔁 **Change job** button in the sticky header clears the slot.

## Sticky header

- **Status badge** — live status of the parent job (refreshes every 30 s).
- **↗ Studio** — opens the run in Azure ML Studio.
- **❌ Cancel** — only when the job is non-terminal.
- **🔄 Resubmit** — only when the job is terminal; re-issues the same config.

## The 5 tabs

### 📊 Overview

- KPIs: total steps · running · completed · failed.
- **Step timeline** grouped by phase (Data / Phase A / Phase B / Phase C
  / Final / Register / Monitor). Each phase is a collapsible expander
  showing per-stage status. Default-expanded when the phase has running
  or failed work.
- Raw metadata expander (full job JSON, for power users).

### 🏆 Live Leaderboard

- **Champion banner** — model name, source phase, score, and a
  one-line breakdown ("Selected from N baseline · M variants · K HPO trials").
- **Per-phase expanders**:
  - 🔵 Phase A — Baseline (PyCaret + FLAML)
  - 🟣 Phase B — Variant search
  - 🟠 Phase C — HPO
- One sortable metrics table per phase, with the champion row marked ★.
- Single CSV download for the full flat leaderboard.

### 📦 Outputs

Three sub-tabs:

1. **Pipeline Summary** — KPIs (job, task, status, champion phase + score),
   then four expanders: baseline aggregate · Phase B aggregate · Phase C
   aggregate · final report. Each renders as JSON when the file exists.
2. **Named Artifacts** — load the API's output list, tick what you want,
   click *Extract selected*. Each artifact opens in its own expander with
   parsed JSON / CSV preview / text preview / download.
3. **Local outputs/** — read-only view of the repo-local `outputs/` folder
   on this compute instance. Useful for inspecting batch logs, downloaded
   artifacts, drift CSVs.

### 📉 Drift

- Same PSI report shown on the dedicated Drift Monitor page, scoped to
  this job. KPIs, stability score, heatmap, top-feature gauges. Empty
  state explains that the drift step is **optional** and only runs when
  the job was submitted with a baseline job.

### 📡 Logs

- Step dropdown — parent job + each child step (with stage prefix).
- Tail of the selected target's logs, refreshing every 60 s.
- For a richer view (severity filter, search, download) use the
  **Live Logs** page.

## Backend endpoints called

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/pipelines/jobs/{job_name}` | Detail + child steps |
| `POST /api/v1/pipelines/jobs/{job_name}/cancel` | Cancel running job |
| `POST /api/v1/pipelines/resubmit` | Resubmit terminal job |
| `GET /api/v1/pipelines/jobs/{job_name}/metrics` | Per-model leaderboard |
| `GET /api/v1/pipelines/jobs/{job_name}/summary` | Aggregated phase reports |
| `GET /api/v1/pipelines/jobs/{job_name}/outputs` | List named artifacts |
| `GET /api/v1/pipelines/jobs/{job_name}/outputs/{name}/content` | Parsed file content |
| `GET /api/v1/pipelines/jobs/{job_name}/outputs/{name}/download` | Raw artifact ZIP |
| `GET /api/v1/pipelines/jobs/{job_name}/drift` | PSI report |
| `GET /api/v1/pipelines/local-outputs` | Local outputs/ inventory |

## Auto-refresh

Each tab body lives inside an `@st.fragment(run_every="30s")`, so only the
visible panel re-runs on its own cadence. The rest of the page stays
interactive and the API isn't hammered.

## Common workflows

- **"Did Phase B beat Phase A?"** → Leaderboard tab → compare champion
  rows in the Phase A vs Phase B expanders.
- **"My job failed at s06"** → Overview → expand Phase B → see the failed
  status → switch to Logs tab → pick the failed step → view its tail.
- **"Download the champion model"** → Outputs → Named Artifacts → Load
  output list → tick `champion_model` → Extract → Download raw artifact.
