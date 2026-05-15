# 0 · Home — Dashboard

**File:** [`ui/app.py`](../app.py)
**Icon:** 🧠
**Single sentence:** Filterable inventory of every Azure ML pipeline job + jump-off point to the rest of the app.

## Who is this for?

Anyone opening the app cold. The first question every user asks is *"what's
running and what isn't?"* — Home answers that on a single screen.

## What you see

1. **Hero header** — product name + one-line value prop.
2. **Quick Actions tiles** — five clickable cards: Focus, Submit, Configs,
   Drift, Live Logs. Each links to the full page for that capability.
3. **Live pipeline activity** (only when API is connected, refreshes every 30s):
   - **Filter bar** — task type · status (multi) · time window · text search.
   - **KPI strip** — matching jobs, running, completed, failed, total configs.
   - **Status donut** — visual breakdown of running / completed / failed.
   - **Jobs table** — paginated list (20 rows/page) with a "Focus" button
     per row that pins the job to the Focus page.

## Backend endpoints called

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/health` | Connection check (5 s cache) |
| `GET /api/v1/pipelines/jobs` | List jobs (15 s cache) |
| `GET /api/v1/configs` | Count configs |

## Common workflows

- **"Find a recent run"** → use the filter bar (last 24 h + status =
  Completed) → click *Focus* on the row.
- **"Spot failures"** → status filter = Failed → review the most recent ones.
- **"Submit a new job"** → click the Submit tile (top-left).

## Known gaps & roadmap

- Job tags are not surfaced in the table — could power richer attribution.
- No per-job duration column (only start time).
- Donut shows global counts; a per-stage breakdown could show where jobs
  typically fail.
