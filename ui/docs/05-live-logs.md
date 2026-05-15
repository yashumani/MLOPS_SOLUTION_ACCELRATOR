# 5 · Live Logs

**File:** [`ui/pages/7_Live_Logs.py`](../pages/7_Live_Logs.py)
**Icon:** 📡
**Single sentence:** Step-level log tail for any job, with severity filter, substring search, configurable tail length, and a download button.

## Who is this for?

Anyone debugging a failure. The Focus → Logs tab gives you a quick peek;
this page is the full investigative view.

## What you see

1. **Job picker** — same component used on Focus and Drift Monitor.
2. **Step picker** with **status emoji** in the label:
   - 🟡 Running · 🟢 Completed · 🔴 Failed · ⚪ Other
   - The label includes the canonical stage key (`s06`, `s05a`, …) so you
     don't have to remember Azure ML's underlying child step names.
3. **Auto-refresh** checkbox — fragment refreshes every 30 s.
4. **Tail (lines)** — number of trailing lines to keep (50…5000, default 500).
5. **Severity filter** — multi-select: ERROR · WARNING · INFO · DEBUG.
6. **Search** — substring (case-insensitive). Empty = no filter.
7. **Status badge** for the selected child step.
8. **Log tail** — filtered + tailed view. Empty state explains how to
   broaden the filter.
9. **Download raw logs** — always offered, ignores filters so you get the
   full unredacted content for offline analysis.

## Backend endpoint

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/pipelines/jobs/{target_job}/logs?download=true` | Raw log text |

The same endpoint is called for the parent job and for every child step
(child step name is passed as `target_job`).

## Filtering implementation

Filtering happens **client-side** inside `ui/components/log_stream.py` via
`filter_logs(logs, levels=..., search=..., tail=...)`. The raw logs come
straight from the API; the UI only formats them, so the download always
matches what's on Azure ML.

The level regex matches whole-word `ERROR | CRITICAL | FATAL | WARN(ING) |
INFO | DEBUG` anywhere on a line. Lines without a recognized level are
hidden when any level filter is active — uncheck *all* levels to see them.

## Common workflows

- **"My s06 step failed — show me only the errors"** → pick the s06 child
  → set Severity = ERROR → review.
- **"I need to grep for `OutOfMemoryError`"** → Search = `OutOfMemoryError`.
- **"I want the full logs for an offline ticket"** → Download raw logs.

## What it does NOT do

- It does not stream logs in real time via websocket — it polls every
  30 s when auto-refresh is on.
- It does not preserve scroll position across refreshes (Streamlit limitation).
- It does not parse Python tracebacks into clickable links (roadmap).
