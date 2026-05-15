"""7️⃣ Live Logs — Stream logs from a running pipeline job.

Pick the job from the experiment tree, then choose which step's log to
stream. Logs are pulled from the parent job by default; child steps can
be selected once they appear in the timeline.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.api_client import get_client
from ui.components.job_picker import pick_single_job
from ui.components.log_stream import filter_logs, render_log_stream
from ui.components.sidebar import render_sidebar
from ui.components.status_badge import status_badge
from ui.components.theme import inject_theme, page_header
from ui.config import REFRESH_INTERVAL
from ui.data_cache import cached_get_job, cached_list_jobs

st.set_page_config(page_title="Live Logs", page_icon="📋", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Live Logs",
    "Stream step-level logs without copy-pasting job names",
    "📡",
)

client = get_client()

sel = st.session_state.get("logs_selected_job")
if not sel:
    try:
        recent_jobs = (cached_list_jobs(max_results=1) or {}).get("jobs") or []
    except Exception:  # noqa: BLE001
        recent_jobs = []
    if recent_jobs:
        sel = recent_jobs[0]
        st.session_state["logs_selected_job"] = sel
    else:
        sel = pick_single_job(
            client,
            key="logs",
            label_experiment="Experiment",
            label_job="Job to stream logs from",
        )

if not sel:
    st.info("Pick a job above to stream its logs.")
    st.stop()

job_name = sel["job_name"]
display_name = sel.get("display_name") or job_name
known_status = sel.get("status")

st.markdown(
    f"**Selected:** `{display_name}` &nbsp;·&nbsp; "
    f"experiment `{sel.get('experiment_name')}` &nbsp;·&nbsp; "
    f"status `{known_status}`"
)

# ── Load job + child steps (cached) ──────────────────────────
try:
    detail = cached_get_job(job_name, known_status=known_status) or {}
    steps = detail.get("steps") or []
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load job: {exc}")
    detail = {}
    steps = []

# Build dropdown options annotated with status, so users see at a glance
# which step is running / done / failed.
step_options: dict[str, str] = {"(parent job)": job_name}
for i, step in enumerate(steps):
    target_name = step.get("name") or f"step_{i}"
    label = step.get("display_name") or target_name
    stage = step.get("stage_key") or ""
    status = step.get("status") or "Unknown"
    icon = {
        "running": "🟡", "preparing": "🟡", "starting": "🟡",
        "completed": "🟢", "finished": "🟢",
        "failed": "🔴", "canceled": "⚪", "cancelled": "⚪",
    }.get(status.lower(), "⚪")
    prefix = f"{stage} · " if stage else ""
    step_options[f"{icon} {prefix}{label}  ({status})"] = target_name

# ── Controls ─────────────────────────────────────────────────
ctl1, ctl2, ctl3 = st.columns([2, 1.2, 1.2])
with ctl1:
    step_choice = st.selectbox(
        "Pipeline step",
        options=list(step_options.keys()),
        key="logs_step",
    )
with ctl2:
    auto_refresh = st.checkbox(
        f"Auto-refresh ({REFRESH_INTERVAL}s)",
        value=False,
        key="logs_autorefresh",
    )
with ctl3:
    tail_lines = st.number_input(
        "Tail (lines)",
        min_value=0,
        max_value=10000,
        value=500,
        step=100,
        key="logs_tail",
        help="Show only the last N lines. Use 0 for all.",
    )

flt1, flt2 = st.columns([1.2, 2])
with flt1:
    levels = st.multiselect(
        "Log level",
        ["ERROR", "WARNING", "INFO", "DEBUG"],
        default=[],
        key="logs_levels",
        help="Empty = show everything",
    )
with flt2:
    search = st.text_input(
        "🔍 Search in logs",
        placeholder="exception, traceback, score=…",
        key="logs_search",
    )

st.divider()


def _resolve_logs(target: str) -> str:
    """Fetch raw logs/text for the given target job (parent or child step)."""
    try:
        log_data = (
            cached_get_job(target, known_status=known_status)
            if step_choice != "(parent job)"
            else detail
        )
    except Exception as exc:  # noqa: BLE001
        return f"Failed to load logs: {exc}"
    log_data = log_data or {}
    logs = log_data.get("logs") or log_data.get("log_text") or ""
    if not logs:
        logs = (
            f"No direct log content for {target}.\n"
            f"Status: {log_data.get('status', 'Unknown')}\n"
        )
        if "error" in log_data:
            logs += f"Error: {log_data['error']}\n"
        studio = log_data.get("studio_url") or detail.get("studio_url")
        if studio:
            logs += f"\nFor live logs, open in Azure ML Studio:\n{studio}\n"
    return logs


target = step_options[step_choice]


@st.fragment(run_every=f"{REFRESH_INTERVAL}s")
def _live_log_panel() -> None:
    raw = _resolve_logs(target)
    if not st.session_state.get("logs_autorefresh", False):
        # Render once; the fragment still re-runs on its own cadence but the
        # toggle controls whether we keep refreshing visually.
        pass
    filtered = filter_logs(
        raw,
        levels=st.session_state.get("logs_levels") or None,
        search=st.session_state.get("logs_search") or None,
        tail=int(st.session_state.get("logs_tail") or 0) or None,
    )
    if not filtered.strip():
        st.info(
            "No log lines match the current filters. "
            "Try clearing the search box or log-level selector."
        )
    else:
        render_log_stream(filtered)

    # Always offer a download of the *unfiltered* logs.
    if raw:
        st.download_button(
            "⬇️ Download raw logs (.log)",
            data=raw,
            file_name=f"{display_name}_{target}.log",
            mime="text/plain",
            key=f"logs_dl_{target}",
        )


# Live status badge for the currently selected target.
if step_choice != "(parent job)":
    target_step = next(
        (s for s in steps if (s.get("name") or "") == target),
        None,
    )
    if target_step:
        st.markdown(
            f"Step status: {status_badge(target_step.get('status', 'Unknown'))}",
            unsafe_allow_html=True,
        )

_live_log_panel()
