"""2️⃣ Job Monitor — Browse experiments → jobs → step timeline.

No more typing job names. Pick the experiment, tick the jobs you want,
then the panel below shows their step-level status, raw metadata, and
provides Cancel / Resubmit / Open-in-Studio actions.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.api_client import get_client
from ui.components.job_picker import pick_jobs
from ui.components.sidebar import render_sidebar
from ui.components.status_badge import status_badge
from ui.components.step_timeline import render_step_timeline
from ui.components.theme import inject_theme, page_header
from ui.config import REFRESH_INTERVAL

st.set_page_config(page_title="Job Monitor", page_icon="📊", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Job Monitor",
    "Browse jobs grouped by experiment — pick one or more to inspect",
    "📊",
)

client = get_client()

# ── Filters ──────────────────────────────────────────────────
filter_col1, filter_col2 = st.columns([2, 1])
with filter_col1:
    status_choice = st.selectbox(
        "Status filter",
        options=[
            "All",
            "Running",
            "Completed",
            "Failed",
            "Canceled",
            "NotStarted",
            "Preparing",
        ],
        key="monitor_status_filter",
    )
with filter_col2:
    auto_refresh = st.checkbox(
        f"Auto-refresh every {REFRESH_INTERVAL}s",
        value=False,
        key="monitor_autorefresh",
    )

status_filter = None if status_choice == "All" else [status_choice]

# ── Picker ───────────────────────────────────────────────────
selected_jobs = pick_jobs(
    client,
    key="monitor",
    multi=True,
    status_filter=status_filter,
    label_experiment="Experiment",
    label_job="Jobs (display name → status → started)",
)

st.divider()

if not selected_jobs:
    st.info("Select at least one job above to inspect details.")
    st.stop()

# ── Quick KPI strip across selected jobs ─────────────────────
running = sum(
    1 for j in selected_jobs
    if (j.get("status") or "").lower() in ("running", "preparing", "starting")
)
completed = sum(
    1 for j in selected_jobs
    if (j.get("status") or "").lower() in ("completed", "finished")
)
failed = sum(
    1 for j in selected_jobs if (j.get("status") or "").lower() == "failed"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected", len(selected_jobs))
c2.metric("Running", running)
c3.metric("Completed", completed)
c4.metric("Failed", failed)

st.divider()

# ── Per-job detail panels ────────────────────────────────────
for sel in selected_jobs:
    job_name = sel.get("job_name")
    display_name = sel.get("display_name") or job_name
    studio_url = sel.get("studio_url")

    with st.expander(
        f"📁 {display_name}  —  {sel.get('status', 'Unknown')}",
        expanded=(len(selected_jobs) == 1),
    ):
        try:
            detail = client.get_job(job_name)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load job detail: {exc}")
            continue

        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown(
                f"**Display name:** `{detail.get('display_name') or display_name}`  \n"
                f"**Job (run) ID:** `{detail.get('job_name', job_name)}`  \n"
                f"**Experiment:** `{detail.get('experiment_name') or sel.get('experiment_name')}`  \n"
                f"**Status:** {status_badge(detail.get('status', 'Unknown'))}",
                unsafe_allow_html=True,
            )
        with head_r:
            if studio_url or detail.get("studio_url"):
                st.link_button(
                    "🔗 Azure ML Studio",
                    studio_url or detail.get("studio_url"),
                    use_container_width=True,
                )

        steps = detail.get("steps") or []
        if steps:
            st.markdown("#### Pipeline steps")
            render_step_timeline(steps)
        else:
            st.caption("No child steps reported yet.")

        with st.expander("📄 Raw metadata", expanded=False):
            st.json(detail)

        # Actions
        a1, a2 = st.columns(2)
        status_lower = (detail.get("status") or "").lower()
        with a1:
            if status_lower in (
                "running", "preparing", "starting", "notstarted", "queued"
            ):
                if st.button("❌ Cancel job", key=f"cancel_{job_name}"):
                    try:
                        client.cancel_job(job_name)
                        st.success("Cancellation requested.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Cancel failed: {exc}")
        with a2:
            if status_lower in (
                "completed", "finished", "failed", "canceled", "cancelled"
            ):
                if st.button("🔄 Resubmit", key=f"resub_{job_name}"):
                    try:
                        result = client.resubmit(job_name)
                        st.success(
                            "Resubmitted → "
                            f"`{result.get('display_name') or result.get('job_name')}`"
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Resubmit failed: {exc}")

# ── Auto-refresh ─────────────────────────────────────────────
if auto_refresh:
    import time

    time.sleep(REFRESH_INTERVAL)
    st.rerun()
