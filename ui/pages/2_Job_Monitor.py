"""2️⃣ Job Monitor — Track running and completed pipeline jobs."""

import streamlit as st

from ui.api_client import get_client
from ui.components.job_card import render_job_card
from ui.components.sidebar import render_sidebar
from ui.components.status_badge import status_badge
from ui.components.step_timeline import render_step_timeline
from ui.config import REFRESH_INTERVAL

st.set_page_config(page_title="Job Monitor", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 Job Monitor")

client = get_client()

# ── Filters ───────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    experiment_filter = st.text_input("Experiment Name", placeholder="All experiments")
with col2:
    status_filter = st.selectbox(
        "Status Filter",
        options=["All", "Running", "Completed", "Failed", "Canceled"],
    )
with col3:
    max_results = st.number_input("Max Results", min_value=5, max_value=200, value=50)

auto_refresh = st.checkbox(
    f"Auto-refresh every {REFRESH_INTERVAL}s", value=False
)

# ── Job List ──────────────────────────────────────────────────
st.divider()

try:
    status_param = None if status_filter == "All" else status_filter.lower()
    jobs_data = client.list_jobs(
        experiment_name=experiment_filter or None,
        status=status_param,
        max_results=max_results,
    )
    jobs = jobs_data.get("jobs", [])
except Exception as exc:
    st.error(f"Failed to load jobs: {exc}")
    jobs = []

if not jobs:
    st.info("No jobs found matching the filter criteria.")
else:
    # Summary row
    running = sum(1 for j in jobs if j.get("status", "").lower() in ("running", "preparing", "starting"))
    completed = sum(1 for j in jobs if j.get("status", "").lower() in ("completed", "finished"))
    failed = sum(1 for j in jobs if j.get("status", "").lower() == "failed")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(jobs))
    c2.metric("🔵 Running", running)
    c3.metric("✅ Completed", completed)
    c4.metric("🔴 Failed", failed)

    st.divider()

    for job in jobs:
        render_job_card(job)


# ── Job Detail Viewer ─────────────────────────────────────────
st.divider()
st.subheader("🔍 Job Detail")

job_name_input = st.text_input(
    "Enter Job Name",
    value=st.query_params.get("job", ""),
    placeholder="e.g. clever_banana_abc123",
)

if job_name_input:
    try:
        detail = client.get_job(job_name_input)
        st.markdown(f"### {detail.get('job_name', job_name_input)}")
        st.markdown(f"**Status:** {status_badge(detail.get('status', 'Unknown'))}")

        # Studio URL link
        studio_url = detail.get("studio_url")
        if studio_url:
            st.markdown(f"🔗 [Open in Azure ML Studio]({studio_url})")

        # Step Timeline
        steps = detail.get("steps", [])
        if steps:
            st.markdown("#### Pipeline Steps")
            render_step_timeline(steps)

        # Raw JSON
        with st.expander("📄 Raw Job Data"):
            st.json(detail)

        # Actions
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if detail.get("status", "").lower() in ("running", "preparing", "starting"):
                if st.button("❌ Cancel Job", type="secondary"):
                    try:
                        client.cancel_job(job_name_input)
                        st.success("Job cancellation requested.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Cancel failed: {exc}")
        with act_col2:
            if detail.get("status", "").lower() in ("completed", "finished", "failed"):
                if st.button("🔄 Resubmit Job"):
                    try:
                        result = client.resubmit(job_name_input)
                        st.success(f"Resubmitted: {result.get('job_name', '')}")
                    except Exception as exc:
                        st.error(f"Resubmit failed: {exc}")

    except Exception as exc:
        st.error(f"Failed to load job: {exc}")

# Auto-refresh via Streamlit
if auto_refresh:
    import time
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
