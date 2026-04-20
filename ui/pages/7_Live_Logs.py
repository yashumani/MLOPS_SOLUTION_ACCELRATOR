"""7️⃣ Live Logs — Stream logs from running pipeline jobs."""

import streamlit as st

from ui.api_client import get_client
from ui.components.log_stream import render_log_stream
from ui.components.sidebar import render_sidebar
from ui.config import REFRESH_INTERVAL

st.set_page_config(page_title="Live Logs", page_icon="📋", layout="wide")
render_sidebar()

st.title("📋 Live Logs")
st.markdown("Stream logs from running or completed pipeline jobs.")

client = get_client()

# ── Job Selection ─────────────────────────────────────────────
job_name = st.text_input(
    "Enter Job Name",
    placeholder="e.g. clever_banana_abc123",
)

auto_refresh = st.checkbox(f"Auto-refresh every {REFRESH_INTERVAL}s", value=False)

if not job_name:
    st.info("Enter a job name above to view logs.")
    st.stop()

# ── Load Job Detail ───────────────────────────────────────────
try:
    detail = client.get_job(job_name)
    steps = detail.get("steps", [])
    step_names = [s.get("name", f"step_{i}") for i, s in enumerate(steps)]
except Exception:
    step_names = []

# ── Step Selection ────────────────────────────────────────────
st.subheader("Select Step")

if step_names:
    selected_step = st.selectbox("Step", options=["(parent job)"] + step_names)
else:
    selected_step = "(parent job)"
    st.info("No steps found — showing parent job logs.")

# ── Fetch Logs ────────────────────────────────────────────────
st.divider()

target = job_name if selected_step == "(parent job)" else selected_step

with st.spinner(f"Loading logs for {target}..."):
    try:
        # Use get_job since the API may return logs in job detail
        log_data = client.get_job(target) if selected_step != "(parent job)" else detail
        logs = log_data.get("logs", log_data.get("log_text", ""))

        if not logs:
            # Fallback: show step status info
            logs = f"No direct log content available for {target}.\n"
            logs += f"Status: {log_data.get('status', 'Unknown')}\n"
            if "error" in log_data:
                logs += f"Error: {log_data['error']}\n"
    except Exception as exc:
        logs = f"Failed to load logs: {exc}"

render_log_stream(logs)

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
