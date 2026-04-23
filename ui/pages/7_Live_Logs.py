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
from ui.components.log_stream import render_log_stream
from ui.components.sidebar import render_sidebar
from ui.components.theme import inject_theme, page_header
from ui.config import REFRESH_INTERVAL

st.set_page_config(page_title="Live Logs", page_icon="📋", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Live Logs",
    "Stream step-level logs without copy-pasting job names",
    "📡",
)

client = get_client()

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

st.markdown(
    f"**Selected:** `{display_name}` &nbsp;·&nbsp; "
    f"experiment `{sel.get('experiment_name')}` &nbsp;·&nbsp; "
    f"status `{sel.get('status')}`"
)

auto_refresh = st.checkbox(
    f"Auto-refresh every {REFRESH_INTERVAL}s", value=False, key="logs_autorefresh"
)

# ── Load job + child steps ───────────────────────────────────
try:
    detail = client.get_job(job_name)
    steps = detail.get("steps") or []
    step_names = [s.get("name", f"step_{i}") for i, s in enumerate(steps)]
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load job: {exc}")
    detail = {}
    step_names = []

# ── Step selection ───────────────────────────────────────────
step_choice = st.selectbox(
    "Pipeline step",
    options=["(parent job)"] + step_names,
    key="logs_step",
)

st.divider()

target = job_name if step_choice == "(parent job)" else step_choice
with st.spinner(f"Loading logs for {target}…"):
    try:
        log_data = (
            client.get_job(target) if step_choice != "(parent job)" else detail
        )
        logs = log_data.get("logs") or log_data.get("log_text") or ""
        if not logs:
            logs = (
                f"No direct log content for {target}.\n"
                f"Status: {log_data.get('status', 'Unknown')}\n"
            )
            if "error" in log_data:
                logs += f"Error: {log_data['error']}\n"
            studio = log_data.get("studio_url")
            if studio:
                logs += f"\nFor live logs, open in Azure ML Studio:\n{studio}\n"
    except Exception as exc:  # noqa: BLE001
        logs = f"Failed to load logs: {exc}"

render_log_stream(logs)

if auto_refresh:
    import time

    time.sleep(REFRESH_INTERVAL)
    st.rerun()
