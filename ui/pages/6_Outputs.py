"""6️⃣ Outputs — Browse job output artifacts."""

import streamlit as st

from ui.api_client import get_client
from ui.components.file_browser import render_file_browser
from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="Outputs", page_icon="📦", layout="wide")
render_sidebar()

st.title("📦 Job Outputs")
st.markdown("Browse and download artifacts from completed pipeline jobs.")

client = get_client()

# ── Job Selection ─────────────────────────────────────────────
job_name = st.text_input(
    "Enter Job Name",
    placeholder="e.g. clever_banana_abc123",
)

if not job_name:
    st.info("Enter a job name above to view outputs.")
    st.stop()

# ── Load Outputs ──────────────────────────────────────────────
with st.spinner("Loading outputs..."):
    try:
        data = client.get_outputs(job_name)
    except Exception as exc:
        st.error(f"Failed to load outputs: {exc}")
        data = {}

if not data:
    st.warning("No outputs found for this job.")
    st.stop()

# ── File Browser ──────────────────────────────────────────────
render_file_browser(data)

# ── Raw JSON View ─────────────────────────────────────────────
with st.expander("📄 Raw Output Data"):
    st.json(data)
