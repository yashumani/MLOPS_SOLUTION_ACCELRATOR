"""1️⃣ Submit Pipeline — Launch new Azure ML pipeline jobs."""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.api_client import get_client
from ui.components.sidebar import render_sidebar
from ui.components.theme import inject_theme, page_header

st.set_page_config(page_title="Submit Pipeline", page_icon="🚀", layout="wide")
inject_theme()
render_sidebar()

page_header("Submit Pipeline", "Launch a new Azure ML V3 pipeline job", "🚀")

client = get_client()

# ── Config Selection ──────────────────────────────────────────
st.subheader("Pipeline Configuration")

try:
    configs_data = client.list_configs()
    config_names = configs_data.get("configs", [])
except Exception as exc:
    st.error(f"Failed to load configs: {exc}")
    config_names = []

if not config_names:
    st.warning("No configurations found. Check your API connection.")
    st.stop()

selected_config = st.selectbox(
    "Select Configuration",
    options=config_names,
    help="Choose a pipeline configuration YAML",
)

# ── Advanced Options ──────────────────────────────────────────
with st.expander("⚙️ Advanced Options", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        compute_override = st.text_input(
            "Compute Target (override)",
            placeholder="mlopsv2computecluster",
            help="Leave empty to use default from config",
        )
        force_rerun = st.checkbox("Force Re-run", value=False)
    with col2:
        baseline_job = st.text_input(
            "Baseline Job Name",
            placeholder="(optional)",
            help="Reference job for drift comparison",
        )
        tag_key = st.text_input("Custom Tag Key", placeholder="team")
        tag_val = st.text_input("Custom Tag Value", placeholder="ml-engineering")

# ── Preview ───────────────────────────────────────────────────
if selected_config:
    with st.expander("📋 Config Preview", expanded=False):
        try:
            cfg = client.get_config(selected_config)
            st.json(cfg)
        except Exception:
            st.info("Could not load config preview.")

# ── Submit ────────────────────────────────────────────────────
st.divider()

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
if st.button("🚀 Submit Pipeline", type="primary", use_container_width=True):
    tags = {}
    if tag_key and tag_val:
        tags[tag_key] = tag_val

    with st.spinner("Submitting pipeline..."):
        try:
            result = client.submit_pipeline(
                config_name=selected_config,
                compute=compute_override or None,
                force_rerun=force_rerun,
                baseline_job=baseline_job or None,
                tags=tags or None,
            )
            st.success(f"✅ Pipeline submitted!")
            st.json(result)

            job_name = result.get("job_name")
            if job_name:
                st.markdown(
                    f"📊 [View in Job Monitor →](/Job_Monitor?job={job_name})"
                )
        except Exception as exc:
            st.error(f"❌ Submission failed: {exc}")
