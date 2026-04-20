"""MLOps V3 Dashboard — Streamlit entry point."""

import streamlit as st

st.set_page_config(
    page_title="MLOps V3 Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.components.sidebar import render_sidebar

render_sidebar()

# ── Home page ─────────────────────────────────────────────────

st.title("🚀 MLOps V3 Pipeline Dashboard")
st.markdown(
    """
    Welcome to the **MLOps V3 Pipeline Management Dashboard**.
    Use the sidebar to navigate between pages.

    ---

    ### Quick Overview

    | Page | Description |
    |------|-------------|
    | **Submit Pipeline** | Launch new Azure ML pipeline jobs |
    | **Job Monitor** | Track running & completed jobs |
    | **Configs** | Browse pipeline configuration files |
    | **Leaderboard** | Compare model performance metrics |
    | **Drift Monitor** | Monitor data & model drift |
    | **Outputs** | Download job artifacts |
    | **Live Logs** | Stream real-time step logs |

    ---
    """
)

# Quick stats if connected
if st.session_state.get("connection_status") == "connected":
    from ui.api_client import get_client

    try:
        client = get_client()
        col1, col2, col3 = st.columns(3)

        # Recent jobs
        jobs_data = client.list_jobs(max_results=100)
        jobs = jobs_data.get("jobs", [])
        running = sum(1 for j in jobs if (j.get("status", "").lower() in ("running", "preparing", "starting")))
        completed = sum(1 for j in jobs if j.get("status", "").lower() in ("completed", "finished"))
        failed = sum(1 for j in jobs if j.get("status", "").lower() == "failed")

        col1.metric("🔵 Running", running)
        col2.metric("✅ Completed", completed)
        col3.metric("🔴 Failed", failed)

        # Configs
        configs_data = client.list_configs()
        st.metric("📋 Available Configs", configs_data.get("total", 0))
    except Exception:
        st.info("Connect to the API to see dashboard stats.")
else:
    st.info("👈 Enter your API key in the sidebar to get started.")
