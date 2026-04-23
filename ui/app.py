"""Savyminds MLOps V3 — Streamlit home page."""

from __future__ import annotations

# ── Bootstrap: ensure project root is on sys.path so `from ui.X` works ────────
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Savyminds MLOps V3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.components.sidebar import render_sidebar  # noqa: E402
from ui.components.theme import inject_theme, page_header, section_label  # noqa: E402
from ui.data_cache import (  # noqa: E402
    cached_list_configs,
    cached_list_jobs,
    prewarm,
)

inject_theme()
render_sidebar()

# Warm list_experiments + list_configs exactly once per session so the
# home page and downstream pages render instantly.
prewarm(st.session_state)

# Initialise the shared "focused job" slot used by Phase 2 (Focus page) so
# other pages can read it without guarding every access.
st.session_state.setdefault("focused_job", None)

# ── Hero ──────────────────────────────────────────────────────────────────────
page_header(
    title="MLOps Intelligence Hub",
    subtitle="End-to-end ML pipeline orchestration powered by Azure ML",
    icon="🧠",
)

# ── Quick actions: real clickable navigation tiles ────────────────────────────
section_label("Quick Actions")

_TILES = [
    ("pages/1_Submit_Pipeline.py", "🚀", "Submit Pipeline",
     "Launch Azure ML pipeline jobs from YAML configs"),
    ("pages/2_Job_Monitor.py",     "📊", "Job Monitor",
     "Real-time job tracking with step-level visibility"),
    ("pages/4_Leaderboard.py",     "🏆", "Leaderboard",
     "Compare model performance across all experiments"),
    ("pages/5_Drift_Monitor.py",   "📉", "Drift Monitor",
     "PSI-based data and model drift analysis"),
    ("pages/6_Outputs.py",         "📦", "Outputs",
     "Download artifacts, models and result files"),
    ("pages/7_Live_Logs.py",       "📡", "Live Logs",
     "Stream real-time step logs from running jobs"),
]

cols = st.columns(3, gap="medium")
for i, (path, icon, title, desc) in enumerate(_TILES):
    with cols[i % 3]:
        with st.container(border=False):
            st.markdown(
                f'<div class="svm-card">'
                f'<div class="svm-card-icon">{icon}</div>'
                f'<div class="svm-card-title">{title}</div>'
                f'<p class="svm-card-desc">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.page_link(path, label=f"Open {title} →")

# ── Live stats ────────────────────────────────────────────────────────────────
if st.session_state.get("connection_status") == "connected":
    section_label("Live Statistics")

    try:
        jobs_data = cached_list_jobs(max_results=100) or {}
        jobs = jobs_data.get("jobs", []) or []

        running   = sum(1 for j in jobs if str(j.get("status", "")).lower() in ("running", "preparing", "starting", "queued"))
        completed = sum(1 for j in jobs if str(j.get("status", "")).lower() in ("completed", "finished"))
        failed    = sum(1 for j in jobs if str(j.get("status", "")).lower() == "failed")

        configs_data = cached_list_configs() or {}
        n_configs = configs_data.get("total") or len(configs_data.get("configs", []) or [])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Running",   running)
        c2.metric("Completed", completed)
        c3.metric("Failed",    failed)
        c4.metric("Configs",   n_configs)

        # Donut chart
        if running + completed + failed > 0:
            try:
                from streamlit_echarts import st_echarts

                section_label("Pipeline Activity")
                st_echarts(
                    options={
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                        "legend": {
                            "bottom": "2%", "left": "center",
                            "textStyle": {"color": "#475569", "fontSize": 12},
                            "itemGap": 20,
                        },
                        "series": [{
                            "name": "Jobs",
                            "type": "pie",
                            "radius": ["50%", "72%"],
                            "center": ["50%", "45%"],
                            "data": [
                                {"value": running,   "name": "Running",
                                 "itemStyle": {"color": "#0EA5E9"}},
                                {"value": completed, "name": "Completed",
                                 "itemStyle": {"color": "#10B981"}},
                                {"value": failed,    "name": "Failed",
                                 "itemStyle": {"color": "#EF4444"}},
                            ],
                            "label": {"color": "#0F172A", "fontSize": 13, "fontWeight": 600},
                            "itemStyle": {
                                "borderRadius": 6,
                                "borderColor": "#FFFFFF",
                                "borderWidth": 3,
                            },
                        }],
                    },
                    height="320px",
                    key="home_donut",
                )
            except ImportError:
                pass
    except Exception as exc:
        st.warning(f"Could not load live stats: {exc}")
else:
    st.info("API connecting — check sidebar for status.")
