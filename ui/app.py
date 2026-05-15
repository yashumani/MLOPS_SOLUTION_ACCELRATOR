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
    ("pages/2_Focus.py",           "🎯", "Focus",
     "One job, five live tabs — leaderboard, drift, outputs, logs"),
    ("pages/1_Submit_Pipeline.py", "🚀", "Submit",
     "Launch Azure ML pipeline jobs from YAML configs"),
    ("pages/3_Configs.py",         "⚙️", "Configs",
     "Browse, create and edit pipeline configurations"),
    ("pages/5_Drift_Monitor.py",   "📉", "Drift",
     "PSI-based data and model drift analysis"),
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

# Warm expensive API caches only after the visible shell is painted. A slow
# Azure ML experiments call should not leave the main panel blank.
prewarm(st.session_state)

# ── Live stats + Filter bar + Jobs table ─────────────────────────────────────
import datetime as _dt  # noqa: E402

_RUNNING = {"running", "preparing", "starting", "queued", "notstarted"}
_COMPLETED = {"completed", "finished"}
_FAILED = {"failed"}


def _task_type_from_experiment(name: str | None) -> str:
    if not name:
        return "—"
    base = name.lower()
    for tt in ("classification", "regression", "clustering", "timeseries"):
        if tt in base:
            return tt
    return "other"


def _within_window(start_time: object, window: str) -> bool:
    if window == "All time":
        return True
    if not start_time:
        return False
    try:
        if isinstance(start_time, str):
            ts = _dt.datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        elif isinstance(start_time, _dt.datetime):
            ts = start_time
        else:
            return False
    except Exception:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.timezone.utc)
    now = _dt.datetime.now(_dt.timezone.utc)
    delta = {"Last 24h": _dt.timedelta(days=1),
             "Last 7d": _dt.timedelta(days=7),
             "Last 30d": _dt.timedelta(days=30)}.get(window)
    return delta is None or (now - ts) <= delta


if st.session_state.get("connection_status") == "connected":
    section_label("Live Pipeline Activity")

    @st.fragment(run_every="30s")
    def _live_panel() -> None:  # noqa: C901
        try:
            jobs_data = cached_list_jobs(max_results=100) or {}
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load live stats: {exc}")
            return

        jobs = list(jobs_data.get("jobs", []) or [])
        configs_data = cached_list_configs() or {}
        n_configs = configs_data.get("total") or len(
            configs_data.get("configs", []) or []
        )

        # ── Filter bar ────────────────────────────────────────
        with st.container(border=True):
            f1, f2, f3, f4 = st.columns([1.2, 1.5, 1.2, 2])
            with f1:
                task_filter = st.selectbox(
                    "Task type",
                    ["All", "classification", "regression",
                     "clustering", "timeseries", "other"],
                    key="home_task_filter",
                )
            with f2:
                status_filter = st.multiselect(
                    "Status",
                    ["Running", "Completed", "Failed", "Canceled", "NotStarted"],
                    default=[],
                    key="home_status_filter",
                )
            with f3:
                window_filter = st.selectbox(
                    "When",
                    ["Last 24h", "Last 7d", "Last 30d", "All time"],
                    index=1,
                    key="home_window_filter",
                )
            with f4:
                substr = st.text_input(
                    "Search (display name / experiment)",
                    placeholder="telecom_churn, college, …",
                    key="home_substr_filter",
                )

        # ── Apply filters ─────────────────────────────────────
        def _matches(j: dict) -> bool:
            if task_filter != "All" and _task_type_from_experiment(
                j.get("experiment_name")
            ) != task_filter:
                return False
            if status_filter:
                lowered = {s.lower() for s in status_filter}
                if str(j.get("status", "")).lower() not in lowered:
                    return False
            if not _within_window(j.get("start_time"), window_filter):
                return False
            if substr:
                hay = " ".join(
                    str(j.get(k, "")) for k in
                    ("display_name", "experiment_name", "job_name")
                ).lower()
                if substr.lower() not in hay:
                    return False
            return True

        filtered = [j for j in jobs if _matches(j)]

        running = sum(
            1 for j in filtered if str(j.get("status", "")).lower() in _RUNNING
        )
        completed = sum(
            1 for j in filtered if str(j.get("status", "")).lower() in _COMPLETED
        )
        failed = sum(
            1 for j in filtered if str(j.get("status", "")).lower() in _FAILED
        )

        # ── KPI strip ────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Matching", len(filtered))
        c2.metric("Running", running)
        c3.metric("Completed", completed)
        c4.metric("Failed", failed)
        c5.metric("Configs", n_configs)

        # ── Donut ────────────────────────────────────────────
        if running + completed + failed > 0:
            try:
                from streamlit_echarts import st_echarts

                st_echarts(
                    options={
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "item",
                                    "formatter": "{b}: {c} ({d}%)"},
                        "legend": {
                            "bottom": "2%", "left": "center",
                            "textStyle": {"color": "#475569", "fontSize": 12},
                            "itemGap": 20,
                        },
                        "series": [{
                            "name": "Jobs", "type": "pie",
                            "radius": ["50%", "72%"], "center": ["50%", "45%"],
                            "data": [
                                {"value": running, "name": "Running",
                                 "itemStyle": {"color": "#0EA5E9"}},
                                {"value": completed, "name": "Completed",
                                 "itemStyle": {"color": "#10B981"}},
                                {"value": failed, "name": "Failed",
                                 "itemStyle": {"color": "#EF4444"}},
                            ],
                            "label": {"color": "#0F172A",
                                      "fontSize": 13, "fontWeight": 600},
                            "itemStyle": {"borderRadius": 6,
                                          "borderColor": "#FFFFFF",
                                          "borderWidth": 3},
                        }],
                    },
                    height="280px",
                    key="home_donut",
                )
            except ImportError:
                pass

        # ── Paginated jobs table with Focus action ───────────
        section_label("Jobs")
        if not filtered:
            st.info("No jobs match the current filters.")
            return

        # Sort: most recent first (start_time desc), missing → bottom
        def _sort_key(j: dict):
            t = j.get("start_time")
            if isinstance(t, str):
                try:
                    t = _dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
                except Exception:
                    return _dt.datetime.min
            return t or _dt.datetime.min

        filtered.sort(key=_sort_key, reverse=True)

        page_size = 20
        n_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        page = st.number_input(
            "Page", min_value=1, max_value=n_pages, value=1,
            step=1, key="home_jobs_page",
        )
        start = (page - 1) * page_size
        page_rows = filtered[start:start + page_size]

        # Table header
        h = st.columns([0.6, 2.6, 1.4, 1.0, 1.4, 0.8])
        for col, label in zip(
            h, ["#", "Display name", "Experiment", "Status", "Started", "Action"]
        ):
            col.markdown(f"**{label}**")

        for i, j in enumerate(page_rows, start=start + 1):
            row = st.columns([0.6, 2.6, 1.4, 1.0, 1.4, 0.8])
            row[0].write(i)
            row[1].write(f"`{j.get('display_name') or j.get('job_name')}`")
            row[2].write(j.get("experiment_name") or "—")
            row[3].write(str(j.get("status") or "—"))
            ts = j.get("start_time")
            if isinstance(ts, _dt.datetime):
                ts = ts.strftime("%Y-%m-%d %H:%M")
            row[4].write(ts or "—")
            with row[5]:
                if st.button(
                    "Focus →",
                    key=f"home_focus_{j.get('job_name')}",
                    use_container_width=True,
                ):
                    st.session_state["focused_job"] = {
                        "job_name": j.get("job_name"),
                        "display_name": j.get("display_name"),
                        "experiment_name": j.get("experiment_name"),
                        "status": j.get("status"),
                        "start_time": j.get("start_time"),
                        "studio_url": j.get("studio_url"),
                    }
                    st.switch_page("pages/2_Focus.py")

        st.caption(
            f"Showing {start + 1}–{min(start + page_size, len(filtered))} "
            f"of {len(filtered)} matching jobs (page {page}/{n_pages})."
        )

    _live_panel()
else:
    st.info("API connecting — check sidebar for status.")
