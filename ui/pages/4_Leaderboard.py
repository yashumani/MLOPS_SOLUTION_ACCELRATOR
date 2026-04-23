"""4️⃣ Leaderboard — Compare Phase A / B / C metrics for a job.

Two-step flow:
  1. Pick the job from the experiment → display-name tree.
  2. Click *Extract metrics* — only then do we hit the (slow) Azure ML
     download. Results are cached server-side for completed jobs.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.api_client import get_client
from ui.components.job_picker import pick_single_job
from ui.components.metrics_table import render_metrics_table
from ui.components.sidebar import render_sidebar
from ui.components.theme import inject_theme, page_header

st.set_page_config(page_title="Leaderboard", page_icon="🏆", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Model Leaderboard",
    "Phase A baseline vs Phase B variants vs Phase C HPO — compare per job",
    "🏆",
)

client = get_client()

# ── Step 1: pick the job ─────────────────────────────────────
sel = pick_single_job(
    client,
    key="leaderboard",
    status_filter=["Completed", "Finished", "Failed"],
    label_experiment="Experiment",
    label_job="Job (only completed/failed jobs have metrics)",
)

if not sel:
    st.info("Select a job above, then press **Extract metrics**.")
    st.stop()

job_name = sel["job_name"]
display_name = sel.get("display_name") or job_name

st.markdown(
    f"**Selected:** `{display_name}` &nbsp;·&nbsp; "
    f"experiment `{sel.get('experiment_name')}` &nbsp;·&nbsp; "
    f"status `{sel.get('status')}`"
)

# ── Step 2: explicit extraction trigger ──────────────────────
trigger = st.button(
    "📈 Extract metrics from this job",
    type="primary",
    key="lb_extract",
)

cache_key = f"lb_metrics_{job_name}"
if trigger:
    st.session_state.pop(cache_key, None)

if cache_key not in st.session_state and not trigger:
    st.stop()

with st.spinner(
    "Downloading aggregate reports from Azure ML… this can take ~30s on first hit."
):
    try:
        st.session_state[cache_key] = client.get_job_metrics(job_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load metrics: {exc}")
        st.session_state.pop(cache_key, None)
        st.stop()

data = st.session_state[cache_key]
metrics = data.get("models") or data.get("metrics") or []
task_type = data.get("task_type")

if not metrics:
    st.warning(
        "No metrics found. The job may not have produced aggregate reports "
        "(e.g. it failed before s05 baseline)."
    )
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────
champion = next((m for m in metrics if m.get("is_champion")), None)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", len(metrics))
k2.metric("Task type", task_type or "—")
k3.metric("Champion phase", (champion or {}).get("phase", "—"))
k4.metric("Champion engine", (champion or {}).get("engine") or "—")
st.divider()

# ── Table ────────────────────────────────────────────────────
st.markdown("#### Metrics table")
df = render_metrics_table(metrics)

# ── Charts ───────────────────────────────────────────────────
if df is not None and not df.empty:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.markdown("#### Visualizations")
        selected_metric = st.selectbox(
            "Bar chart metric", options=numeric_cols, index=0, key="lb_metric"
        )
        try:
            from streamlit_echarts import st_echarts

            name_col = next(
                (
                    c for c in df.columns
                    if c.lower() in ("model", "name", "run", "experiment")
                ),
                df.columns[0],
            )
            labels = df[name_col].astype(str).tolist()
            values = [float(v) for v in df[selected_metric].tolist()]

            bar_option = {
                "backgroundColor": "transparent",
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "grid": {"left": "3%", "right": "4%", "bottom": "10%", "containLabel": True},
                "xAxis": {
                    "type": "category",
                    "data": labels,
                    "axisLabel": {"color": "#475569", "rotate": 30, "fontSize": 11},
                    "axisLine": {"lineStyle": {"color": "#E2E8F0"}},
                },
                "yAxis": {
                    "type": "value",
                    "axisLabel": {"color": "#475569", "fontSize": 11},
                    "splitLine": {"lineStyle": {"color": "#E2E8F0"}},
                },
                "series": [{
                    "type": "bar",
                    "data": values,
                    "barMaxWidth": 50,
                    "itemStyle": {
                        "color": {
                            "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": "#2563EB"},
                                {"offset": 1, "color": "#06B6D4"},
                            ],
                        },
                        "borderRadius": [4, 4, 0, 0],
                    },
                }],
            }
            st_echarts(options=bar_option, height="320px", key="lb_bar")

            if len(numeric_cols) >= 2:
                radar_metrics = st.multiselect(
                    "Radar metrics",
                    options=numeric_cols,
                    default=numeric_cols[: min(4, len(numeric_cols))],
                    key="lb_radar_select",
                )
                if radar_metrics:
                    max_vals = {m: max(df[m].max(), 0.001) for m in radar_metrics}
                    radar_data = []
                    for _, row in df.iterrows():
                        radar_data.append({
                            "value": [float(row[m]) for m in radar_metrics],
                            "name": str(row[name_col]),
                        })
                    radar_option = {
                        "backgroundColor": "transparent",
                        "tooltip": {},
                        "legend": {
                            "data": [d["name"] for d in radar_data],
                            "textStyle": {"color": "#475569"},
                            "bottom": "0%",
                        },
                        "radar": {
                            "indicator": [
                                {"name": m, "max": max_vals[m]} for m in radar_metrics
                            ],
                            "splitLine": {"lineStyle": {"color": "#E2E8F0"}},
                            "axisLine": {"lineStyle": {"color": "#CBD5E1"}},
                            "axisName": {"color": "#475569"},
                            "splitArea": {"show": False},
                        },
                        "series": [{
                            "type": "radar",
                            "data": radar_data[:5],
                            "areaStyle": {"opacity": 0.15},
                            "lineStyle": {"width": 2},
                            "symbol": "circle",
                            "symbolSize": 5,
                        }],
                    }
                    st_echarts(options=radar_option, height="360px", key="lb_radar")
        except ImportError:
            st.caption("Install streamlit-echarts for charts.")

# ── Download ─────────────────────────────────────────────────
if df is not None and not df.empty:
    st.divider()
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False),
        file_name=f"{display_name}_metrics.csv",
        mime="text/csv",
    )
