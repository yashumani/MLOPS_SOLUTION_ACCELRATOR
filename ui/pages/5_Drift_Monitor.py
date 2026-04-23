"""5️⃣ Drift Monitor — Population Stability Index analysis.

Workflow:
  1. Pick a completed job (drift report only exists once s13 has run).
  2. Press *Extract drift report* — the API downloads ``drift_report``
     and returns parsed PSI scores.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from ui.api_client import get_client
from ui.components.job_picker import pick_single_job
from ui.components.sidebar import render_sidebar
from ui.components.theme import inject_theme, page_header

st.set_page_config(page_title="Drift Monitor", page_icon="📉", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Drift Monitor",
    "Population Stability Index per feature, with stability score & cadence",
    "📉",
)

client = get_client()

sel = pick_single_job(
    client,
    key="drift",
    status_filter=["Completed", "Finished"],
    label_experiment="Experiment",
    label_job="Job (must be Completed for drift_report to exist)",
)

if not sel:
    st.info("Pick a completed job above, then press **Extract drift report**.")
    st.stop()

job_name = sel["job_name"]
display_name = sel.get("display_name") or job_name

st.markdown(
    f"**Selected:** `{display_name}` &nbsp;·&nbsp; "
    f"experiment `{sel.get('experiment_name')}` &nbsp;·&nbsp; "
    f"status `{sel.get('status')}`"
)

trigger = st.button(
    "📉 Extract drift report from this job",
    type="primary",
    key="drift_extract",
)

cache_key = f"drift_data_{job_name}"
if trigger:
    st.session_state.pop(cache_key, None)

if cache_key not in st.session_state and not trigger:
    st.stop()

with st.spinner("Downloading drift_report from Azure ML…"):
    try:
        st.session_state[cache_key] = client.get_job_drift(job_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load drift data: {exc}")
        st.session_state.pop(cache_key, None)
        st.stop()

data = st.session_state[cache_key]
features = data.get("features") or []
studio_url = data.get("studio_url") or sel.get("studio_url")

if studio_url:
    st.markdown(f"🔗 [Open in Azure ML Studio]({studio_url})")

if not features:
    st.warning(
        "No drift results found for this job. Verify that step `s13_drift_monitor` "
        "completed successfully and produced the `drift_report` output."
    )
    st.stop()

# ── Summary KPIs ─────────────────────────────────────────────
total = len(features)
moderate = sum(1 for f in features if f.get("severity") == "moderate")
severe = sum(1 for f in features if f.get("severity") == "severe")
none = total - moderate - severe

c1, c2, c3, c4 = st.columns(4)
c1.metric("Features analyzed", total)
c2.metric("No drift", none)
c3.metric("Moderate drift", moderate)
c4.metric("Severe drift", severe)

stability_score = data.get("stability_score")
if stability_score is not None:
    st.markdown(
        f"**Stability score:** `{stability_score:.2f} / 100`  &nbsp; "
        f"**Drift type:** `{data.get('drift_type', '—')}`  &nbsp; "
        f"**Overall drift detected:** "
        f"`{'YES' if data.get('overall_drift_detected') else 'no'}`"
    )

st.divider()

# ── PSI bar chart ────────────────────────────────────────────
st.markdown("#### PSI per feature")
df = pd.DataFrame(features)
try:
    from streamlit_echarts import st_echarts

    feat_names = df["feature"].tolist()
    psi_vals = [float(v) for v in df["psi"].tolist()]

    bar_colors = []
    for v in psi_vals:
        if v < 0.10:
            bar_colors.append("#10B981")
        elif v < 0.25:
            bar_colors.append("#F59E0B")
        else:
            bar_colors.append("#EF4444")

    heatmap_option = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "4%", "bottom": "8%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": feat_names,
            "axisLabel": {"color": "#475569", "rotate": 35, "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#E2E8F0"}},
        },
        "yAxis": {
            "type": "value",
            "name": "PSI",
            "nameTextStyle": {"color": "#475569"},
            "axisLabel": {"color": "#475569"},
            "splitLine": {"lineStyle": {"color": "#E2E8F0"}},
            "max": max(max(psi_vals) * 1.2, 0.3) if psi_vals else 0.3,
        },
        "series": [{
            "type": "bar",
            "data": [
                {"value": v, "itemStyle": {"color": bar_colors[i],
                                            "borderRadius": [4, 4, 0, 0]}}
                for i, v in enumerate(psi_vals)
            ],
            "barMaxWidth": 40,
            "markLine": {
                "silent": True,
                "data": [
                    {"yAxis": 0.10, "lineStyle": {"color": "#10B981", "type": "dashed"},
                     "label": {"formatter": "Low (0.10)"}},
                    {"yAxis": 0.25, "lineStyle": {"color": "#EF4444", "type": "dashed"},
                     "label": {"formatter": "High (0.25)"}},
                ],
            },
        }],
    }
    st_echarts(options=heatmap_option, height="360px", key="drift_bar")
except ImportError:
    st.dataframe(df, use_container_width=True)

# ── Feature drill-down (gauges) ──────────────────────────────
st.markdown("#### Feature drill-down")
selected_features = st.multiselect(
    "Pick features to visualize as gauges",
    options=df["feature"].tolist(),
    default=df["feature"].tolist()[: min(4, len(df))],
    key="drift_gauges",
)

if selected_features:
    try:
        from streamlit_echarts import st_echarts

        cols = st.columns(min(4, len(selected_features)))
        for i, feat_name in enumerate(selected_features):
            psi = float(
                df.loc[df["feature"] == feat_name, "psi"].iloc[0]
            ) if feat_name in df["feature"].values else 0.0
            psi_pct = min(psi / 0.5 * 100, 100)
            color = "#10B981" if psi < 0.10 else ("#F59E0B" if psi < 0.25 else "#EF4444")
            gauge_option = {
                "backgroundColor": "transparent",
                "series": [{
                    "type": "gauge",
                    "startAngle": 200, "endAngle": -20,
                    "min": 0, "max": 100, "splitNumber": 4,
                    "radius": "90%",
                    "center": ["50%", "60%"],
                    "axisLine": {"lineStyle": {"width": 10, "color": [
                        [0.2, "#10B981"],
                        [0.5, "#F59E0B"],
                        [1, "#EF4444"],
                    ]}},
                    "pointer": {"length": "60%", "width": 4, "itemStyle": {"color": color}},
                    "axisTick": {"show": False},
                    "splitLine": {"show": False},
                    "axisLabel": {"show": False},
                    "title": {"offsetCenter": ["0%", "82%"], "color": "#475569", "fontSize": 11},
                    "detail": {"formatter": f"{psi:.4f}", "color": color,
                                "fontSize": 16, "fontWeight": "bold",
                                "offsetCenter": ["0%", "30%"]},
                    "data": [{"value": psi_pct, "name": feat_name}],
                }],
            }
            with cols[i % len(cols)]:
                st_echarts(options=gauge_option, height="200px", key=f"gauge_{i}_{feat_name}")
    except ImportError:
        for feat_name in selected_features:
            psi = float(df.loc[df["feature"] == feat_name, "psi"].iloc[0])
            st.write(f"**{feat_name}**: PSI = {psi:.4f}")

# ── Download ─────────────────────────────────────────────────
st.divider()
st.download_button(
    "⬇️ Download drift report (CSV)",
    data=df.to_csv(index=False),
    file_name=f"{display_name}_drift.csv",
    mime="text/csv",
)
