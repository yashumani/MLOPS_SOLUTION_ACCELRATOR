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
from ui.data_cache import cached_job_drift, cached_list_jobs

st.set_page_config(page_title="Drift Monitor", page_icon="📉", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Drift Monitor",
    "Population Stability Index per feature, with stability score & cadence",
    "📉",
)

client = get_client()


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 50) -> str:
    if df.empty:
        return ""
    display_df = df.head(max_rows).copy()
    display_df.columns = [str(col) for col in display_df.columns]
    for col in display_df.columns:
        if display_df[col].dtype == "object":
            display_df[col] = display_df[col].map(lambda value: "" if value is None else str(value))
    columns = list(display_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in display_df.iterrows():
        values = [str(row[col]).replace("|", "\\|").replace("\n", " ") for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    suffix = ""
    if len(df) > max_rows:
        suffix = f"\n\n_Showing first {max_rows} of {len(df)} rows. Download CSV for the full table._"
    return "\n".join([header, separator, *rows]) + suffix

sel = st.session_state.get("drift_selected_job")
if not sel:
    try:
        recent_completed = (
            cached_list_jobs(status="Completed", max_results=1) or {}
        ).get("jobs") or []
    except Exception:  # noqa: BLE001
        recent_completed = []
    if recent_completed:
        sel = recent_completed[0]
        st.session_state["drift_selected_job"] = sel
    else:
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

if not trigger and not st.session_state.get(f"drift_loaded_{job_name}"):
    st.stop()

st.session_state[f"drift_loaded_{job_name}"] = True

with st.spinner("Downloading drift_report from Azure ML…"):
    try:
        data = cached_job_drift(job_name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load drift data: {exc}")
        st.session_state.pop(f"drift_loaded_{job_name}", None)
        st.stop()
features = data.get("features") or []
studio_url = data.get("studio_url") or sel.get("studio_url")

if studio_url:
    st.markdown(f"🔗 [Open in Azure ML Studio]({studio_url})")

if not features:
    st.warning(
        "No drift results found for this job. Drift analysis is an **optional** "
        "step (`s13_drift_monitor`) — it is only present when the pipeline was "
        "submitted with a baseline job for comparison. Re-submit with a "
        "`baseline_job` set on the Submit Pipeline page to enable it."
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

# ── PSI legend ───────────────────────────────────────────────
with st.expander("ℹ️  How to read PSI", expanded=False):
    st.markdown(
        """
**Population Stability Index (PSI)** measures how much a feature's
distribution has shifted relative to the baseline used during training.

| PSI range | Severity | Action |
|-----------|----------|--------|
| `< 0.10`  | 🟢 Negligible | None — distribution is stable |
| `0.10 – 0.25` | 🟠 Moderate | Investigate; consider monitoring closely |
| `> 0.25` | 🔴 Severe | Retrain — distribution has materially changed |

PSI is symmetric and unitless. The **Stability score** above is
`100 × (1 − mean(PSI))` clipped to `[0, 100]`.
"""
    )

st.divider()

# ── PSI bar chart ────────────────────────────────────────────
df = pd.DataFrame(features)
# Ensure psi is numeric for downstream sort/filter.
if "psi" in df.columns:
    df["psi"] = pd.to_numeric(df["psi"], errors="coerce").fillna(0.0)

with st.expander("📊  PSI per feature (chart)", expanded=True):
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
        st.markdown(_markdown_table(df))


# ── Searchable / sortable feature table ──────────────────────
with st.expander("🔍  Feature table (search + sort)", expanded=True):
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        feat_search = st.text_input(
            "Search feature name",
            placeholder="age, region_id, …",
            key="drift_feat_search",
        )
    with fc2:
        sev_filter = st.multiselect(
            "Severity",
            ["none", "moderate", "severe"],
            default=[],
            key="drift_sev_filter",
        )

    table_df = df.copy()
    if "severity" not in table_df.columns and "psi" in table_df.columns:
        def _sev(v: float) -> str:
            return "none" if v < 0.10 else ("moderate" if v < 0.25 else "severe")
        table_df["severity"] = table_df["psi"].apply(_sev)

    if feat_search:
        table_df = table_df[
            table_df["feature"].str.contains(feat_search, case=False, na=False)
        ]
    if sev_filter:
        table_df = table_df[table_df["severity"].isin(sev_filter)]

    table_df = table_df.sort_values("psi", ascending=False)
    st.caption(f"{len(table_df)} of {len(df)} features shown")
    st.markdown(_markdown_table(table_df))

# ── Top-N gauges (worst offenders) ───────────────────────────
worst_n = df.sort_values("psi", ascending=False).head(4)
if not worst_n.empty:
    with st.expander("🎯  Top-4 most-drifted features (gauges)", expanded=False):
        try:
            from streamlit_echarts import st_echarts

            cols = st.columns(min(4, len(worst_n)))
            for i, (_, row) in enumerate(worst_n.iterrows()):
                feat_name = row["feature"]
                psi = float(row["psi"])
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
                    st_echarts(options=gauge_option, height="200px",
                               key=f"gauge_{i}_{feat_name}")
        except ImportError:
            for _, row in worst_n.iterrows():
                st.write(f"**{row['feature']}**: PSI = {float(row['psi']):.4f}")

# ── Download ─────────────────────────────────────────────────
st.divider()
st.download_button(
    "⬇️ Download drift report (CSV)",
    data=df.to_csv(index=False),
    file_name=f"{display_name}_drift.csv",
    mime="text/csv",
)
