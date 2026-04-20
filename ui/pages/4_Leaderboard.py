"""4️⃣ Leaderboard — Model metrics and performance comparison."""

import streamlit as st

from ui.api_client import get_client
from ui.components.metrics_charts import render_metrics_bar_chart, render_metrics_radar
from ui.components.metrics_table import render_metrics_table
from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="Leaderboard", page_icon="🏆", layout="wide")
render_sidebar()

st.title("🏆 Model Leaderboard")
st.markdown("Compare model performance metrics across pipeline jobs.")

client = get_client()

# ── Job Selection ─────────────────────────────────────────────
job_name = st.text_input(
    "Enter Job Name",
    placeholder="e.g. clever_banana_abc123",
    help="Pipeline job to fetch metrics from",
)

if not job_name:
    st.info("Enter a job name above to load the leaderboard.")
    st.stop()

# ── Load Metrics ──────────────────────────────────────────────
with st.spinner("Loading metrics..."):
    try:
        data = client.get_metrics(job_name)
        metrics = data.get("metrics", [])
    except Exception as exc:
        st.error(f"Failed to load metrics: {exc}")
        metrics = []

if not metrics:
    st.warning("No metrics found for this job.")
    st.stop()

st.success(f"Loaded **{len(metrics)}** metric records")

# ── Leaderboard Table ─────────────────────────────────────────
st.subheader("📊 Metrics Table")
df = render_metrics_table(metrics)

# ── Charts ────────────────────────────────────────────────────
st.subheader("📈 Visualizations")

if df is not None and not df.empty:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_cols:
        selected_metric = st.selectbox(
            "Select Metric for Chart",
            options=numeric_cols,
            index=0,
        )
        render_metrics_bar_chart(metrics, metric_key=selected_metric)

        # Radar chart with multi-select
        if len(numeric_cols) >= 2:
            radar_metrics = st.multiselect(
                "Select Metrics for Radar",
                options=numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))],
            )
            if radar_metrics:
                render_metrics_radar(metrics, radar_metrics)

# ── Download ──────────────────────────────────────────────────
if df is not None and not df.empty:
    st.divider()
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False),
        file_name=f"{job_name}_metrics.csv",
        mime="text/csv",
    )
