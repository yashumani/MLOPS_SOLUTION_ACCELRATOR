"""5️⃣ Drift Monitor — Data and model drift analysis."""

import streamlit as st

from ui.api_client import get_client
from ui.components.drift_gauge import render_drift_gauge
from ui.components.drift_heatmap import render_drift_heatmap
from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="Drift Monitor", page_icon="📉", layout="wide")
render_sidebar()

st.title("📉 Drift Monitor")
st.markdown("Analyze data drift using PSI (Population Stability Index) across features.")

client = get_client()

# ── Job Selection ─────────────────────────────────────────────
job_name = st.text_input(
    "Enter Job Name",
    placeholder="e.g. clever_banana_abc123",
    help="Pipeline job containing drift analysis results",
)

if not job_name:
    st.info("Enter a job name above to view drift results.")
    st.stop()

# ── Load Drift Data ───────────────────────────────────────────
with st.spinner("Loading drift report..."):
    try:
        data = client.get_drift(job_name)
        drift_results = data.get("drift_results", [])
        summary = data.get("summary", {})
    except Exception as exc:
        st.error(f"Failed to load drift data: {exc}")
        drift_results = []
        summary = {}

if not drift_results:
    st.warning("No drift results found for this job. Ensure s13_drift_monitor ran successfully.")
    st.stop()

# ── Summary ───────────────────────────────────────────────────
st.subheader("📊 Drift Summary")

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Features Analyzed", summary.get("total_features", len(drift_results)))
sc2.metric("No Drift", summary.get("no_drift", 0))
sc3.metric("Moderate Drift", summary.get("moderate_drift", 0))
sc4.metric("High Drift", summary.get("high_drift", 0))

# ── Heatmap ───────────────────────────────────────────────────
st.subheader("🔥 Drift Heatmap")
df = render_drift_heatmap(drift_results)

# ── Detailed Gauges ───────────────────────────────────────────
st.subheader("🎯 Feature Drill-Down")

# Let user select features for gauge view
feature_names = [d.get("feature", f"feature_{i}") for i, d in enumerate(drift_results)]
selected_features = st.multiselect(
    "Select features for gauge view",
    options=feature_names,
    default=feature_names[:min(4, len(feature_names))],
)

if selected_features:
    cols = st.columns(min(4, len(selected_features)))
    for i, feat_name in enumerate(selected_features):
        col = cols[i % len(cols)]
        # Find PSI value
        psi = 0.0
        for d in drift_results:
            if d.get("feature") == feat_name:
                psi = d.get("psi", 0.0)
                break
        with col:
            render_drift_gauge(psi, feat_name)

# ── Download ──────────────────────────────────────────────────
if df is not None and not df.empty:
    st.divider()
    st.download_button(
        "⬇️ Download Drift Report (CSV)",
        data=df.to_csv(index=False),
        file_name=f"{job_name}_drift.csv",
        mime="text/csv",
    )
