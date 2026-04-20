"""Step timeline component — visual pipeline progress."""

import streamlit as st

from ui.components.status_badge import status_badge, status_color

# Canonical V3 step order
V3_STEPS = [
    "s00_data_validation",
    "s01_ingestion",
    "s02_preparation",
    "s03_preprocessing",
    "s04_feature_engineering",
    "s05a_pycaret_train",
    "s05b_flaml_train",
    "s05t_timeseries_train",
    "s05z_aggregate_baseline",
    "s06_phaseb_variant_runner",
    "s07_pipeline_attribution",
    "s08_model_selection",
    "s09_phasec_hpo",
    "s10_final_evaluation",
    "s11_aggregate_phasec",
    "s12_model_registration",
    "s13_drift_monitor",
]


def render_step_timeline(steps: list[dict]):
    """Render a horizontal step timeline with status indicators."""
    if not steps:
        st.info("No step data available.")
        return

    # Build lookup by name
    step_map = {}
    for s in steps:
        name = s.get("name", "")
        # Match to canonical name
        for canon in V3_STEPS:
            if canon in name.lower() or name.lower() in canon:
                step_map[canon] = s
                break
        else:
            step_map[name] = s

    # Render columns
    n = len(V3_STEPS)
    cols = st.columns(min(n, 17))
    for i, step_name in enumerate(V3_STEPS):
        col = cols[i % len(cols)]
        step_data = step_map.get(step_name, {})
        step_status = step_data.get("status", "NotStarted")
        short_name = step_name.split("_", 1)[0]  # e.g., "s00"

        with col:
            color = status_color(step_status)
            st.markdown(f":{color}[**{short_name}**]")
            st.caption(step_status[:8])
