"""Config summary card — compact overview of a pipeline config."""

import streamlit as st


def render_config_summary_card(name: str, config: dict):
    """Render a compact summary card for a pipeline configuration."""
    dataset = config.get("dataset", {})
    azure = config.get("azure_ml", {})
    phases = config.get("phases", {})

    target = dataset.get("target_column", "—")
    task = config.get("task_type", "—")
    compute = azure.get("compute_target", "—")
    n_phases = len(phases)

    with st.container(border=True):
        st.markdown(f"#### 📋 {name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Task", task)
        c2.metric("Target", target)
        c3.metric("Compute", compute)
        c4.metric("Phases", n_phases)
