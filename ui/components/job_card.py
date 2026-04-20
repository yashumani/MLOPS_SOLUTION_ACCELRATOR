"""Job card component for the Job Monitor page."""

import streamlit as st

from ui.components.status_badge import status_badge


def render_job_card(job: dict):
    """Render a single job as an expandable card."""
    name = job.get("job_name", "unknown")
    status = job.get("status", "Unknown")
    experiment = job.get("experiment_name", "—")
    created = job.get("created_time", "—")

    badge = status_badge(status)

    with st.expander(f"{badge}  **{name}**  •  {experiment}  •  {created}", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Job Name:** `{name}`")
        col2.markdown(f"**Status:** {badge}")
        col3.markdown(f"**Experiment:** {experiment}")

        studio_url = job.get("studio_url")
        if studio_url:
            st.markdown(f"🔗 [Open in Azure ML Studio]({studio_url})")

        tags = job.get("tags", {})
        if tags:
            st.markdown("**Tags:**")
            tag_str = ", ".join(f"`{k}={v}`" for k, v in tags.items())
            st.markdown(tag_str)

        # Steps
        steps = job.get("steps", [])
        if steps:
            st.markdown("---")
            st.markdown("**Steps:**")
            for step in steps:
                step_name = step.get("name", "?")
                step_status = step.get("status", "Unknown")
                st.markdown(f"  {status_badge(step_status)}  {step_name}")
