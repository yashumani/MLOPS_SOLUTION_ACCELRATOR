"""Status badge component for job/step statuses."""

import streamlit as st

# Status → (emoji, color name for st.markdown)
_STATUS_MAP = {
    "completed": ("✅", "green"),
    "finished": ("✅", "green"),
    "running": ("🔵", "blue"),
    "starting": ("🔵", "blue"),
    "preparing": ("🔵", "blue"),
    "queued": ("🟡", "orange"),
    "notstarted": ("⚪", "gray"),
    "notreported": ("⚪", "gray"),
    "failed": ("🔴", "red"),
    "canceled": ("⚫", "gray"),
    "cancelled": ("⚫", "gray"),
}


def status_badge(status: str) -> str:
    """Return a colored status string for display."""
    key = (status or "unknown").lower().replace(" ", "").replace("_", "")
    emoji, color = _STATUS_MAP.get(key, ("❓", "gray"))
    return f"{emoji} :{color}[{status}]"


def status_color(status: str) -> str:
    """Return the color name for a given status."""
    key = (status or "unknown").lower().replace(" ", "").replace("_", "")
    _, color = _STATUS_MAP.get(key, ("❓", "gray"))
    return color


def render_status_badge(status: str):
    """Render a status badge in Streamlit."""
    st.markdown(status_badge(status))
