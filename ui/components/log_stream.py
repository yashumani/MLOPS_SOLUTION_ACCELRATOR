"""Log stream component — display job logs."""

import streamlit as st


def render_log_stream(logs: str, height: int = 500):
    """Render a scrollable log viewer."""
    if not logs:
        st.info("No logs available.")
        return

    st.code(logs, language="log", line_numbers=True)
