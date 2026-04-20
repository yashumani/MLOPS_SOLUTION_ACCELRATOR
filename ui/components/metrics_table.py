"""Metrics table component — sortable model leaderboard."""

import streamlit as st
import pandas as pd


def render_metrics_table(metrics: list[dict]):
    """Render a list of metric dicts as a sortable table."""
    if not metrics:
        st.info("No metrics data available.")
        return

    df = pd.DataFrame(metrics)
    if df.empty:
        st.info("No metrics data available.")
        return

    # Reorder columns — put name/model first
    priority = ["run_name", "model", "name", "step"]
    leading = [c for c in priority if c in df.columns]
    remaining = [c for c in df.columns if c not in leading]
    df = df[leading + remaining]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 40 + 35 * len(df)),
    )

    return df
