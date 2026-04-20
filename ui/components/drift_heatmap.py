"""Drift heatmap component — feature drift overview."""

import streamlit as st
import plotly.express as px
import pandas as pd


def render_drift_heatmap(drift_results: list[dict]):
    """Render a heatmap of PSI values across features."""
    if not drift_results:
        st.info("No drift data to visualize.")
        return

    df = pd.DataFrame(drift_results)
    if df.empty:
        st.info("No drift data to visualize.")
        return

    feature_col = "feature" if "feature" in df.columns else df.columns[0]
    psi_col = "psi" if "psi" in df.columns else None

    if psi_col is None:
        # Try to find numeric column
        numeric = df.select_dtypes(include=["number"]).columns
        if len(numeric) > 0:
            psi_col = numeric[0]
        else:
            st.warning("No numeric PSI column found.")
            return

    # Sort by PSI descending
    df_sorted = df.sort_values(psi_col, ascending=False)

    # Bar chart (more readable than heatmap for single-column data)
    fig = px.bar(
        df_sorted,
        x=feature_col,
        y=psi_col,
        color=psi_col,
        color_continuous_scale=[
            [0, "green"],
            [0.4, "yellow"],
            [1, "red"],
        ],
        title="Feature Drift (PSI)",
    )

    # Add threshold lines
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="High")

    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    return df_sorted
