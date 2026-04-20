"""Metrics charts — Plotly visualizations for model performance."""

import streamlit as st
import plotly.express as px
import pandas as pd


def render_metrics_bar_chart(metrics: list[dict], metric_key: str = "value"):
    """Render a bar chart of model metrics."""
    if not metrics:
        return

    df = pd.DataFrame(metrics)
    if df.empty or metric_key not in df.columns:
        return

    name_col = "run_name" if "run_name" in df.columns else "name"
    if name_col not in df.columns:
        return

    fig = px.bar(
        df.sort_values(metric_key, ascending=False),
        x=name_col,
        y=metric_key,
        color=metric_key,
        color_continuous_scale="Viridis",
        title=f"Model Performance — {metric_key}",
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metrics_radar(metrics: list[dict], metric_keys: list[str]):
    """Render a radar/spider chart comparing models across metrics."""
    if not metrics or not metric_keys:
        return

    df = pd.DataFrame(metrics)
    available_keys = [k for k in metric_keys if k in df.columns]
    if not available_keys:
        return

    name_col = "run_name" if "run_name" in df.columns else "name"
    if name_col not in df.columns:
        return

    # Take top 5 models for readability
    df_top = df.head(5)

    fig = px.line_polar(
        df_top.melt(id_vars=[name_col], value_vars=available_keys),
        r="value",
        theta="variable",
        color=name_col,
        line_close=True,
        title="Model Comparison Radar",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
