"""Drift gauge component — visual PSI severity indicator."""

import streamlit as st
import plotly.graph_objects as go


def render_drift_gauge(psi_value: float, feature_name: str = "Feature"):
    """Render a gauge chart showing PSI-based drift severity."""
    # PSI thresholds: <0.1 = No drift, 0.1-0.2 = Moderate, >0.2 = High
    if psi_value < 0.1:
        severity = "No Drift"
        color = "green"
    elif psi_value < 0.2:
        severity = "Moderate"
        color = "orange"
    else:
        severity = "High Drift"
        color = "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=psi_value,
            title={"text": f"{feature_name}<br><span style='font-size:0.7em'>{severity}</span>"},
            gauge={
                "axis": {"range": [0, max(0.5, psi_value * 1.2)]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 0.1], "color": "lightgreen"},
                    {"range": [0.1, 0.2], "color": "lightyellow"},
                    {"range": [0.2, max(0.5, psi_value * 1.2)], "color": "lightsalmon"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": 0.2,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=80, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
