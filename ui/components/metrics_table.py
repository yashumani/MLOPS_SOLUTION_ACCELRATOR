"""Metrics table component — sortable model leaderboard."""

import pandas as pd
import streamlit as st


def _flatten_model_metric(item: dict) -> dict:
    """Flatten a ModelMetric dict {model_name, engine, phase, metrics:{...}, is_champion}
    into a single flat row suitable for a DataFrame."""
    if not isinstance(item, dict):
        return {"value": str(item)}

    metrics_dict = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    flat = {
        "🏆": "★" if item.get("is_champion") else "",
        "model": item.get("model_name") or item.get("model") or item.get("name") or "—",
        "phase": item.get("phase") or "—",
        "engine": item.get("engine") or "—",
    }
    # Append metric columns (round floats for display)
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            flat[k] = round(float(v), 4)
        else:
            flat[k] = v
    # Include any extra top-level numeric/string fields not already mapped
    for k, v in item.items():
        if k in {"model_name", "model", "name", "engine", "phase",
                 "metrics", "is_champion"}:
            continue
        if k not in flat:
            flat[k] = v
    return flat


def render_metrics_table(metrics: list[dict]):
    """Render a list of ModelMetric (or generic) dicts as a sortable DataFrame."""
    if not metrics:
        st.info("No metrics data available.")
        return None

    rows = [_flatten_model_metric(m) for m in metrics]
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No metrics data available.")
        return None

    # Reorder: champion flag, model, phase, engine, then everything else
    leading = [c for c in ("🏆", "model", "phase", "engine") if c in df.columns]
    remaining = [c for c in df.columns if c not in leading]
    df = df[leading + remaining]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(420, 60 + 38 * len(df)),
    )
    return df
