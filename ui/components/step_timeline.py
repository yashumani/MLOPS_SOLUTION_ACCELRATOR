"""Step timeline component for the current 14-stage Azure ML pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from ui.components.status_badge import status_badge, status_color


CANONICAL_STEPS = [
    {"key": "s1", "id": "S01", "label": "Ingestion", "phase": "Data"},
    {"key": "s2", "id": "S02", "label": "Preparation", "phase": "Data"},
    {"key": "s3", "id": "S03", "label": "Preprocessing", "phase": "Data"},
    {"key": "s4", "id": "S04", "label": "Feature engineering", "phase": "Data"},
    {"key": "s5a", "id": "S05a", "label": "PyCaret baseline", "phase": "Baseline"},
    {"key": "s5b", "id": "S05b", "label": "FLAML baseline", "phase": "Baseline"},
    {"key": "s5t", "id": "S05t", "label": "Time-series baseline", "phase": "Baseline"},
    {"key": "s5z", "id": "S05z", "label": "Baseline aggregate", "phase": "Baseline"},
    {"key": "s06", "id": "S06", "label": "Variant runner", "phase": "Phase B"},
    {"key": "s08", "id": "S08", "label": "Optuna HPO", "phase": "Phase C"},
    {"key": "s09", "id": "S09", "label": "Phase C aggregate", "phase": "Phase C"},
    {"key": "s10", "id": "S10", "label": "Final evaluation", "phase": "Final"},
    {"key": "s12", "id": "S12", "label": "Model registration", "phase": "Register"},
    {"key": "s13", "id": "S13", "label": "Drift monitor", "phase": "Monitor"},
]

_STAGE_ALIASES = {
    "s01": "s1",
    "s02": "s2",
    "s03": "s3",
    "s04": "s4",
    "s05a": "s5a",
    "s05b": "s5b",
    "s05t": "s5t",
    "s05z": "s5z",
    "s6": "s06",
    "s8": "s08",
    "s9": "s09",
}

_KEYWORDS = {
    "ingestion": "s1",
    "preparation": "s2",
    "preprocessing": "s3",
    "feature_engineering": "s4",
    "pycaret": "s5a",
    "flaml": "s5b",
    "timeseries": "s5t",
    "forecasting": "s5t",
    "baseline_aggregate": "s5z",
    "aggregate_baseline": "s5z",
    "variant_runner": "s06",
    "phase_b": "s06",
    "phaseb": "s06",
    "optuna": "s08",
    "phasec_hpo": "s08",
    "phase_c_hpo": "s08",
    "phasec_aggregate": "s09",
    "aggregate_phasec": "s09",
    "final_evaluation": "s10",
    "model_registration": "s12",
    "drift_monitor": "s13",
    "drift": "s13",
}


def _infer_stage_key(step: dict[str, Any]) -> str | None:
    explicit = step.get("stage_key")
    if explicit:
        return _STAGE_ALIASES.get(str(explicit).lower(), str(explicit).lower())

    text = " ".join(
        str(step.get(k) or "")
        for k in ("display_name", "name", "job_name", "component")
    ).lower()
    for token in re.findall(r"s\d{1,2}[a-z]?", text):
        normalized = _STAGE_ALIASES.get(token, token)
        if normalized in {s["key"] for s in CANONICAL_STEPS}:
            return normalized

    compact = text.replace("-", "_").replace(" ", "_")
    for keyword, key in _KEYWORDS.items():
        if keyword in compact:
            return key
    return None


def _format_time(value: Any) -> str:
    if not value:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value).replace("T", " ").replace("Z", "")[:16]


def render_step_timeline(steps: list[dict]):
    """Render the actual V3 stages and avoid false NotStarted placeholders."""
    if not steps:
        st.info("No child step data available from Azure ML yet.")
        return

    step_map: dict[str, dict] = {}
    extras: list[dict] = []
    for step in steps:
        key = _infer_stage_key(step)
        if key:
            step_map[key] = step
        else:
            extras.append(step)

    cols = st.columns(7)
    for i, stage in enumerate(CANONICAL_STEPS):
        step = step_map.get(stage["key"])
        status = step.get("status") if step else "Not reported"
        caption = f"{status} (artifact)" if step and step.get("is_inferred") else status
        color = status_color(status)
        label = stage["id"]
        with cols[i % len(cols)]:
            st.markdown(f":{color}[**{label}**]")
            st.caption(caption)

    rows = []
    for stage in CANONICAL_STEPS:
        step = step_map.get(stage["key"], {})
        source = (
            "Output artifact"
            if step.get("is_inferred")
            else "Azure child"
            if step
            else "Not reported"
        )
        rows.append(
            {
                "Stage": stage["id"],
                "Name": stage["label"],
                "Phase": stage["phase"],
                "Status": step.get("status") or "Not reported",
                "Source": source,
                "Azure child job": step.get("name") or "—",
                "Started": _format_time(step.get("start_time")),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    if extras:
        with st.expander(f"Unmapped Azure child steps ({len(extras)})", expanded=False):
            for step in extras:
                label = step.get("display_name") or step.get("name") or "unknown"
                st.markdown(
                    f"{status_badge(step.get('status', 'Unknown'))} `{label}`",
                    unsafe_allow_html=True,
                )
