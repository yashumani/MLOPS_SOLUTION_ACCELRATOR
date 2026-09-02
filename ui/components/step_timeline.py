"""Step timeline component for the current 14-stage Azure ML pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import streamlit as st

from ui.components.status_badge import status_badge, status_color


CANONICAL_STEPS = [
    {"key": "s0", "id": "S00", "label": "Data validation", "phase": "Data"},
    {"key": "s1", "id": "S01", "label": "Ingestion", "phase": "Data"},
    {"key": "s2", "id": "S02", "label": "Preparation", "phase": "Data"},
    {"key": "s3", "id": "S03", "label": "Preprocessing", "phase": "Data"},
    {"key": "s4", "id": "S04", "label": "Feature engineering", "phase": "Data"},
    {"key": "s5a", "id": "S05a", "label": "PyCaret baseline", "phase": "Phase A (Baseline)"},
    {"key": "s5b", "id": "S05b", "label": "FLAML baseline", "phase": "Phase A (Baseline)"},
    {"key": "s5t", "id": "S05t", "label": "Time-series baseline", "phase": "Phase A (Baseline)"},
    {"key": "s5z", "id": "S05z", "label": "Baseline aggregate", "phase": "Phase A (Baseline)"},
    {"key": "s06", "id": "S06", "label": "Variant runner", "phase": "Phase B (Variants)"},
    {"key": "s08", "id": "S08", "label": "Optuna HPO", "phase": "Phase C (HPO)"},
    {"key": "s09", "id": "S09", "label": "Phase C aggregate", "phase": "Phase C (HPO)"},
    {"key": "s10", "id": "S10", "label": "Final evaluation", "phase": "Final"},
    {"key": "s12", "id": "S12", "label": "Model registration", "phase": "Register"},
    {"key": "s13", "id": "S13", "label": "Drift monitor (optional)", "phase": "Monitor"},
]

_STAGE_ALIASES = {
    "s00": "s0",
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
    "data_validation": "s0",
    "validation": "s0",
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

    # Group stages by phase and render each group inside an expander so users
    # can collapse data-prep noise and focus on a single phase.
    from itertools import groupby

    for phase_name, phase_iter in groupby(CANONICAL_STEPS, key=lambda s: s["phase"]):
        phase_stages = list(phase_iter)
        present = [s for s in phase_stages if step_map.get(s["key"])]
        running = sum(
            1 for s in present
            if (step_map[s["key"]].get("status") or "").lower()
            in ("running", "preparing", "starting")
        )
        completed = sum(
            1 for s in present
            if (step_map[s["key"]].get("status") or "").lower()
            in ("completed", "finished")
        )
        failed = sum(
            1 for s in present
            if (step_map[s["key"]].get("status") or "").lower() == "failed"
        )
        summary = f" — {completed}/{len(phase_stages)} done"
        if running:
            summary += f", {running} running"
        if failed:
            summary += f", {failed} failed"

        # Default-expand phases with active or failed work; collapse the rest.
        expand = bool(running or failed) or phase_name.startswith("Phase")
        with st.expander(f"**{phase_name}**{summary}", expanded=expand):
            cols = st.columns(min(4, len(phase_stages)) or 1)
            for i, stage in enumerate(phase_stages):
                step = step_map.get(stage["key"])
                status = step.get("status") if step else "Not reported"
                caption = (
                    f"{status} (artifact)"
                    if step and step.get("is_inferred")
                    else status
                )
                color = status_color(status)
                with cols[i % len(cols)]:
                    st.markdown(f":{color}[**{stage['id']}**]  {stage['label']}")
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

    st.markdown("#### Step details")
    header = "| Stage | Name | Phase | Status | Source | Azure child job | Started |"
    separator = "|---|---|---|---|---|---|---|"
    body = "\n".join(
        "| " + " | ".join(str(row[col]).replace("|", "\\|") for col in row) + " |"
        for row in rows
    )
    st.markdown("\n".join([header, separator, body]))

    if extras:
        with st.expander(f"Unmapped Azure child steps ({len(extras)})", expanded=False):
            for step in extras:
                label = step.get("display_name") or step.get("name") or "unknown"
                safe_label = str(label).replace("`", "'")
                st.markdown(
                    f"{status_badge(step.get('status', 'Unknown'))} "
                    f"`{safe_label}`",
                )
