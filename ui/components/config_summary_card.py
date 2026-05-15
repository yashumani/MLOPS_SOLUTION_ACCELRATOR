"""Config summary card — compact overview of a pipeline config."""

from __future__ import annotations

import streamlit as st


def _phase_b_variant_count(phases: dict) -> int | None:
    pb = phases.get("phase_b_recipes") or phases.get("phase_b") or []
    if isinstance(pb, list):
        return len(pb)
    if isinstance(pb, dict):
        recs = pb.get("recipes") or pb.get("variants")
        if isinstance(recs, list):
            return len(recs)
        for key in ("max_variants", "max_recipes"):
            if isinstance(pb.get(key), int):
                return pb[key]
    return None


def _phase_c_trial_budget(phases: dict) -> int | None:
    pc = phases.get("phase_c_hpo") or phases.get("phase_c") or {}
    if isinstance(pc, dict):
        for key in ("n_trials", "trials", "budget"):
            if isinstance(pc.get(key), int):
                return pc[key]
    return None


def _source_badge(name: str) -> str:
    """Heuristic — built-in configs follow `config_<task>_<dataset>_azureml` naming."""
    n = name.lower()
    if n.startswith("config_") and n.endswith("_azureml"):
        return "🔷 Built-in"
    if n.startswith("config_") and n.endswith("_local"):
        return "🟦 Built-in (local)"
    if "_copy" in n or "_user_" in n:
        return "🟧 User copy"
    return "⚪ Custom"


def render_config_summary_card(name: str, config: dict) -> None:
    """Render a compact summary card for a pipeline configuration."""
    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    azure = (
        config.get("azureml")
        or config.get("azure_ml")
        or {}
        if isinstance(config, dict)
        else {}
    )
    phases = config.get("phases", {}) if isinstance(config, dict) else {}

    target = dataset.get("target_column") or "—"
    task = config.get("task_type") if isinstance(config, dict) else None
    task = task or "—"
    compute = azure.get("compute_target") or "—"
    n_phases = len(phases) if isinstance(phases, dict) else 0
    pb_count = _phase_b_variant_count(phases) if isinstance(phases, dict) else None
    pc_trials = _phase_c_trial_budget(phases) if isinstance(phases, dict) else None

    badge = _source_badge(name)

    with st.container(border=True):
        st.markdown(f"#### 📋 `{name}`  &nbsp;<span style='font-size:0.85em;'>{badge}</span>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Task", task)
        c2.metric("Target", target)
        c3.metric("Compute", compute)
        c4.metric("Phases", n_phases)

        chips: list[str] = []
        if pb_count is not None:
            chips.append(f"🟣 Phase B: **{pb_count}** variant{'s' if pb_count != 1 else ''}")
        if pc_trials is not None:
            chips.append(f"🟠 Phase C: **{pc_trials}** HPO trials")
        dataset_path = dataset.get("path") or dataset.get("blob_path") or dataset.get("local_path")
        if dataset_path:
            chips.append(f"📂 Dataset: `{dataset_path}`")
        if chips:
            st.markdown(" &nbsp;·&nbsp; ".join(chips))
