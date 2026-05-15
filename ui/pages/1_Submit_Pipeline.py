"""1️⃣ Submit Pipeline — Launch new Azure ML pipeline jobs (async)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.api_client import get_client
from ui.components.config_summary_card import render_config_summary_card
from ui.components.sidebar import render_sidebar
from ui.components.theme import inject_theme, page_header
from ui.data_cache import (
    cached_get_config,
    cached_list_configs,
    invalidate_job_caches,
    prewarm,
)

st.set_page_config(page_title="Submit Pipeline", page_icon="🚀", layout="wide")
inject_theme()
render_sidebar()

page_header("Submit Pipeline", "Launch a new Azure ML V3 pipeline job", "🚀")

# Warm expensive API caches only after the page header is visible.
prewarm(st.session_state)

client = get_client()

# ── Config selection ──────────────────────────────────────────
try:
    configs_data = cached_list_configs()
    raw = configs_data.get("configs", [])
    if raw and isinstance(raw[0], dict):
        config_names = [c.get("config_name") for c in raw if c.get("config_name")]
    else:
        config_names = list(raw)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load configs: {exc}")
    config_names = []

if not config_names:
    st.warning("No configurations found. Check your API connection.")
    st.stop()

selected_config = st.selectbox(
    "Select Configuration",
    options=config_names,
    key="submit_config_select",
    help="Choose a pipeline configuration YAML",
)

# ── Summary card + YAML preview ───────────────────────────────
cfg_loaded: dict | None = None
if selected_config:
    try:
        cfg_loaded = cached_get_config(selected_config)
        # ConfigDetail wraps the YAML body in `content`.
        cfg_body = (
            cfg_loaded.get("content")
            if isinstance(cfg_loaded, dict) and "content" in cfg_loaded
            else cfg_loaded
        )
        with st.container(border=True):
            render_config_summary_card(selected_config, cfg_body)

        with st.expander("📄  Preview YAML before submit", expanded=False):
            try:
                import yaml as _yaml

                st.code(
                    _yaml.safe_dump(cfg_body, sort_keys=False, default_flow_style=False),
                    language="yaml",
                )
            except Exception:  # noqa: BLE001
                st.json(cfg_body)
            st.caption(
                "Hint: edit this config on the **Configs** page. The API "
                "validates against the config schema on save."
            )
    except Exception:  # noqa: BLE001
        st.info("Could not load config preview.")

# ── Advanced options ──────────────────────────────────────────
with st.expander("⚙️ Advanced Options", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        compute_override = st.text_input(
            "Compute target (override)",
            placeholder="cpu-cluster",
            help="Leave empty to use the compute target defined in the config.",
            key="submit_compute_override",
        )
        force_rerun = st.checkbox(
            "Disable component cache (force re-run all steps)",
            value=False,
            key="submit_force_rerun",
            help=(
                "When checked, every pipeline step is re-executed instead of "
                "reusing cached outputs from previous runs."
            ),
        )
    with col2:
        baseline_job = st.text_input(
            "Baseline job for drift comparison (optional)",
            placeholder="e.g. previous successful job_name",
            help=(
                "If set, the optional `s13_drift_monitor` step compares this "
                "run's data distribution against the baseline job and produces "
                "a PSI report viewable on the Drift Monitor page."
            ),
            key="submit_baseline_job",
        )
        tag_key = st.text_input(
            "Custom tag key",
            placeholder="team",
            help="Stored as Azure ML job tag for attribution.",
            key="submit_tag_key",
        )
        tag_val = st.text_input(
            "Custom tag value",
            placeholder="ml-engineering",
            key="submit_tag_val",
        )

# ── Submit ────────────────────────────────────────────────────
st.divider()
st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

if st.button("🚀 Submit Pipeline", type="primary", use_container_width=True, key="submit_btn"):
    tags = {}
    if tag_key and tag_val:
        tags[tag_key] = tag_val
    try:
        record = client.submit_pipeline_async(
            config_name=selected_config,
            compute=compute_override or None,
            force_rerun=force_rerun,
            baseline_job=baseline_job or None,
            tags=tags or None,
        )
        st.session_state["submit_request_id"] = record.get("request_id")
        st.session_state["submit_record"] = record
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Submission request failed: {exc}")

# ── Poll & hand off to Focus ──────────────────────────────────
request_id = st.session_state.get("submit_request_id")
if request_id:
    placeholder = st.empty()
    record = st.session_state.get("submit_record") or {}
    status = record.get("status", "pending")

    if status == "pending":
        with st.spinner(f"Submitting to Azure ML… (request_id={request_id})"):
            for _ in range(60):
                try:
                    record = client.get_submit_status(request_id)
                except Exception as exc:  # noqa: BLE001
                    placeholder.error(f"Polling failed: {exc}")
                    break
                status = record.get("status", "pending")
                if status != "pending":
                    st.session_state["submit_record"] = record
                    break
                time.sleep(1)
            else:
                placeholder.warning(
                    "Still pending after 60s — Azure ML is slow to respond. "
                    "Refresh this page; the submission continues in the background."
                )

    if status == "submitted":
        job_name = record.get("job_name")
        placeholder.success(f"✅ Pipeline submitted: **{job_name}**")
        st.session_state["focused_job"] = {
            "job_name": job_name,
            "display_name": record.get("display_name"),
            "experiment_name": record.get("experiment_name"),
            "status": "Starting",
            "start_time": record.get("completed_at"),
            "studio_url": record.get("studio_url"),
        }
        invalidate_job_caches()
        st.session_state.pop("submit_request_id", None)
        st.session_state.pop("submit_record", None)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "🎯 Open Focus", type="primary", use_container_width=True, key="submit_open_focus"
            ):
                st.switch_page("pages/2_Focus.py")
        with col_b:
            studio_url = record.get("studio_url")
            if studio_url:
                st.link_button(
                    "Open in Azure ML Studio →", studio_url, use_container_width=True
                )
    elif status == "failed":
        placeholder.error(f"❌ Submission failed: {record.get('error', 'unknown error')}")
        st.session_state.pop("submit_request_id", None)
        st.session_state.pop("submit_record", None)
