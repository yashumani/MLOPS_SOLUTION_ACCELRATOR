"""Auto-retrain operations page."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from ui.api_client import get_client
from ui.components.sidebar import render_sidebar
from ui.components.theme import inject_theme, page_header
from ui.data_cache import cached_list_configs

st.set_page_config(page_title="Auto Retrain", page_icon="🔁", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Auto Retrain",
    "Schedules, approved baselines, controller plans, and decision ledger",
    "🔁",
)

client = get_client()


def _config_names() -> list[str]:
    try:
        configs_data = cached_list_configs() or {}
    except Exception:  # noqa: BLE001
        return []
    raw = configs_data.get("configs", [])
    if raw and isinstance(raw[0], dict):
        return [item.get("config_name") for item in raw if item.get("config_name")]
    return [str(item) for item in raw]


def _display_records(records: list[dict], *, key: str) -> None:
    if not records:
        st.info("No ledger records found.")
        return
    rows = []
    for record in records:
        rows.append({
            "timestamp": record.get("timestamp_utc"),
            "config": record.get("config_name"),
            "task": record.get("task_type"),
            "dataset": record.get("dataset_name"),
            "outcome": record.get("outcome"),
            "promotion": record.get("promotion_status"),
            "candidate_job": record.get("candidate_job_name"),
            "baseline_job": record.get("baseline_job_name"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Ledger JSON", expanded=False):
        st.json(records, expanded=False)


configs = _config_names()

tab_schedules, tab_baselines, tab_controller, tab_ledger = st.tabs(
    ["Schedules", "Baseline Approval", "Controller Plan", "Decision Ledger"]
)

with tab_schedules:
    try:
        schedule_data = client.list_auto_retrain_schedules(limit_records=10) or {}
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load auto-retrain schedules: {exc}")
        schedule_data = {}

    schedules = schedule_data.get("schedules") or []
    ledger_path = schedule_data.get("ledger_path") or "outputs/auto_retrain_decisions.jsonl"
    c1, c2 = st.columns(2)
    c1.metric("Planned schedules", schedule_data.get("total") or len(schedules))
    c2.metric("Ledger", Path(ledger_path).name)

    if schedules:
        st.dataframe(pd.DataFrame(schedules), use_container_width=True, hide_index=True)
    else:
        st.info("No planned schedules were returned by the API.")

    st.markdown("#### Recent decisions")
    _display_records(schedule_data.get("latest_records") or [], key="recent")

with tab_baselines:
    st.markdown("#### Approve Drift Baseline")
    if not configs:
        st.warning("No configs are available from the API.")
    else:
        with st.form("approve_baseline_form"):
            selected_config = st.selectbox(
                "Config",
                options=configs,
                key="auto_retrain_baseline_config",
            )
            baseline_job_name = st.text_input(
                "Baseline job name",
                placeholder="Completed job with drift_baseline output",
                key="auto_retrain_baseline_job",
            )
            output_baseline_uri = st.text_input(
                "Baseline URI override",
                placeholder="azureml://.../drift_baseline/...",
                key="auto_retrain_baseline_uri",
            )
            schedule_name = st.text_input(
                "Schedule name",
                placeholder="auto-retrain-classification-telecom-churn-daily",
                key="auto_retrain_baseline_schedule",
            )
            reason = st.text_area(
                "Approval note",
                value="Operator approved drift baseline for future auto-retrain.",
                key="auto_retrain_baseline_reason",
            )
            submitted = st.form_submit_button("Approve Baseline", type="primary")

        if submitted:
            payload = {
                "config_name": selected_config,
                "baseline_job_name": baseline_job_name or None,
                "output_baseline_uri": output_baseline_uri or None,
                "schedule_name": schedule_name or None,
                "reason": reason,
            }
            try:
                result = client.approve_auto_retrain_baseline(payload)
                st.success(f"Baseline approved and written to {result.get('ledger_path')}")
                st.json(result.get("record") or {}, expanded=False)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Baseline approval failed: {exc}")

with tab_controller:
    st.markdown("#### Build Controller Dry Run")
    if not configs:
        st.warning("No configs are available from the API.")
    else:
        with st.form("controller_plan_form"):
            selected_config = st.selectbox(
                "Config",
                options=configs,
                key="auto_retrain_plan_config",
            )
            trigger = st.text_input(
                "Trigger",
                value="manual_ui",
                key="auto_retrain_plan_trigger",
            )
            schedule_name = st.text_input(
                "Schedule name",
                placeholder="auto-retrain-classification-telecom-churn-daily",
                key="auto_retrain_plan_schedule",
            )
            decision_path = st.text_input(
                "S14 decision path",
                placeholder="retrain_decision.json",
                key="auto_retrain_plan_decision_path",
            )
            col1, col2 = st.columns(2)
            with col1:
                experiment_name = st.text_input(
                    "Experiment override",
                    placeholder="classification_telecom_churn_auto_retrain",
                    key="auto_retrain_plan_experiment",
                )
            with col2:
                display_name = st.text_input(
                    "Display name override",
                    placeholder="auto_retrain_controller_...",
                    key="auto_retrain_plan_display",
                )
            force_submit = st.checkbox(
                "Allow duplicate override in generated command",
                value=False,
                key="auto_retrain_plan_force",
            )
            force_reason = st.text_area(
                "Override reason",
                value="",
                key="auto_retrain_plan_force_reason",
            )
            planned = st.form_submit_button("Build Plan", type="primary")

        if planned:
            if not decision_path.strip():
                st.error("S14 decision path is required.")
                st.stop()
            payload = {
                "config_name": selected_config,
                "decision_path": decision_path.strip(),
                "trigger": trigger or "manual_ui",
                "schedule_name": schedule_name or None,
                "experiment_name": experiment_name or None,
                "display_name": display_name or None,
                "force_submit": force_submit,
                "force_reason": force_reason or None,
            }
            try:
                result = client.plan_auto_retrain(payload)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Plan failed: {exc}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Task", result.get("task_type") or "—")
                c2.metric("Dataset", result.get("dataset_name") or "—")
                c3.metric("Experiment", result.get("experiment_name") or "—")
                st.text_area("Canonical command", result.get("command") or "", height=140)
                st.json(result.get("pending_decision_record") or {}, expanded=False)

with tab_ledger:
    limit = st.slider("Records", min_value=10, max_value=500, value=100, step=10)
    if st.button("Refresh Ledger", key="auto_retrain_ledger_refresh"):
        st.session_state["auto_retrain_ledger_refresh_count"] = (
            st.session_state.get("auto_retrain_ledger_refresh_count", 0) + 1
        )
    try:
        decision_data = client.list_auto_retrain_decisions(limit=limit) or {}
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load decision ledger: {exc}")
        decision_data = {}
    st.caption(decision_data.get("ledger_path") or "outputs/auto_retrain_decisions.jsonl")
    _display_records(decision_data.get("records") or [], key="ledger")
