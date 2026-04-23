"""2️⃣ Focus — single-job command center.

Replaces Job Monitor + Leaderboard + Outputs with one focused cockpit
view of *one* job. The job is identified by ``st.session_state["focused_job"]``
which is set from:

* the Dashboard's "Focus" action,
* the Submit page after a successful submission, or
* the picker rendered here when the slot is empty.

Each tab body lives inside an ``@st.fragment(run_every=…)`` so that only
the active panel re-runs on its own cadence — the rest of the page stays
interactive and the FastAPI backend isn't hammered.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from ui.api_client import get_client
from ui.components.drift_gauge import render_drift_gauge
from ui.components.drift_heatmap import render_drift_heatmap
from ui.components.job_picker import pick_single_job
from ui.components.log_stream import render_log_stream
from ui.components.metrics_table import render_metrics_table
from ui.components.sidebar import render_sidebar
from ui.components.status_badge import status_badge
from ui.components.step_timeline import render_step_timeline
from ui.components.theme import inject_theme, page_header
from ui.config import REFRESH_INTERVAL
from ui.data_cache import (
    cached_get_job,
    cached_job_drift,
    cached_job_metrics,
    cached_pipeline_summary,
    invalidate_job_caches,
)

st.set_page_config(page_title="Focus", page_icon="🎯", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Focus",
    "One job, one cockpit — Overview · Leaderboard · Outputs · Drift · Logs",
    "🎯",
)

client = get_client()

# ── Resolve focused job ──────────────────────────────────────
focused: dict | None = st.session_state.get("focused_job")

if not focused:
    st.info(
        "Pick a job to focus on. Once selected, every tab below targets the "
        "same job and refreshes independently."
    )
    sel = pick_single_job(
        client,
        key="focus_picker",
        label_experiment="Experiment",
        label_job="Job to focus",
    )
    if not sel:
        st.stop()
    st.session_state["focused_job"] = sel
    focused = sel
    st.rerun()

job_name: str = focused["job_name"]
display_name: str = focused.get("display_name") or job_name
known_status: str | None = focused.get("status")
studio_url: str | None = focused.get("studio_url")


# ── Sticky header (live status badge + actions) ──────────────
@st.fragment(run_every=f"{REFRESH_INTERVAL}s")
def _focus_header() -> None:
    detail = cached_get_job(job_name, known_status=known_status) or {}
    live_status = detail.get("status") or known_status or "Unknown"
    live_studio = detail.get("studio_url") or studio_url

    h1, h2, h3 = st.columns([5, 2, 3])
    with h1:
        st.markdown(
            f"### `{display_name}`  &nbsp;{status_badge(live_status)}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Job ID `{job_name}` · experiment `"
            f"{detail.get('experiment_name') or focused.get('experiment_name') or '—'}`"
        )
    with h2:
        if st.button("🔁 Change job", key="focus_change_btn", use_container_width=True):
            st.session_state["focused_job"] = None
            st.rerun()
        if live_studio:
            st.link_button(
                "↗ Studio", live_studio, use_container_width=True
            )
    with h3:
        status_lower = (live_status or "").lower()
        if status_lower in ("running", "preparing", "starting", "notstarted", "queued"):
            if st.button("❌ Cancel", key="focus_cancel_btn", use_container_width=True):
                try:
                    client.cancel_job(job_name)
                    invalidate_job_caches(job_name)
                    st.success("Cancellation requested.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Cancel failed: {exc}")
        elif status_lower in ("completed", "finished", "failed", "canceled", "cancelled"):
            if st.button("🔄 Resubmit", key="focus_resub_btn", use_container_width=True):
                try:
                    result = client.resubmit(job_name)
                    invalidate_job_caches()
                    new_name = result.get("job_name") or result.get("display_name")
                    st.success(f"Resubmitted → `{new_name}`")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Resubmit failed: {exc}")


_focus_header()
st.divider()


# ── Tabs ─────────────────────────────────────────────────────
tab_overview, tab_lb, tab_outputs, tab_drift, tab_logs = st.tabs(
    ["📊 Overview", "🏆 Live Leaderboard", "📦 Outputs", "📉 Drift", "📡 Logs"]
)


# ── Tab 1: Overview (step timeline + KPI strip) ──────────────
with tab_overview:

    @st.fragment(run_every=f"{REFRESH_INTERVAL}s")
    def _overview_panel() -> None:
        detail = cached_get_job(job_name, known_status=known_status) or {}
        steps = detail.get("steps") or []
        running = sum(
            1
            for s in steps
            if (s.get("status") or "").lower() in ("running", "preparing", "starting")
        )
        completed = sum(
            1
            for s in steps
            if (s.get("status") or "").lower() in ("completed", "finished")
        )
        failed = sum(1 for s in steps if (s.get("status") or "").lower() == "failed")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", len(steps))
        c2.metric("Running", running)
        c3.metric("Completed", completed)
        c4.metric("Failed", failed)

        st.markdown("#### Pipeline steps")
        if steps:
            render_step_timeline(steps)
        else:
            st.caption("No child steps reported yet.")

        with st.expander("📄 Raw metadata", expanded=False):
            st.json(detail)

    _overview_panel()


# ── Tab 2: Live Leaderboard (partial) ────────────────────────
with tab_lb:
    st.caption(
        "Phase A baseline → Phase B variants → Phase C HPO. Refreshes "
        "automatically; while a job is still running this shows whatever "
        "aggregate reports have been written so far."
    )

    @st.fragment(run_every=f"{REFRESH_INTERVAL}s")
    def _leaderboard_panel() -> None:
        try:
            data = cached_job_metrics(job_name, known_status=known_status) or {}
        except Exception as exc:  # noqa: BLE001
            st.warning(
                f"No metrics yet (`{exc}`). Aggregate reports appear after "
                "the corresponding stage finishes."
            )
            return

        metrics = data.get("models") or data.get("metrics") or []
        if not metrics:
            st.info(
                "No leaderboard rows yet — waiting for `s05`/`s06`/`s09` "
                "aggregate reports."
            )
            return

        task_type = data.get("task_type")
        champion = next((m for m in metrics if m.get("is_champion")), None)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", len(metrics))
        k2.metric("Task type", task_type or "—")
        k3.metric("Champion phase", (champion or {}).get("phase", "—"))
        k4.metric("Champion engine", (champion or {}).get("engine") or "—")

        st.markdown("#### Metrics")
        df = render_metrics_table(metrics)
        if df is not None and not df.empty:
            st.download_button(
                "⬇️ Download CSV",
                data=df.to_csv(index=False),
                file_name=f"{display_name}_metrics.csv",
                mime="text/csv",
                key="focus_lb_dl",
            )

    _leaderboard_panel()


# ── Tab 3: Outputs (summary + on-demand artifact extraction) ─
with tab_outputs:

    sub_summary, sub_artifacts = st.tabs(
        ["📊 Pipeline Summary", "📁 Named Artifacts"]
    )

    with sub_summary:

        @st.fragment(run_every=f"{REFRESH_INTERVAL}s")
        def _summary_panel() -> None:
            try:
                summary = cached_pipeline_summary(
                    job_name, known_status=known_status
                ) or {}
            except Exception as exc:  # noqa: BLE001
                st.info(f"No pipeline summary yet ({exc}).")
                return
            if not summary:
                st.info("Waiting for aggregate reports…")
                return

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Job", summary.get("job_name") or display_name)
            c2.metric("Task type", summary.get("task_type") or "—")
            c3.metric("Status", summary.get("status") or "—")
            champ = summary.get("champion_phase") or "—"
            score = summary.get("champion_score")
            disp = (
                f"{champ} ({score:.4f})"
                if isinstance(score, (int, float))
                else champ
            )
            c4.metric("Champion", disp)

            for title, key, accent in (
                ("Baseline aggregate", "baseline_aggregate", "#06B6D4"),
                ("Phase B aggregate", "phaseb_aggregate", "#7C3AED"),
                ("Phase C aggregate (HPO)", "phasec_aggregate", "#F59E0B"),
                ("Final report", "final_report", "#10B981"),
            ):
                payload = summary.get(key)
                with st.expander(f"📂 {title}", expanded=False):
                    if not payload:
                        st.caption("_(not available — output absent for this job)_")
                    else:
                        st.json(payload)

            avail = summary.get("available_outputs") or []
            st.caption(f"Available outputs: `{', '.join(avail) or '—'}`")

        _summary_panel()

    with sub_artifacts:
        st.caption(
            "List the job's named outputs, tick what you want, then download "
            "and parse each on demand. Heavy operations stay opt-in."
        )

        list_cache = f"focus_outputs_{job_name}"
        if st.button(
            "📁 Load output list", type="primary", key="focus_outputs_list_btn"
        ):
            st.session_state.pop(list_cache, None)

        if list_cache not in st.session_state:
            try:
                with st.spinner("Listing named outputs…"):
                    st.session_state[list_cache] = client.list_outputs(job_name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to list outputs: {exc}")
                st.stop()

        data = st.session_state.get(list_cache) or {}
        outputs = data.get("outputs", []) if isinstance(data, dict) else []

        if not outputs:
            st.info("No named outputs reported (yet) for this job.")
        else:
            selected: list[str] = []
            grid = st.columns(2)
            for i, o in enumerate(outputs):
                name = o.get("name") if isinstance(o, dict) else str(o)
                type_ = (o.get("type") if isinstance(o, dict) else None) or "—"
                with grid[i % 2]:
                    if st.checkbox(
                        f"`{name}` · {type_}",
                        key=f"focus_out_chk_{job_name}_{name}",
                    ):
                        selected.append(name)

            extract = st.button(
                "⬇️ Extract selected",
                type="primary",
                key="focus_outputs_extract_btn",
                disabled=not selected,
            )
            if extract and selected:
                for name in selected:
                    with st.expander(f"📦 {name}", expanded=True):
                        try:
                            with st.spinner(f"Downloading `{name}`…"):
                                content = client.get_output_content(
                                    job_name, name
                                )
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed: {exc}")
                            continue

                        files = content.get("files") or []
                        primary = content.get("primary_file")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Files", len(files))
                        m2.metric("Primary", primary or "—")
                        m3.metric("Truncated", "yes" if content.get("truncated") else "no")

                        if files:
                            st.dataframe(
                                pd.DataFrame(files),
                                use_container_width=True,
                                hide_index=True,
                            )
                        if content.get("json_content") is not None:
                            st.markdown("##### JSON content")
                            st.json(content["json_content"])
                        if content.get("csv_preview"):
                            st.markdown("##### CSV preview")
                            st.dataframe(
                                pd.DataFrame(content["csv_preview"]),
                                use_container_width=True,
                                hide_index=True,
                            )
                        if content.get("text_preview"):
                            st.markdown("##### Text preview")
                            st.code(content["text_preview"], language="text")

                        try:
                            st.download_button(
                                "⬇️ Download raw artifact",
                                data=client.download_output(job_name, name),
                                file_name=f"{display_name}_{name}.zip",
                                key=f"focus_dl_{name}",
                            )
                        except Exception as exc:  # noqa: BLE001
                            st.caption(f"Download not available: {exc}")


# ── Tab 4: Drift ─────────────────────────────────────────────
with tab_drift:
    st.caption(
        "PSI per feature from `s13_drift_monitor`. Only available once the "
        "drift step has produced its `drift_report` output."
    )

    @st.fragment(run_every=f"{REFRESH_INTERVAL}s")
    def _drift_panel() -> None:
        try:
            data = cached_job_drift(job_name, known_status=known_status) or {}
        except Exception as exc:  # noqa: BLE001
            st.info(f"No drift report yet ({exc}).")
            return

        features = data.get("features") or []
        if not features:
            st.info("Drift report not available for this job (yet).")
            return

        total = len(features)
        moderate = sum(1 for f in features if f.get("severity") == "moderate")
        severe = sum(1 for f in features if f.get("severity") == "severe")
        none = total - moderate - severe

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Features", total)
        c2.metric("No drift", none)
        c3.metric("Moderate", moderate)
        c4.metric("Severe", severe)

        stability = data.get("stability_score")
        if stability is not None:
            st.markdown(
                f"**Stability score:** `{stability:.2f} / 100` &nbsp; "
                f"**Drift type:** `{data.get('drift_type', '—')}` &nbsp; "
                f"**Overall drift:** "
                f"`{'YES' if data.get('overall_drift_detected') else 'no'}`"
            )

        st.markdown("#### PSI heatmap")
        try:
            render_drift_heatmap(features)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not render heatmap: {exc}")
            st.dataframe(pd.DataFrame(features), use_container_width=True)

        # Top-N gauges for the worst offenders
        worst = sorted(
            features,
            key=lambda f: float(f.get("psi") or 0.0),
            reverse=True,
        )[:4]
        if worst:
            st.markdown("#### Top features")
            cols = st.columns(len(worst))
            for col, f in zip(cols, worst):
                with col:
                    try:
                        render_drift_gauge(
                            float(f.get("psi") or 0.0), f.get("feature", "?")
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.caption(f"Gauge error: {exc}")

    _drift_panel()


# ── Tab 5: Logs ──────────────────────────────────────────────
with tab_logs:
    st.caption(
        "Tail logs from the parent job or any child step. Refreshes "
        "in-place via a Streamlit fragment — no full page reload."
    )

    detail_for_steps = cached_get_job(job_name, known_status=known_status) or {}
    steps = detail_for_steps.get("steps") or []
    step_names = [s.get("name", f"step_{i}") for i, s in enumerate(steps)]

    step_choice = st.selectbox(
        "Pipeline step",
        options=["(parent job)"] + step_names,
        key="focus_logs_step",
    )
    target = job_name if step_choice == "(parent job)" else step_choice

    @st.fragment(run_every=f"{REFRESH_INTERVAL * 2}s")
    def _logs_panel() -> None:
        try:
            log_data = (
                cached_get_job(target, known_status=known_status)
                if step_choice != "(parent job)"
                else (cached_get_job(job_name, known_status=known_status) or {})
            )
        except Exception as exc:  # noqa: BLE001
            render_log_stream(f"Failed to load logs: {exc}")
            return
        log_data = log_data or {}
        logs = log_data.get("logs") or log_data.get("log_text") or ""
        if not logs:
            studio = log_data.get("studio_url") or studio_url
            logs = (
                f"No direct log content for {target}.\n"
                f"Status: {log_data.get('status', 'Unknown')}\n"
            )
            if studio:
                logs += f"\nFor live logs, open in Azure ML Studio:\n{studio}\n"
        render_log_stream(logs)

    _logs_panel()
