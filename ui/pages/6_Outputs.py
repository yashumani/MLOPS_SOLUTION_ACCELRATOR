"""6️⃣ Outputs — Pipeline Control Panel.

Two-step workflow:
  1. Pick a job from the experiment → display-name tree.
  2. Press *Load output list* → see metadata only (no downloads yet).
  3. Tick the named outputs you want to inspect, then *Extract selected* —
     each ticked output is downloaded and parsed (JSON / CSV / text).

A separate **Pipeline Summary** tab fetches the four aggregate reports
(``baseline_aggregate_report``, ``phaseb_aggregate_report``,
``phasec_aggregate_report``, ``final_report``) in one call.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from ui.api_client import get_client
from ui.components.job_picker import pick_single_job
from ui.components.sidebar import render_sidebar
from ui.components.theme import inject_theme, page_header

st.set_page_config(page_title="Outputs", page_icon="📦", layout="wide")
inject_theme()
render_sidebar()

page_header(
    "Pipeline Control Panel",
    "Browse named outputs and aggregate reports for any job — no run-IDs needed",
    "📦",
)

client = get_client()


# ── Helpers ──────────────────────────────────────────────────
def _fmt_size(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def render_value_card(label: str, value):
    if isinstance(value, float):
        value = round(value, 4)
    st.metric(label, "—" if value is None else value)


def render_json_as_table(data, depth: int = 0):
    if data is None:
        st.caption("_(no content)_")
        return
    if isinstance(data, list):
        if not data:
            st.caption("_(empty list)_")
            return
        if all(isinstance(item, dict) for item in data):
            df = pd.json_normalize(data, sep=".", max_level=1)
            st.dataframe(df, use_container_width=True, hide_index=True)
            return
        st.dataframe(
            pd.DataFrame({"value": data}), use_container_width=True, hide_index=True
        )
        return
    if not isinstance(data, dict):
        st.write(data)
        return

    scalars: dict = {}
    nested: dict = {}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            nested[k] = v
        else:
            scalars[k] = v
    if scalars:
        df = pd.DataFrame([{"field": k, "value": v} for k, v in scalars.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)
    for k, v in nested.items():
        if depth == 0:
            with st.expander(f"📂 {k}", expanded=False):
                render_json_as_table(v, depth + 1)
        else:
            st.markdown(f"**{k}**")
            render_json_as_table(v, depth + 1)


def render_summary_card(title: str, payload, *, accent: str = "#2563EB"):
    if not payload:
        st.markdown(
            f"<div style='border:1px solid #E2E8F0;border-left:3px solid {accent};"
            f"padding:0.75rem 1rem;border-radius:6px;background:#FFFFFF;"
            f"margin-bottom:0.75rem;'><div style='color:#475569;font-size:0.85rem;'>"
            f"{title}</div><div style='color:#94A3B8;font-size:0.8rem;margin-top:0.25rem;'>"
            "(not available — output absent for this job)</div></div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"<div style='border-left:3px solid {accent};padding-left:0.75rem;"
        f"margin:0.5rem 0;'><span style='color:#0F172A;font-weight:600;'>"
        f"{title}</span></div>",
        unsafe_allow_html=True,
    )
    render_json_as_table(payload, depth=0)


# ── Step 1: Pick the job ─────────────────────────────────────
sel = pick_single_job(
    client,
    key="outputs",
    label_experiment="Experiment",
    label_job="Job (display name → status → started)",
)

if not sel:
    st.info("Pick a job above to begin.")
    st.stop()

job_name = sel["job_name"]
display_name = sel.get("display_name") or job_name
studio_url = sel.get("studio_url")

st.markdown(
    f"**Selected:** `{display_name}` &nbsp;·&nbsp; "
    f"experiment `{sel.get('experiment_name')}` &nbsp;·&nbsp; "
    f"status `{sel.get('status')}`"
)
if studio_url:
    st.markdown(f"🔗 [Open in Azure ML Studio]({studio_url})")

tab_summary, tab_artifacts = st.tabs(
    ["📊 Pipeline Summary", "📁 Artifacts (selectable)"]
)


# ─────────────────────────────────────────────────────────────
# TAB 1 — Aggregated summary (fetches all four reports at once)
# ─────────────────────────────────────────────────────────────
with tab_summary:
    st.caption(
        "Combined view of `baseline_aggregate_report`, `phaseb_aggregate_report`, "
        "`phasec_aggregate_report` and `final_report` — fetched only on demand."
    )

    sum_trigger = st.button(
        "📊 Extract pipeline summary", type="primary", key="outputs_summary_btn"
    )
    sum_cache = f"summary_{job_name}"
    if sum_trigger:
        st.session_state.pop(sum_cache, None)

    if sum_cache in st.session_state or sum_trigger:
        with st.spinner("Downloading aggregate reports…"):
            try:
                st.session_state[sum_cache] = client.get_pipeline_summary(job_name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to load pipeline summary: {exc}")
                st.session_state.pop(sum_cache, None)

    summary = st.session_state.get(sum_cache)
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_value_card("Job", summary.get("job_name"))
        with c2:
            render_value_card("Task type", summary.get("task_type") or "—")
        with c3:
            render_value_card("Status", summary.get("status") or "—")
        with c4:
            champ = summary.get("champion_phase") or "—"
            score = summary.get("champion_score")
            display = (
                f"{champ} ({score:.4f})"
                if isinstance(score, (int, float)) else champ
            )
            render_value_card("Champion", display)

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            render_summary_card(
                "Baseline aggregate", summary.get("baseline_aggregate"),
                accent="#06B6D4",
            )
            render_summary_card(
                "Phase B aggregate", summary.get("phaseb_aggregate"),
                accent="#7C3AED",
            )
        with col_b:
            render_summary_card(
                "Phase C aggregate (HPO)", summary.get("phasec_aggregate"),
                accent="#F59E0B",
            )
            render_summary_card(
                "Final report", summary.get("final_report"),
                accent="#10B981",
            )

        st.caption(
            f"Available outputs in this job: "
            f"`{', '.join(summary.get('available_outputs', [])) or '—'}`"
        )


# ─────────────────────────────────────────────────────────────
# TAB 2 — Selectable artifacts: metadata first → tick → extract
# ─────────────────────────────────────────────────────────────
with tab_artifacts:
    list_trigger = st.button(
        "📁 Load output list", type="primary", key="outputs_list_btn"
    )
    list_cache = f"output_list_{job_name}"
    if list_trigger:
        st.session_state.pop(list_cache, None)

    if list_cache in st.session_state or list_trigger:
        with st.spinner("Listing named outputs…"):
            try:
                st.session_state[list_cache] = client.list_outputs(job_name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to list outputs: {exc}")
                st.session_state.pop(list_cache, None)

    data = st.session_state.get(list_cache)
    if not data:
        st.info("Press **Load output list** to see this job's named outputs.")
        st.stop()

    outputs = data.get("outputs", []) if isinstance(data, dict) else []
    if not outputs:
        st.warning("No named outputs reported for this job.")
        st.stop()

    st.caption(
        f"{len(outputs)} named output(s). Tick the ones you want, then click "
        "**Extract selected** to download and parse each."
    )

    # Render checkboxes in 2-column grid
    selected_outputs: list[str] = []
    grid_cols = st.columns(2)
    for i, o in enumerate(outputs):
        name = o.get("name") if isinstance(o, dict) else str(o)
        type_ = (o.get("type") if isinstance(o, dict) else None) or "—"
        with grid_cols[i % 2]:
            if st.checkbox(
                f"`{name}` · {type_}", key=f"out_chk_{job_name}_{name}"
            ):
                selected_outputs.append(name)

    bt_l, bt_r = st.columns([1, 5])
    with bt_l:
        extract_clicked = st.button(
            "⬇️ Extract selected", key="outputs_extract_btn", type="primary",
            disabled=not selected_outputs,
        )
    with bt_r:
        if not selected_outputs:
            st.caption("Tick at least one output to enable extraction.")

    if extract_clicked and selected_outputs:
        for name in selected_outputs:
            with st.expander(f"📦 {name}", expanded=True):
                with st.spinner(f"Downloading & parsing `{name}`…"):
                    try:
                        content = client.get_output_content(job_name, name)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed: {exc}")
                        continue

                files = content.get("files") or []
                primary = content.get("primary_file")
                total = sum((f.get("size_bytes") or 0) for f in files)

                m1, m2, m3 = st.columns(3)
                m1.metric("Files", len(files))
                m2.metric("Total size", _fmt_size(total))
                m3.metric("Primary", primary or "—")

                if files:
                    files_df = pd.DataFrame(files)
                    if "size_bytes" in files_df.columns:
                        files_df["size"] = files_df["size_bytes"].apply(_fmt_size)
                        files_df = files_df.drop(columns=["size_bytes"])
                    st.dataframe(
                        files_df, use_container_width=True, hide_index=True
                    )

                json_content = content.get("json_content")
                if json_content is not None:
                    st.markdown("##### JSON content")
                    render_json_as_table(json_content)

                csv_preview = content.get("csv_preview")
                if csv_preview:
                    st.markdown("##### CSV preview (first rows)")
                    st.dataframe(
                        pd.DataFrame(csv_preview),
                        use_container_width=True,
                        hide_index=True,
                    )

                text_preview = content.get("text_preview")
                if text_preview:
                    st.markdown("##### Text preview")
                    st.code(text_preview, language="text")

                if content.get("truncated"):
                    st.caption("⚠️ Output truncated for preview.")

                st.download_button(
                    "⬇️ Download raw artifact (file or zip)",
                    data=client.download_output(job_name, name),
                    file_name=f"{display_name}_{name}.zip",
                    key=f"dl_{name}",
                )
