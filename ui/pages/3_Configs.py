"""3️⃣ Configs — Browse, inspect, and edit pipeline configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import yaml

from ui.api_client import get_client
from ui.components.config_summary_card import render_config_summary_card
from ui.components.config_viewer import render_config_viewer
from ui.components.sidebar import render_sidebar
from ui.components.theme import inject_theme, page_header
from ui.data_cache import (
    cached_get_config,
    cached_list_configs,
    invalidate_config_caches,
    prewarm,
)

st.set_page_config(page_title="Configurations", page_icon="⚙️", layout="wide")
inject_theme()
render_sidebar()

page_header("Pipeline Configs", "Browse, inspect, and edit pipeline configuration files", "⚙️")

# Warm expensive API caches only after the page header is visible.
prewarm(st.session_state)

with st.expander("ℹ️  What is a pipeline config?", expanded=False):
    st.markdown(
        """
A **pipeline config** is a single YAML file that fully describes one Azure ML
training run:

- **`dataset`** — datastore path + target column
- **`azureml`** — workspace, compute target, environment names
- **`stage1`…`stage5`** — validation, preparation, preprocessing, feature engineering, baseline settings
- **`phases`** — what to train:
  - `phase_a_baseline` → quick PyCaret + FLAML baseline
  - `phase_b_recipes` → top-N recommended variants × engines
  - `phase_c_hpo` → Optuna search on the Phase B champion
- **`final_evaluation`** — holdout metrics & plots, when present

You can:
1. Browse every config (cards below).
2. Open one in **Detail** to read / edit its YAML, duplicate it, or delete it.
3. Submit it from the **Submit Pipeline** page.

The API validates against `src/orchestration/config_schema.py` on save, so
invalid YAML is rejected before it ever reaches Azure ML.
"""
    )

client = get_client()


def _config_names_from_response(data) -> list[str]:
    raw = data.get("configs", []) if isinstance(data, dict) else []
    if raw and isinstance(raw[0], dict):
        return [c.get("config_name") for c in raw if c.get("config_name")]
    return list(raw)


def _refresh_and_rerun() -> None:
    invalidate_config_caches()
    st.rerun()


def _config_body(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    content = payload.get("content")
    return content if isinstance(content, dict) else payload


# ── Config List ───────────────────────────────────────────────
try:
    data = cached_list_configs()
    config_names = _config_names_from_response(data)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to list configs: {exc}")
    config_names = []

# ── Search / filter bar ───────────────────────────────────────
fc1, fc2, fc3 = st.columns([2, 1.5, 1.5])
with fc1:
    search = st.text_input(
        "🔍 Search by name",
        placeholder="telecom_churn, college, …",
        key="cfg_search",
    )
with fc2:
    task_filter = st.selectbox(
        "Task type",
        ["All", "classification", "regression", "clustering", "timeseries"],
        key="cfg_task_filter",
    )
with fc3:
    source_filter = st.selectbox(
        "Source",
        ["All", "Built-in", "User copy", "Custom"],
        key="cfg_source_filter",
    )

# Pre-load configs we'll show — single shot, cached.
loaded: list[tuple[str, dict]] = []
for name in config_names:
    try:
        cfg = _config_body(cached_get_config(name) or {})
    except Exception:  # noqa: BLE001
        cfg = {}
    loaded.append((name, cfg))


def _matches(name: str, cfg: dict) -> bool:
    if search and search.lower() not in name.lower():
        return False
    if task_filter != "All":
        if (cfg.get("task_type") or "").lower() != task_filter:
            return False
    if source_filter != "All":
        n = name.lower()
        is_builtin = n.startswith("config_") and (
            n.endswith("_azureml") or n.endswith("_local")
        )
        is_user_copy = "_copy" in n or "_user_" in n
        if source_filter == "Built-in" and not is_builtin:
            return False
        if source_filter == "User copy" and not is_user_copy:
            return False
        if source_filter == "Custom" and (is_builtin or is_user_copy):
            return False
    return True


visible = [(n, c) for n, c in loaded if _matches(n, c)]
st.info(
    f"Showing **{len(visible)}** of **{len(config_names)}** configuration(s)"
    + (" (filtered)" if len(visible) != len(config_names) else "")
)

# ── Summary Cards ─────────────────────────────────────────────
for name, cfg in visible:
    if cfg:
        render_config_summary_card(name, cfg)
    else:
        st.markdown(f"📋 **{name}** — _unable to load_")

# ── Detail / Edit / Delete ────────────────────────────────────
st.divider()
st.subheader("🔍 Config Detail")

if config_names:
    selected = st.selectbox("Select config", config_names, key="cfg_select")
else:
    selected = None
    st.warning("No configs available to inspect.")

if selected:
    try:
        cfg_detail = cached_get_config(selected)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load config: {exc}")
        cfg_detail = None

    if cfg_detail:
        # Detail viewer + download
        with st.container(border=True):
            render_config_viewer(cfg_detail, title=selected)
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(cfg_detail, indent=2, default=str),
                file_name=f"{selected}.json",
                mime="application/json",
                key="cfg_download_btn",
            )

        # Edit / Duplicate / Delete tabs
        tab_edit, tab_dup, tab_del = st.tabs(["✏️ Edit", "📑 Duplicate", "🗑️ Delete"])

        # The detail payload is `ConfigDetail` which embeds `content`. Strip
        # it for the YAML editor.
        edit_content = (
            cfg_detail.get("content") if isinstance(cfg_detail, dict) and "content" in cfg_detail
            else cfg_detail
        )
        edit_yaml = yaml.safe_dump(edit_content, sort_keys=False, default_flow_style=False)

        with tab_edit:
            st.caption("Edit YAML in place. Saving validates against the config schema.")
            new_yaml = st.text_area(
                "YAML",
                value=edit_yaml,
                height=420,
                key=f"cfg_edit_yaml_{selected}",
            )
            if st.button("💾 Save", type="primary", key="cfg_edit_save"):
                try:
                    parsed = yaml.safe_load(new_yaml)
                    if not isinstance(parsed, dict):
                        st.error("Top-level YAML must be a mapping.")
                    else:
                        client.update_config(selected, parsed)
                        st.success(f"✅ Saved `{selected}`")
                        _refresh_and_rerun()
                except yaml.YAMLError as exc:
                    st.error(f"YAML parse error: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Save failed: {exc}")

        with tab_dup:
            st.caption("Save the current YAML under a new config name.")
            new_name = st.text_input(
                "New config name",
                placeholder="config_classification_my_copy_azureml",
                key="cfg_dup_name",
            )
            dup_yaml = st.text_area(
                "YAML",
                value=edit_yaml,
                height=320,
                key=f"cfg_dup_yaml_{selected}",
            )
            if st.button("📑 Create copy", type="primary", key="cfg_dup_create"):
                if not new_name.strip():
                    st.error("Please provide a name.")
                else:
                    try:
                        parsed = yaml.safe_load(dup_yaml)
                        if not isinstance(parsed, dict):
                            st.error("Top-level YAML must be a mapping.")
                        else:
                            client.create_config(new_name.strip(), parsed)
                            st.success(f"✅ Created `{new_name.strip()}`")
                            _refresh_and_rerun()
                    except yaml.YAMLError as exc:
                        st.error(f"YAML parse error: {exc}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Create failed: {exc}")

        with tab_del:
            st.caption(
                "Deletion is refused if a non-terminal Azure ML job is using this config."
            )
            confirm = st.text_input(
                f"Type the config name to confirm: **{selected}**",
                key="cfg_del_confirm",
            )
            if st.button("🗑️ Delete", type="primary", key="cfg_del_btn"):
                if confirm.strip() != selected:
                    st.error("Confirmation text does not match.")
                else:
                    try:
                        client.delete_config(selected)
                        st.success(f"✅ Deleted `{selected}`")
                        _refresh_and_rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Delete failed: {exc}")

# ── Create new config ─────────────────────────────────────────
st.divider()
st.subheader("➕ Create New Config")

with st.expander("Create a new configuration from scratch", expanded=False):
    create_name = st.text_input(
        "Config name",
        placeholder="config_classification_my_dataset_azureml",
        key="cfg_create_name",
    )
    create_yaml = st.text_area(
        "YAML",
        value=(
            "task_type: classification\n"
            "dataset:\n"
            "  name: my_dataset\n"
            "  target_column: target\n"
        ),
        height=320,
        key="cfg_create_yaml",
    )
    if st.button("➕ Create", type="primary", key="cfg_create_btn"):
        if not create_name.strip():
            st.error("Please provide a name.")
        else:
            try:
                parsed = yaml.safe_load(create_yaml)
                if not isinstance(parsed, dict):
                    st.error("Top-level YAML must be a mapping.")
                else:
                    client.create_config(create_name.strip(), parsed)
                    st.success(f"✅ Created `{create_name.strip()}`")
                    _refresh_and_rerun()
            except yaml.YAMLError as exc:
                st.error(f"YAML parse error: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Create failed: {exc}")
