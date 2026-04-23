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
prewarm(st.session_state)

page_header("Pipeline Configs", "Browse, inspect, and edit pipeline configuration files", "⚙️")

client = get_client()


def _config_names_from_response(data) -> list[str]:
    raw = data.get("configs", []) if isinstance(data, dict) else []
    if raw and isinstance(raw[0], dict):
        return [c.get("config_name") for c in raw if c.get("config_name")]
    return list(raw)


def _refresh_and_rerun() -> None:
    invalidate_config_caches()
    st.rerun()


# ── Config List ───────────────────────────────────────────────
try:
    data = cached_list_configs()
    config_names = _config_names_from_response(data)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to list configs: {exc}")
    config_names = []

st.info(f"Found **{len(config_names)}** configuration(s)")

# ── Summary Cards ─────────────────────────────────────────────
for name in config_names:
    try:
        cfg = cached_get_config(name)
        render_config_summary_card(name, cfg)
    except Exception:  # noqa: BLE001
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
