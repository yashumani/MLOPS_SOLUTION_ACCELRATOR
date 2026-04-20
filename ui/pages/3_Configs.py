"""3️⃣ Configs — Browse and inspect pipeline configurations."""

import streamlit as st

from ui.api_client import get_client
from ui.components.config_summary_card import render_config_summary_card
from ui.components.config_viewer import render_config_viewer
from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="Configurations", page_icon="⚙️", layout="wide")
render_sidebar()

st.title("⚙️ Pipeline Configurations")
st.markdown("Browse available V3 pipeline configurations and inspect their contents.")

client = get_client()

# ── Config List ───────────────────────────────────────────────
try:
    data = client.list_configs()
    config_names = data.get("configs", [])
except Exception as exc:
    st.error(f"Failed to list configs: {exc}")
    config_names = []

if not config_names:
    st.warning("No configurations found.")
    st.stop()

st.info(f"Found **{len(config_names)}** configuration(s)")

# ── Summary Cards ─────────────────────────────────────────────
for name in config_names:
    try:
        cfg = client.get_config(name)
        render_config_summary_card(name, cfg)
    except Exception:
        st.markdown(f"📋 **{name}** — _unable to load_")

# ── Detail View ───────────────────────────────────────────────
st.divider()
st.subheader("🔍 Config Detail View")

selected = st.selectbox("Select config to inspect", config_names)

if selected:
    try:
        cfg = client.get_config(selected)
        render_config_viewer(cfg, title=selected)

        # Download raw JSON
        import json
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(cfg, indent=2, default=str),
            file_name=f"{selected}.json",
            mime="application/json",
        )
    except Exception as exc:
        st.error(f"Failed to load config: {exc}")
