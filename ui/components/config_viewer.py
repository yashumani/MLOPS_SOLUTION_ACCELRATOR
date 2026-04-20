"""Config viewer component — renders YAML configs as formatted output."""

import streamlit as st


def render_config_viewer(config: dict, title: str = "Configuration"):
    """Render a configuration dict as a structured view."""
    st.markdown(f"### {title}")

    # Top-level sections
    for section, content in config.items():
        if isinstance(content, dict):
            with st.expander(f"📂 **{section}**", expanded=False):
                _render_dict(content)
        elif isinstance(content, list):
            with st.expander(f"📂 **{section}** ({len(content)} items)", expanded=False):
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        st.markdown(f"**Item {i + 1}:**")
                        _render_dict(item)
                        st.divider()
                    else:
                        st.markdown(f"- `{item}`")
        else:
            st.markdown(f"**{section}:** `{content}`")


def _render_dict(d: dict, indent: int = 0):
    """Recursively render a dict."""
    for k, v in d.items():
        prefix = "&nbsp;" * (indent * 4)
        if isinstance(v, dict):
            st.markdown(f"{prefix}**{k}:**")
            _render_dict(v, indent + 1)
        elif isinstance(v, list):
            st.markdown(f"{prefix}**{k}:** `{v}`")
        else:
            st.markdown(f"{prefix}**{k}:** `{v}`")
