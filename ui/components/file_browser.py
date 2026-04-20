"""File browser component — navigate job output artifacts."""

import streamlit as st


def render_file_browser(outputs: dict):
    """Render a file browser for job output artifacts."""
    if not outputs:
        st.info("No output artifacts available.")
        return

    artifacts = outputs.get("artifacts", outputs.get("files", []))
    if isinstance(artifacts, dict):
        # Dict format: {step_name: [files]}
        for step, files in artifacts.items():
            with st.expander(f"📁 **{step}** ({len(files)} file{'s' if len(files) != 1 else ''})", expanded=False):
                for f in files:
                    if isinstance(f, dict):
                        name = f.get("name", f.get("path", "unknown"))
                        size = f.get("size", "")
                        st.markdown(f"  📄 `{name}` {f'({size})' if size else ''}")
                    else:
                        st.markdown(f"  📄 `{f}`")
    elif isinstance(artifacts, list):
        for f in artifacts:
            if isinstance(f, dict):
                name = f.get("name", f.get("path", "unknown"))
                size = f.get("size", "")
                st.markdown(f"📄 `{name}` {f'({size})' if size else ''}")
            else:
                st.markdown(f"📄 `{f}`")
    else:
        st.json(outputs)
