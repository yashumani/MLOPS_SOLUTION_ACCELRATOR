"""Sidebar component: navigation, API key input, connection status."""

import streamlit as st

from ui.config import API_BASE_URL, API_KEY


def render_sidebar():
    """Render the shared sidebar with API connection settings."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/artificial-intelligence.png",
            width=48,
        )
        st.title("MLOps V3")
        st.caption("Pipeline Management Dashboard")

        st.divider()

        # API connection settings
        st.subheader("🔌 API Connection")
        base_url = st.text_input(
            "API URL",
            value=st.session_state.get("api_base_url", API_BASE_URL),
            key="sidebar_api_url",
        )
        api_key = st.text_input(
            "API Key",
            value=st.session_state.get("api_key", API_KEY),
            type="password",
            key="sidebar_api_key",
        )

        st.session_state["api_base_url"] = base_url
        st.session_state["api_key"] = api_key

        # Connection test
        if st.button("Test Connection", use_container_width=True):
            _test_connection()

        # Status indicator
        status = st.session_state.get("connection_status")
        if status == "connected":
            st.success("✅ Connected")
        elif status == "error":
            st.error(f"❌ {st.session_state.get('connection_error', 'Failed')}")
        elif status is None and api_key:
            _test_connection()

        st.divider()

        # Navigation links
        st.subheader("📖 Links")
        st.markdown(
            "- [Azure ML Studio](https://ml.azure.com)\n"
            "- [API Docs](/docs)\n"
            "- [GitHub Repo](https://github.com/SAVYMINDS/YS_MVP)"
        )

        st.divider()
        st.caption("v0.1.0 • Streamlit + FastAPI")


def _test_connection():
    """Test API connection and update session state."""
    from ui.api_client import get_client

    try:
        client = get_client()
        result = client.health()
        if result and result.get("status") == "healthy":
            st.session_state["connection_status"] = "connected"
            st.session_state["connection_error"] = ""
        else:
            st.session_state["connection_status"] = "error"
            st.session_state["connection_error"] = "Unhealthy response"
    except Exception as exc:
        st.session_state["connection_status"] = "error"
        st.session_state["connection_error"] = str(exc)[:100]
