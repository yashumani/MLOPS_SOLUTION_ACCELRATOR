"""Sidebar — Savyminds branding + real Streamlit page navigation."""

from __future__ import annotations

from html import escape

import streamlit as st

from ui.components.theme import LOGO_HTML
from ui.config import API_BASE_URL, API_KEY


# Pages declared via the multipage `pages/` directory.
# Slimmed cockpit nav: Dashboard → Focus (per-job command center) → Submit → Configs → Drift → Logs.
NAV_ITEMS: list[tuple[str, str, str]] = [
    ("app.py",                       ":material/home:",          "Dashboard"),
    ("pages/2_Focus.py",             ":material/center_focus_strong:", "Focus"),
    ("pages/1_Submit_Pipeline.py",   ":material/rocket_launch:", "Submit"),
    ("pages/3_Configs.py",           ":material/settings:",      "Configs"),
    ("pages/5_Drift_Monitor.py",     ":material/trending_down:", "Drift"),
    ("pages/7_Live_Logs.py",         ":material/terminal:",      "Live Logs"),
]


def render_sidebar() -> None:
    """Render the Savyminds sidebar (logo, connection status, nav)."""
    with st.sidebar:
        st.markdown(LOGO_HTML, unsafe_allow_html=True)

        # Initialise session state once
        if "api_base_url" not in st.session_state:
            st.session_state["api_base_url"] = API_BASE_URL
        if "api_key" not in st.session_state:
            st.session_state["api_key"] = API_KEY

        # Auto-test on first render
        if "connection_status" not in st.session_state:
            _test_connection()

        _render_connection_badge()

        _render_focused_job()

        # API settings (collapsed by default)
        with st.expander("API Settings", expanded=False):
            new_url = st.text_input(
                "API URL",
                value=st.session_state.get("api_base_url", API_BASE_URL),
                key="sidebar_api_url_input",
            )
            if new_url != st.session_state.get("api_base_url"):
                st.session_state["api_base_url"] = new_url
            if st.button("Reconnect", use_container_width=True, key="sidebar_reconnect_btn"):
                _test_connection()
                st.rerun()

        # Navigation
        st.markdown('<div class="svm-section-label">Navigation</div>', unsafe_allow_html=True)
        for page_path, icon, label in NAV_ITEMS:
            try:
                st.page_link(page_path, label=label, icon=icon)
            except Exception:
                # Fallback if material icons unsupported on this Streamlit version
                st.page_link(page_path, label=label)

        # Footer
        api_docs_url = escape(
            f"{st.session_state.get('api_base_url', API_BASE_URL).rstrip('/')}/docs",
            quote=True,
        )
        st.markdown('<div class="svm-section-label">Resources</div>', unsafe_allow_html=True)
        st.markdown(
            '<a href="https://ml.azure.com" target="_blank" '
            'style="display:block;color:#475569;font-size:0.82rem;'
            'text-decoration:none;padding:6px 0;">↗ Azure ML Studio</a>'
            f'<a href="{api_docs_url}" target="_blank" '
            'style="display:block;color:#475569;font-size:0.82rem;'
            'text-decoration:none;padding:6px 0;">↗ API Reference</a>',
            unsafe_allow_html=True,
        )
        st.caption("v0.2.0 · Streamlit + FastAPI")


def _render_connection_badge() -> None:
    status = st.session_state.get("connection_status")
    if status == "connected":
        st.markdown(
            '<div class="svm-conn svm-conn-ok">'
            '<span class="svm-conn-dot"></span>API Connected</div>',
            unsafe_allow_html=True,
        )
    elif status == "error":
        err = st.session_state.get("connection_error", "Connection failed")
        st.markdown(
            f'<div class="svm-conn svm-conn-err">'
            f'<span class="svm-conn-dot"></span>{err[:50]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="svm-conn svm-conn-pending">'
            '<span class="svm-conn-dot"></span>Connecting…</div>',
            unsafe_allow_html=True,
        )


def _test_connection() -> None:
    """Hit /api/v1/health and update session state."""
    from ui.api_client import get_client

    try:
        client = get_client()
        result = client.health()
        if result and result.get("status") in ("ok", "healthy"):
            st.session_state["connection_status"] = "connected"
            st.session_state["connection_error"] = ""
        else:
            st.session_state["connection_status"] = "error"
            st.session_state["connection_error"] = "API responded but status not ok"
    except Exception as exc:
        st.session_state["connection_status"] = "error"
        st.session_state["connection_error"] = str(exc)[:80]


def _render_focused_job() -> None:
    """Show the currently focused job (if any) with quick Open / Clear actions."""
    focused = st.session_state.get("focused_job") or {}
    job_name = focused.get("job_name")
    if not job_name:
        return

    display = focused.get("display_name") or job_name
    status = focused.get("status") or "—"
    short = display if len(display) <= 28 else display[:25] + "…"

    st.markdown('<div class="svm-section-label">Focused Job</div>', unsafe_allow_html=True)
    st.caption(f"**{short}**  ·  `{status}`")
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.page_link("pages/2_Focus.py", label="Open", icon=":material/center_focus_strong:")
        except Exception:
            st.page_link("pages/2_Focus.py", label="Open")
    with col2:
        if st.button("Clear", key="sidebar_clear_focus", use_container_width=True):
            st.session_state.pop("focused_job", None)
            st.rerun()
