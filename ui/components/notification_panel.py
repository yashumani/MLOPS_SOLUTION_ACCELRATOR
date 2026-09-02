"""Reusable Streamlit panel for job notification report generation."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_notification_panel(client: Any, job_name: str, display_name: str, *, key: str) -> None:
    """Render controls for generating and sending job notification reports."""
    with st.expander("Email notification report", expanded=False):
        st.caption(
            "Generates Markdown, JSON, drift-feature CSV, and step-status CSV files. "
            "The backend sends only to the configured recipient; SMTP server details "
            "come from environment variables."
        )
        c1, c2 = st.columns(2)
        with c1:
            generate = st.button(
                "Generate files only",
                key=f"{key}_notify_generate",
                use_container_width=True,
            )
        with c2:
            send = st.button(
                "Send email",
                type="primary",
                key=f"{key}_notify_send",
                use_container_width=True,
            )

        if generate or send:
            try:
                with st.spinner("Building notification report..."):
                    result = client.send_job_notification(job_name, dry_run=generate)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Notification failed: {exc}")
                return
            _render_notification_result(result, display_name)


def _render_notification_result(result: dict[str, Any], display_name: str) -> None:
    status = result.get("status") or "unknown"
    if result.get("sent"):
        st.success(f"Email sent for `{display_name}`.")
    elif status == "dry_run":
        st.info("Report files generated; email was not sent.")
    elif status == "not_configured":
        st.warning(result.get("message") or "SMTP is not configured.")
    else:
        st.error(result.get("message") or f"Notification status: {status}")

    st.markdown(f"**Subject:** `{result.get('subject') or 'n/a'}`")
    st.markdown(f"**Recipient:** `{result.get('recipient') or 'n/a'}`")
    st.markdown(f"**Report folder:** `{result.get('report_dir') or 'n/a'}`")
    artifacts = result.get("artifacts") or []
    if artifacts:
        rows = [
            [
                item.get("name") or "n/a",
                item.get("mime_type") or "n/a",
                item.get("size_bytes") or 0,
                item.get("included_in_email"),
            ]
            for item in artifacts
        ]
        st.markdown(_markdown_table(["Artifact", "Type", "Bytes", "Attached"], rows))


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    rendered_rows = [
        "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *rendered_rows])