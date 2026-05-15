"""Log stream component — display job logs with light severity colouring."""

from __future__ import annotations

import re

import streamlit as st


_LEVEL_RE = re.compile(
    r"\b(ERROR|CRITICAL|FATAL|WARN(?:ING)?|INFO|DEBUG)\b",
    re.IGNORECASE,
)


def filter_logs(
    logs: str,
    *,
    levels: list[str] | None = None,
    search: str | None = None,
    tail: int | None = None,
) -> str:
    """Filter log lines by level / substring search and optionally tail N lines."""
    if not logs:
        return ""
    lines = logs.splitlines()

    if levels:
        wanted = {lvl.upper() for lvl in levels}

        def _line_matches(line: str) -> bool:
            m = _LEVEL_RE.search(line)
            if not m:
                # Lines without a level token are kept only when INFO selected,
                # so users always see context unless they explicitly ask for
                # ERROR-only.
                return "INFO" in wanted
            tok = m.group(1).upper()
            if tok.startswith("WARN"):
                tok = "WARNING"
            if tok in {"CRITICAL", "FATAL"}:
                tok = "ERROR"
            return tok in wanted

        lines = [ln for ln in lines if _line_matches(ln)]

    if search:
        s = search.lower()
        lines = [ln for ln in lines if s in ln.lower()]

    if tail and tail > 0:
        lines = lines[-tail:]

    return "\n".join(lines)


def render_log_stream(logs: str, height: int = 500) -> None:
    """Render a scrollable log viewer."""
    if not logs:
        st.info("No logs available.")
        return
    st.code(logs, language="log", line_numbers=True)
