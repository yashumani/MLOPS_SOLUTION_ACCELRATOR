"""Streamlit-cached wrappers around :class:`ui.api_client.APIClient`.

The underlying client is already cached via ``st.cache_resource`` in
:mod:`ui.api_client`; here we cache *responses* with per-endpoint TTLs so
that rapid reruns (filter clicks, tab switches, fragment refreshes) do not
hammer the FastAPI backend. Each wrapper exposes a plain-function signature
and participates in Streamlit's hashing.

Design notes
------------
* TTLs are chosen to align with the API's own warm-loop (``/experiments``
  refreshes every ~20s server-side) and with user expectation of "live-ish"
  data while a job is running.
* For job-scoped endpoints we pass the *status* (when known) so terminal
  jobs can use a much longer TTL than running ones.
* Callers that need a force-refresh should use ``<fn>.clear()`` to evict
  the specific entry, or rely on the TTL.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.api_client import get_client

# ── TTLs (seconds) ───────────────────────────────────────────────────────────
_TTL_CONFIGS = 300           # configs rarely change; 5 min
_TTL_EXPERIMENTS = 20        # API warms every ~20s
_TTL_JOBS = 15               # list_jobs is cheap server-side but hit often
_TTL_JOB_RUNNING = 10        # job detail while running
_TTL_JOB_TERMINAL = 600      # job detail for completed/failed/canceled
_TTL_SUMMARY_RUNNING = 30
_TTL_SUMMARY_TERMINAL = 3_600
_TTL_LOCAL_OUTPUTS = 30
_TTL_HEALTH = 5

_TERMINAL_STATUSES = {
    "completed",
    "finished",
    "failed",
    "canceled",
    "cancelled",
    "cancelrequested",
    "notresponding",
}

_EXPECTED_STAGE_KEYS = {
    "s1",
    "s2",
    "s3",
    "s4",
    "s5a",
    "s5b",
    "s5z",
    "s06",
    "s08",
    "s09",
    "s10",
    "s12",
    "s13",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _is_terminal(status: str | None) -> bool:
    return bool(status) and str(status).lower() in _TERMINAL_STATUSES


def _job_steps_incomplete(detail: dict[str, Any]) -> bool:
    """Return True when a terminal job detail should not get the long TTL."""
    steps = detail.get("steps") or []
    if not steps:
        return True

    stage_keys = {
        str(step.get("stage_key") or "").lower()
        for step in steps
        if isinstance(step, dict) and step.get("stage_key")
    }
    if not stage_keys:
        return True

    status = str(detail.get("status") or "").lower()
    if status in {"completed", "finished"}:
        return len(stage_keys & _EXPECTED_STAGE_KEYS) < 8

    return False


# ── Health ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=_TTL_HEALTH, show_spinner=False)
def cached_health() -> dict[str, Any]:
    return get_client().health() or {}


# ── Configs ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=_TTL_CONFIGS, show_spinner=False)
def cached_list_configs() -> dict[str, Any]:
    return get_client().list_configs() or {}


@st.cache_data(ttl=_TTL_CONFIGS, show_spinner=False)
def cached_get_config(config_name: str) -> dict[str, Any]:
    return get_client().get_config(config_name) or {}


# ── Jobs / experiments ───────────────────────────────────────────────────────
@st.cache_data(ttl=_TTL_EXPERIMENTS, show_spinner=False)
def cached_list_experiments(max_results_per_experiment: int = 100) -> dict[str, Any]:
    return get_client().list_experiments(max_results_per_experiment) or {}


@st.cache_data(ttl=_TTL_JOBS, show_spinner=False)
def cached_list_jobs(
    experiment_name: str | None = None,
    status: str | None = None,
    max_results: int = 100,
) -> dict[str, Any]:
    return (
        get_client().list_jobs(
            experiment_name=experiment_name,
            status=status,
            max_results=max_results,
        )
        or {}
    )


@st.cache_data(ttl=_TTL_JOB_RUNNING, show_spinner=False)
def _cached_get_job_running(job_name: str) -> dict[str, Any]:
    return get_client().get_job(job_name) or {}


@st.cache_data(ttl=_TTL_JOB_TERMINAL, show_spinner=False)
def _cached_get_job_terminal(job_name: str) -> dict[str, Any]:
    return get_client().get_job(job_name) or {}


def cached_get_job(job_name: str, known_status: str | None = None) -> dict[str, Any]:
    """Return a job detail, using a longer TTL for terminal states.

    Pass ``known_status`` if you already have it (e.g. from a list view) to
    avoid a redundant short-TTL fetch; otherwise we default to the running
    TTL, which is the safe choice for still-updating jobs.
    """
    if _is_terminal(known_status):
        detail = _cached_get_job_terminal(job_name)
        if _job_steps_incomplete(detail):
            _cached_get_job_terminal.clear()
            return _cached_get_job_running(job_name)
        return detail
    return _cached_get_job_running(job_name)


# ── Summary / metrics / drift ────────────────────────────────────────────────
@st.cache_data(ttl=_TTL_SUMMARY_RUNNING, show_spinner=False)
def _cached_summary_running(job_name: str) -> dict[str, Any]:
    return get_client().get_pipeline_summary(job_name) or {}


@st.cache_data(ttl=_TTL_SUMMARY_TERMINAL, show_spinner=False)
def _cached_summary_terminal(job_name: str) -> dict[str, Any]:
    return get_client().get_pipeline_summary(job_name) or {}


def cached_pipeline_summary(
    job_name: str, known_status: str | None = None
) -> dict[str, Any]:
    if _is_terminal(known_status):
        return _cached_summary_terminal(job_name)
    return _cached_summary_running(job_name)


@st.cache_data(ttl=_TTL_SUMMARY_RUNNING, show_spinner=False)
def _cached_metrics_running(job_name: str) -> dict[str, Any]:
    return get_client().get_job_metrics(job_name) or {}


@st.cache_data(ttl=_TTL_SUMMARY_TERMINAL, show_spinner=False)
def _cached_metrics_terminal(job_name: str) -> dict[str, Any]:
    return get_client().get_job_metrics(job_name) or {}


def cached_job_metrics(
    job_name: str, known_status: str | None = None
) -> dict[str, Any]:
    if _is_terminal(known_status):
        return _cached_metrics_terminal(job_name)
    return _cached_metrics_running(job_name)


@st.cache_data(ttl=_TTL_SUMMARY_RUNNING, show_spinner=False)
def _cached_drift_running(job_name: str) -> dict[str, Any]:
    return get_client().get_job_drift(job_name) or {}


@st.cache_data(ttl=_TTL_SUMMARY_TERMINAL, show_spinner=False)
def _cached_drift_terminal(job_name: str) -> dict[str, Any]:
    return get_client().get_job_drift(job_name) or {}


def cached_job_drift(
    job_name: str, known_status: str | None = None
) -> dict[str, Any]:
    if _is_terminal(known_status):
        return _cached_drift_terminal(job_name)
    return _cached_drift_running(job_name)


@st.cache_data(ttl=_TTL_LOCAL_OUTPUTS, show_spinner=False)
def cached_local_outputs(max_depth: int = 4, max_files: int = 500) -> dict[str, Any]:
    return get_client().list_local_outputs(max_depth=max_depth, max_files=max_files) or {}


# ── Cache control ────────────────────────────────────────────────────────────
def invalidate_job_caches(job_name: str | None = None) -> None:
    """Drop job-scoped cached entries. Call after submit/cancel/resubmit."""
    cached_list_jobs.clear()
    cached_list_experiments.clear()
    _cached_get_job_running.clear()
    _cached_get_job_terminal.clear()
    _cached_summary_running.clear()
    _cached_summary_terminal.clear()
    _cached_metrics_running.clear()
    _cached_metrics_terminal.clear()
    _cached_drift_running.clear()
    _cached_drift_terminal.clear()
    # job_name arg kept for future per-key eviction; st.cache_data doesn't
    # expose per-key invalidation today, so we clear the whole function cache.
    _ = job_name


def invalidate_config_caches() -> None:
    """Drop config-scoped cached entries. Call after create/edit/delete."""
    cached_list_configs.clear()
    cached_get_config.clear()


def prewarm(session_state: Any) -> None:
    """Warm inexpensive UI caches exactly once per session.

    Pass ``st.session_state``. Idempotent: subsequent calls are no-ops.
    """
    if session_state.get("_prewarmed"):
        return
    try:
        cached_list_configs()
    except Exception:
        # Prewarm is best-effort; swallow so the UI can still render and
        # surface the real error on the first data call.
        pass
    session_state["_prewarmed"] = True
