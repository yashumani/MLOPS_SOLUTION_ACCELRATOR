"""Hierarchical Experiment → Job picker.

Replaces the manual ``st.text_input("Job Name")`` pattern across pages.
Users select an experiment from a dropdown, then pick one or more jobs
by their human-friendly display name (no run-IDs to copy/paste).

Usage
-----
Single-select::

    job_name = pick_single_job(client, key="leaderboard")

Multi-select::

    job_names = pick_jobs(client, key="outputs", multi=True)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from ui.api_client import APIClient

_STATUS_BADGES = {
    "completed": "🟢",
    "finished": "🟢",
    "running": "🔵",
    "preparing": "🟡",
    "starting": "🟡",
    "queued": "⚪",
    "notstarted": "⚪",
    "failed": "🔴",
    "canceled": "⚫",
    "cancelled": "⚫",
    "cancelrequested": "⚫",
}


def _badge(status: str) -> str:
    return _STATUS_BADGES.get((status or "").lower().replace(" ", ""), "•")


def _fmt_when(when: Any) -> str:
    if not when:
        return ""
    if isinstance(when, str):
        try:
            when = datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            return when[:19]
    try:
        return when.strftime("%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return str(when)[:19]


@st.cache_data(ttl=20, show_spinner=False)
def _load_tree(_client: APIClient, max_per_experiment: int) -> dict:
    """Fetch the experiment → jobs tree once, cached for 20s."""
    return _client.list_experiments(max_results_per_experiment=max_per_experiment)


def _render_picker(
    client: APIClient,
    key: str,
    multi: bool,
    *,
    status_filter: list[str] | None,
    label_experiment: str,
    label_job: str,
    max_per_experiment: int,
) -> list[dict]:
    """Internal: render the two-step picker and return selected job dicts."""
    refresh_col, _ = st.columns([1, 6])
    with refresh_col:
        if st.button("🔄 Refresh", key=f"{key}_refresh", use_container_width=True):
            _load_tree.clear()

    try:
        with st.spinner("Loading experiments..."):
            tree = _load_tree(client, max_per_experiment)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load experiment list: {exc}")
        return []

    experiments = tree.get("experiments", []) or []
    if not experiments:
        st.info("No experiments found in this workspace yet.")
        return []

    # Optional status filter (applied per-experiment to job list)
    def _filter_jobs(jobs: list[dict]) -> list[dict]:
        if not status_filter:
            return jobs
        wanted = {s.lower() for s in status_filter}
        return [j for j in jobs if (j.get("status") or "").lower() in wanted]

    # ── Experiment selector ──
    exp_labels: list[str] = []
    for n in experiments:
        eligible = len(_filter_jobs(n.get("jobs", [])))
        if eligible == 0 and status_filter:
            continue
        exp_labels.append(
            f"{n.get('experiment_name')} ({eligible} job"
            f"{'s' if eligible != 1 else ''})"
        )

    if not exp_labels:
        st.warning(
            "No jobs match the current status filter "
            f"({', '.join(status_filter or [])})."
        )
        return []

    label_to_node = {
        f"{n.get('experiment_name')} "
        f"({len(_filter_jobs(n.get('jobs', [])))} job"
        f"{'s' if len(_filter_jobs(n.get('jobs', []))) != 1 else ''})": n
        for n in experiments
        if (not status_filter or _filter_jobs(n.get("jobs", [])))
    }

    selected_label = st.selectbox(
        label_experiment,
        options=list(label_to_node.keys()),
        key=f"{key}_exp",
    )
    node = label_to_node[selected_label]
    jobs = _filter_jobs(node.get("jobs", []))

    if not jobs:
        st.info("No jobs in this experiment.")
        return []

    # ── Job selector (display_name shown; job_name kept internally) ──
    job_options: list[str] = []
    label_to_job: dict[str, dict] = {}
    for j in jobs:
        display = j.get("display_name") or j.get("job_name", "(unnamed)")
        when = _fmt_when(j.get("start_time"))
        status = j.get("status") or "Unknown"
        # Make label unique even if two jobs share a display_name
        label = f"{_badge(status)} {display} — {status} {when}".strip()
        suffix = 1
        base = label
        while label in label_to_job:
            suffix += 1
            label = f"{base} ({suffix})"
        job_options.append(label)
        label_to_job[label] = j

    if multi:
        chosen = st.multiselect(
            label_job,
            options=job_options,
            default=[job_options[0]] if job_options else [],
            key=f"{key}_jobs",
            help="Tick one or more jobs to operate on.",
        )
    else:
        single = st.selectbox(
            label_job,
            options=job_options,
            key=f"{key}_job",
        )
        chosen = [single] if single else []

    return [label_to_job[c] for c in chosen]


# ─── Public API ────────────────────────────────────────────────


def pick_jobs(
    client: APIClient,
    key: str,
    *,
    multi: bool = True,
    status_filter: list[str] | None = None,
    label_experiment: str = "Experiment",
    label_job: str = "Jobs",
    max_per_experiment: int = 100,
) -> list[dict]:
    """Render an experiment → jobs picker; return selected job objects.

    Each returned dict has at least ``job_name``, ``display_name``,
    ``experiment_name``, ``status``, ``start_time``, ``studio_url``.
    """
    return _render_picker(
        client,
        key=key,
        multi=multi,
        status_filter=status_filter,
        label_experiment=label_experiment,
        label_job=label_job,
        max_per_experiment=max_per_experiment,
    )


def pick_single_job(
    client: APIClient,
    key: str,
    *,
    status_filter: list[str] | None = None,
    label_experiment: str = "Experiment",
    label_job: str = "Job",
    max_per_experiment: int = 100,
) -> dict | None:
    """Convenience wrapper: render the picker in single-select mode."""
    chosen = _render_picker(
        client,
        key=key,
        multi=False,
        status_filter=status_filter,
        label_experiment=label_experiment,
        label_job=label_job,
        max_per_experiment=max_per_experiment,
    )
    return chosen[0] if chosen else None
