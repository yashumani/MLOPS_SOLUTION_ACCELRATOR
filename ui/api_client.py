"""API client wrapper for the MLOps V3 FastAPI backend."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st


class APIClient:
    """Synchronous client for the MLOps V3 Pipeline Management API."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": api_key})

    # ── helpers ────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> dict | list | None:
        kwargs.setdefault("timeout", self.timeout)
        resp = self._session.request(method, self._url(path), **kwargs)
        resp.raise_for_status()
        if resp.status_code == 204:
            return None
        return resp.json()

    def _get(self, path: str, **params) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, json: dict | None = None) -> Any:
        return self._request("POST", path, json=json)

    def _put(self, path: str, json: dict | None = None) -> Any:
        return self._request("PUT", path, json=json)

    def _delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # ── health ─────────────────────────────────────────────

    def health(self) -> dict:
        return self._get("/api/v1/health")

    # ── configs ────────────────────────────────────────────

    def list_configs(self) -> dict:
        return self._get("/api/v1/configs")

    def get_config(self, config_name: str) -> dict:
        return self._get(f"/api/v1/configs/{config_name}")

    # ── pipelines ──────────────────────────────────────────

    def submit_pipeline(
        self,
        config_name: str,
        compute: str | None = None,
        force_rerun: bool = False,
        baseline_job: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> dict:
        body: dict[str, Any] = {"config_name": config_name}
        if compute:
            body["compute"] = compute
        if force_rerun:
            body["force_rerun"] = True
        if baseline_job:
            body["baseline_job"] = baseline_job
        if tags:
            body["tags"] = tags
        return self._post("/api/v1/pipelines/submit", json=body)

    def list_jobs(
        self,
        experiment_name: str | None = None,
        status: str | None = None,
        max_results: int = 50,
    ) -> dict:
        params: dict[str, Any] = {"max_results": max_results}
        if experiment_name:
            params["experiment_name"] = experiment_name
        if status:
            params["status"] = status
        return self._get("/api/v1/pipelines/jobs", **params)

    def list_experiments(self, max_results_per_experiment: int = 100) -> dict:
        """Return jobs grouped by experiment for hierarchical UI pickers."""
        return self._get(
            "/api/v1/pipelines/experiments",
            max_results_per_experiment=max_results_per_experiment,
        )

    def get_job(self, job_name: str) -> dict:
        return self._get(f"/api/v1/pipelines/jobs/{job_name}")

    def cancel_job(self, job_name: str) -> dict:
        return self._post(f"/api/v1/pipelines/jobs/{job_name}/cancel")

    def list_outputs(self, job_name: str) -> dict:
        return self._get(f"/api/v1/pipelines/jobs/{job_name}/outputs")

    def get_output_content(self, job_name: str, output_name: str) -> dict:
        """Fetch parsed file content of a named output (for UI rendering)."""
        return self._get(
            f"/api/v1/pipelines/jobs/{job_name}/outputs/{output_name}/content"
        )

    def get_pipeline_summary(self, job_name: str) -> dict:
        """Fetch combined baseline/phaseB/phaseC/final reports for a job."""
        return self._get(f"/api/v1/pipelines/jobs/{job_name}/summary")

    def list_local_outputs(self, max_depth: int = 4, max_files: int = 500) -> dict:
        """Fetch a read-only inventory of the repo-local outputs/ folder."""
        return self._get(
            "/api/v1/pipelines/local-outputs",
            max_depth=max_depth,
            max_files=max_files,
        )

    def download_output(self, job_name: str, output_name: str) -> bytes:
        resp = self._session.get(
            self._url(f"/api/v1/pipelines/jobs/{job_name}/outputs/{output_name}/download"),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.content

    # ── metrics (Phase 0a) ─────────────────────────────────

    def get_job_metrics(self, job_name: str) -> dict:
        return self._get(f"/api/v1/pipelines/jobs/{job_name}/metrics")

    # ── drift (Phase 0b) ──────────────────────────────────

    def get_job_drift(self, job_name: str) -> dict:
        return self._get(f"/api/v1/pipelines/jobs/{job_name}/drift")

    # ── resubmit (Phase 0d) ───────────────────────────────

    def resubmit(self, job_name: str, force_rerun: bool = True) -> dict:
        return self._post(
            "/api/v1/pipelines/resubmit",
            json={"job_name": job_name, "force_rerun": force_rerun},
        )

    # ── async submit (Phase 4) ────────────────────────────

    def submit_pipeline_async(
        self,
        config_name: str,
        compute: str | None = None,
        force_rerun: bool = False,
        baseline_job: str | None = None,
        tags: dict | None = None,
    ) -> dict:
        body = {
            "config_name": config_name,
            "force_rerun": force_rerun,
            "tags": tags or {},
        }
        if compute:
            body["compute"] = compute
        if baseline_job:
            body["baseline_job"] = baseline_job
        return self._post("/api/v1/pipelines/submit/async", json=body)

    def get_submit_status(self, request_id: str) -> dict:
        return self._get(f"/api/v1/pipelines/submit/status/{request_id}")

    # ── configs CRUD (Phase 4) ────────────────────────────

    def create_config(self, config_name: str, content: dict) -> dict:
        return self._post(f"/api/v1/configs/{config_name}", json={"content": content})

    def update_config(self, config_name: str, content: dict) -> dict:
        return self._put(f"/api/v1/configs/{config_name}", json={"content": content})

    def delete_config(self, config_name: str) -> dict:
        return self._delete(f"/api/v1/configs/{config_name}")


def _build_client(base_url: str, api_key: str) -> APIClient:
    """Construct a fresh APIClient. Split out so `get_client` can cache it."""
    return APIClient(base_url=base_url, api_key=api_key)


# Cache the client (+ its underlying requests.Session for HTTP keep-alive)
# across Streamlit reruns. Keyed on (base_url, api_key) so API Settings
# reconnects still produce a new session.
_cached_builder = st.cache_resource(show_spinner=False)(_build_client)


def get_client() -> APIClient:
    """Return a cached APIClient keyed on (base_url, api_key).

    Uses ``st.cache_resource`` so the underlying ``requests.Session`` is
    reused across reruns / page switches / user interactions.
    """
    api_key = st.session_state.get("api_key") or ""
    base_url = st.session_state.get("api_base_url") or "http://localhost:8000"
    return _cached_builder(base_url, api_key)
