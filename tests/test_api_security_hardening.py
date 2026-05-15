"""API and registry security-hardening regression tests."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest import mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


def test_submit_request_rejects_path_traversal_config_name():
    from api.schemas.pipeline import SubmitRequest

    with pytest.raises(ValidationError):
        SubmitRequest(config_name="../config_classification_telecom_churn_azureml")


def test_submit_request_accepts_safe_config_name():
    from api.schemas.pipeline import SubmitRequest

    req = SubmitRequest(config_name="config_classification_telecom_churn_azureml")

    assert req.config_name == "config_classification_telecom_churn_azureml"


def test_verify_api_key_uses_configured_secret(monkeypatch):
    from api.core import security

    monkeypatch.setattr(security.settings, "api_key", "expected-secret")

    assert asyncio.run(security.verify_api_key("expected-secret")) == "expected-secret"
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(security.verify_api_key("wrong-secret"))
    assert exc_info.value.status_code == 401


def test_root_endpoint_returns_api_metadata():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MLOps V3 Pipeline Management API"
    assert data["status"] == "ok"
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/v1/health"
    assert data["api_base"] == "/api/v1"
    assert data["frontend"]["service"] == "Streamlit dashboard"


def test_root_endpoint_redirects_browsers_to_dashboard(monkeypatch):
    from fastapi.testclient import TestClient

    from api.core.config import settings
    from api.main import app

    monkeypatch.setattr(settings, "ui_base_url", "https://example.test/dashboard")

    client = TestClient(app)
    response = client.get(
        "/", headers={"Accept": "text/html"}, follow_redirects=False
    )

    assert response.status_code == 307
    assert response.headers["location"] == "https://example.test/dashboard/"


def test_root_endpoint_derives_azureml_dashboard_url(monkeypatch):
    from fastapi.testclient import TestClient

    from api.core.config import settings
    from api.main import app

    monkeypatch.setattr(settings, "ui_base_url", "")

    client = TestClient(
        app,
        base_url="http://mlopspipelinev2-8000.eastus2.instances.azureml.ms",
    )
    response = client.get("/", headers={"Accept": "application/json"})

    assert response.status_code == 200
    assert (
        response.json()["dashboard"]
        == "https://mlopspipelinev2-8501.eastus2.instances.azureml.ms/"
    )


def test_healthz_liveness_probe_returns_ok():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_mutation_guard_fails_closed_when_status_check_fails(monkeypatch):
    from api.routers import configs

    monkeypatch.setattr(
        configs.pipeline_service,
        "list_jobs",
        mock.Mock(side_effect=RuntimeError("azure unavailable")),
    )

    with pytest.raises(HTTPException) as exc_info:
        configs._guard_no_running_jobs("config_classification_telecom_churn_azureml")
    assert exc_info.value.status_code == 503


def test_stage_key_inference_matches_current_pipeline_names():
    from api.services import pipeline_service

    assert pipeline_service._infer_stage_key("s1") == "s1"
    assert pipeline_service._infer_stage_key("V3 Stage 1 - Ingestion") == "s1"
    assert pipeline_service._infer_stage_key("s06 - V3 Phase B - Variant Runner") == "s06"
    assert pipeline_service._infer_stage_key("s13 - Drift Monitor") == "s13"


def test_list_local_outputs_is_read_only_and_bounded(tmp_path, monkeypatch):
    from api.services import pipeline_service

    (tmp_path / "batch").mkdir()
    report = tmp_path / "batch" / "report.json"
    report.write_text('{"ok": true}')
    monkeypatch.setattr(pipeline_service, "_LOCAL_OUTPUTS_DIR", tmp_path)

    result = pipeline_service.list_local_outputs(max_depth=2, max_files=10)

    paths = {item.relative_path for item in result.files}
    assert "batch" in paths
    assert "batch/report.json" in paths
    assert result.truncated is False


def test_s12_skips_registration_when_quality_gate_failed(tmp_path, monkeypatch):
    import src.steps.s12_model_registration as s12

    manifest = tmp_path / "final_report.json"
    manifest.write_text(
        json.dumps(
            {
                "champion_valid": True,
                "quality_gate_passed": False,
                "selection": {"key": "baseline", "score": 0.1},
            }
        )
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output = tmp_path / "registry_info.json"

    class _Logger:
        def log_metric(self, *_args, **_kwargs):
            return None

        def log_param(self, *_args, **_kwargs):
            return None

        def end_run(self):
            return None

    monkeypatch.setattr(s12, "create_metrics_logger", lambda *args, **kwargs: _Logger())
    monkeypatch.setattr(s12.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(s12, "_safe_disable_autolog", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "s12_model_registration.py",
            "--champion_manifest",
            str(manifest),
            "--champion_model",
            str(model_dir),
            "--config_name",
            "config_classification_telecom_churn_azureml.yml",
            "--registry_info",
            str(output),
        ],
    )

    s12.main()

    data = json.loads(output.read_text())
    assert data["registration_skipped"] is True
    assert data["skip_reason"] == "quality_gate_failed"
