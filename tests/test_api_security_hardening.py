"""API and registry security-hardening regression tests."""

from __future__ import annotations

import asyncio
import json
import sys
import zipfile
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


def test_private_api_profile_accepts_only_hardened_single_operator_config(tmp_path):
    from api.core.config import Settings

    settings = Settings(
        _env_file=None,
        api_deployment_profile="private_single_operator",
        api_key="x" * 32,
        api_reload=False,
        api_config_mutation_enabled=False,
        cors_allow_origins="https://mlops-ui.example.test",
        ui_base_url="https://mlops-ui.example.test",
        mlops_state_dir=str(tmp_path / "submitter"),
        mlops_submission_request_root=str(tmp_path / "requests"),
        mlops_auto_retrain_ledger_root=str(tmp_path / "retrain"),
        notification_report_dir=str(tmp_path / "notifications"),
    )

    settings.validate_runtime_security()


def test_private_api_profile_rejects_unsafe_defaults():
    from api.core.config import Settings

    settings = Settings(
        _env_file=None,
        api_deployment_profile="private_single_operator",
        api_key="short",
    )

    with pytest.raises(RuntimeError, match="Unsafe private_single_operator") as exc_info:
        settings.validate_runtime_security()
    message = str(exc_info.value)
    assert "API_KEY" in message
    assert "API_CONFIG_MUTATION_ENABLED" in message
    assert "MLOPS_SUBMISSION_REQUEST_ROOT" in message


def test_multi_user_api_profile_fails_closed():
    from api.core.config import Settings

    settings = Settings(_env_file=None, api_deployment_profile="multi_user")

    with pytest.raises(RuntimeError, match="Entra/OIDC"):
        settings.validate_runtime_security()


def test_config_mutation_can_be_disabled_by_deployment(monkeypatch):
    from api.core import security

    monkeypatch.setattr(security.settings, "api_config_mutation_enabled", False)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(security.require_config_mutation_enabled())
    assert exc_info.value.status_code == 403


def test_react_runtime_cannot_embed_backend_api_key():
    repo_root = Path(__file__).resolve().parents[1]
    runtime_source = (repo_root / "react-ui/src/services/runtimeConfig.ts").read_text()
    public_config = (repo_root / "react-ui/public/runtime-config.js").read_text()
    gate_source = (repo_root / "react-ui/src/components/ApiKeyGate.tsx").read_text()

    assert "VITE_API_KEY" not in runtime_source
    assert "runtime.apiKey" not in runtime_source
    assert "apiKey:" not in public_config
    assert "runtimeConfig.apiKey" not in gate_source


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


def test_root_endpoint_does_not_redirect_to_untrusted_host(monkeypatch):
    from fastapi.testclient import TestClient

    from api.core.config import settings
    from api.main import app

    monkeypatch.setattr(settings, "ui_base_url", "")
    client = TestClient(app, base_url="https://attacker.example")

    response = client.get(
        "/",
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_healthz_liveness_probe_returns_ok():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_application_lifespan_runs_runtime_security_validation(monkeypatch):
    from fastapi.testclient import TestClient

    from api.core import azure_ml
    from api.core.config import settings
    from api.main import app

    validated: list[object] = []
    monkeypatch.setattr(azure_ml, "get_ml_client", lambda: object())
    monkeypatch.setattr(settings, "experiment_cache_enabled", False)
    monkeypatch.setattr(
        type(settings),
        "validate_runtime_security",
        lambda current: validated.append(current),
    )

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200

    assert validated == [settings]


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


def test_download_output_temp_directory_is_removed_after_response(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api.core.config import settings
    from api.main import app
    from api.routers import pipelines

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "report.json").write_text('{"ok": true}')
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(
        pipelines.pipeline_service,
        "download_output",
        lambda *_args, **_kwargs: artifact_dir,
    )

    response = TestClient(app).get(
        "/api/v1/pipelines/jobs/job-1/outputs/report/download",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.content == b'{"ok": true}'
    assert not artifact_dir.exists()


def test_download_output_zip_and_temp_directory_are_removed(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api.core.config import settings
    from api.main import app
    from api.routers import pipelines

    artifact_dir = tmp_path / "artifact-multi"
    artifact_dir.mkdir()
    (artifact_dir / "one.txt").write_text("one")
    (artifact_dir / "two.txt").write_text("two")
    archive_path = Path(f"{artifact_dir}.zip")
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(
        pipelines.pipeline_service,
        "download_output",
        lambda *_args, **_kwargs: artifact_dir,
    )

    response = TestClient(app).get(
        "/api/v1/pipelines/jobs/job-1/outputs/report/download",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(__import__("io").BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"one.txt", "two.txt"}
    assert not artifact_dir.exists()
    assert not archive_path.exists()


def test_download_output_cleans_temp_directory_when_azure_download_fails(
    tmp_path, monkeypatch
):
    from api.services import pipeline_service

    artifact_dir = tmp_path / "failed-download"

    class _Jobs:
        def download(self, *_args, **_kwargs):
            raise RuntimeError("download failed")

    class _Client:
        jobs = _Jobs()

    def _make_temp_dir(*_args, **_kwargs):
        artifact_dir.mkdir()
        return str(artifact_dir)

    monkeypatch.setattr(pipeline_service, "get_ml_client", lambda: _Client())
    monkeypatch.setattr(pipeline_service.tempfile, "mkdtemp", _make_temp_dir)

    with pytest.raises(RuntimeError, match="download failed"):
        pipeline_service.download_output("job-1", "report")
    assert not artifact_dir.exists()


def test_large_json_preview_is_bounded(tmp_path, monkeypatch):
    from api.services import pipeline_service

    artifact_dir = tmp_path / "preview"

    class _Jobs:
        def download(self, *_args, **kwargs):
            path = Path(kwargs["download_path"])
            path.mkdir(parents=True, exist_ok=True)
            (path / "large.json").write_text(
                json.dumps({"payload": "x" * (pipeline_service._TEXT_PREVIEW_BYTES + 50)})
            )

    class _Client:
        jobs = _Jobs()

    def _make_temp_dir(*_args, **_kwargs):
        artifact_dir.mkdir()
        return str(artifact_dir)

    monkeypatch.setattr(pipeline_service, "get_ml_client", lambda: _Client())
    monkeypatch.setattr(pipeline_service.tempfile, "mkdtemp", _make_temp_dir)

    result = pipeline_service.get_output_content("job-1", "report")

    assert result.truncated is True
    assert result.json_content is None
    assert result.text_preview is not None
    assert len(result.text_preview.encode("utf-8")) <= pipeline_service._TEXT_PREVIEW_BYTES
    assert not artifact_dir.exists()


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
