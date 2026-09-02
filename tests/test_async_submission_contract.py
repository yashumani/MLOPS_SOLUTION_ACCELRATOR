from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from api.schemas.pipeline import SubmitRequest, SubmitResponse
from api.services import pipeline_service


class _ImmediateExecutor:
    def submit(self, function, *args, **kwargs):
        function(*args, **kwargs)
        return None


def test_async_submission_persists_and_correlates_job(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))
    monkeypatch.setattr(pipeline_service, "_submit_executor", _ImmediateExecutor())
    captured: dict = {}

    def fake_submit(req, *, replay_context=None, internal_tags=None):
        captured["request"] = req
        captured["internal_tags"] = internal_tags
        return SubmitResponse(
            job_name="azure-job-1",
            experiment_name="experiment-1",
            display_name="display-1",
            status="Submitted",
            studio_url="https://ml.azure.com/runs/azure-job-1",
        )

    monkeypatch.setattr(pipeline_service, "submit_pipeline", fake_submit)

    queued = pipeline_service.submit_pipeline_async(
        SubmitRequest(
            config_name="config_classification",
            tags={"client": "test"},
        )
    )
    loaded = pipeline_service.get_submit_request(queued["request_id"])

    assert queued["status"] == "pending"
    assert loaded is not None
    assert loaded["status"] == "submitted"
    assert loaded["job_name"] == "azure-job-1"
    assert captured["internal_tags"] == {
        "submission_request_id": queued["request_id"]
    }
    assert (tmp_path / f"{queued['request_id']}.json").is_file()


def test_async_request_state_is_not_process_local() -> None:
    source = Path("api/services/pipeline_service.py").read_text(encoding="utf-8")

    assert "_submit_requests" not in source
    assert "create_request_record(record)" in source
    assert "get_request_record(request_id)" in source


def test_react_submit_page_polls_async_request_to_terminal_state() -> None:
    source = Path("react-ui/src/pages/Submit.tsx").read_text(encoding="utf-8")

    assert "api.submitStatus(asyncRequestId as string)" in source
    assert 'status === "reconciliation_required"' in source
    assert "refetchInterval" in source
    assert 'useState("client=react-ui")' in source


def test_stale_pending_request_recovers_job_identity(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))
    monkeypatch.setenv("MLOPS_SUBMISSION_RECONCILE_AFTER_SECONDS", "0")
    request_id = "req-abcdef012345"
    pipeline_service.create_request_record(
        {
            "request_id": request_id,
            "status": "pending",
            "config_name": "config_classification",
            "submitted_at": "2026-08-02T12:00:00Z",
        }
    )
    job = SimpleNamespace(
        name="recovered-job",
        experiment_name="classification-v3",
        display_name="classification",
        tags={"submission_request_id": request_id},
    )
    client = SimpleNamespace(
        jobs=SimpleNamespace(list=lambda: [job]),
        subscription_id="sub",
        resource_group_name="rg",
        workspace_name="ws",
    )
    monkeypatch.setattr(pipeline_service, "get_ml_client", lambda: client)

    recovered = pipeline_service.get_submit_request(request_id)

    assert recovered is not None
    assert recovered["status"] == "submitted"
    assert recovered["job_name"] == "recovered-job"
    assert recovered["reconciled_at"]


def test_stale_pending_request_never_blindly_resubmits(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))
    monkeypatch.setenv("MLOPS_SUBMISSION_RECONCILE_AFTER_SECONDS", "0")
    request_id = "req-123456abcdef"
    pipeline_service.create_request_record(
        {
            "request_id": request_id,
            "status": "pending",
            "config_name": "config_classification",
            "submitted_at": "2026-08-02T12:00:00Z",
        }
    )
    client = SimpleNamespace(
        jobs=SimpleNamespace(list=lambda: []),
        subscription_id="sub",
        resource_group_name="rg",
        workspace_name="ws",
    )
    monkeypatch.setattr(pipeline_service, "get_ml_client", lambda: client)

    reconciled = pipeline_service.get_submit_request(request_id)

    assert reconciled is not None
    assert reconciled["status"] == "reconciliation_required"
    assert "before retrying" in reconciled["error"]
