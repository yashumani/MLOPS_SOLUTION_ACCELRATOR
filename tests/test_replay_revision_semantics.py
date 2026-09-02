from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from api.schemas.pipeline import ResubmitRequest, SubmitRequest, SubmitResponse
from api.services import pipeline_service
from pipelines import submit_pipeline as canonical_submitter


ORIGINAL_TAGS = {
    "config_name": "config_regression_college_azureml.yml",
    "execution_id": "execution-1",
    "compiled_config_hash": "config-sha",
    "source_identity": "source-sha",
}


def _manifest(execution_id: str = "execution-1") -> SimpleNamespace:
    return SimpleNamespace(execution_id=execution_id)


def _patch_original_job(monkeypatch, tags: dict[str, str]) -> None:
    original = SimpleNamespace(
        tags=tags,
        experiment_name="regression_college_v3",
    )
    client = SimpleNamespace(jobs=SimpleNamespace(get=lambda _name: original))
    monkeypatch.setattr(pipeline_service, "get_ml_client", lambda: client)


def test_exact_replay_accepts_only_matching_immutable_identity() -> None:
    canonical_submitter._validate_submission_revision_identity(
        revision_kind="exact_replay",
        execution_manifest=_manifest(),
        config_hash="config-sha",
        source_identity="source-sha",
        parent_execution_id="execution-1",
        parent_config_hash="config-sha",
        parent_source_identity="source-sha",
        expected_execution_id="execution-1",
        expected_config_hash="config-sha",
        expected_source_identity="source-sha",
    )

    with pytest.raises(ValueError, match="current immutable inputs changed: source_identity"):
        canonical_submitter._validate_submission_revision_identity(
            revision_kind="exact_replay",
            execution_manifest=_manifest(),
            config_hash="config-sha",
            source_identity="changed-source",
            parent_execution_id="execution-1",
            parent_config_hash="config-sha",
            parent_source_identity="source-sha",
            expected_execution_id="execution-1",
            expected_config_hash="config-sha",
            expected_source_identity="source-sha",
        )


def test_new_revision_requires_an_actual_identity_change() -> None:
    with pytest.raises(ValueError, match="identity are unchanged"):
        canonical_submitter._validate_submission_revision_identity(
            revision_kind="new_revision",
            execution_manifest=_manifest(),
            config_hash="config-sha",
            source_identity="source-sha",
            parent_execution_id="execution-1",
            parent_config_hash="config-sha",
            parent_source_identity="source-sha",
            expected_execution_id=None,
            expected_config_hash=None,
            expected_source_identity=None,
        )

    canonical_submitter._validate_submission_revision_identity(
        revision_kind="new_revision",
        execution_manifest=_manifest("execution-2"),
        config_hash="changed-config",
        source_identity="changed-source",
        parent_execution_id="execution-1",
        parent_config_hash="config-sha",
        parent_source_identity="source-sha",
        expected_execution_id=None,
        expected_config_hash=None,
        expected_source_identity=None,
    )


def test_api_exact_resubmit_passes_parent_identity_to_canonical_submitter(
    monkeypatch,
) -> None:
    _patch_original_job(monkeypatch, ORIGINAL_TAGS)
    captured = {}

    def fake_submit(req, *, replay_context=None):
        captured["request"] = req
        captured["context"] = replay_context
        return SubmitResponse(
            job_name="new-job",
            experiment_name="regression_college_v3",
            display_name="replay",
            status="Submitted",
            studio_url="https://ml.azure.com/runs/new-job",
        )

    monkeypatch.setattr(pipeline_service, "submit_pipeline", fake_submit)

    pipeline_service.resubmit_pipeline(ResubmitRequest(job_name="original-job"))

    assert captured["request"].config_name == "config_regression_college_azureml"
    context = captured["context"]
    assert context.revision_kind == "exact_replay"
    assert context.parent_execution_id == "execution-1"
    assert context.parent_config_hash == "config-sha"
    assert context.parent_source_identity == "source-sha"


def test_api_resubmit_rejects_missing_identity_and_unreasoned_new_revision(
    monkeypatch,
) -> None:
    _patch_original_job(monkeypatch, {"config_name": ORIGINAL_TAGS["config_name"]})
    with pytest.raises(ValueError, match="lacks immutable replay identity tags"):
        pipeline_service.resubmit_pipeline(ResubmitRequest(job_name="legacy-job"))

    _patch_original_job(monkeypatch, ORIGINAL_TAGS)
    with pytest.raises(ValueError, match="new_revision requires"):
        pipeline_service.resubmit_pipeline(
            ResubmitRequest(job_name="original-job", revision_mode="new_revision")
        )


def test_api_new_revision_is_explicit_and_does_not_claim_exact_expectations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_original_job(monkeypatch, ORIGINAL_TAGS)
    captured = {}

    def fake_submit(req, *, replay_context=None):
        captured["context"] = replay_context
        return SubmitResponse(
            job_name="new-job",
            experiment_name="regression_college_v3",
            display_name="new-revision",
            status="Submitted",
            studio_url="https://ml.azure.com/runs/new-job",
        )

    monkeypatch.setattr(pipeline_service, "submit_pipeline", fake_submit)
    pipeline_service.resubmit_pipeline(
        ResubmitRequest(
            job_name="original-job",
            revision_mode="new_revision",
            revision_reason="approved config correction",
        )
    )

    context = captured["context"]
    assert context.revision_kind == "new_revision"
    assert context.revision_reason == "approved config correction"

    command, _, _ = pipeline_service._build_canonical_submit_command(
        SubmitRequest(config_name="config_regression_college_azureml"),
        result_path=tmp_path / "result.json",
        baseline_uri=None,
        replay_context=context,
    )
    assert "--revision_reason" in command
    assert "--expected_execution_id" not in command

