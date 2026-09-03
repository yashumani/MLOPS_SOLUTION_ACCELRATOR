"""Contract tests for isolated registered-model qualification smokes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.registered_model_inference_smoke.score import (
    _parse_model_uri,
    _validate_model_version,
)
from scripts.submit_registered_model_smoke import (
    SCORE_ROOT,
    _environment_id,
    build_evidence_output,
    validate_registry_info,
)


def _registry_info() -> dict:
    model_name = "mlops-v3-qualification-classification-healthcare"
    version = "3"
    return {
        "model_name": model_name,
        "version": version,
        "model_uri": f"models:/{model_name}/{version}",
        "stage": "None",
        "lifecycle_stage": "Unassigned",
        "promotion_mode": "manual",
        "promotion_performed": False,
        "execution_id": "a" * 64,
        "code_sha": "b" * 64,
        "dataset_content_sha256": "c" * 64,
        "task_type": "classification",
    }


def test_registry_info_binds_exact_unpromoted_model_version() -> None:
    result = validate_registry_info(_registry_info())

    assert result["model_uri"].endswith("/3")
    assert result["promotion_performed"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_uri", "models:/other/3", "exact version"),
        ("model_name", "production-model", "qualification model"),
        ("promotion_mode", "automatic", "must be manual"),
        ("promotion_performed", True, "unapproved promotion"),
        ("lifecycle_stage", "Staging", "remain unassigned"),
        ("code_sha", "not-a-digest", "SHA-256"),
    ],
)
def test_registry_info_rejects_unqualified_records(
    field,
    value,
    message,
) -> None:
    payload = _registry_info()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_registry_info(payload)


def test_smoke_model_version_requires_lineage_and_no_promotion() -> None:
    model_version = SimpleNamespace(
        current_stage="None",
        aliases=[],
        tags={
            "execution_id": "a" * 64,
            "code_sha": "b" * 64,
            "dataset_content_sha256": "c" * 64,
            "promotion_mode": "manual",
            "promotion_performed": "false",
            "lifecycle_stage": "Unassigned",
        },
    )

    tags = _validate_model_version(
        model_version,
        expected_execution_id="a" * 64,
        expected_code_sha="b" * 64,
        expected_dataset_sha="c" * 64,
    )

    assert tags["execution_id"] == "a" * 64


def test_smoke_model_version_rejects_stage_or_protected_alias() -> None:
    tags = {
        "execution_id": "a" * 64,
        "code_sha": "b" * 64,
        "dataset_content_sha256": "c" * 64,
        "promotion_mode": "manual",
        "promotion_performed": "false",
        "lifecycle_stage": "Unassigned",
    }
    with pytest.raises(RuntimeError, match="promoted before approval"):
        _validate_model_version(
            SimpleNamespace(current_stage="Staging", aliases=[], tags=tags),
            expected_execution_id="a" * 64,
            expected_code_sha="b" * 64,
            expected_dataset_sha="c" * 64,
        )
    with pytest.raises(RuntimeError, match="protected aliases"):
        _validate_model_version(
            SimpleNamespace(current_stage="None", aliases=["champion"], tags=tags),
            expected_execution_id="a" * 64,
            expected_code_sha="b" * 64,
            expected_dataset_sha="c" * 64,
        )


def test_smoke_code_upload_is_isolated_from_repository_model_modules() -> None:
    assert SCORE_ROOT.is_dir()
    assert (SCORE_ROOT / "score.py").is_file()
    assert (SCORE_ROOT / ".amlignore").is_file()
    assert not (SCORE_ROOT / "utils").exists()
    assert not any(path.name == "model_bundle.py" for path in SCORE_ROOT.rglob("*"))


def test_model_uri_and_environment_normalization() -> None:
    assert _parse_model_uri("models:/model-name/12") == ("model-name", "12")
    with pytest.raises(ValueError, match="positive-version"):
        _parse_model_uri("models:/model-name/latest")
    assert _environment_id("mlops-v3-unified:33") == (
        "azureml:mlops-v3-unified:33"
    )
    assert _environment_id("azureml:mlops-v3-unified:33") == (
        "azureml:mlops-v3-unified:33"
    )


def test_smoke_evidence_uses_explicit_governed_datastore_path() -> None:
    output = build_evidence_output(
        datastore_name="mlops_blob",
        scenario_id="classification-healthcare-heart-disease",
        parent_job="parent_job_123",
        submission_id="d" * 32,
    )

    assert output.type == "uri_folder"
    assert output.mode == "rw_mount"
    assert output.path == (
        "azureml://datastores/mlops_blob/paths/qualification/"
        "registered-model-smoke/classification-healthcare-heart-disease/"
        f"parent_job_123/{'d' * 32}/evidence/"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("datastore_name", "../workspaceblobstore", "datastore"),
        ("scenario_id", "classification/escape", "scenario"),
        ("parent_job", "parent/job", "parent job"),
        ("submission_id", "not-a-uuid", "submission ID"),
    ],
)
def test_smoke_evidence_rejects_invalid_uri_segments(
    field: str,
    value: str,
    message: str,
) -> None:
    kwargs = {
        "datastore_name": "mlops_blob",
        "scenario_id": "classification-healthcare-heart-disease",
        "parent_job": "parent_job_123",
        "submission_id": "d" * 32,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        build_evidence_output(**kwargs)
