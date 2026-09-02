from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from api.services import pipeline_service


def test_direct_submission_uses_strict_baseline_validator(
    monkeypatch,
    tmp_path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    config_path = configs / "config_classification.yml"
    config_path.write_text("task_type: classification\n", encoding="utf-8")
    monkeypatch.setattr(pipeline_service, "_CONFIGS_DIR", configs)
    metadata = SimpleNamespace(
        task_type="classification",
        dataset_name="sample",
    )
    monkeypatch.setattr(
        pipeline_service,
        "load_config_metadata",
        lambda path: metadata,
    )
    captured = {}

    def validate(**kwargs):
        captured.update(kwargs)
        return "azureml://verified-baseline/", None, {"verified": True}

    monkeypatch.setattr(pipeline_service, "validate_baseline_job", validate)

    uri = pipeline_service._resolve_baseline_uri(
        "baseline-job",
        config_name="config_classification",
    )

    assert uri == "azureml://verified-baseline/"
    assert captured == {
        "config_path": config_path.resolve(),
        "metadata": metadata,
        "baseline_job_name": "baseline-job",
        "requested_uri": None,
    }


def test_direct_submission_cannot_keep_legacy_weak_baseline_resolution() -> None:
    source = Path("api/services/pipeline_service.py").read_text(encoding="utf-8")

    function = source[
        source.index("def _resolve_baseline_uri"):
        source.index("def _append_cli_option")
    ]
    assert "validate_baseline_job(" in function
    assert 'outputs["drift_baseline"]' not in function
