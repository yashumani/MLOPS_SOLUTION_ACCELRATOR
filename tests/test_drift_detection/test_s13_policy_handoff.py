"""Regression tests for the S13 evidence-only ownership boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from orchestration.contracts import ExecutionManifest
from steps.s13_drift_monitor import (
    _collect_upstream_identity,
    _drift_alert_feature_names,
)


S13_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "steps"
    / "s13_drift_monitor.py"
)


def test_s13_emits_evidence_without_evaluating_retrain_policy() -> None:
    source = S13_SOURCE.read_text(encoding="utf-8")

    assert "evaluate_auto_retrain_policy" not in source
    assert "_build_auto_retrain_handoff" not in source
    assert '"auto_retrain_decision"' not in source
    assert '"auto_retrain_trigger"' not in source


def test_s13_contains_no_child_pipeline_submission_path() -> None:
    source = S13_SOURCE.read_text(encoding="utf-8")

    assert "PipelineTrigger" not in source
    assert "submit_pipeline.py" not in source
    assert "subprocess" not in source


def test_s13_preserves_upstream_identity_without_synthesizing_values(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

    identity = _collect_upstream_identity(
        {
            "identity": {
                "execution_id": "upstream-run",
                "config_hash": "config-sha",
                "model_bundle_id": "bundle-7",
            }
        },
        {
            "identity": {
                "model_version": "12",
                "source_sha": "source-sha",
            }
        },
    )

    assert identity == {
        "execution_id": "upstream-run",
        "config_hash": "config-sha",
        "model_bundle_id": "bundle-7",
        "model_version": "12",
        "source_sha": "source-sha",
    }


def test_s13_normalizes_structured_drift_features_for_alerting() -> None:
    assert _drift_alert_feature_names(
        [
            {"feature": "age", "psi": 0.12, "severity": "yellow"},
            {"feature": "income", "psi": 0.28, "severity": "red"},
            "legacy_feature",
        ]
    ) == ["age", "income", "legacy_feature"]


def _final_report(task_type="classification"):
    manifest = ExecutionManifest(
        config_hash="config-sha", task_type=task_type,
        dataset={"name": "qualification", "version": "v1", "content_sha256": "data-sha"},
        split_policy={"strategy": "random", "seed": 42}, engines=("pycaret",),
        recipe_paths=(f"configs/recipes/{task_type}/baseline.yml",),
        recipe_ids=("baseline",), candidate_ids=("candidate-1",), budgets={},
        code_sha="source-sha", environment_hashes={"training": "env-sha"},
        recipe_catalog_hash="catalog-sha",
    )
    return {
        "execution_manifest": manifest.to_dict(),
        "lineage": {
            "execution_id": manifest.execution_id, "config_hash": manifest.config_hash,
            "code_sha": manifest.code_sha, "split_id": "split-sha",
            "source_bundle_id": "pre-evaluation-bundle",
        },
        "model_bundle": {"bundle_id": "evaluated-bundle", "candidate_id": "candidate-1"},
    }


@pytest.mark.parametrize("task_type", ["classification", "regression", "clustering"])
def test_s13_maps_actual_final_evaluation_contract(task_type):
    report = _final_report(task_type)
    identity = _collect_upstream_identity(report, {"version": "7", "code_sha": "source-sha"})
    assert identity["source_sha"] == "source-sha"
    assert identity["data_fingerprint"] == "data-sha"
    assert identity["model_bundle_id"] == "evaluated-bundle"
    assert identity["candidate_id"] == "candidate-1"
    assert identity["dataset_version"] == "v1"
    assert identity["split_fingerprint"] == "split-sha"
    assert identity["model_version"] == "7"
    assert identity["execution_id"] == report["execution_manifest"]["execution_id"]


@pytest.mark.parametrize("field", ["source_sha", "config_hash", "execution_id", "model_bundle_id", "data_fingerprint"])
def test_s13_rejects_conflicting_baseline_identity(field):
    with pytest.raises(ValueError, match=f"Conflicting upstream baseline identity: {field}"):
        _collect_upstream_identity(_final_report(), {"identity": {field: "conflicting"}})


def test_s13_rejects_tampered_execution_manifest():
    report = _final_report()
    report["execution_manifest"]["code_sha"] = "tampered"
    with pytest.raises(ValueError, match="execution_id does not match"):
        _collect_upstream_identity(report, {})


def test_s13_does_not_write_baseline_with_incomplete_identity(monkeypatch):
    from types import SimpleNamespace
    from steps import s13_drift_monitor as stage

    monkeypatch.setattr(stage, "_safe_disable_autolog", lambda: None)
    monkeypatch.setattr(stage, "_load_config", lambda _: {"dataset": {"name": "qualification"}})
    monkeypatch.setattr(stage, "_load_json_safe", lambda *_: {})
    with pytest.raises(ValueError, match="Cannot produce a reusable drift baseline"):
        stage.run_drift_monitor(SimpleNamespace(
            config_name="qualification.yml", final_report="report.json", registry_info=None,
        ))


def test_s13_keeps_azureml_tracking_uri_unchanged() -> None:
    source = S13_SOURCE.read_text(encoding="utf-8")

    assert 'replace("azureml://"' not in source
    assert "mlflow.set_tracking_uri" not in source


def test_s13_requires_reference_data_before_comparison_is_available() -> None:
    source = S13_SOURCE.read_text(encoding="utf-8")
    reference_guard = source.index("if ref_data is not None and not ref_data.empty:")
    available_true = source.index('comparison_drift["available"] = True', reference_guard)
    loaded_status = source.index('comparison_drift["baseline_status"] = "loaded"', reference_guard)

    assert reference_guard < available_true < loaded_status
    assert 'comparison_drift["available"] = False' in source[loaded_status:]
    assert 'comparison_drift["baseline_status"] = "loaded_no_reference_data"' in source[loaded_status:]
