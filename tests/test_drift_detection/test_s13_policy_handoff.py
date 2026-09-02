"""Regression tests for the S13 evidence-only ownership boundary."""

from __future__ import annotations

from pathlib import Path

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
