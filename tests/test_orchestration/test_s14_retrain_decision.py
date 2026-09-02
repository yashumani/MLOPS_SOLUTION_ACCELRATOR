"""Tests for s14 retrain decision artifact generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestration.contracts import ExecutionManifest
from steps.s14_retrain_decision import (
    _load_retrain_policy,
    build_retrain_decision_payload,
    run_retrain_decision,
)


def _execution_manifest() -> dict:
    return ExecutionManifest(
        config_hash="config-sha",
        task_type="regression",
        dataset={"name": "college", "version": "1"},
        split_policy={"strategy": "random", "seed": 42},
        engines=("pycaret",),
        recipe_paths=("configs/recipes/regression/baseline_recipe.yml",),
        recipe_ids=("baseline-regression",),
        candidate_ids=("candidate-1",),
        budgets={"round1_max_variants": 1},
        code_sha="source-sha",
        environment_hashes={"training": "environment-sha"},
        recipe_catalog_hash="catalog-sha",
    ).to_dict()


def _stable_report() -> dict:
    return {
        "config_name": "config_regression_college_azureml.yml",
        "task_type": "regression",
        "dataset_name": "college",
        "feature_psi_scores": {
            "admit_rate": 0.02,
            "tuition": 0.03,
        },
        "stability_assessment": {
            "stability_score": 82,
            "recommended_days": 90,
        },
        "comparison_drift": {
            "available": True,
            "baseline_status": "loaded",
            "baseline_path": "azureml://baseline/",
            "feature_psi_scores": {
                "admit_rate": 0.02,
                "tuition": 0.03,
            },
            "evidently": {"dataset_drift": False},
            "concept_drift": {
                "detected": False,
                "baseline": 0.72,
                "current": 0.73,
                "drop": 0.0,
            },
        },
        "champion_info": {"registered": True},
    }


def test_s14_payload_observes_stable_comparison_report(tmp_path: Path) -> None:
    report = _stable_report()
    report["identity"] = {
        "execution_id": "s10-run",
        "config_hash": "config-sha",
        "model_bundle_id": "bundle-7",
    }
    payload, record = build_retrain_decision_payload(
        config_name="config_regression_college_azureml.yml",
        drift_report=report,
        final_report={"selection": {"score": 0.73}},
        registry_info={
            "model_name": "college_model",
            "version": "1",
            "identity": {"model_version": "1", "source_sha": "source-sha"},
        },
        candidate_baseline_path=tmp_path / "drift_baseline",
    )

    assert payload["stage"] == "s14_retrain_decision"
    assert payload["decision"]["outcome"] == "observe_only"
    assert payload["decision"]["should_submit"] is False
    assert payload["retrain_decision"]["contract_type"] == "RetrainDecision"
    assert payload["retrain_decision"]["schema_version"] == "2.0"
    assert payload["retrain_decision"]["decision_id"] == payload["decision_id"]
    assert payload["retrain_decision"]["outcome"] == payload["decision"]["outcome"]
    assert payload["identity"] == {
        "execution_id": "s10-run",
        "config_hash": "config-sha",
        "model_bundle_id": "bundle-7",
        "model_version": "1",
        "source_sha": "source-sha",
    }
    assert payload["source_revision"] == {
        "schema_version": "1.0",
        "execution_id": "s10-run",
        "config_hash": "config-sha",
        "source_sha": "source-sha",
    }
    assert payload["revision_validation"]["status"] == "incomplete"
    assert payload["revision_validation"]["required_contract_present"] is False
    assert payload["retrain_decision"]["source_revision"] == payload["source_revision"]
    assert record.outcome == "observe_only"
    assert record.input_baseline_uri == "azureml://baseline/"
    assert record.approved_for_future_baseline is False
    assert record.metadata["candidate_baseline_path"] == str(tmp_path / "drift_baseline")


def test_s14_payload_requests_candidate_retrain_for_drift(tmp_path: Path) -> None:
    report = _stable_report()
    report["feature_psi_scores"]["tuition"] = 0.44
    report["comparison_drift"]["feature_psi_scores"]["tuition"] = 0.44
    execution_manifest = _execution_manifest()

    payload, record = build_retrain_decision_payload(
        config_name="config_regression_college_azureml.yml",
        drift_report=report,
        final_report={
            "selection": {"score": 0.73},
            "execution_manifest": execution_manifest,
        },
        registry_info={"model_name": "college_model", "version": "1"},
        candidate_baseline_path=tmp_path / "drift_baseline",
        trigger="schedule_s14",
        schedule_name="auto-retrain-regression-college-daily",
    )

    assert payload["decision"]["outcome"] == "candidate_retrain"
    assert payload["decision"]["should_submit"] is True
    assert payload["decision"]["severity"] == "severe"
    assert record.trigger == "schedule_s14"
    assert record.schedule_name == "auto-retrain-regression-college-daily"
    assert record.promotion_status == "manual_pending"
    assert record.signals["severe_feature_count"] == 1
    assert payload["planned_schedules_table"]["summary"]["total_planned_schedules"] == 3
    assert record.metadata["planned_schedules_table"] == payload["planned_schedules_table"]
    assert payload["source_revision"]["execution_id"] == execution_manifest["execution_id"]
    assert payload["source_revision"]["source_sha"] == "source-sha"
    assert payload["revision_validation"]["status"] == "verified"
    current_rows = [row for row in payload["planned_schedules_table"]["rows"] if row["is_current"]]
    assert len(current_rows) == 1
    assert current_rows[0]["schedule_name"] == "auto-retrain-regression-college-daily"
    assert current_rows[0]["task_type"] == "regression"
    assert current_rows[0]["dataset_name"] == "college"
    assert current_rows[0]["outcome"] == "candidate_retrain"
    assert current_rows[0]["input_baseline_uri"] == "azureml://baseline/"


def test_s14_blocks_submission_when_source_revision_is_incomplete(tmp_path: Path) -> None:
    report = _stable_report()
    report["feature_psi_scores"]["tuition"] = 0.44
    report["comparison_drift"]["feature_psi_scores"]["tuition"] = 0.44

    payload, record = build_retrain_decision_payload(
        config_name="config_regression_college_azureml.yml",
        drift_report=report,
        final_report={"selection": {"score": 0.73}},
        registry_info={"model_name": "college_model", "version": "1"},
        candidate_baseline_path=tmp_path / "drift_baseline",
    )

    assert payload["revision_validation"]["status"] == "incomplete"
    assert payload["decision"]["outcome"] == "blocked"
    assert payload["decision"]["should_submit"] is False
    assert record.outcome == "blocked"
    assert "Immutable source revision is incomplete" in payload["decision"]["reasons"][-1]


def test_s14_blocks_conflicting_execution_identity(tmp_path: Path) -> None:
    report = _stable_report()
    report["feature_psi_scores"]["tuition"] = 0.44
    report["comparison_drift"]["feature_psi_scores"]["tuition"] = 0.44
    report["identity"] = {
        "execution_id": "drift-execution",
        "config_hash": "config-sha",
        "source_sha": "source-sha",
    }

    execution_manifest = _execution_manifest()
    payload, _ = build_retrain_decision_payload(
        config_name="config_regression_college_azureml.yml",
        drift_report=report,
        final_report={
            "selection": {"score": 0.73},
            "execution_manifest": execution_manifest,
        },
        registry_info={"model_name": "college_model", "version": "1"},
        candidate_baseline_path=tmp_path / "drift_baseline",
    )

    assert payload["revision_validation"]["status"] == "conflict"
    assert payload["revision_validation"]["conflicts"]["execution_id"] == sorted(
        ["drift-execution", execution_manifest["execution_id"]]
    )
    assert payload["decision"]["outcome"] == "blocked"
    assert payload["decision"]["should_submit"] is False


def test_s14_blocks_tampered_execution_manifest(tmp_path: Path) -> None:
    report = _stable_report()
    report["feature_psi_scores"]["tuition"] = 0.44
    report["comparison_drift"]["feature_psi_scores"]["tuition"] = 0.44
    execution_manifest = _execution_manifest()
    execution_manifest["execution_id"] = "forged-execution-id"

    payload, _ = build_retrain_decision_payload(
        config_name="config_regression_college_azureml.yml",
        drift_report=report,
        final_report={
            "selection": {"score": 0.73},
            "execution_manifest": execution_manifest,
        },
        registry_info={"model_name": "college_model", "version": "1"},
        candidate_baseline_path=tmp_path / "drift_baseline",
    )

    assert payload["revision_validation"]["status"] == "invalid"
    assert "execution_id does not match" in payload["revision_validation"][
        "contract_error"
    ]
    assert payload["decision"]["outcome"] == "blocked"
    assert payload["decision"]["should_submit"] is False


def test_s14_run_uses_configured_feature_psi_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = _stable_report()
    report["identity"] = {"execution_id": "execution-1"}
    report["feature_psi_scores"]["tuition"] = 0.12
    report["comparison_drift"]["feature_psi_scores"]["tuition"] = 0.12
    report_path = tmp_path / "drift_report.json"
    decision_path = tmp_path / "retrain_decision.json"
    ledger_record_path = tmp_path / "decision_ledger_record.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    policy_path = Path(__file__).resolve().parents[2] / "configs" / "drift_config.yaml"

    monkeypatch.setattr("steps.s14_retrain_decision._log_decision", lambda _payload: None)
    result = run_retrain_decision(
        argparse.Namespace(
            config_name="config_regression_college_azureml.yml",
            drift_report=str(report_path),
            candidate_baseline=str(tmp_path / "drift_baseline"),
            final_report=None,
            registry_info=None,
            drift_policy_config=str(policy_path),
            trigger="test",
            schedule_name=None,
            retrain_decision=str(decision_path),
            decision_ledger_record=str(ledger_record_path),
        )
    )

    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["decision"]["outcome"] == "observe_only"
    assert payload["decision"]["should_submit"] is False
    assert payload["policy"]["source"] == "drift_policy_config"
    assert payload["policy"]["effective"]["moderate_feature_psi"] == 0.15
    assert len(payload["policy"]["config_sha256"]) == 64


def test_s14_component_wires_default_drift_policy_config() -> None:
    component = (
        Path(__file__).resolve().parents[2]
        / "components"
        / "s14_retrain_decision.yml"
    ).read_text(encoding="utf-8")
    policy, metadata = _load_retrain_policy(
        str(Path(__file__).resolve().parents[2] / "configs" / "drift_config.yaml")
    )

    assert "drift_policy_config:" in component
    assert "default: configs/drift_config.yaml" in component
    assert "--drift_policy_config ${{inputs.drift_policy_config}}" in component
    assert policy.moderate_feature_psi == 0.15
    assert metadata["effective"]["concept_drift_drop"] == 0.05


def test_s14_payload_refreshes_baseline_when_comparison_missing(tmp_path: Path) -> None:
    report = _stable_report()
    report["comparison_drift"] = {
        "available": False,
        "baseline_status": "not_provided",
    }

    payload, record = build_retrain_decision_payload(
        config_name="config_regression_college_azureml.yml",
        drift_report=report,
        final_report={},
        registry_info={},
        candidate_baseline_path=tmp_path / "drift_baseline",
    )

    assert payload["decision"]["outcome"] == "refresh_baseline"
    assert payload["decision"]["should_submit"] is False
    assert record.outcome == "refresh_baseline"
    assert record.output_baseline_uri is None


def test_s14_payload_includes_planned_schedules_for_classification(tmp_path: Path) -> None:
    report = _stable_report()
    report["config_name"] = "config_classification_telecom_churn_azureml.yml"
    report["task_type"] = "classification"
    report["dataset_name"] = "telecom_churn"

    payload, record = build_retrain_decision_payload(
        config_name="config_classification_telecom_churn_azureml.yml",
        drift_report=report,
        final_report={},
        registry_info={},
        candidate_baseline_path=tmp_path / "drift_baseline",
    )

    current_rows = [row for row in payload["planned_schedules_table"]["rows"] if row["is_current"]]
    assert len(current_rows) == 1
    assert current_rows[0]["schedule_name"] == "auto-retrain-classification-telecom-churn-daily"
    assert current_rows[0]["task_type"] == "classification"
    assert record.metadata["planned_schedules_table"] == payload["planned_schedules_table"]


def test_s14_payload_includes_planned_schedules_for_clustering(tmp_path: Path) -> None:
    report = _stable_report()
    report["config_name"] = "config_clustering_online_retail_azureml.yml"
    report["task_type"] = "clustering"
    report["dataset_name"] = "online_retail_clustering"

    payload, record = build_retrain_decision_payload(
        config_name="config_clustering_online_retail_azureml.yml",
        drift_report=report,
        final_report={},
        registry_info={},
        candidate_baseline_path=tmp_path / "drift_baseline",
        schedule_name="auto-retrain-clustering-online-retail-daily",
    )

    current_rows = [row for row in payload["planned_schedules_table"]["rows"] if row["is_current"]]
    assert len(current_rows) == 1
    assert current_rows[0]["schedule_name"] == "auto-retrain-clustering-online-retail-daily"
    assert current_rows[0]["task_type"] == "clustering"
    assert record.metadata["planned_schedules_table"] == payload["planned_schedules_table"]


def test_s14_has_no_submission_path_and_keeps_azureml_uri_unchanged() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "steps"
        / "s14_retrain_decision.py"
    ).read_text(encoding="utf-8")

    assert "submit_pipeline.py" not in source
    assert "subprocess" not in source
    assert 'replace("azureml://"' not in source
    assert "mlflow.set_tracking_uri" not in source
