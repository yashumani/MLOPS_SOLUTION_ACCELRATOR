"""Regression tests for truthful S13 evidence and S14 decision reporting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _s13_report() -> dict:
    return {
        "identity": {"execution_id": "execution-1", "config_hash": "config-1"},
        "config_name": "config_regression_college_azureml.yml",
        "task_type": "regression",
        "dataset_name": "college",
        "feature_psi_scores": {"self_only": 0.31, "shared": 0.12},
        "smoke_test": {"status": "WARN", "overall_psi": 0.21},
        "stability_assessment": {
            "stability_score": 75,
            "recommended_cadence": "monthly",
            "recommended_days": 30,
        },
        "comparison_drift": {
            "available": True,
            "baseline_status": "loaded",
            "baseline_metadata": {"dataset_name": "college"},
            "feature_psi_scores": {"comparison_only": 0.16, "shared": 0.12},
            "evidently": {"dataset_drift": False, "drifted_columns": []},
            "concept_drift": {"detected": False, "drop": 0.0},
        },
        "auto_retrain_decision": {"outcome": "wrong_s13_owner"},
        "warnings": [],
    }


def _s14_payload(*, execution_id: str = "execution-1") -> dict:
    return {
        "stage": "s14_retrain_decision",
        "decision_id": "decision-1",
        "identity": {"execution_id": execution_id, "config_hash": "config-1"},
        "config_name": "config_regression_college_azureml.yml",
        "task_type": "regression",
        "dataset_name": "college",
        "policy": {
            "source": "drift_policy_config",
            "effective": {
                "moderate_feature_psi": 0.15,
                "severe_feature_psi": 0.25,
            },
        },
        "retrain_decision": {
            "contract_type": "RetrainDecision",
            "schema_version": "2.0",
            "decision_id": "decision-1",
            "outcome": "candidate_retrain",
            "should_submit": True,
            "eligible_for_promotion": False,
            "severity": "moderate",
            "reasons": ["comparison PSI exceeded configured policy"],
            "signals": {"max_feature_psi": 0.16},
        },
    }


class _FakeJobs:
    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads
        self.downloaded: list[str] = []

    def get(self, _job_name: str):
        return SimpleNamespace(
            outputs={name: object() for name in self.payloads},
            status="Completed",
        )

    def download(self, _job_name: str, *, download_path: str, output_name: str) -> None:
        self.downloaded.append(output_name)
        destination = Path(download_path) / f"{output_name}.json"
        destination.write_text(json.dumps(self.payloads[output_name]), encoding="utf-8")


def _run_drift(monkeypatch, payloads: dict[str, dict]):
    from api.services import pipeline_service

    jobs = _FakeJobs(payloads)
    client = SimpleNamespace(jobs=jobs)
    pipeline_service._response_cache.clear()
    monkeypatch.setattr(pipeline_service, "get_ml_client", lambda: client)
    monkeypatch.setattr(
        pipeline_service,
        "build_studio_url",
        lambda _client, job_name: f"https://studio/{job_name}",
    )
    return pipeline_service.get_job_drift("job-1"), jobs


def test_drift_detail_uses_comparison_psi_and_identity_matched_s14(monkeypatch) -> None:
    response, jobs = _run_drift(
        monkeypatch,
        {
            "drift_report": _s13_report(),
            "retrain_decision": _s14_payload(),
        },
    )

    assert jobs.downloaded == ["drift_report", "retrain_decision"]
    assert [item.feature for item in response.features] == ["comparison_only", "shared"]
    assert response.features[0].drift_detected is True
    assert response.features[0].severity == "moderate"
    assert response.features[1].drift_detected is False
    assert response.overall_drift_detected is True
    assert response.auto_retrain_decision["outcome"] == "candidate_retrain"
    assert response.auto_retrain_decision["join_status"] == "matched"
    assert response.auto_retrain_decision["matched_execution_id"] == "execution-1"
    assert response.auto_retrain_trigger["contract_type"] == "S13DriftEvidence"
    assert response.auto_retrain_trigger["ownership"] == "evidence_only"
    assert "self_only" in response.auto_retrain_trigger["self_check"]["feature_psi_scores"]
    assert "comparison_only" in response.auto_retrain_trigger["comparison"]["feature_psi_scores"]


def test_drift_detail_withholds_s14_decision_on_identity_mismatch(monkeypatch) -> None:
    response, _jobs = _run_drift(
        monkeypatch,
        {
            "drift_report": _s13_report(),
            "retrain_decision": _s14_payload(execution_id="different-execution"),
        },
    )

    assert response.auto_retrain_decision["available"] is False
    assert response.auto_retrain_decision["join_status"] == "identity_mismatch"
    assert "outcome" not in response.auto_retrain_decision
    assert any("identity mismatch" in warning for warning in response.warnings)


def test_drift_detail_withholds_inconsistent_s14_decision_identity(monkeypatch) -> None:
    s14_payload = _s14_payload()
    s14_payload["retrain_decision"]["decision_id"] = "different-decision"
    response, _jobs = _run_drift(
        monkeypatch,
        {
            "drift_report": _s13_report(),
            "retrain_decision": s14_payload,
        },
    )

    assert response.auto_retrain_decision["available"] is False
    assert response.auto_retrain_decision["join_status"] == "decision_identity_mismatch"
    assert "outcome" not in response.auto_retrain_decision
    assert any("decision identity" in warning for warning in response.warnings)
