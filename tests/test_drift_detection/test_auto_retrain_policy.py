"""Tests for pure auto-retrain policy decisions."""

from __future__ import annotations

from src.orchestration.auto_retrain_policy import (
    AutoRetrainPolicyConfig,
    evaluate_auto_retrain_policy,
)


def _base_report() -> dict:
    return {
        "feature_psi_scores": {
            "age": 0.02,
            "income": 0.03,
            "region": 0.04,
        },
        "stability_assessment": {
            "stability_score": 88,
            "recommended_days": 90,
        },
        "comparison_drift": {
            "available": True,
            "baseline_status": "loaded",
            "feature_psi_scores": {
                "age": 0.02,
                "income": 0.03,
                "region": 0.04,
            },
            "evidently": {"dataset_drift": False},
            "concept_drift": {
                "detected": False,
                "baseline": 0.82,
                "current": 0.83,
                "drop": 0.0,
            },
        },
        "champion_info": {"registered": True},
    }


def test_policy_observes_stable_report() -> None:
    decision = evaluate_auto_retrain_policy(_base_report())
    assert decision.outcome == "observe_only"
    assert decision.should_submit is False
    assert decision.eligible_for_promotion is False
    assert decision.signals["max_feature_psi"] == 0.04


def test_policy_refreshes_baseline_when_comparison_missing() -> None:
    report = _base_report()
    report["comparison_drift"] = {
        "available": False,
        "baseline_status": "not_provided",
    }
    decision = evaluate_auto_retrain_policy(report)
    assert decision.outcome == "refresh_baseline"
    assert decision.should_submit is False
    assert "baseline" in decision.reasons[0].lower()


def test_policy_requests_candidate_retrain_for_severe_feature_drift() -> None:
    report = _base_report()
    report["feature_psi_scores"]["income"] = 0.42
    report["comparison_drift"]["feature_psi_scores"]["income"] = 0.42
    decision = evaluate_auto_retrain_policy(report)
    assert decision.outcome == "candidate_retrain"
    assert decision.should_submit is True
    assert decision.severity == "severe"
    assert decision.signals["severe_feature_count"] == 1


def test_policy_requests_candidate_retrain_for_concept_drift() -> None:
    report = _base_report()
    report["comparison_drift"]["concept_drift"] = {
        "detected": True,
        "baseline": 0.86,
        "current": 0.77,
        "drop": 0.09,
    }
    decision = evaluate_auto_retrain_policy(report)
    assert decision.outcome == "candidate_retrain"
    assert decision.should_submit is True
    assert decision.signals["concept_drift_detected"] is True


def test_policy_can_promote_only_when_explicitly_allowed() -> None:
    report = _base_report()
    report["feature_psi_scores"]["region"] = 0.31
    report["comparison_drift"]["feature_psi_scores"]["region"] = 0.31
    report["comparison_drift"]["concept_drift"] = {
        "detected": False,
        "baseline": 0.80,
        "current": 0.84,
        "drop": 0.0,
    }
    final_report = {"selection": {"score": 0.84}}
    policy = AutoRetrainPolicyConfig(allow_auto_promotion=True, minimum_promotion_delta=0.01)

    decision = evaluate_auto_retrain_policy(report, final_report=final_report, policy=policy)

    assert decision.outcome == "promote_candidate"
    assert decision.should_submit is True
    assert decision.eligible_for_promotion is True


def test_policy_does_not_submit_from_smoke_test_psi_alone() -> None:
    report = _base_report()
    report["feature_psi_scores"]["income"] = 0.42
    report["comparison_drift"].pop("feature_psi_scores")

    decision = evaluate_auto_retrain_policy(report)

    assert decision.outcome == "observe_only"
    assert decision.should_submit is False
    assert decision.signals["max_feature_psi"] == 0.0
    assert decision.signals["smoke_test_max_feature_psi"] == 0.42
