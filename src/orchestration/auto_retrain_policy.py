"""Auto-retrain decision policy for V3 drift reports.

This module is intentionally pure: it does not submit Azure ML jobs, touch
MLflow, or read/write datastore assets. It turns already-produced pipeline
artifacts into an auditable decision that a controller can act on later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Literal

AutoRetrainOutcome = Literal[
    "observe_only",
    "refresh_baseline",
    "candidate_retrain",
    "promote_candidate",
    "blocked",
]


@dataclass(frozen=True)
class AutoRetrainPolicyConfig:
    """Thresholds and gates used to evaluate an auto-retrain decision."""

    moderate_feature_psi: float = 0.10
    severe_feature_psi: float = 0.25
    severe_drifted_share: float = 0.30
    concept_drift_drop: float = 0.05
    low_stability_score: float = 40.0
    urgent_cadence_days: int = 14
    minimum_promotion_delta: float = 0.01
    allow_auto_promotion: bool = False
    require_registered_candidate_for_promotion: bool = True


@dataclass(frozen=True)
class AutoRetrainDecision:
    """Policy result emitted by the auto-retrain controller."""

    outcome: AutoRetrainOutcome
    should_submit: bool
    eligible_for_promotion: bool
    severity: str
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "should_submit": self.should_submit,
            "eligible_for_promotion": self.eligible_for_promotion,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
        }


def evaluate_auto_retrain_policy(
    drift_report: dict[str, Any],
    final_report: dict[str, Any] | None = None,
    registry_info: dict[str, Any] | None = None,
    policy: AutoRetrainPolicyConfig | None = None,
) -> AutoRetrainDecision:
    """Evaluate whether drift artifacts warrant retraining or promotion.

    The function intentionally separates two decisions:
    - `should_submit`: whether a candidate retrain/evaluation should be run.
    - `eligible_for_promotion`: whether the candidate may replace production.

    First-release production posture should leave `allow_auto_promotion=False`.
    """
    policy = policy or AutoRetrainPolicyConfig()
    final_report = final_report or {}
    registry_info = registry_info or {}

    comparison = drift_report.get("comparison_drift") or {}
    stability = drift_report.get("stability_assessment") or {}
    smoke_psi_scores = _coerce_psi_scores(drift_report.get("feature_psi_scores") or {})
    psi_scores = _coerce_psi_scores(
        comparison.get("feature_psi_scores")
        or comparison.get("comparison_feature_psi_scores")
        or comparison.get("psi_scores")
        or {}
    )
    evidently = comparison.get("evidently") or {}
    concept = comparison.get("concept_drift") or {}

    max_psi = max(psi_scores.values(), default=0.0)
    mean_psi = mean(psi_scores.values()) if psi_scores else 0.0
    smoke_max_psi = max(smoke_psi_scores.values(), default=0.0)
    smoke_mean_psi = mean(smoke_psi_scores.values()) if smoke_psi_scores else 0.0
    feature_count = len(psi_scores)
    drifted_count = sum(1 for value in psi_scores.values() if value >= policy.moderate_feature_psi)
    severe_count = sum(1 for value in psi_scores.values() if value >= policy.severe_feature_psi)
    drifted_share = drifted_count / feature_count if feature_count else 0.0

    concept_drop = _as_float(concept.get("drop"), 0.0)
    concept_detected = bool(concept.get("detected")) or concept_drop >= policy.concept_drift_drop
    dataset_drift = bool(evidently.get("dataset_drift"))
    comparison_available = bool(comparison.get("available"))
    baseline_status = comparison.get("baseline_status") or (
        "loaded" if comparison_available else "not_available"
    )
    stability_score = _as_float(stability.get("stability_score"), 100.0)
    recommended_days = _as_int(stability.get("recommended_days"), 0)
    registered_candidate = _candidate_registered(drift_report, registry_info)

    reasons: list[str] = []
    severity = "none"

    if not comparison_available:
        return AutoRetrainDecision(
            outcome="refresh_baseline",
            should_submit=False,
            eligible_for_promotion=False,
            severity="none",
            reasons=[
                "No previous baseline comparison is available; capture this run's drift_baseline first."
            ],
            signals={
                "baseline_status": baseline_status,
                "comparison_available": False,
                "max_feature_psi": max_psi,
                "mean_feature_psi": mean_psi,
                "smoke_test_max_feature_psi": smoke_max_psi,
                "smoke_test_mean_feature_psi": smoke_mean_psi,
                "stability_score": stability_score,
                "recommended_days": recommended_days,
            },
        )

    if concept_detected:
        severity = "severe"
        reasons.append(f"Concept drift detected with metric drop {concept_drop:.4f}.")

    if dataset_drift:
        severity = "severe"
        reasons.append("Evidently reported dataset drift against the approved baseline.")

    if max_psi >= policy.severe_feature_psi:
        severity = "severe"
        reasons.append(f"Max feature PSI {max_psi:.4f} exceeds severe threshold {policy.severe_feature_psi:.2f}.")
    elif drifted_count:
        severity = "moderate"
        reasons.append(f"{drifted_count} feature(s) exceed moderate PSI threshold {policy.moderate_feature_psi:.2f}.")

    if drifted_share >= policy.severe_drifted_share:
        severity = "severe"
        reasons.append(f"Drifted feature share {drifted_share:.2%} exceeds {policy.severe_drifted_share:.0%}.")

    if stability_score <= policy.low_stability_score:
        severity = "severe" if severity != "moderate" else severity
        reasons.append(f"Stability score {stability_score:.2f} is at or below {policy.low_stability_score:.2f}.")

    if recommended_days and recommended_days <= policy.urgent_cadence_days:
        if severity == "none":
            severity = "moderate"
        reasons.append(f"Recommended cadence is urgent: every {recommended_days} day(s).")

    score_delta = _score_delta(final_report, concept)
    promotion_blockers = []
    if policy.require_registered_candidate_for_promotion and not registered_candidate:
        promotion_blockers.append("candidate model is not registered")
    if score_delta is None or score_delta < policy.minimum_promotion_delta:
        promotion_blockers.append("candidate improvement delta is below promotion threshold")
    if not policy.allow_auto_promotion:
        promotion_blockers.append("auto-promotion is disabled by policy")

    should_submit = bool(reasons)
    eligible_for_promotion = should_submit and not promotion_blockers
    outcome: AutoRetrainOutcome
    if eligible_for_promotion:
        outcome = "promote_candidate"
    elif should_submit:
        outcome = "candidate_retrain"
    else:
        outcome = "observe_only"
        reasons.append("No drift or cadence signal exceeded retrain thresholds.")

    signals = {
        "baseline_status": baseline_status,
        "comparison_available": comparison_available,
        "dataset_drift": dataset_drift,
        "concept_drift_detected": concept_detected,
        "concept_drift_drop": concept_drop,
        "max_feature_psi": max_psi,
        "mean_feature_psi": mean_psi,
        "smoke_test_max_feature_psi": smoke_max_psi,
        "smoke_test_mean_feature_psi": smoke_mean_psi,
        "drifted_feature_count": drifted_count,
        "severe_feature_count": severe_count,
        "feature_count": feature_count,
        "drifted_feature_share": drifted_share,
        "stability_score": stability_score,
        "recommended_days": recommended_days,
        "registered_candidate": registered_candidate,
        "score_delta": score_delta,
        "promotion_blockers": promotion_blockers,
    }

    return AutoRetrainDecision(
        outcome=outcome,
        should_submit=should_submit,
        eligible_for_promotion=eligible_for_promotion,
        severity=severity,
        reasons=reasons,
        signals=signals,
    )


def _coerce_psi_scores(raw_scores: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for feature_name, value in raw_scores.items():
        if isinstance(value, dict):
            value = value.get("psi") or value.get("score") or value.get("value")
        try:
            scores[str(feature_name)] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def _candidate_registered(drift_report: dict[str, Any], registry_info: dict[str, Any]) -> bool:
    if registry_info:
        return not bool(registry_info.get("registration_skipped")) and bool(
            registry_info.get("model_name") or registry_info.get("version")
        )
    champion = drift_report.get("champion_info") or {}
    return bool(champion.get("registered"))


def _score_delta(final_report: dict[str, Any], concept: dict[str, Any]) -> float | None:
    current = (final_report.get("selection") or {}).get("score")
    if current is None:
        current = concept.get("current")
    baseline = concept.get("baseline")
    try:
        if current is None or baseline is None:
            return None
        return float(current) - float(baseline)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
