from src.steps.final_evaluation import make_quality_decision
import pytest

from src.orchestration.contracts import ContractValidationError
from src.steps.s12_model_registration import (
    resolve_quality_decision,
    validate_quality_decision_bundle,
)


def test_warn_allows_registration_but_forbids_promotion():
    quality = make_quality_decision(
        champion_valid=True,
        observed_value=0.45,
        threshold=0.5,
        metric_name="balanced_accuracy",
        block_on_quality_fail=False,
        candidate_id="candidate-1",
        evaluated_bundle_hash="bundle-1",
    )
    assert quality["decision"] == "warn"
    assert quality["registration_allowed"] is True
    assert quality["registration_tags"]["promotion_allowed"] == "false"
    assert quality["decision_hash"]
    assert resolve_quality_decision({"quality_decision": quality}) == "warn"


def test_block_forbids_registration():
    quality = make_quality_decision(
        champion_valid=True,
        observed_value=0.45,
        threshold=0.5,
        metric_name="balanced_accuracy",
        block_on_quality_fail=True,
        candidate_id="candidate-1",
        evaluated_bundle_hash="bundle-1",
    )
    assert quality["decision"] == "block"
    assert quality["registration_allowed"] is False
    assert quality["registration_tags"]["promotion_allowed"] == "false"


def test_pass_is_the_only_promotable_decision():
    quality = make_quality_decision(
        champion_valid=True,
        observed_value=0.8,
        threshold=0.5,
        metric_name="balanced_accuracy",
        block_on_quality_fail=False,
        candidate_id="candidate-1",
        evaluated_bundle_hash="bundle-1",
    )
    assert quality["decision"] == "pass"
    assert quality["registration_tags"]["promotion_allowed"] == "true"


def test_missing_quality_decision_registers_as_warning_without_promotion():
    assert resolve_quality_decision({}) == "warn"


def test_explicit_legacy_quality_failure_remains_blocked():
    assert resolve_quality_decision({"quality_gate_passed": False}) == "block"


def test_schema_v2_quality_decision_rejects_tampering():
    quality = make_quality_decision(
        champion_valid=True,
        observed_value=0.8,
        threshold=0.5,
        metric_name="balanced_accuracy",
        block_on_quality_fail=False,
        candidate_id="candidate-1",
        evaluated_bundle_hash="bundle-1",
    )
    quality["decision"] = "warn"

    with pytest.raises(Exception, match="decision_hash"):
        resolve_quality_decision(
            {"schema_version": 2, "quality_decision": quality}
        )


def test_schema_v2_quality_decision_must_match_exact_bundle():
    quality = make_quality_decision(
        champion_valid=True,
        observed_value=0.8,
        threshold=0.5,
        metric_name="balanced_accuracy",
        block_on_quality_fail=False,
        candidate_id="candidate-1",
        evaluated_bundle_hash="bundle-1",
    )

    class ExactBundle:
        candidate_id = "candidate-1"
        bundle_id = "different-bundle"

    with pytest.raises(Exception, match="does not match"):
        validate_quality_decision_bundle(
            {"schema_version": 2, "quality_decision": quality},
            ExactBundle(),
        )
