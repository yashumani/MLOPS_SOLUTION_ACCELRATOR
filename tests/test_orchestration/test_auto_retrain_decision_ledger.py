import pytest

from orchestration.auto_retrain_decision_ledger import (
    DecisionReservationConflict,
    append_decision_record,
    build_decision_record,
    latest_approved_baseline_uri,
    load_decision_records,
    reserve_candidate_submission,
    validate_decision_record,
)


def test_append_load_and_resolve_latest_approved_baseline(tmp_path):
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"

    older = build_decision_record(
        config_name="config_regression_college_azureml.yml",
        task_type="regression",
        dataset_name="college",
        decision={"outcome": "refresh_baseline", "severity": "none"},
        output_baseline_uri="azureml://old-baseline/",
        promotion_status="manual_pending",
    )
    newer = build_decision_record(
        config_name="config_regression_college_azureml.yml",
        task_type="regression",
        dataset_name="college",
        decision={"outcome": "candidate_retrain", "severity": "moderate"},
        output_baseline_uri="azureml://new-baseline/",
        approved_for_future_baseline=True,
    )

    append_decision_record(ledger_path, older)
    append_decision_record(ledger_path, newer)

    records = load_decision_records(ledger_path)

    assert len(records) == 2
    assert latest_approved_baseline_uri(
        records,
        config_name="config_regression_college_azureml.yml",
        task_type="regression",
        dataset_name="college",
    ) == "azureml://new-baseline/"


def test_latest_approved_baseline_uri_ignores_unapproved_records():
    records = [
        {
            "decision_id": "pending-record",
            "timestamp_utc": "2026-05-16T00:00:00+00:00",
            "config_name": "config_regression_college_azureml.yml",
            "task_type": "regression",
            "dataset_name": "college",
            "outcome": "candidate_retrain",
            "output_baseline_uri": "azureml://pending/",
            "promotion_status": "manual_pending",
        }
    ]

    assert latest_approved_baseline_uri(records, config_name="config_regression_college_azureml.yml") is None


def test_load_decision_records_requires_lineage_fields(tmp_path):
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    ledger_path.write_text(
        '{"decision_id":"bad-record","timestamp_utc":"2026-05-16T00:00:00Z",'
        '"config_name":"config_regression_college_azureml",'
        '"dataset_name":"college","outcome":"candidate_retrain",'
        '"promotion_status":"manual_pending"}\n'
    )

    try:
        load_decision_records(ledger_path)
    except ValueError as exc:
        assert "task_type" in str(exc)
        assert str(ledger_path) in str(exc)
    else:
        raise AssertionError("Expected missing task_type to fail validation")


def test_approved_baseline_records_require_output_baseline_uri():
    record = build_decision_record(
        config_name="config_regression_college_azureml",
        task_type="regression",
        dataset_name="college",
        decision={"outcome": "refresh_baseline"},
        approved_for_future_baseline=True,
    ).as_dict()

    try:
        validate_decision_record(record, source="unit-test")
    except ValueError as exc:
        assert "output_baseline_uri" in str(exc)
        assert "unit-test" in str(exc)
    else:
        raise AssertionError("Expected approved baseline without output URI to fail validation")


def test_append_decision_record_validates_payload(tmp_path):
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"

    try:
        append_decision_record(
            ledger_path,
            {
                "decision_id": "missing-lineage",
                "timestamp_utc": "2026-05-16T00:00:00Z",
                "config_name": "config_regression_college_azureml",
            },
        )
    except ValueError as exc:
        assert "task_type" in str(exc)
    else:
        raise AssertionError("Expected invalid append payload to fail validation")

    assert not ledger_path.exists()


def test_candidate_reservation_is_atomic_and_duplicate_safe(tmp_path):
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    first = build_decision_record(
        config_name="config_regression_college_azureml",
        task_type="regression",
        dataset_name="college",
        decision={"outcome": "candidate_retrain"},
        input_baseline_uri="azureml://baseline/",
        promotion_status="submitting",
    )
    second = build_decision_record(
        config_name="config_regression_college_azureml",
        task_type="regression",
        dataset_name="college",
        decision={"outcome": "candidate_retrain"},
        input_baseline_uri="azureml://baseline/",
        promotion_status="submitting",
    )

    reserve_candidate_submission(ledger_path, first)
    with pytest.raises(DecisionReservationConflict):
        reserve_candidate_submission(ledger_path, second)

    records = load_decision_records(ledger_path)
    assert [record["decision_id"] for record in records] == [first.decision_id]
    assert not ledger_path.with_name(ledger_path.name + ".lock").exists()


def test_candidate_reservation_requires_submitting_status(tmp_path):
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    pending = build_decision_record(
        config_name="config_regression_college_azureml",
        task_type="regression",
        dataset_name="college",
        decision={"outcome": "candidate_retrain"},
        input_baseline_uri="azureml://baseline/",
        promotion_status="manual_pending",
    )

    with pytest.raises(ValueError, match="submitting"):
        reserve_candidate_submission(ledger_path, pending)
