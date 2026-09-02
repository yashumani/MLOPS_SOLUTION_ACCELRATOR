import json
from pathlib import Path

import pytest
import yaml

from orchestration.auto_retrain_controller import (
    AutoRetrainControllerError,
    AutoRetrainControllerRequest,
    AzureSubmissionContext,
    build_controller_plan,
    build_pending_decision_record,
    parse_submitted_job_name,
)


APPROVED_BASELINE = "azureml://baseline/"


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "task_type": "regression",
                "dataset": {"name": "college", "target_column": "Grad.Rate"},
            }
        )
    )


def _write_s14_decision(
    path: Path,
    *,
    should_submit: bool = True,
    baseline_uri: str = APPROVED_BASELINE,
    config_name: str = "config_regression_college_azureml.yml",
    task_type: str = "regression",
    dataset_name: str = "college",
) -> None:
    outcome = "candidate_retrain" if should_submit else "observe_only"
    decision = {
        "outcome": outcome,
        "severity": "severe" if should_submit else "none",
        "should_submit": should_submit,
        "eligible_for_promotion": False,
        "reasons": ["policy approved candidate"] if should_submit else ["stable evidence"],
        "signals": {},
    }
    source_revision = {
        "schema_version": "1.0",
        "execution_id": "source-execution",
        "config_hash": "config-sha",
        "source_sha": "source-sha",
    }
    path.write_text(
        json.dumps(
            {
                "stage": "s14_retrain_decision",
                "stage_id": "S14",
                "decision_id": "s14-decision-1",
                "config_name": config_name,
                "task_type": task_type,
                "dataset_name": dataset_name,
                "identity": {
                    "execution_id": "source-execution",
                    "config_hash": "config-sha",
                    "source_sha": "source-sha",
                },
                "source_revision": source_revision,
                "revision_validation": {
                    "status": "verified",
                    "missing_fields": [],
                    "conflicts": {},
                },
                "decision": decision,
                "retrain_decision": {
                    "contract_type": "RetrainDecision",
                    "schema_version": "2.0",
                    "decision_id": "s14-decision-1",
                    "source_revision": source_revision,
                    **decision,
                },
                "comparison": {
                    "available": True,
                    "input_baseline_uri": baseline_uri,
                },
            }
        )
    )


def _write_approved_ledger(path: Path, *, include_duplicate: bool = False) -> None:
    records = [
        {
            "decision_id": "approved-baseline",
            "timestamp_utc": "2026-05-16T00:00:00Z",
            "config_name": "config_regression_college_azureml",
            "task_type": "regression",
            "dataset_name": "college",
            "outcome": "refresh_baseline",
            "approved_for_future_baseline": True,
            "output_baseline_uri": APPROVED_BASELINE,
            "promotion_status": "manual_pending",
        }
    ]
    if include_duplicate:
        records.append(
            {
                "decision_id": "pending-candidate",
                "timestamp_utc": "2026-05-16T01:00:00Z",
                "config_name": "config_regression_college_azureml",
                "task_type": "regression",
                "dataset_name": "college",
                "outcome": "candidate_retrain",
                "input_baseline_uri": APPROVED_BASELINE,
                "candidate_job_name": "job1",
                "promotion_status": "manual_pending",
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _context() -> AzureSubmissionContext:
    return AzureSubmissionContext(
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
        compute="cpu-cluster",
    )


def _request(
    config_path: Path,
    ledger_path: Path,
    decision_path: Path,
    **kwargs,
) -> AutoRetrainControllerRequest:
    return AutoRetrainControllerRequest(
        config_path=config_path,
        ledger_path=ledger_path,
        decision_path=decision_path,
        azure_context=_context(),
        **kwargs,
    )


def test_controller_consumes_s14_decision_and_builds_canonical_dry_run(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)
    _write_s14_decision(decision_path)

    plan = build_controller_plan(
        _request(
            config_path,
            ledger_path,
            decision_path,
            mode="dry_run",
            python_executable="python",
        )
    )

    assert plan.baseline_uri == APPROVED_BASELINE
    assert plan.decision_payload["decision_id"] == "s14-decision-1"
    assert plan.experiment_name == "regression_college_auto_retrain"
    assert any(
        Path(part).name == "submit_pipeline.py"
        and Path(part).parent.name == "pipelines"
        for part in plan.command
    )
    assert "--drift_baseline_in" in plan.command
    assert APPROVED_BASELINE in plan.command
    assert "--dry_run" in plan.command
    assert plan.command[plan.command.index("--submission_revision_kind") + 1] == (
        "decision_retrain"
    )
    assert plan.command[plan.command.index("--expected_execution_id") + 1] == (
        "source-execution"
    )
    assert plan.command[plan.command.index("--expected_config_hash") + 1] == (
        "config-sha"
    )
    assert plan.command[plan.command.index("--expected_source_identity") + 1] == (
        "source-sha"
    )
    assert plan.command[plan.command.index("--source_decision_id") + 1] == (
        "s14-decision-1"
    )


def test_controller_refuses_when_s14_should_submit_is_false(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)
    _write_s14_decision(decision_path, should_submit=False)

    with pytest.raises(AutoRetrainControllerError, match="S14 policy refused"):
        build_controller_plan(_request(config_path, ledger_path, decision_path))


def test_controller_requires_explicit_s14_decision(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)

    with pytest.raises(AutoRetrainControllerError, match="S14 retrain decision not found"):
        build_controller_plan(
            _request(config_path, ledger_path, tmp_path / "missing-decision.json")
        )


def test_controller_rejects_legacy_decision_without_retrain_contract(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)
    _write_s14_decision(decision_path)
    payload = json.loads(decision_path.read_text())
    payload.pop("retrain_decision")
    decision_path.write_text(json.dumps(payload))

    with pytest.raises(AutoRetrainControllerError, match="RetrainDecision contract"):
        build_controller_plan(_request(config_path, ledger_path, decision_path))


def test_controller_rejects_decision_without_verified_source_revision(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)
    _write_s14_decision(decision_path)
    payload = json.loads(decision_path.read_text())
    payload["revision_validation"]["status"] = "incomplete"
    payload["revision_validation"]["missing_fields"] = ["source_sha"]
    decision_path.write_text(json.dumps(payload))

    with pytest.raises(AutoRetrainControllerError, match="was not verified"):
        build_controller_plan(_request(config_path, ledger_path, decision_path))


def test_controller_rejects_extra_arg_override_of_protected_lineage(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)
    _write_s14_decision(decision_path)

    with pytest.raises(AutoRetrainControllerError, match="protected canonical argument"):
        build_controller_plan(
            _request(
                config_path,
                ledger_path,
                decision_path,
                extra_args=("--drift_baseline=https://attacker.invalid/",),
            )
        )


def test_controller_rejects_non_azureml_ledger_baseline(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)
    payload = json.loads(ledger_path.read_text())
    payload["output_baseline_uri"] = "https://example.invalid/baseline"
    ledger_path.write_text(json.dumps(payload) + "\n")
    _write_s14_decision(
        decision_path,
        baseline_uri="https://example.invalid/baseline",
    )

    with pytest.raises(AutoRetrainControllerError, match="must be an azureml:// URI"):
        build_controller_plan(_request(config_path, ledger_path, decision_path))


def test_controller_refuses_identity_and_baseline_mismatches(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)
    _write_s14_decision(decision_path, dataset_name="other")

    with pytest.raises(AutoRetrainControllerError, match="dataset identity"):
        build_controller_plan(_request(config_path, ledger_path, decision_path))

    _write_s14_decision(decision_path, baseline_uri="azureml://different/")
    with pytest.raises(AutoRetrainControllerError, match="does not match"):
        build_controller_plan(_request(config_path, ledger_path, decision_path))


def test_controller_plan_fails_without_approved_baseline(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    ledger_path.write_text("")
    _write_s14_decision(decision_path)

    with pytest.raises(AutoRetrainControllerError, match="No approved drift_baseline"):
        build_controller_plan(_request(config_path, ledger_path, decision_path))


def test_submit_plan_blocks_duplicate_but_dry_run_allows_preview(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path, include_duplicate=True)
    _write_s14_decision(decision_path)

    with pytest.raises(AutoRetrainControllerError, match="pending-candidate"):
        build_controller_plan(
            _request(config_path, ledger_path, decision_path, mode="submit")
        )

    plan = build_controller_plan(
        _request(config_path, ledger_path, decision_path, mode="dry_run")
    )
    assert plan.baseline_uri == APPROVED_BASELINE


def test_force_submit_preserves_s14_lineage_in_pending_record(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path, include_duplicate=True)
    _write_s14_decision(decision_path)

    plan = build_controller_plan(
        _request(
            config_path,
            ledger_path,
            decision_path,
            mode="submit",
            force_submit=True,
            force_reason="operator approved duplicate validation run",
        )
    )
    record = build_pending_decision_record(plan, candidate_job_name="job2")

    assert "--force" in plan.command
    reason_index = plan.command.index("--force_reason")
    assert plan.command[reason_index + 1] == "operator approved duplicate validation run"
    assert record.metadata["force_submit"] is True
    assert record.metadata["source_s14_decision_id"] == "s14-decision-1"
    assert record.metadata["source_identity"] == {
        "execution_id": "source-execution",
        "config_hash": "config-sha",
        "source_sha": "source-sha",
    }
    assert record.metadata["source_revision"]["execution_id"] == "source-execution"


def test_force_submit_without_reason_fails_closed(tmp_path):
    config_path = tmp_path / "config_regression_college_azureml.yml"
    ledger_path = tmp_path / "auto_retrain_decisions.jsonl"
    decision_path = tmp_path / "retrain_decision.json"
    _write_config(config_path)
    _write_approved_ledger(ledger_path)
    _write_s14_decision(decision_path)

    with pytest.raises(AutoRetrainControllerError, match="force_reason"):
        build_controller_plan(
            _request(
                config_path,
                ledger_path,
                decision_path,
                mode="submit",
                force_submit=True,
            )
        )


def test_parse_submitted_job_name():
    assert (
        parse_submitted_job_name("Submitted job: loyal_owl_0h0rz9krcn\n")
        == "loyal_owl_0h0rz9krcn"
    )
