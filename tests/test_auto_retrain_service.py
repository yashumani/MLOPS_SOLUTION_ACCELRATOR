from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from api.schemas.pipeline import (
    AutoRetrainBaselineApprovalRequest,
    AutoRetrainControllerPlanRequest,
)
from api.services import auto_retrain_service


def _write_config(configs_dir: Path) -> Path:
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / "config_regression_college_azureml.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "college-regression",
                "task_type": "regression",
                "dataset": {"name": "college", "target_column": "Grad.Rate"},
            }
        )
    )
    return path


def _patch_baseline_job_client(monkeypatch, config_path: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = auto_retrain_service.compile_config(
        raw,
        source_name=config_path.name,
    )["compiled_config_hash"]
    baseline_uri = "azureml://datastores/mlops_blob/paths/baseline/"
    job = SimpleNamespace(
        name="baseline-job",
        status="Completed",
        outputs={"drift_baseline": SimpleNamespace(path=baseline_uri)},
        tags={
            "task": "regression",
            "dataset": "college",
            "compiled_config_hash": config_hash,
            "execution_id": "execution-1",
            "source_identity": "source-sha",
        },
    )

    def download(*, name, output_name, download_path):
        assert name == "baseline-job"
        assert output_name == "drift_baseline"
        output_dir = Path(download_path) / "named-outputs" / output_name
        output_dir.mkdir(parents=True)
        (output_dir / "feature_baseline.json").write_text(
            json.dumps(
                {
                    "dataset_name": "college",
                    "task_type": "regression",
                    "identity": {
                        "execution_id": "execution-1",
                        "config_hash": config_hash,
                        "source_sha": "source-sha",
                        "model_bundle_id": "bundle-1",
                        "data_fingerprint": "data-sha",
                    },
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "reference_data.csv").write_text(
            "feature\n1\n",
            encoding="utf-8",
        )

    client = SimpleNamespace(
        jobs=SimpleNamespace(get=lambda name: job, download=download),
        subscription_id="sub",
        resource_group_name="rg",
        workspace_name="ws",
    )
    monkeypatch.setattr(auto_retrain_service, "get_ml_client", lambda: client)


def _write_approved_baseline(ledger_path: Path) -> None:
    ledger_path.write_text(
        '{"decision_id":"approved-baseline","timestamp_utc":"2026-05-16T00:00:00Z",'
        '"config_name":"config_regression_college_azureml","task_type":"regression",'
        '"dataset_name":"college","approved_for_future_baseline":true,'
        '"outcome":"refresh_baseline","promotion_status":"approved",'
        '"output_baseline_uri":"azureml://baseline/"}\n'
    )


def _write_s14_decision(decision_path: Path) -> None:
    source_revision = {
        "schema_version": "1.0",
        "execution_id": "source-execution",
        "config_hash": "config-sha",
        "source_sha": "source-sha",
    }
    decision_path.write_text(
        json.dumps(
            {
                "stage": "s14_retrain_decision",
                "stage_id": "S14",
                "decision_id": "s14-decision-1",
                "config_name": "config_regression_college_azureml.yml",
                "task_type": "regression",
                "dataset_name": "college",
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
                "decision": {
                    "outcome": "candidate_retrain",
                    "severity": "severe",
                    "should_submit": True,
                    "eligible_for_promotion": False,
                    "reasons": ["policy approved candidate"],
                    "signals": {},
                },
                "retrain_decision": {
                    "contract_type": "RetrainDecision",
                    "schema_version": "2.0",
                    "decision_id": "s14-decision-1",
                    "source_revision": source_revision,
                    "outcome": "candidate_retrain",
                    "severity": "severe",
                    "should_submit": True,
                    "eligible_for_promotion": False,
                    "reasons": ["policy approved candidate"],
                    "signals": {},
                },
                "comparison": {
                    "available": True,
                    "input_baseline_uri": "azureml://baseline/",
                },
            }
        )
    )


def _patch_service_context(
    monkeypatch,
    configs_dir: Path,
    ledger_root: Path,
) -> None:
    monkeypatch.setattr(auto_retrain_service, "_CONFIGS_DIR", configs_dir)
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))
    monkeypatch.delenv("MLOPS_AUTO_RETRAIN_LEDGER", raising=False)
    monkeypatch.setattr(auto_retrain_service.settings, "azure_subscription_id", "sub")
    monkeypatch.setattr(auto_retrain_service.settings, "azure_resource_group", "rg")
    monkeypatch.setattr(auto_retrain_service.settings, "azure_workspace_name", "ws")
    monkeypatch.setattr(auto_retrain_service.settings, "compute_target", "cpu-cluster")


def test_auto_retrain_service_builds_controller_plan(monkeypatch, tmp_path) -> None:
    configs_dir = tmp_path / "configs"
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    ledger_path = ledger_root / "auto_retrain_decisions.jsonl"
    decision_path = ledger_root / "retrain_decision.json"
    _write_config(configs_dir)
    _write_approved_baseline(ledger_path)
    _write_s14_decision(decision_path)
    _patch_service_context(monkeypatch, configs_dir, ledger_root)

    result = auto_retrain_service.build_auto_retrain_controller_plan(
        AutoRetrainControllerPlanRequest(
            config_name="config_regression_college_azureml.yml",
            ledger_path=ledger_path.name,
            decision_path=decision_path.name,
            trigger="unit_test",
        )
    )

    assert result.baseline_uri == "azureml://baseline/"
    assert result.task_type == "regression"
    assert result.dataset_name == "college"
    assert Path(result.decision_path) == decision_path
    assert "--drift_baseline_in azureml://baseline/" in result.command
    assert result.pending_decision_record["trigger"] == "unit_test"


def test_auto_retrain_service_approves_explicit_baseline_uri(monkeypatch, tmp_path) -> None:
    configs_dir = tmp_path / "configs"
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    ledger_path = ledger_root / "auto_retrain_decisions.jsonl"
    config_path = _write_config(configs_dir)
    _patch_service_context(monkeypatch, configs_dir, ledger_root)
    _patch_baseline_job_client(monkeypatch, config_path, tmp_path)

    result = auto_retrain_service.approve_auto_retrain_baseline(
        AutoRetrainBaselineApprovalRequest(
            config_name="config_regression_college_azureml.yml",
            baseline_job_name="baseline-job",
            output_baseline_uri="azureml://datastores/mlops_blob/paths/baseline/",
            ledger_path=ledger_path.name,
            reason="unit test approval",
        )
    )

    assert result.status == "approved"
    assert result.baseline_uri == "azureml://datastores/mlops_blob/paths/baseline/"
    assert result.record["approved_for_future_baseline"] is True
    assert result.record["promotion_status"] == "approved"
    assert result.record["metadata"]["baseline_identity_verified"] is True
    assert ledger_path.exists()


def test_auto_retrain_service_requires_producing_job_for_approval(
    monkeypatch,
    tmp_path,
) -> None:
    configs_dir = tmp_path / "configs"
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    _write_config(configs_dir)
    _patch_service_context(monkeypatch, configs_dir, ledger_root)

    with pytest.raises(ValueError, match="baseline_job_name is required"):
        auto_retrain_service.approve_auto_retrain_baseline(
            AutoRetrainBaselineApprovalRequest(
                config_name="config_regression_college_azureml.yml",
                output_baseline_uri="azureml://datastores/mlops_blob/paths/baseline/",
            )
        )


def test_baseline_validation_accepts_verified_system_generated_output(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path / "configs")
    _patch_baseline_job_client(monkeypatch, config_path, tmp_path)
    client = auto_retrain_service.get_ml_client()
    job = client.jobs.get("baseline-job")
    job.outputs["drift_baseline"] = SimpleNamespace(path=None, type="uri_folder")

    uri, _, identity = auto_retrain_service.validate_baseline_job(
        config_path=config_path,
        metadata=auto_retrain_service.load_config_metadata(config_path),
        baseline_job_name="baseline-job",
        requested_uri=None,
    )

    assert uri == "azureml://jobs/baseline-job/outputs/drift_baseline/paths/"
    assert identity["baseline_execution_id"] == "execution-1"


@pytest.mark.parametrize("failure", ["missing", "wrong_type", "wrong_job", "unsafe_job"])
def test_baseline_job_reference_requires_matching_folder_output(monkeypatch, tmp_path, failure):
    config_path = _write_config(tmp_path / "configs")
    _patch_baseline_job_client(monkeypatch, config_path, tmp_path)
    client = auto_retrain_service.get_ml_client()
    job = client.jobs.get("baseline-job")
    job.outputs["drift_baseline"] = SimpleNamespace(path=None, type="uri_folder")
    name = "baseline-job"
    if failure == "missing":
        job.outputs.clear()
    elif failure == "wrong_type":
        job.outputs["drift_baseline"].type = "uri_file"
    elif failure == "wrong_job":
        job.name = "different-job"
    else:
        job.name = name = "baseline-job/other"

    with pytest.raises(ValueError, match="does not expose a reusable"):
        auto_retrain_service.validate_baseline_job(
            config_path=config_path,
            metadata=auto_retrain_service.load_config_metadata(config_path),
            baseline_job_name=name,
            requested_uri=None,
        )


@pytest.mark.parametrize("failure", ["empty_artifacts", "wrong_identity", "requested_uri"])
def test_baseline_job_reference_preserves_content_and_identity_gates(monkeypatch, tmp_path, failure):
    config_path = _write_config(tmp_path / "configs")
    _patch_baseline_job_client(monkeypatch, config_path, tmp_path)
    client = auto_retrain_service.get_ml_client()
    job = client.jobs.get("baseline-job")
    job.outputs["drift_baseline"] = SimpleNamespace(path=None, type="uri_folder")
    requested_uri = None
    if failure == "empty_artifacts":
        client.jobs.download = lambda **kwargs: None
        expected = "exactly one feature_baseline"
    elif failure == "wrong_identity":
        job.tags["execution_id"] = "different-execution"
        expected = "does not match its producing job"
    else:
        requested_uri = "azureml://jobs/other-job/outputs/drift_baseline/paths/"
        expected = "does not match the producing job output"

    with pytest.raises(ValueError, match=expected):
        auto_retrain_service.validate_baseline_job(
            config_path=config_path,
            metadata=auto_retrain_service.load_config_metadata(config_path),
            baseline_job_name="baseline-job",
            requested_uri=requested_uri,
        )


def test_auto_retrain_service_rejects_external_baseline_uri(monkeypatch, tmp_path) -> None:
    configs_dir = tmp_path / "configs"
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    _write_config(configs_dir)
    _patch_service_context(monkeypatch, configs_dir, ledger_root)

    with pytest.raises(ValueError, match="must be an azureml:// URI"):
        auto_retrain_service.approve_auto_retrain_baseline(
            AutoRetrainBaselineApprovalRequest(
                config_name="config_regression_college_azureml.yml",
                output_baseline_uri="https://example.invalid/baseline",
            )
        )


def test_request_ledger_path_rejects_absolute_path(monkeypatch, tmp_path) -> None:
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))

    with pytest.raises(ValueError, match="must be relative"):
        auto_retrain_service._resolve_ledger_path(
            str(tmp_path / "outside.jsonl")
        )


@pytest.mark.parametrize(
    "raw_path",
    (
        "../outside.json",
        "nested/../../outside.json",
        "nested/decision.txt",
    ),
)
def test_request_decision_path_rejects_escape_or_wrong_type(
    monkeypatch,
    tmp_path,
    raw_path,
) -> None:
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))

    with pytest.raises((ValueError, FileNotFoundError)):
        auto_retrain_service._resolve_decision_path(raw_path)


def test_request_decision_path_must_exist(monkeypatch, tmp_path) -> None:
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))

    with pytest.raises(FileNotFoundError, match="S14 retrain decision not found"):
        auto_retrain_service._resolve_decision_path("missing.json")


@pytest.mark.parametrize(
    "raw_path",
    (
        "../outside.jsonl",
        "nested/../../outside.jsonl",
        "nested/ledger.txt",
    ),
)
def test_request_ledger_path_rejects_escape_or_wrong_type(
    monkeypatch,
    tmp_path,
    raw_path,
) -> None:
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))

    with pytest.raises(ValueError):
        auto_retrain_service._resolve_ledger_path(raw_path)


def test_trusted_environment_selects_contained_default(
    monkeypatch,
    tmp_path,
) -> None:
    ledger_root = tmp_path / "trusted-ledgers"
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))
    monkeypatch.setenv(
        "MLOPS_AUTO_RETRAIN_LEDGER",
        "daily/auto_retrain_decisions.jsonl",
    )

    resolved = auto_retrain_service._resolve_ledger_path(None)

    assert resolved == (
        ledger_root / "daily" / "auto_retrain_decisions.jsonl"
    ).resolve()


def test_trusted_default_must_remain_inside_trusted_root(
    monkeypatch,
    tmp_path,
) -> None:
    ledger_root = tmp_path / "trusted-ledgers"
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))
    monkeypatch.setenv(
        "MLOPS_AUTO_RETRAIN_LEDGER",
        str(tmp_path / "outside.jsonl"),
    )

    with pytest.raises(ValueError, match="must resolve under"):
        auto_retrain_service._resolve_ledger_path(None)


def test_schedule_listing_reconciles_live_azure_state(monkeypatch, tmp_path) -> None:
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))
    live = SimpleNamespace(
        name="auto-retrain-regression-college-daily",
        is_enabled=True,
        provisioning_status="Succeeded",
    )
    client = SimpleNamespace(
        schedules=SimpleNamespace(list=lambda: [live]),
    )
    monkeypatch.setattr(auto_retrain_service, "get_ml_client", lambda: client)

    result = auto_retrain_service.list_auto_retrain_schedules()

    rows = {row.schedule_name: row for row in result.schedules}
    regression = rows["auto-retrain-regression-college-daily"]
    assert regression.live_state == "enabled"
    assert regression.actual_enabled is True
    assert regression.provisioning_status == "Succeeded"
    assert regression.source == "azure_ml"
    assert rows["auto-retrain-classification-telecom-churn-daily"].live_state == "missing"
    assert result.azure_error is None
    assert result.azure_checked_at


def test_schedule_listing_reports_unverified_when_azure_read_fails(
    monkeypatch,
    tmp_path,
) -> None:
    ledger_root = tmp_path / "ledgers"
    ledger_root.mkdir()
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER_ROOT", str(ledger_root))

    def fail_client():
        raise ConnectionError("schedule read unavailable")

    monkeypatch.setattr(auto_retrain_service, "get_ml_client", fail_client)

    result = auto_retrain_service.list_auto_retrain_schedules()

    assert {row.live_state for row in result.schedules} == {"unverified"}
    assert {row.source for row in result.schedules} == {"planned_only"}
    assert result.azure_error == "ConnectionError: schedule read unavailable"
