from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest
from azure.core.exceptions import ResourceNotFoundError

from orchestration import automated_retrain_controller as controller
from orchestration.auto_retrain_controller import AzureSubmissionContext
from orchestration.auto_retrain_decision_ledger import append_decision_record, load_decision_records, latest_decision_records
from test_orchestration.test_auto_retrain_controller import _write_config, _write_s14_decision


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    monkeypatch.setenv("MLOPS_OPERATIONAL_STATE_DB", str(tmp_path / "state.sqlite3"))
    config = tmp_path / "config_regression_college_azureml.yml"
    decision = tmp_path / "source.json"
    _write_config(config)
    _write_s14_decision(decision)
    payload = json.loads(decision.read_text())
    now = datetime.now(timezone.utc)
    payload["timestamp_utc"] = now.isoformat()
    ledger = tmp_path / "ledger.jsonl"
    append_decision_record(ledger, {
        "decision_id": "baseline", "timestamp_utc": now.isoformat(),
        "config_name": config.stem, "task_type": "regression", "dataset_name": "college",
        "outcome": "approve_baseline", "promotion_status": "approved", "output_baseline_uri": "azureml://baseline/",
    })
    job = SimpleNamespace(name="parent", type="pipeline", status="Completed", experiment_name="watched", outputs={"retrain_decision": object()}, creation_context=SimpleNamespace(created_at=now-timedelta(hours=1)), tags={"config_name": config.stem, "execution_id": "source-execution", "compiled_config_hash": "config-sha", "source_identity": "source-sha"})

    class Jobs:
        def get(self, name):
            return job if name == "parent" else SimpleNamespace(tags={"source_decision_id": payload["decision_id"]})

        def download(self, *, name, output_name, download_path):
            assert name == "parent" and output_name == "retrain_decision"
            (Path(download_path) / "retrain_decision.json").write_text(json.dumps(payload))

    return SimpleNamespace(client=SimpleNamespace(jobs=Jobs()), target=controller.WatchTarget(config, "watched"), ledger=ledger, payload=payload, job=job, now=now, context=AzureSubmissionContext("sub", "rg", "ws", "cluster"))


def _process(scenario, **kwargs):
    return controller.process_source_job(scenario.client, scenario.target, "parent", context=scenario.context, ledger=scenario.ledger, now=scenario.now, **kwargs)


def _successful_submit(command, **kwargs):
    assert kwargs["shell"] is False
    assert "--force" not in command and "--skip-active-check" not in command
    assert "--expected_execution_id" in command and "--source_decision_id" in command
    Path(command[command.index("--result_json")+1]).write_text(json.dumps({
        "job_name": "candidate", "experiment_name": command[command.index("--experiment_name")+1],
        "display_name": command[command.index("--display_name")+1],
    }))
    return SimpleNamespace(returncode=0)


def test_dry_run_has_no_reservation_or_submission(scenario, monkeypatch):
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry run submitted"))
    assert _process(scenario)["status"] == "eligible_dry_run"
    assert len(load_decision_records(scenario.ledger)) == 1


@pytest.mark.parametrize("change", ["policy", "stale", "future", "identity", "config", "baseline", "naive", "precreation"])
def test_invalid_evidence_never_submits(scenario, monkeypatch, change):
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid decision submitted"))
    if change == "policy":
        scenario.payload["decision"]["should_submit"] = False
        scenario.payload["retrain_decision"]["should_submit"] = False
    elif change in {"stale", "future", "naive", "precreation"}:
        offsets = {"stale": -90000, "future": 120, "naive": 0, "precreation": -7200}
        stamp = scenario.now + timedelta(seconds=offsets[change])
        scenario.payload["timestamp_utc"] = (stamp.replace(tzinfo=None) if change == "naive" else stamp).isoformat()
    elif change == "identity":
        scenario.job.tags["source_identity"] = "another-source"
    elif change == "config":
        scenario.job.tags["config_name"] = "another-config"
    else:
        scenario.payload["comparison"]["input_baseline_uri"] = None
    assert _process(scenario, execute=True)["status"] == "blocked"
    assert len(load_decision_records(scenario.ledger)) == 1


def test_competing_controllers_submit_once_and_never_promote(scenario, monkeypatch):
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        return _successful_submit(command, **kwargs)
    monkeypatch.setattr(controller.subprocess, "run", submit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _process(scenario, execute=True), range(2)))
    assert len(calls) == 1
    assert sum(result["status"] == "submitted" for result in results) == 1
    latest = latest_decision_records(load_decision_records(scenario.ledger))[-1]
    assert latest["promotion_status"] == "manual_pending"
    assert latest["approved_for_future_baseline"] is False


def test_controller_propagates_explicit_credential_mode_to_submitter(
    scenario,
    monkeypatch,
):
    captured = {}

    def submit(command, **kwargs):
        captured["credential_mode"] = kwargs["env"][
            "MLOPS_AZURE_CREDENTIAL_MODE"
        ]
        return _successful_submit(command, **kwargs)

    monkeypatch.setattr(controller.subprocess, "run", submit)

    result = _process(
        scenario,
        execute=True,
        credential_mode="azureml_obo",
    )

    assert result["status"] == "submitted"
    assert captured["credential_mode"] == "azureml_obo"


@pytest.mark.parametrize("failure", ["timeout", "exit", "missing_result", "identity"])
def test_ambiguous_submission_requires_reconciliation_and_blocks_replay(scenario, monkeypatch, failure):
    def submit(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure in {"exit", "missing_result"}:
            return SimpleNamespace(returncode=1 if failure == "exit" else 0)
        return _successful_submit(command, **kwargs)
    if failure == "identity":
        original = scenario.client.jobs.get
        scenario.client.jobs.get = lambda name: original(name) if name == "parent" else SimpleNamespace(tags={})
    monkeypatch.setattr(controller.subprocess, "run", submit)
    with pytest.raises(controller.ControllerReconciliationRequired):
        _process(scenario, execute=True)
    assert latest_decision_records(load_decision_records(scenario.ledger))[-1]["promotion_status"] == "reconciliation_required"
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: pytest.fail("uncertain job retried"))
    assert _process(scenario, execute=True)["status"] == "blocked"


def test_discovery_is_ordered_bounded_and_experiment_scoped(scenario):
    calls = []
    def query(*args, body, **kwargs):
        calls.append((args, body))
        return SimpleNamespace(value=[
            SimpleNamespace(run_id="parent", status="Completed", parent_run_id=None, end_time_utc=scenario.now),
            SimpleNamespace(run_id="old", status="Completed", parent_run_id=None, end_time_utc=scenario.now-timedelta(days=2)),
        ], continuation_token="unused")
    scenario.client.jobs._runs_operations = SimpleNamespace(_operation=SimpleNamespace(get_by_query_by_experiment_name=query))
    assert controller.discover_completed_runs(scenario.client, scenario.context, "watched", now=scenario.now, max_age_seconds=86400) == ["parent"]
    assert calls[0][0] == ("sub", "rg", "ws", "watched")
    assert calls[0][1].order_by == "endTimeUtc desc"


def test_discovery_supports_item_paged_responses(scenario):
    class Pages:
        def __init__(self):
            self.continuation_token = None
            self._pages = iter([
                ([SimpleNamespace(run_id="parent", status="Completed", parent_run_id=None, end_time_utc=scenario.now)], "next"),
                ([SimpleNamespace(run_id="child", status="Completed", parent_run_id="parent", end_time_utc=scenario.now-timedelta(minutes=1))], None),
            ])

        def __iter__(self):
            return self

        def __next__(self):
            page, self.continuation_token = next(self._pages)
            return iter(page)

    class Paged:
        def by_page(self):
            return Pages()

    query = lambda *args, **kwargs: Paged()
    scenario.client.jobs._runs_operations = SimpleNamespace(_operation=SimpleNamespace(get_by_query_by_experiment_name=query))
    assert controller.discover_completed_runs(
        scenario.client, scenario.context, "watched", now=scenario.now,
        max_age_seconds=86400,
    ) == ["parent"]


def test_discovery_treats_only_missing_experiment_as_empty(scenario):
    missing = ResourceNotFoundError("Experiment watched not found in workspace ws")
    missing.status_code = 404
    scenario.client.jobs._runs_operations = SimpleNamespace(
        _operation=SimpleNamespace(get_by_query_by_experiment_name=lambda *args, **kwargs: (_ for _ in ()).throw(missing))
    )
    assert controller.discover_completed_runs(
        scenario.client, scenario.context, "watched", now=scenario.now,
        max_age_seconds=86400,
    ) == []

    unrelated = ResourceNotFoundError("Workspace ws not found")
    unrelated.status_code = 404
    scenario.client.jobs._runs_operations = SimpleNamespace(
        _operation=SimpleNamespace(get_by_query_by_experiment_name=lambda *args, **kwargs: (_ for _ in ()).throw(unrelated))
    )
    with pytest.raises(ResourceNotFoundError, match="Workspace ws not found"):
        controller.discover_completed_runs(
            scenario.client, scenario.context, "watched", now=scenario.now,
            max_age_seconds=86400,
        )


def test_scan_limit_never_silently_truncates(scenario):
    runs = [SimpleNamespace(run_id=str(index), status="Completed", parent_run_id=None, end_time_utc=scenario.now) for index in range(3)]
    query = lambda *args, **kwargs: SimpleNamespace(value=runs, continuation_token=None)
    scenario.client.jobs._runs_operations = SimpleNamespace(_operation=SimpleNamespace(get_by_query_by_experiment_name=query))
    with pytest.raises(RuntimeError, match="scan limit"):
        controller.discover_completed_runs(scenario.client, scenario.context, "watched", now=scenario.now, max_age_seconds=86400, max_runs=2)
