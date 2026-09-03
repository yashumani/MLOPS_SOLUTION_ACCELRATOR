from __future__ import annotations

import json
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from api.services import submission_request_store as requests
from orchestration import operational_state as state
from orchestration.auto_retrain_decision_ledger import (
    AutoRetrainDecisionRecord,
    DecisionReservationConflict,
    append_decision_record,
    load_decision_records,
    reserve_candidate_submission,
)
from scripts.migrate_operational_state import migrate_state


@pytest.fixture
def database(monkeypatch, tmp_path):
    path = tmp_path / "state.sqlite3"
    monkeypatch.setenv("MLOPS_OPERATIONAL_STATE_DB", str(path))
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path / "requests"))
    return path


def test_transaction_rolls_back_on_failure(database):
    with pytest.raises(RuntimeError):
        with state.transaction() as connection:
            state.put_document(connection, "test", "one", {"value": 1})
            raise RuntimeError("abort")
    with state.transaction() as connection:
        assert state.get_document(connection, "test", "one") is None


def test_request_updates_preserve_concurrent_fields(database):
    request_id = "req-012345abcdef"
    requests.create_request_record({"request_id": request_id, "status": "pending"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: requests.update_request_record(request_id, {f"field_{index}": index}), range(20)))
    record = requests.get_request_record(request_id)
    assert all(record[f"field_{index}"] == index for index in range(20))
    with pytest.raises(requests.SubmissionRequestStoreError, match="already exists"):
        requests.create_request_record({"request_id": request_id, "status": "pending"})
    with pytest.raises(requests.SubmissionRequestStoreError, match="identity"):
        requests.update_request_record(request_id, {"request_id": "req-fedcba987654"})


def _reservation(**values):
    return AutoRetrainDecisionRecord(
        config_name="config_classification", task_type="classification", dataset_name="test",
        outcome="candidate_retrain", promotion_status="submitting", input_baseline_uri="azureml://baseline",
        **values,
    )


def test_only_one_concurrent_reservation_is_accepted(database, tmp_path):
    path = tmp_path / "ledger.jsonl"

    def reserve(index):
        try:
            reserve_candidate_submission(path, _reservation(decision_id=f"decision-{index}"))
            return True
        except DecisionReservationConflict:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(12))) == 1
    assert len(load_decision_records(path)) == 1


@pytest.mark.parametrize("sqlite_enabled", [True, False])
def test_terminal_reservation_does_not_block_new_decision(database, tmp_path, monkeypatch, sqlite_enabled):
    if not sqlite_enabled:
        monkeypatch.delenv("MLOPS_OPERATIONAL_STATE_DB")
    path = tmp_path / "ledger.jsonl"
    first = _reservation()
    reserve_candidate_submission(path, first)
    append_decision_record(path, replace(first, promotion_status="submission_failed"))
    reserve_candidate_submission(path, _reservation())
    assert len(load_decision_records(path)) == 3


def test_replaying_same_source_decision_is_refused(database, tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = _reservation(metadata={"source_s14_decision_id": "source-decision"})
    reserve_candidate_submission(path, first)
    append_decision_record(path, replace(first, promotion_status="submission_failed"))
    with pytest.raises(DecisionReservationConflict):
        reserve_candidate_submission(path, _reservation(metadata=first.metadata))
    with pytest.raises(DecisionReservationConflict):
        reserve_candidate_submission(path, first)


def test_legacy_request_migration_is_explicit_and_idempotent(database, tmp_path):
    root = tmp_path / "requests"
    root.mkdir()
    payload = {"request_id": "req-012345abcdef", "status": "pending"}
    (root / "req-012345abcdef.json").write_text(json.dumps(payload))
    with pytest.raises(state.OperationalStateError, match="import"):
        requests.get_request_record(payload["request_id"])
    assert migrate_state(database, root, [], apply=False)["applied"] is False
    assert migrate_state(database, root, [], apply=True)["counts"] == {"submission_requests": 1}
    assert migrate_state(database, root, [], apply=True)["applied"] is True
    assert requests.get_request_record(payload["request_id"]) == payload


def test_ledger_import_preserves_history_and_rejects_changed_source(database, tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = _reservation().as_dict()
    path.write_text(json.dumps(first) + "\n")
    migrate_state(database, None, [path], apply=True)
    migrate_state(database, None, [path], apply=True)
    assert load_decision_records(path) == [first]
    path.write_text(json.dumps(_reservation().as_dict()) + "\n")
    with pytest.raises(state.OperationalStateError, match="changed"):
        migrate_state(database, None, [path], apply=True)


def test_relative_or_unc_database_paths_are_rejected(monkeypatch):
    for path in ("state.sqlite3", "//server/share/state.sqlite3"):
        monkeypatch.setenv("MLOPS_OPERATIONAL_STATE_DB", path)
        with pytest.raises(state.OperationalStateError, match="local-disk"):
            state.database_path()


def _cold_process_writer(path, barrier, index):
    from pathlib import Path
    barrier.wait(timeout=30)
    with state.transaction(path=Path(path)) as connection:
        state.put_document(connection, "cold_start", str(index), {"index": index})


def test_cross_process_cold_database_initialization(database):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    processes = [context.Process(target=_cold_process_writer, args=(str(database), barrier, index)) for index in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=40)
            assert process.exitcode == 0
        with state.transaction() as connection:
            assert connection.execute("SELECT COUNT(*) FROM documents WHERE namespace='cold_start'").fetchone()[0] == 4
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_wal_initialization_retries_busy_but_not_other_errors(monkeypatch):
    calls = []
    class Connection:
        def execute(self, statement):
            calls.append(statement)
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            return self
        def fetchone(self):
            return ("wal",)
    monkeypatch.setattr(state.time, "sleep", lambda delay: None)
    state._enable_wal(Connection())
    assert len(calls) == 2
    class Broken:
        def execute(self, statement):
            raise sqlite3.OperationalError("disk I/O error")
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        state._enable_wal(Broken())
