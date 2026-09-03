from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from api.services import submission_request_store as store


def _record(request_id: str = "req-012345abcdef") -> dict:
    return {
        "request_id": request_id,
        "status": "pending",
        "config_name": "config_classification",
        "submitted_at": "2026-08-02T12:00:00Z",
        "job_name": None,
    }


def test_request_record_survives_process_memory_loss(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))

    created = store.create_request_record(_record())
    loaded = store.get_request_record(created["request_id"])
    updated = store.update_request_record(
        created["request_id"],
        {"status": "submitted", "job_name": "azure-job-1"},
    )

    assert loaded == created
    assert updated["status"] == "submitted"
    assert store.get_request_record(created["request_id"])["job_name"] == "azure-job-1"
    on_disk = json.loads((tmp_path / "req-012345abcdef.json").read_text())
    assert on_disk == updated


def test_request_store_rejects_paths_and_duplicate_ids(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))
    store.create_request_record(_record())

    with pytest.raises(store.SubmissionRequestStoreError, match="already exists"):
        store.create_request_record(_record())
    with pytest.raises(store.SubmissionRequestStoreError, match="Invalid submission request ID"):
        store.get_request_record("../outside")


def test_request_store_serializes_concurrent_updates(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))
    store.create_request_record(_record())

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda index: store.update_request_record(
                    "req-012345abcdef",
                    {"attempt": index},
                ),
                range(12),
            )
        )

    loaded = store.get_request_record("req-012345abcdef")
    assert loaded is not None
    assert loaded["attempt"] in range(12)
    assert not list(tmp_path.glob("*.tmp"))


def test_request_store_retries_windows_permission_error_for_existing_lock(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))
    store.create_request_record(_record())
    real_open = store.os.open
    injected = False
    lock_path = tmp_path / ".submission-request-store.lock"

    def windows_open(path, flags, mode=0o777):
        nonlocal injected
        if not injected and Path(path).name == ".submission-request-store.lock":
            injected = True
            Path(path).write_text("{}", encoding="utf-8")
            raise PermissionError(13, "simulated Windows lock contention", path)
        return real_open(path, flags, mode)

    def release_contended_lock(_seconds):
        lock_path.unlink(missing_ok=True)

    monkeypatch.setattr(store.os, "open", windows_open)
    monkeypatch.setattr(store.time, "sleep", release_contended_lock)

    updated = store.update_request_record(
        "req-012345abcdef",
        {"status": "submitted", "job_name": "azure-job-2"},
    )

    assert injected is True
    assert updated["job_name"] == "azure-job-2"
    assert not lock_path.exists()


def test_request_store_retries_windows_permission_error_when_lock_disappears(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))
    store.create_request_record(_record())
    real_open = store.os.open
    injected = False

    def windows_open(path, flags, mode=0o777):
        nonlocal injected
        if not injected and Path(path).name == ".submission-request-store.lock":
            injected = True
            raise PermissionError(13, "simulated released Windows lock", path)
        return real_open(path, flags, mode)

    monkeypatch.setattr(store.os, "open", windows_open)

    updated = store.update_request_record(
        "req-012345abcdef",
        {"status": "submitted", "job_name": "azure-job-3"},
    )

    assert injected is True
    assert updated["job_name"] == "azure-job-3"


def test_request_store_surfaces_persistent_permission_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLOPS_SUBMISSION_REQUEST_ROOT", str(tmp_path))
    attempts = 0

    def denied_open(path, flags, mode=0o777):
        nonlocal attempts
        attempts += 1
        raise PermissionError(13, "simulated directory permission failure", path)

    monkeypatch.setattr(store.os, "open", denied_open)
    monkeypatch.setattr(store.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError, match="simulated directory permission failure"):
        store.create_request_record(_record())

    assert attempts == 20
