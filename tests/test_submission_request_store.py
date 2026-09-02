from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

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
