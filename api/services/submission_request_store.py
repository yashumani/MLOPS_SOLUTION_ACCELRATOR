"""Durable, server-owned state for asynchronous pipeline submissions."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4


_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUEST_ID = re.compile(r"^req-[0-9a-f]{12}$")
_VALID_STATUSES = {
    "pending",
    "submitted",
    "failed",
    "reconciliation_required",
}
_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_STALE_SECONDS = 15 * 60.0
_MAX_RECORD_BYTES = 1024 * 1024


class SubmissionRequestStoreError(RuntimeError):
    """Raised when durable request state is invalid or unavailable."""


def request_store_root() -> Path:
    configured = os.environ.get("MLOPS_SUBMISSION_REQUEST_ROOT")
    root = Path(configured).expanduser() if configured else _REPO_ROOT / "outputs" / "submission_requests"
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root.resolve()


def create_request_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_record(dict(record))
    path = _record_path(payload["request_id"])
    with _store_lock(path.parent):
        if path.exists():
            raise SubmissionRequestStoreError(
                f"Submission request already exists: {payload['request_id']}"
            )
        _write_atomic(path, payload)
    return dict(payload)


def update_request_record(
    request_id: str,
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    path = _record_path(request_id)
    with _store_lock(path.parent):
        current = _read_record(path)
        if current is None:
            raise SubmissionRequestStoreError(
                f"Unknown submission request: {request_id}"
            )
        current.update(dict(updates))
        payload = _validate_record(current)
        _write_atomic(path, payload)
    return dict(payload)


def get_request_record(request_id: str) -> dict[str, Any] | None:
    path = _record_path(request_id)
    with _store_lock(path.parent):
        record = _read_record(path)
    return dict(record) if record is not None else None


def _record_path(request_id: str) -> Path:
    normalized = str(request_id or "").strip()
    if not _REQUEST_ID.fullmatch(normalized):
        raise SubmissionRequestStoreError("Invalid submission request ID")
    return request_store_root() / f"{normalized}.json"


def _validate_record(record: dict[str, Any]) -> dict[str, Any]:
    request_id = str(record.get("request_id") or "").strip()
    if not _REQUEST_ID.fullmatch(request_id):
        raise SubmissionRequestStoreError("Invalid submission request ID")
    status = str(record.get("status") or "").strip().lower()
    if status not in _VALID_STATUSES:
        raise SubmissionRequestStoreError(
            f"Invalid submission request status: {status or '<missing>'}"
        )
    record["request_id"] = request_id
    record["status"] = status
    encoded = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    if len(encoded) > _MAX_RECORD_BYTES:
        raise SubmissionRequestStoreError("Submission request record is too large")
    return record


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionRequestStoreError(
            f"Invalid durable submission request record: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise SubmissionRequestStoreError(
            f"Invalid durable submission request record: {path}"
        )
    validated = _validate_record(payload)
    if path.stem != validated["request_id"]:
        raise SubmissionRequestStoreError(
            f"Submission request ID does not match record path: {path}"
        )
    return validated


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _store_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".submission-request-store.lock"
    token = uuid4().hex
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    missing_lock_permission_errors = 0
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"token": token, "pid": os.getpid(), "created_at": time.time()},
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            break
        except (FileExistsError, PermissionError) as exc:
            # Windows can report EACCES rather than EEXIST while another
            # process owns an O_EXCL lock file. A successful stat confirms
            # contention; a missing lock preserves genuine directory errors.
            try:
                lock_stat = lock_path.stat()
            except FileNotFoundError:
                if isinstance(exc, PermissionError):
                    missing_lock_permission_errors += 1
                    if missing_lock_permission_errors >= 20:
                        raise exc
                    time.sleep(0.01)
                continue
            except PermissionError:
                lock_stat = None
            missing_lock_permission_errors = 0
            if lock_stat is not None and time.time() - lock_stat.st_mtime > _LOCK_STALE_SECONDS:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    continue
                except PermissionError:
                    pass
                else:
                    continue
            if time.monotonic() >= deadline:
                raise SubmissionRequestStoreError(
                    f"Timed out waiting for submission request store lock: {lock_path}"
                ) from exc
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                lock_path.unlink()
        except FileNotFoundError:
            pass
