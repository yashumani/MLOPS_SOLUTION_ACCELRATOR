"""Append-only decision ledger helpers for auto-retrain orchestration."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from . import operational_state


APPROVED_BASELINE_STATUSES = {"approved", "production", "baseline_approved"}
REQUIRED_LEDGER_FIELDS = (
    "decision_id",
    "timestamp_utc",
    "config_name",
    "task_type",
    "dataset_name",
    "outcome",
    "promotion_status",
)
LEDGER_LOCK_TIMEOUT_SECONDS = 10.0
LEDGER_LOCK_STALE_SECONDS = 15 * 60.0
UNRESOLVED_CANDIDATE_STATUSES = {
    "reconciliation_required",
    "submitting",
    "manual_pending",
    "submitted",
    "candidate_submitted",
    "running",
    "queued",
    "in_progress",
}
SUBMISSION_ELIGIBLE_OUTCOMES = {"candidate_retrain", "promote_candidate"}


class DecisionReservationConflict(RuntimeError):
    """Raised when another unresolved candidate owns the same baseline key."""


@dataclass(frozen=True)
class AutoRetrainDecisionRecord:
    """One auditable auto-retrain controller decision."""

    config_name: str
    task_type: str
    dataset_name: str
    outcome: str
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trigger: str = "manual"
    schedule_name: str | None = None
    decision_mode: str = "candidate_retrain"
    promotion_mode: str = "manual"
    promotion_status: str = "manual_pending"
    approved_for_future_baseline: bool = False
    input_baseline_uri: str | None = None
    output_baseline_uri: str | None = None
    baseline_job_name: str | None = None
    candidate_job_name: str | None = None
    severity: str = "none"
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    azure_context: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_decision_record(
    *,
    config_name: str,
    task_type: str,
    dataset_name: str,
    decision: dict[str, Any],
    trigger: str = "manual",
    schedule_name: str | None = None,
    input_baseline_uri: str | None = None,
    output_baseline_uri: str | None = None,
    baseline_job_name: str | None = None,
    candidate_job_name: str | None = None,
    promotion_status: str = "manual_pending",
    approved_for_future_baseline: bool = False,
    azure_context: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AutoRetrainDecisionRecord:
    """Create a ledger record from a policy decision dictionary."""
    return AutoRetrainDecisionRecord(
        config_name=config_name,
        task_type=task_type,
        dataset_name=dataset_name,
        outcome=str(decision.get("outcome") or "observe_only"),
        trigger=trigger,
        schedule_name=schedule_name,
        promotion_status=promotion_status,
        approved_for_future_baseline=approved_for_future_baseline,
        input_baseline_uri=input_baseline_uri,
        output_baseline_uri=output_baseline_uri,
        baseline_job_name=baseline_job_name,
        candidate_job_name=candidate_job_name,
        severity=str(decision.get("severity") or "none"),
        reasons=list(decision.get("reasons") or []),
        signals=dict(decision.get("signals") or {}),
        azure_context=dict(azure_context or {}),
        metadata=dict(metadata or {}),
    )


def append_decision_record(
    ledger_path: str | Path,
    record: AutoRetrainDecisionRecord | dict[str, Any],
) -> Path:
    """Append one decision record to a JSONL ledger."""
    payload = record.as_dict() if isinstance(record, AutoRetrainDecisionRecord) else dict(record)
    path = Path(ledger_path)
    payload = validate_decision_record(payload, source=str(path))
    if operational_state.database_path() is not None:
        with operational_state.transaction() as connection:
            namespace = _sqlite_ledger_namespace(connection, path)
            operational_state.append_event(connection, namespace, payload)
        return path
    with _ledger_lock(path):
        _append_decision_record_unlocked(path, payload)
    return path


def reserve_candidate_submission(
    ledger_path: str | Path,
    record: AutoRetrainDecisionRecord | dict[str, Any],
) -> Path:
    """Atomically reject duplicates and reserve a candidate submission."""

    path = Path(ledger_path)
    payload = record.as_dict() if isinstance(record, AutoRetrainDecisionRecord) else dict(record)
    payload = validate_decision_record(payload, source=str(path))
    if str(payload.get("promotion_status") or "").lower() != "submitting":
        raise ValueError("Candidate reservation must use promotion_status='submitting'")
    if operational_state.database_path() is not None:
        with operational_state.transaction() as connection:
            namespace = _sqlite_ledger_namespace(connection, path)
            records = operational_state.load_events(connection, namespace)
            _reject_duplicate_reservation(records, payload)
            operational_state.append_event(connection, namespace, payload)
        return path
    with _ledger_lock(path):
        records = _load_decision_records_unlocked(path)
        _reject_duplicate_reservation(records, payload)
        _append_decision_record_unlocked(path, payload)
    return path


def load_decision_records(ledger_path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL auto-retrain decision ledger."""
    path = Path(ledger_path)
    if operational_state.database_path() is not None:
        with operational_state.transaction() as connection:
            namespace = _sqlite_ledger_namespace(connection, path)
            return [validate_decision_record(value) for value in operational_state.load_events(connection, namespace)]
    with _ledger_lock(path):
        return _load_decision_records_unlocked(path)


def _sqlite_ledger_namespace(connection: Any, path: Path) -> str:
    namespace = "retrain_ledger:" + str(path.resolve())
    operational_state.require_legacy_import(connection, namespace, path.exists())
    return namespace


def latest_decision_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record["decision_id"])] = record
    return list(latest.values())


def _reject_duplicate_reservation(records: Iterable[dict[str, Any]], payload: dict[str, Any]) -> None:
    records = list(records)
    if payload.get("trigger") == "automated_controller":
        approved = latest_approved_baseline_uri(
            records, config_name=payload["config_name"],
            task_type=payload["task_type"], dataset_name=payload["dataset_name"],
        )
        if approved != payload.get("input_baseline_uri"):
            raise DecisionReservationConflict("Approved baseline changed before reservation")
    for existing in latest_decision_records(records):
        source_decision = (payload.get("metadata") or {}).get("source_s14_decision_id")
        same_source_decision = bool(source_decision) and source_decision == (
            existing.get("metadata") or {}
        ).get("source_s14_decision_id")
        if existing.get("decision_id") == payload["decision_id"] or same_source_decision or _is_unresolved_duplicate(existing, payload):
            raise DecisionReservationConflict(
                "Candidate submission is already reserved for "
                f"config={payload['config_name']!r}, dataset={payload['dataset_name']!r}, "
                f"baseline={payload.get('input_baseline_uri')!r}; "
                f"decision_id={existing.get('decision_id')!r}"
            )


def _load_decision_records_unlocked(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}") from exc
            records.append(validate_decision_record(payload, source=f"{path}:{line_number}"))
    return records


def _append_decision_record_unlocked(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _ledger_lock(path: Path):
    """Use an atomic sidecar lock with bounded stale-lock recovery."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    token = str(uuid4())
    deadline = time.monotonic() + LEDGER_LOCK_TIMEOUT_SECONDS
    missing_lock_permission_errors = 0
    while True:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "token": token,
                            "pid": os.getpid(),
                            "created_at": time.time(),
                        },
                        sort_keys=True,
                    )
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
            if (
                lock_stat is not None
                and time.time() - lock_stat.st_mtime > LEDGER_LOCK_STALE_SECONDS
            ):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    continue
                except PermissionError:
                    pass
                else:
                    continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for decision ledger lock: {lock_path}"
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


def _is_unresolved_duplicate(
    existing: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    if existing.get("decision_id") == candidate.get("decision_id"):
        return False
    return (
        existing.get("config_name") == candidate.get("config_name")
        and existing.get("task_type") == candidate.get("task_type")
        and existing.get("dataset_name") == candidate.get("dataset_name")
        and existing.get("input_baseline_uri") == candidate.get("input_baseline_uri")
        and str(existing.get("outcome") or "").lower()
        in SUBMISSION_ELIGIBLE_OUTCOMES
        and str(existing.get("promotion_status") or "").lower()
        in UNRESOLVED_CANDIDATE_STATUSES
    )


def validate_decision_record(record: dict[str, Any], *, source: str = "record") -> dict[str, Any]:
    """Validate and normalize one auto-retrain ledger record."""
    if not isinstance(record, dict):
        raise ValueError(f"Invalid auto-retrain ledger record at {source}: expected object")

    missing = [field for field in REQUIRED_LEDGER_FIELDS if _blank(record.get(field))]
    if missing:
        raise ValueError(
            f"Invalid auto-retrain ledger record at {source}: missing required field(s): "
            + ", ".join(missing)
        )

    normalized = dict(record)
    normalized.setdefault("decision_mode", "candidate_retrain")
    normalized.setdefault("promotion_mode", "manual")
    normalized.setdefault("approved_for_future_baseline", False)
    normalized.setdefault("input_baseline_uri", None)
    normalized.setdefault("output_baseline_uri", None)
    normalized.setdefault("baseline_job_name", None)
    normalized.setdefault("candidate_job_name", None)
    normalized.setdefault("schedule_name", None)
    normalized.setdefault("severity", "none")
    normalized["reasons"] = _coerce_list(normalized.get("reasons"))
    normalized["signals"] = _coerce_dict(normalized.get("signals"), field_name="signals", source=source)
    normalized["azure_context"] = _coerce_dict(
        normalized.get("azure_context"), field_name="azure_context", source=source
    )
    normalized["metadata"] = _coerce_dict(normalized.get("metadata"), field_name="metadata", source=source)

    if _is_approved_baseline(normalized) and _blank(normalized.get("output_baseline_uri")):
        raise ValueError(
            f"Invalid auto-retrain ledger record at {source}: approved baseline records require output_baseline_uri"
        )
    return normalized


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_dict(value: Any, *, field_name: str, source: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(f"Invalid auto-retrain ledger record at {source}: {field_name} must be an object")


def latest_approved_baseline_uri(
    records: Iterable[dict[str, Any]],
    *,
    config_name: str | None = None,
    task_type: str | None = None,
    dataset_name: str | None = None,
) -> str | None:
    """Return the newest approved output baseline URI matching the filters."""
    candidates = [record for record in records if _matches(record, config_name, task_type, dataset_name)]
    candidates.sort(key=lambda record: str(record.get("timestamp_utc") or ""), reverse=True)
    for record in candidates:
        if _is_approved_baseline(record):
            uri = record.get("output_baseline_uri")
            if uri:
                return str(uri)
    return None


def _matches(
    record: dict[str, Any],
    config_name: str | None,
    task_type: str | None,
    dataset_name: str | None,
) -> bool:
    if config_name and record.get("config_name") != config_name:
        return False
    if task_type and record.get("task_type") != task_type:
        return False
    if dataset_name and record.get("dataset_name") != dataset_name:
        return False
    return True


def _is_approved_baseline(record: dict[str, Any]) -> bool:
    if bool(record.get("approved_for_future_baseline")):
        return True
    return str(record.get("promotion_status") or "").lower() in APPROVED_BASELINE_STATUSES
