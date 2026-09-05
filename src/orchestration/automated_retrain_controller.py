"""Bounded Azure S14 discovery and fail-closed automated candidate submission."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import operational_state as state
from .auto_retrain_controller import (
    REPO_ROOT, AutoRetrainControllerError, AutoRetrainControllerRequest,
    AzureSubmissionContext, build_controller_plan, build_pending_decision_record,
)
from .auto_retrain_decision_ledger import (
    DecisionReservationConflict, append_decision_record, reserve_candidate_submission,
)


class ControllerReconciliationRequired(RuntimeError):
    """A submission may have succeeded; automatic retries are forbidden."""


@dataclass(frozen=True)
class WatchTarget:
    config_path: Path
    experiment_name: str


def utc_timestamp(value: Any) -> datetime:
    stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Decision and run timestamps must include a timezone")
    return stamp.astimezone(timezone.utc)


def discover_completed_runs(client, context: AzureSubmissionContext, experiment: str, *, now: datetime, max_age_seconds: int, max_runs: int = 200) -> list[str]:
    """Use the existing Run History SDK adapter; never truncate silently."""
    from azure.ai.ml._restclient.runhistory.models import QueryParams
    from azure.core.exceptions import ResourceNotFoundError

    operation = client.jobs._runs_operations._operation
    continuation = None
    seen_tokens: set[str] = set()
    names: list[str] = []
    scanned = 0
    previous_end: datetime | None = None
    cutoff = now - timedelta(seconds=max_age_seconds)

    def consume(runs) -> bool:
        nonlocal scanned, previous_end
        if runs is None:
            raise RuntimeError("Run History returned no run collection")
        for run in runs:
            ended = utc_timestamp(run.end_time_utc)
            if previous_end is not None and ended > previous_end:
                raise RuntimeError("Run History ordering could not be verified")
            previous_end = ended
            if ended < cutoff:
                return True
            scanned += 1
            if scanned > max_runs:
                raise RuntimeError("Recent-run scan limit exceeded; no submissions permitted")
            if not run.parent_run_id:
                if not run.run_id or run.status != "Completed":
                    raise RuntimeError("Invalid completed parent in Run History")
                names.append(run.run_id)
        return False

    def query(token=None):
        return operation.get_by_query_by_experiment_name(
            context.subscription_id, context.resource_group, context.workspace_name, experiment,
            body=QueryParams(filter="status eq 'Completed'", order_by="endTimeUtc desc", top=min(100, max_runs), continuation_token=token),
            connection_timeout=10, read_timeout=30,
        )

    try:
        response = query()
        if not hasattr(response, "value") and hasattr(response, "by_page"):
            pages = response.by_page()
            if not hasattr(pages, "continuation_token"):
                raise RuntimeError("Run History page iterator has no continuation token")
            while True:
                try:
                    page = next(pages)
                except StopIteration:
                    return names
                if consume(page):
                    return names
                continuation = pages.continuation_token
                if not continuation:
                    return names
                if continuation in seen_tokens:
                    raise RuntimeError("Run History repeated a continuation token")
                seen_tokens.add(continuation)
                if len(seen_tokens) >= 10:
                    raise RuntimeError("Run History page limit exceeded")

        while True:
            if consume(response.value):
                return names
            continuation = response.continuation_token
            if not continuation:
                return names
            if continuation in seen_tokens:
                raise RuntimeError("Run History repeated a continuation token")
            seen_tokens.add(continuation)
            if len(seen_tokens) >= 10:
                raise RuntimeError("Run History page limit exceeded")
            response = query(continuation)
    except ResourceNotFoundError as exc:
        missing = f"experiment {experiment} not found in workspace {context.workspace_name}".casefold()
        if getattr(exc, "status_code", None) == 404 and missing in str(exc).casefold():
            return []
        raise


def _observe(job_name: str, config_name: str, status: str, **details) -> dict:
    record = {"source_job_name": job_name, "config_name": config_name, "status": status, "checked_at": datetime.now(timezone.utc).isoformat(), **details}
    with state.transaction() as connection:
        state.append_event(connection, "controller_audit", record)
    return record


def validate_source_job(job, target: WatchTarget, payload: dict, *, now: datetime, max_age_seconds: int) -> None:
    if job.status != "Completed" or job.experiment_name != target.experiment_name or str(job.type).lower() != "pipeline":
        raise AutoRetrainControllerError("Only completed pipeline parents in the configured experiment are eligible")
    if "retrain_decision" not in (job.outputs or {}):
        raise AutoRetrainControllerError("Parent job has no named S14 output")
    tags = job.tags or {}
    if str(tags.get("config_name", "")).removesuffix(".yml").removesuffix(".yaml") != target.config_path.stem:
        raise AutoRetrainControllerError("Parent config identity does not match the configured target")
    revision = payload.get("source_revision") or {}
    for field, tag in (("execution_id", "execution_id"), ("config_hash", "compiled_config_hash"), ("source_sha", "source_identity")):
        if not revision.get(field) or revision[field] != tags.get(tag):
            raise AutoRetrainControllerError(f"S14 does not belong to this parent: {field}")
    if not (payload.get("comparison") or {}).get("input_baseline_uri"):
        raise AutoRetrainControllerError("Automatic submission requires an explicit compared baseline")
    try:
        stamp = utc_timestamp(payload.get("timestamp_utc"))
        created = utc_timestamp(job.creation_context.created_at)
    except (ValueError, AttributeError, TypeError) as exc:
        raise AutoRetrainControllerError("Missing or invalid source timestamps") from exc
    if not -60 <= (now - stamp).total_seconds() <= max_age_seconds or stamp < created - timedelta(seconds=60):
        raise AutoRetrainControllerError("S14 decision is stale, future-dated, or predates its source job")


def process_source_job(client, target: WatchTarget, job_name: str, *, context: AzureSubmissionContext, ledger: Path, execute: bool = False, max_age_seconds: int = 86400, timeout_seconds: int = 3600, now: datetime | None = None, credential_mode: str | None = None) -> dict:
    """Evaluate one downloaded S14; the ledger serializes competing controllers."""
    now = now or datetime.now(timezone.utc)
    job = client.jobs.get(job_name)
    if job.status != "Completed" or "retrain_decision" not in (job.outputs or {}):
        return _observe(job_name, target.config_path.stem, "not_eligible")
    with tempfile.TemporaryDirectory(prefix="mlops-controller-") as directory:
        folder = Path(directory)
        client.jobs.download(name=job_name, output_name="retrain_decision", download_path=str(folder))
        # Azure named uri_file outputs may lose the original filename extension.
        files = [path for path in folder.rglob("*")
                 if path.name in {"retrain_decision", "retrain_decision.json"} and path.is_file()]
        if (len(files) != 1 or files[0].is_symlink()
                or not files[0].resolve().is_relative_to(folder.resolve())
                or files[0].stat().st_size > 2_000_000):
            raise AutoRetrainControllerError("Expected one bounded retrain_decision artifact")
        try:
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("S14 must be an object")
            validate_source_job(job, target, payload, now=now, max_age_seconds=max_age_seconds)
            request = AutoRetrainControllerRequest(
                config_path=target.config_path, ledger_path=ledger, decision_path=files[0],
                azure_context=context, mode="submit" if execute else "dry_run",
                trigger="automated_controller", experiment_name=target.experiment_name,
            )
            plan = build_controller_plan(request)
        except (AutoRetrainControllerError, ValueError) as exc:
            return _observe(job_name, target.config_path.stem, "blocked", reason=str(exc))
        if not execute:
            return _observe(job_name, target.config_path.stem, "eligible_dry_run", source_decision_id=payload["decision_id"])

        reservation = build_pending_decision_record(plan, promotion_status="submitting")
        reservation = replace(reservation, metadata={**reservation.metadata, "source_job_name": job_name})
        try:
            reserve_candidate_submission(ledger, reservation)
        except DecisionReservationConflict as exc:
            return _observe(job_name, target.config_path.stem, "duplicate_blocked", reason=str(exc))
        result_path = folder / "submission-result.json"
        command = [*plan.command, "--result_json", str(result_path)]
        try:
            child_environment = None
            if credential_mode:
                child_environment = dict(os.environ)
                child_environment["MLOPS_AZURE_CREDENTIAL_MODE"] = credential_mode
            result = subprocess.run(command, cwd=str(REPO_ROOT), env=child_environment, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, shell=False, timeout=timeout_seconds)
            if result.returncode != 0:
                raise RuntimeError(f"Canonical submitter returned {result.returncode}")
            submitted = json.loads(result_path.read_text(encoding="utf-8"))
            name = submitted.get("job_name")
            if not name or submitted.get("experiment_name") != plan.experiment_name or submitted.get("display_name") != plan.display_name:
                raise RuntimeError("Canonical result is missing or inconsistent")
            candidate = client.jobs.get(name)
            if (candidate.tags or {}).get("source_decision_id") != payload["decision_id"]:
                raise RuntimeError("Submitted candidate identity could not be verified")
            append_decision_record(ledger, replace(reservation, candidate_job_name=name, promotion_status="manual_pending"))
            return _observe(job_name, target.config_path.stem, "submitted", candidate_job_name=name, source_decision_id=payload["decision_id"])
        except BaseException as exc:
            append_decision_record(ledger, replace(reservation, promotion_status="reconciliation_required", metadata={**reservation.metadata, "error_type": type(exc).__name__}))
            _observe(job_name, target.config_path.stem, "reconciliation_required", decision_id=reservation.decision_id, error_type=type(exc).__name__)
            raise ControllerReconciliationRequired("Submission outcome uncertain; inspect Azure jobs and the ledger before any retry") from exc
