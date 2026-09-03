"""Controller planning utilities for auto-retrain submissions."""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from orchestration.auto_retrain_decision_ledger import (
    AutoRetrainDecisionRecord,
    build_decision_record,
    latest_approved_baseline_uri,
    latest_decision_records,
    load_decision_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_PIPELINE = REPO_ROOT / "pipelines" / "submit_pipeline.py"
DUPLICATE_CANDIDATE_STATUSES = {
    "reconciliation_required",
    "submitting",
    "manual_pending",
    "submitted",
    "candidate_submitted",
    "running",
    "queued",
    "in_progress",
}
DUPLICATE_CANDIDATE_OUTCOMES = {"candidate_retrain", "promote_candidate"}
PROTECTED_SUBMIT_ARGS = frozenset(
    {
        "--config",
        "--subscription_id",
        "--resource_group",
        "--workspace_name",
        "--compute",
        "--experiment_name",
        "--display_name",
        "--drift_baseline_in",
        "--force",
        "--force_reason",
        "--dry_run",
        "--expected_execution_id",
        "--expected_config_hash",
        "--expected_source_identity",
        "--submission_revision_kind",
        "--parent_execution_id",
        "--parent_config_hash",
        "--parent_source_identity",
        "--source_decision_id",
        "--revision_reason",
    }
)


class AutoRetrainControllerError(RuntimeError):
    """Raised when an auto-retrain controller plan cannot be built."""


@dataclass(frozen=True)
class AzureSubmissionContext:
    """Azure ML workspace context for canonical pipeline submissions."""

    subscription_id: str
    resource_group: str
    workspace_name: str
    compute: str


@dataclass(frozen=True)
class AutoRetrainConfigMetadata:
    """Task and dataset metadata loaded from a V3 config file."""

    config_path: Path
    config_stem: str
    task_type: str
    dataset_name: str


@dataclass(frozen=True)
class AutoRetrainControllerRequest:
    """Inputs required to plan a controller-driven submission."""

    config_path: Path
    ledger_path: Path
    decision_path: Path
    azure_context: AzureSubmissionContext
    mode: str = "dry_run"
    trigger: str = "manual_controller"
    experiment_name: str | None = None
    display_name: str | None = None
    schedule_name: str | None = None
    python_executable: str = sys.executable
    force_submit: bool = False
    force_reason: str | None = None
    skip_active_check: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AutoRetrainControllerPlan:
    """Resolved command and ledger context for one controller action."""

    request: AutoRetrainControllerRequest
    metadata: AutoRetrainConfigMetadata
    baseline_uri: str
    experiment_name: str
    display_name: str
    command: tuple[str, ...]
    decision_path: Path
    decision_payload: dict[str, Any]

    @property
    def command_text(self) -> str:
        return shlex.join(self.command)


def build_controller_plan(request: AutoRetrainControllerRequest) -> AutoRetrainControllerPlan:
    """Resolve the approved baseline and build a canonical submit command."""
    if request.mode not in {"dry_run", "submit"}:
        raise AutoRetrainControllerError("mode must be one of: dry_run, submit")

    metadata = load_config_metadata(request.config_path)
    decision_payload = load_s14_decision(request.decision_path, metadata)
    records = load_decision_records(request.ledger_path)
    baseline_uri = resolve_latest_baseline_uri(records, metadata)
    if not baseline_uri:
        raise AutoRetrainControllerError(
            f"No approved drift_baseline found for {metadata.config_stem} in {request.ledger_path}"
        )
    baseline_uri = require_azureml_uri(
        baseline_uri,
        source="latest approved drift baseline",
    )
    decision_baseline_uri = (
        (decision_payload.get("comparison") or {}).get("input_baseline_uri")
        or (decision_payload.get("comparison") or {}).get("baseline_uri")
    )
    if decision_baseline_uri and str(decision_baseline_uri) != baseline_uri:
        raise AutoRetrainControllerError(
            "S14 decision baseline does not match the latest approved baseline: "
            f"decision={decision_baseline_uri!r}, approved={baseline_uri!r}"
        )
    if decision_baseline_uri:
        require_azureml_uri(
            str(decision_baseline_uri),
            source="S14 decision baseline",
        )
    if request.mode == "submit" and not request.force_submit:
        duplicate = find_duplicate_candidate_record(records, metadata, baseline_uri)
        if duplicate:
            raise AutoRetrainControllerError(
                "Duplicate candidate retrain detected for "
                f"{metadata.config_stem} using baseline {baseline_uri}. "
                f"Existing decision_id={duplicate.get('decision_id')} "
                f"candidate_job_name={duplicate.get('candidate_job_name') or 'unknown'}. "
                "Use --force-submit with --force-reason only for an intentional override."
            )

    experiment_name = request.experiment_name or default_experiment_name(metadata.config_stem)
    display_name = request.display_name or default_display_name(metadata.config_stem)
    command = build_submit_command(
        request=request,
        baseline_uri=baseline_uri,
        experiment_name=experiment_name,
        display_name=display_name,
        source_revision=decision_payload["source_revision"],
        decision_id=str(decision_payload["decision_id"]),
    )
    return AutoRetrainControllerPlan(
        request=request,
        metadata=metadata,
        baseline_uri=baseline_uri,
        experiment_name=experiment_name,
        display_name=display_name,
        command=tuple(command),
        decision_path=Path(request.decision_path),
        decision_payload=decision_payload,
    )


def load_config_metadata(config_path: str | Path) -> AutoRetrainConfigMetadata:
    """Load task and dataset metadata from a config YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise AutoRetrainControllerError(f"Config not found: {path}")
    try:
        config = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise AutoRetrainControllerError(f"Invalid config YAML: {path}") from exc
    if not isinstance(config, dict):
        raise AutoRetrainControllerError(f"Config did not parse to a mapping: {path}")

    dataset = config.get("dataset") or {}
    return AutoRetrainConfigMetadata(
        config_path=path,
        config_stem=path.stem,
        task_type=str(config.get("task_type") or "unknown"),
        dataset_name=str(dataset.get("name") or "unknown"),
    )


def load_s14_decision(
    decision_path: str | Path,
    metadata: AutoRetrainConfigMetadata,
) -> dict[str, Any]:
    """Load and validate the explicit S14 policy artifact for this submission."""
    path = Path(decision_path)
    if path.is_dir():
        path = path / "retrain_decision.json"
    if not path.is_file():
        raise AutoRetrainControllerError(f"S14 retrain decision not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutoRetrainControllerError(f"Invalid S14 retrain decision: {path}") from exc
    if not isinstance(payload, dict):
        raise AutoRetrainControllerError("S14 retrain decision must be a JSON object")
    if payload.get("stage_id") != "S14" or payload.get("stage") != "s14_retrain_decision":
        raise AutoRetrainControllerError(
            "Controller accepts only an explicit s14_retrain_decision artifact"
        )

    decision = payload.get("retrain_decision")
    if not isinstance(decision, dict):
        raise AutoRetrainControllerError(
            "S14 artifact is missing its RetrainDecision contract"
        )
    if decision.get("contract_type") != "RetrainDecision":
        raise AutoRetrainControllerError("S14 artifact has an invalid retrain decision contract")
    if decision.get("schema_version") != "2.0":
        raise AutoRetrainControllerError("S14 retrain decision schema_version must be 2.0")
    if decision.get("decision_id") != payload.get("decision_id"):
        raise AutoRetrainControllerError("S14 retrain decision identity is inconsistent")
    source_revision = payload.get("source_revision")
    if not isinstance(source_revision, dict):
        raise AutoRetrainControllerError(
            "S14 artifact is missing its immutable source_revision contract"
        )
    if decision.get("source_revision") != source_revision:
        raise AutoRetrainControllerError(
            "S14 retrain decision source revision is inconsistent"
        )
    revision_validation = payload.get("revision_validation")
    if not isinstance(revision_validation, dict) or (
        revision_validation.get("status") != "verified"
    ):
        raise AutoRetrainControllerError(
            "S14 source revision was not verified by the decision stage"
        )
    if source_revision.get("schema_version") != "1.0":
        raise AutoRetrainControllerError(
            "S14 source_revision schema_version must be 1.0"
        )
    missing_revision_fields = [
        field
        for field in ("execution_id", "config_hash", "source_sha")
        if not str(source_revision.get(field) or "").strip()
    ]
    if missing_revision_fields:
        raise AutoRetrainControllerError(
            "S14 source_revision is missing required fields: "
            + ", ".join(missing_revision_fields)
        )
    artifact_identity = payload.get("identity") or {}
    if not isinstance(artifact_identity, dict):
        raise AutoRetrainControllerError("S14 artifact identity must be an object")
    for field in ("execution_id", "config_hash", "source_sha"):
        if str(artifact_identity.get(field) or "") != str(source_revision[field]):
            raise AutoRetrainControllerError(
                f"S14 artifact {field} does not match source_revision"
            )
    if not isinstance(decision.get("should_submit"), bool):
        raise AutoRetrainControllerError("S14 decision must declare boolean should_submit")
    if not decision["should_submit"]:
        raise AutoRetrainControllerError(
            "S14 policy refused candidate submission: "
            + "; ".join(str(reason) for reason in decision.get("reasons") or [])
        )
    if str(decision.get("outcome") or "") not in DUPLICATE_CANDIDATE_OUTCOMES:
        raise AutoRetrainControllerError(
            f"S14 outcome {decision.get('outcome')!r} is not submission-eligible"
        )
    if not payload.get("decision_id"):
        raise AutoRetrainControllerError("S14 artifact is missing decision_id")

    expected_config_names = {
        metadata.config_stem,
        f"{metadata.config_stem}.yml",
    }
    if payload.get("config_name") not in expected_config_names:
        raise AutoRetrainControllerError("S14 decision config identity does not match request")
    if payload.get("task_type") != metadata.task_type:
        raise AutoRetrainControllerError("S14 decision task identity does not match request")
    if payload.get("dataset_name") != metadata.dataset_name:
        raise AutoRetrainControllerError("S14 decision dataset identity does not match request")
    return payload


def resolve_latest_baseline_uri(
    records: Sequence[dict[str, Any]],
    metadata: AutoRetrainConfigMetadata,
) -> str | None:
    """Resolve a baseline URI using stem and filename-compatible ledger keys."""
    config_names = (metadata.config_stem, f"{metadata.config_stem}.yml")
    for config_name in config_names:
        baseline_uri = latest_approved_baseline_uri(
            records,
            config_name=config_name,
            task_type=metadata.task_type,
            dataset_name=metadata.dataset_name,
        )
        if baseline_uri:
            return baseline_uri
    return None


def require_azureml_uri(value: str, *, source: str) -> str:
    """Return a canonical AML URI or fail closed on external/relative locations."""
    uri = str(value or "").strip()
    if not uri.startswith("azureml://"):
        raise AutoRetrainControllerError(f"{source} must be an azureml:// URI")
    return uri


def find_duplicate_candidate_record(
    records: Sequence[dict[str, Any]],
    metadata: AutoRetrainConfigMetadata,
    baseline_uri: str,
) -> dict[str, Any] | None:
    """Find the newest unresolved candidate already submitted for this baseline."""
    config_names = {metadata.config_stem, f"{metadata.config_stem}.yml"}
    matches = [
        record
        for record in latest_decision_records(records)
        if record.get("config_name") in config_names
        and record.get("task_type") == metadata.task_type
        and record.get("dataset_name") == metadata.dataset_name
        and record.get("input_baseline_uri") == baseline_uri
        and str(record.get("outcome") or "").lower() in DUPLICATE_CANDIDATE_OUTCOMES
        and str(record.get("promotion_status") or "").lower() in DUPLICATE_CANDIDATE_STATUSES
    ]
    matches.sort(key=lambda record: str(record.get("timestamp_utc") or ""), reverse=True)
    return matches[0] if matches else None


def build_submit_command(
    *,
    request: AutoRetrainControllerRequest,
    baseline_uri: str,
    experiment_name: str,
    display_name: str,
    source_revision: dict[str, Any],
    decision_id: str,
) -> list[str]:
    """Build the canonical submit_pipeline.py invocation."""
    baseline_uri = require_azureml_uri(
        baseline_uri,
        source="approved drift baseline",
    )
    for token in request.extra_args:
        option = str(token).split("=", 1)[0]
        is_protected_abbreviation = (
            option.startswith("--")
            and len(option) > 2
            and any(protected.startswith(option) for protected in PROTECTED_SUBMIT_ARGS)
        )
        if option in PROTECTED_SUBMIT_ARGS or is_protected_abbreviation:
            raise AutoRetrainControllerError(
                f"extra_args may not override protected canonical argument {option}"
            )
    command = [
        request.python_executable,
        str(SUBMIT_PIPELINE),
        "--config",
        str(request.config_path),
        "--subscription_id",
        request.azure_context.subscription_id,
        "--resource_group",
        request.azure_context.resource_group,
        "--workspace_name",
        request.azure_context.workspace_name,
        "--compute",
        request.azure_context.compute,
        "--experiment_name",
        experiment_name,
        "--display_name",
        display_name,
        "--drift_baseline_in",
        baseline_uri,
    ]
    command.extend(
        [
            "--submission_revision_kind",
            "decision_retrain",
            "--parent_execution_id",
            str(source_revision["execution_id"]),
            "--parent_config_hash",
            str(source_revision["config_hash"]),
            "--parent_source_identity",
            str(source_revision["source_sha"]),
            "--expected_execution_id",
            str(source_revision["execution_id"]),
            "--expected_config_hash",
            str(source_revision["config_hash"]),
            "--expected_source_identity",
            str(source_revision["source_sha"]),
            "--source_decision_id",
            decision_id,
        ]
    )
    if request.force_submit:
        force_reason = str(request.force_reason or "").strip()
        if not force_reason:
            raise AutoRetrainControllerError(
                "force_submit requires a non-empty force_reason"
            )
        command.extend(["--force", "--force_reason", force_reason])
    if request.mode == "dry_run":
        command.append("--dry_run")
    command.extend(request.extra_args)
    return command


def build_pending_decision_record(
    plan: AutoRetrainControllerPlan,
    *,
    candidate_job_name: str | None = None,
    command: Sequence[str] | None = None,
    promotion_status: str = "manual_pending",
) -> AutoRetrainDecisionRecord:
    """Build the ledger record for a planned or submitted candidate run."""
    command_text = shlex.join(command or plan.command)
    source_decision = plan.decision_payload["retrain_decision"]
    return build_decision_record(
        config_name=plan.metadata.config_stem,
        task_type=plan.metadata.task_type,
        dataset_name=plan.metadata.dataset_name,
        decision=source_decision,
        trigger=plan.request.trigger,
        schedule_name=plan.request.schedule_name,
        input_baseline_uri=plan.baseline_uri,
        candidate_job_name=candidate_job_name,
        promotion_status=promotion_status,
        approved_for_future_baseline=False,
        azure_context={
            "subscription_id": plan.request.azure_context.subscription_id,
            "resource_group": plan.request.azure_context.resource_group,
            "workspace_name": plan.request.azure_context.workspace_name,
            "compute": plan.request.azure_context.compute,
        },
        metadata={
            "controller_mode": plan.request.mode,
            "experiment_name": plan.experiment_name,
            "display_name": plan.display_name,
            "command": command_text,
            "force_submit": bool(plan.request.force_submit),
            "force_reason": plan.request.force_reason,
            "duplicate_guard_overridden": bool(plan.request.force_submit),
            "active_check_skipped": bool(plan.request.skip_active_check),
            "active_check_skip_reason": plan.request.force_reason if plan.request.skip_active_check else None,
            "source_s14_decision_id": plan.decision_payload["decision_id"],
            "source_s14_decision_path": str(plan.decision_path),
            "source_identity": dict(plan.decision_payload.get("identity") or {}),
            "source_revision": dict(
                plan.decision_payload.get("source_revision") or {}
            ),
        },
    )


def parse_submitted_job_name(stdout: str) -> str | None:
    """Extract the Azure ML job name from submit_pipeline.py output."""
    for line in stdout.splitlines():
        if "Submitted job:" in line:
            return line.split("Submitted job:", 1)[1].strip()
    return None


def default_experiment_name(config_stem: str) -> str:
    normalized = config_stem.replace("config_", "").replace("_azureml", "").replace("_local", "")
    return f"{normalized}_auto_retrain"


def default_display_name(config_stem: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    normalized = config_stem.replace("config_", "").replace("_azureml", "").replace("_local", "")
    return f"auto_retrain_controller_{normalized}_{timestamp}"
