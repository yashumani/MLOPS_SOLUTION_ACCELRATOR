"""Auto-retrain schedule, ledger, and controller planning service."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from api.core.azure_ml import get_ml_client
from api.core.config import settings
from api.schemas.pipeline import (
    AutoRetrainBaselineApprovalRequest,
    AutoRetrainBaselineApprovalResponse,
    AutoRetrainControllerPlanRequest,
    AutoRetrainControllerPlanResponse,
    AutoRetrainDecisionListResponse,
    AutoRetrainScheduleResponse,
    AutoRetrainScheduleRow,
)
from api.utils.azure_links import build_studio_url

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs"
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from orchestration.auto_retrain_controller import (  # noqa: E402
    AutoRetrainControllerError,
    AutoRetrainControllerRequest,
    AzureSubmissionContext,
    build_controller_plan,
    build_pending_decision_record,
    load_config_metadata,
    require_azureml_uri,
)
from orchestration.config_compiler import compile_config  # noqa: E402
from orchestration.auto_retrain_decision_ledger import (  # noqa: E402
    append_decision_record,
    build_decision_record,
    load_decision_records,
)
from orchestration.auto_retrain_schedule_catalog import (  # noqa: E402
    PLANNED_AUTO_RETRAIN_SCHEDULES,
)


def _ledger_root() -> Path:
    """Return the trusted server-owned root for every API ledger operation."""
    configured = os.environ.get("MLOPS_AUTO_RETRAIN_LEDGER_ROOT")
    root = Path(configured).expanduser() if configured else _REPO_ROOT / "outputs"
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root.resolve()


def _require_contained_jsonl(path: Path, root: Path, *, source: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{source} must resolve under the configured auto-retrain ledger root"
        ) from exc
    if resolved.suffix.lower() != ".jsonl":
        raise ValueError(f"{source} must reference a .jsonl file")
    return resolved


def _default_ledger_path() -> Path:
    root = _ledger_root()
    override = os.environ.get("MLOPS_AUTO_RETRAIN_LEDGER")
    candidate = Path(override).expanduser() if override else Path(
        "auto_retrain_decisions.jsonl"
    )
    if not candidate.is_absolute():
        candidate = root / candidate
    return _require_contained_jsonl(
        candidate,
        root,
        source="MLOPS_AUTO_RETRAIN_LEDGER",
    )


def _resolve_ledger_path(raw_path: str | None) -> Path:
    if raw_path is None or not raw_path.strip():
        return _default_ledger_path()

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        raise ValueError(
            "ledger_path must be relative to the configured auto-retrain ledger root"
        )
    root = _ledger_root()
    return _require_contained_jsonl(
        root / candidate,
        root,
        source="ledger_path",
    )


def _resolve_decision_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        raise ValueError(
            "decision_path must be relative to the configured auto-retrain ledger root"
        )
    root = _ledger_root()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "decision_path must resolve under the configured auto-retrain ledger root"
        ) from exc
    if resolved.suffix.lower() != ".json":
        raise ValueError("decision_path must reference a .json file")
    if not resolved.is_file():
        raise FileNotFoundError(f"S14 retrain decision not found: {resolved}")
    return resolved


def _resolve_config_path(config_name: str) -> Path:
    candidate = Path(config_name)
    if candidate.suffix not in {".yml", ".yaml"}:
        candidate = candidate.with_suffix(".yml")
    if not candidate.is_absolute():
        candidate = _CONFIGS_DIR / candidate.name
    candidate = candidate.resolve()
    if _CONFIGS_DIR.resolve() not in candidate.parents:
        raise ValueError(f"Config must live under {_CONFIGS_DIR}: {config_name}")
    if not candidate.exists():
        raise FileNotFoundError(f"Config not found: {config_name}")
    return candidate


def _azure_context() -> AzureSubmissionContext:
    missing = [
        name
        for name, value in (
            ("AZURE_SUBSCRIPTION_ID", settings.azure_subscription_id),
            ("AZURE_RESOURCE_GROUP", settings.azure_resource_group),
            ("AZURE_WORKSPACE_NAME", settings.azure_workspace_name),
            ("AZURE_COMPUTE", settings.compute_target),
        )
        if not value
    ]
    if missing:
        raise ValueError("Missing Azure context: " + ", ".join(missing))
    return AzureSubmissionContext(
        subscription_id=settings.azure_subscription_id,
        resource_group=settings.azure_resource_group,
        workspace_name=settings.azure_workspace_name,
        compute=settings.compute_target,
    )


def _latest_records(ledger_path: Path, limit: int) -> list[dict[str, Any]]:
    records = load_decision_records(ledger_path)
    records.sort(key=lambda item: str(item.get("timestamp_utc") or ""), reverse=True)
    return records[:limit]


def list_auto_retrain_schedules(limit_records: int = 10) -> AutoRetrainScheduleResponse:
    ledger_path = _resolve_ledger_path(None)
    checked_at = datetime.now(timezone.utc).isoformat()
    azure_error: str | None = None
    live_schedules: dict[str, Any] | None = None
    try:
        ml_client = get_ml_client()
        live_schedules = {
            str(schedule.name): schedule
            for schedule in ml_client.schedules.list()
            if getattr(schedule, "name", None)
        }
    except Exception as exc:  # noqa: BLE001 - surface read-plane uncertainty to operators
        azure_error = f"{type(exc).__name__}: {exc}"

    schedules: list[AutoRetrainScheduleRow] = []
    for planned in PLANNED_AUTO_RETRAIN_SCHEDULES:
        payload = planned.as_dict()
        if live_schedules is None:
            payload.update(
                live_state="unverified",
                actual_enabled=None,
                provisioning_status=None,
                source="planned_only",
            )
        else:
            live = live_schedules.get(planned.schedule_name)
            if live is None:
                payload.update(
                    live_state="missing",
                    actual_enabled=False,
                    provisioning_status=None,
                    source="azure_ml",
                )
            else:
                enabled = getattr(live, "is_enabled", None)
                payload.update(
                    live_state=(
                        "enabled"
                        if enabled is True
                        else "disabled"
                        if enabled is False
                        else "unknown"
                    ),
                    actual_enabled=enabled if isinstance(enabled, bool) else None,
                    provisioning_status=(
                        str(getattr(live, "provisioning_status", "") or "") or None
                    ),
                    source="azure_ml",
                )
        schedules.append(AutoRetrainScheduleRow(**payload))
    return AutoRetrainScheduleResponse(
        schedules=schedules,
        total=len(schedules),
        ledger_path=str(ledger_path),
        latest_records=_latest_records(ledger_path, limit=max(0, limit_records)),
        azure_checked_at=checked_at,
        azure_error=azure_error,
    )


def list_auto_retrain_decisions(limit: int = 100) -> AutoRetrainDecisionListResponse:
    ledger_path = _resolve_ledger_path(None)
    records = _latest_records(ledger_path, limit=max(1, min(limit, 500)))
    return AutoRetrainDecisionListResponse(
        ledger_path=str(ledger_path),
        records=records,
        total=len(load_decision_records(ledger_path)),
    )


def build_auto_retrain_controller_plan(
    req: AutoRetrainControllerPlanRequest,
) -> AutoRetrainControllerPlanResponse:
    if req.force_submit and not (req.force_reason or "").strip():
        raise ValueError("force_reason is required when force_submit is true")

    config_path = _resolve_config_path(req.config_name)
    ledger_path = _resolve_ledger_path(req.ledger_path)
    decision_path = _resolve_decision_path(req.decision_path)
    request = AutoRetrainControllerRequest(
        config_path=config_path,
        ledger_path=ledger_path,
        decision_path=decision_path,
        azure_context=_azure_context(),
        mode="dry_run",
        trigger=req.trigger,
        schedule_name=req.schedule_name,
        experiment_name=req.experiment_name,
        display_name=req.display_name,
        force_submit=req.force_submit,
        force_reason=req.force_reason,
    )
    try:
        plan = build_controller_plan(request)
    except AutoRetrainControllerError as exc:
        raise ValueError(str(exc)) from exc

    pending_record = build_pending_decision_record(plan).as_dict()
    return AutoRetrainControllerPlanResponse(
        config_name=plan.metadata.config_stem,
        task_type=plan.metadata.task_type,
        dataset_name=plan.metadata.dataset_name,
        baseline_uri=plan.baseline_uri,
        experiment_name=plan.experiment_name,
        display_name=plan.display_name,
        command=plan.command_text,
        ledger_path=str(ledger_path),
        decision_path=str(plan.decision_path),
        pending_decision_record=pending_record,
    )


def validate_baseline_job(
    *,
    config_path: Path,
    metadata: Any,
    baseline_job_name: str,
    requested_uri: str | None,
) -> tuple[str, str | None, dict[str, Any]]:
    """Bind an approved baseline to one completed, identity-matched Azure job."""

    ml_client = get_ml_client()
    job = ml_client.jobs.get(baseline_job_name)
    status = str(getattr(job, "status", "") or "").lower()
    if status not in {"completed", "succeeded"}:
        raise ValueError(
            f"Baseline job {baseline_job_name!r} is not completed (status={status or 'unknown'})"
        )

    outputs = getattr(job, "outputs", None) or {}
    output = outputs.get("drift_baseline")
    output_uri = getattr(output, "path", None) if output is not None else None
    if not output_uri:
        if (
            output is None
            or getattr(output, "type", None) != "uri_folder"
            or getattr(job, "name", None) != baseline_job_name
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,254}", baseline_job_name) is None
        ):
            raise ValueError(
                f"Baseline job {baseline_job_name!r} does not expose a reusable drift_baseline URI"
            )
        # Azure can leave a system-generated parent output path unset. The
        # job-output reference is usable only after the content checks below.
        output_uri = f"azureml://jobs/{baseline_job_name}/outputs/drift_baseline/paths/"
    try:
        output_uri = require_azureml_uri(
            str(output_uri),
            source="baseline job drift_baseline output",
        )
    except AutoRetrainControllerError as exc:
        raise ValueError(str(exc)) from exc
    if requested_uri and requested_uri.rstrip("/") != output_uri.rstrip("/"):
        raise ValueError(
            "Requested baseline URI does not match the producing job output: "
            f"requested={requested_uri!r}, job_output={output_uri!r}"
        )

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    compiled = compile_config(raw_config, source_name=config_path.name)
    tags = dict(getattr(job, "tags", None) or {})
    expected_tags = {
        "task": metadata.task_type,
        "dataset": metadata.dataset_name,
        "compiled_config_hash": compiled["compiled_config_hash"],
    }
    for key, expected in expected_tags.items():
        actual = str(tags.get(key) or "")
        if actual != str(expected):
            raise ValueError(
                f"Baseline job tag {key!r} does not match the selected config: "
                f"expected={expected!r}, actual={actual!r}"
            )
    required_identity_tags = ("execution_id", "source_identity")
    missing_tags = [name for name in required_identity_tags if not tags.get(name)]
    if missing_tags:
        raise ValueError(
            "Baseline job is missing immutable identity tag(s): " + ", ".join(missing_tags)
        )

    with tempfile.TemporaryDirectory(prefix="mlops-baseline-approval-") as temp_dir:
        ml_client.jobs.download(
            name=baseline_job_name,
            output_name="drift_baseline",
            download_path=temp_dir,
        )
        root = Path(temp_dir)
        metadata_files = list(root.rglob("feature_baseline.json"))
        reference_files = list(root.rglob("reference_data.csv"))
        if len(metadata_files) != 1 or len(reference_files) != 1:
            raise ValueError(
                "Baseline output must contain exactly one feature_baseline.json and reference_data.csv"
            )
        with reference_files[0].open("r", encoding="utf-8") as handle:
            header = handle.readline().strip()
            first_row = handle.readline().strip()
        if not header or not first_row:
            raise ValueError("Baseline reference_data.csv has no data rows")
        try:
            baseline_metadata = json.loads(
                metadata_files[0].read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Baseline feature_baseline.json is invalid") from exc

    if not isinstance(baseline_metadata, dict):
        raise ValueError("Baseline feature_baseline.json must contain an object")
    if str(baseline_metadata.get("task_type") or "") != metadata.task_type:
        raise ValueError("Baseline artifact task_type does not match the selected config")
    if str(baseline_metadata.get("dataset_name") or "") != metadata.dataset_name:
        raise ValueError("Baseline artifact dataset_name does not match the selected config")
    identity = baseline_metadata.get("identity") or {}
    if not isinstance(identity, dict):
        raise ValueError("Baseline artifact identity must contain an object")
    identity_checks = {
        "execution_id": tags["execution_id"],
        "config_hash": tags["compiled_config_hash"],
        "source_sha": tags["source_identity"],
    }
    for key, expected in identity_checks.items():
        if str(identity.get(key) or "") != str(expected):
            raise ValueError(
                f"Baseline artifact identity {key!r} does not match its producing job"
            )
    missing_artifact_identity = [
        field
        for field in ("model_bundle_id", "data_fingerprint")
        if not str(identity.get(field) or "").strip()
    ]
    if missing_artifact_identity:
        raise ValueError(
            "Baseline artifact is missing model/data identity field(s): "
            + ", ".join(missing_artifact_identity)
        )

    return (
        output_uri,
        build_studio_url(ml_client, baseline_job_name),
        {
            "baseline_execution_id": tags["execution_id"],
            "baseline_config_hash": tags["compiled_config_hash"],
            "baseline_source_identity": tags["source_identity"],
            "baseline_model_bundle_id": identity["model_bundle_id"],
            "baseline_data_fingerprint": identity["data_fingerprint"],
            "baseline_job_status": status,
            "baseline_identity_verified": True,
        },
    )


def approve_auto_retrain_baseline(
    req: AutoRetrainBaselineApprovalRequest,
    *,
    actor_tags: dict[str, str] | None = None,
) -> AutoRetrainBaselineApprovalResponse:
    config_path = _resolve_config_path(req.config_name)
    metadata = load_config_metadata(config_path)
    ledger_path = _resolve_ledger_path(req.ledger_path)
    baseline_uri = (req.output_baseline_uri or "").strip() or None
    if baseline_uri:
        try:
            baseline_uri = require_azureml_uri(
                baseline_uri,
                source="approved drift baseline",
            )
        except AutoRetrainControllerError as exc:
            raise ValueError(str(exc)) from exc
    if not req.baseline_job_name:
        raise ValueError(
            "baseline_job_name is required so baseline ownership and identity can be verified"
        )
    baseline_uri, studio_url, verified_identity = validate_baseline_job(
        config_path=config_path,
        metadata=metadata,
        baseline_job_name=req.baseline_job_name,
        requested_uri=baseline_uri,
    )

    record = build_decision_record(
        config_name=metadata.config_stem,
        task_type=metadata.task_type,
        dataset_name=metadata.dataset_name,
        decision={
            "outcome": "approve_baseline",
            "severity": "none",
            "reasons": [req.reason],
            "signals": {"baseline_status": "approved", "comparison_available": True},
        },
        trigger="manual_ui_approval",
        schedule_name=req.schedule_name,
        output_baseline_uri=baseline_uri,
        baseline_job_name=req.baseline_job_name,
        promotion_status="approved",
        approved_for_future_baseline=True,
        azure_context={
            "subscription_id": settings.azure_subscription_id,
            "resource_group": settings.azure_resource_group,
            "workspace_name": settings.azure_workspace_name,
            "compute": settings.compute_target,
        },
        metadata={
            "approved_via": "api",
            "actor_tags": dict(actor_tags or {}),
            "config_path": str(config_path),
            **verified_identity,
        },
    )
    append_decision_record(ledger_path, record)
    payload = record.as_dict()
    return AutoRetrainBaselineApprovalResponse(
        status="approved",
        ledger_path=str(ledger_path),
        record=payload,
        baseline_uri=baseline_uri,
        studio_url=studio_url,
    )
