"""SMTP-backed job notification reports for Azure ML pipeline runs."""

from __future__ import annotations

import csv
import json
import logging
import re
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any, Callable

from api.core.config import settings
from api.schemas.pipeline import (
    DriftResponse,
    JobStatus,
    NotificationArtifact,
    NotificationEmailResponse,
    OutputListResponse,
    PipelineSummaryResponse,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAFE_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")
_TASK_TYPES = ("classification", "regression", "clustering", "timeseries")
_DATASET_STOP_TOKENS = {
    "auto",
    "retrain",
    "daily",
    "weekly",
    "monthly",
    "pipeline",
    "run",
    "job",
    "v3",
    "azure",
    "azureml",
    "candidate",
}
pipeline_service: Any | None = None


@dataclass(frozen=True)
class JobNotificationPackage:
    job_name: str
    recipient: str
    subject: str
    report_dir: Path
    markdown_body: str
    payload: dict[str, Any]
    artifacts: list[NotificationArtifact]


def send_job_notification(job_name: str, *, dry_run: bool = False) -> NotificationEmailResponse:
    """Generate a job report package and optionally send it by SMTP."""
    package = build_job_notification_package(job_name)

    if dry_run:
        return NotificationEmailResponse(
            job_name=package.job_name,
            recipient=package.recipient,
            subject=package.subject,
            status="dry_run",
            sent=False,
            report_dir=str(package.report_dir),
            artifacts=package.artifacts,
            message="Report files generated; SMTP send skipped by dry_run.",
            smtp_host=_configured_smtp_host(),
        )

    missing = _missing_smtp_settings()
    if missing:
        return NotificationEmailResponse(
            job_name=package.job_name,
            recipient=package.recipient,
            subject=package.subject,
            status="not_configured",
            sent=False,
            report_dir=str(package.report_dir),
            artifacts=package.artifacts,
            message="Report files generated, but SMTP settings are missing: " + ", ".join(missing),
            smtp_host=_configured_smtp_host(),
        )

    try:
        _send_smtp_email(package)
    except Exception as exc:  # noqa: BLE001 - return structured status to UI/API clients
        logger.warning("job notification email failed for %s: %s", job_name, exc)
        return NotificationEmailResponse(
            job_name=package.job_name,
            recipient=package.recipient,
            subject=package.subject,
            status="failed",
            sent=False,
            report_dir=str(package.report_dir),
            artifacts=package.artifacts,
            message=f"Report files generated, but SMTP send failed: {exc}",
            smtp_host=_configured_smtp_host(),
        )

    return NotificationEmailResponse(
        job_name=package.job_name,
        recipient=package.recipient,
        subject=package.subject,
        status="sent",
        sent=True,
        report_dir=str(package.report_dir),
        artifacts=package.artifacts,
        message="Notification email sent with generated report attachments.",
        smtp_host=_configured_smtp_host(),
    )


def build_job_notification_package(job_name: str) -> JobNotificationPackage:
    """Collect job state, write report files, and return an email-ready package."""
    service = _pipeline_service()
    job = service.get_job(job_name)
    drift, drift_error = _safe_call(service.get_job_drift, job_name)
    summary, summary_error = _safe_call(service.get_pipeline_summary, job_name)
    outputs, outputs_error = _safe_call(service.list_outputs, job_name)

    payload = _build_payload(
        job=job,
        drift=drift,
        summary=summary,
        outputs=outputs,
        errors={
            "drift": drift_error,
            "summary": summary_error,
            "outputs": outputs_error,
        },
    )
    subject = _build_subject(payload)
    markdown_body = _render_markdown(payload)
    report_dir = _make_report_dir(job_name)
    artifacts = _write_report_artifacts(report_dir, job_name, markdown_body, payload)

    return JobNotificationPackage(
        job_name=job_name,
        recipient=_recipient_address(),
        subject=subject,
        report_dir=report_dir,
        markdown_body=markdown_body,
        payload=payload,
        artifacts=artifacts,
    )


def _pipeline_service() -> Any:
    global pipeline_service
    if pipeline_service is None:
        from api.services import pipeline_service as loaded_service

        pipeline_service = loaded_service
    return pipeline_service


def _safe_call(func: Callable[[str], Any], job_name: str) -> tuple[Any | None, str | None]:
    try:
        return func(job_name), None
    except Exception as exc:  # noqa: BLE001 - partial reports are still useful
        return None, str(exc)


def _build_payload(
    *,
    job: JobStatus,
    drift: DriftResponse | None,
    summary: PipelineSummaryResponse | None,
    outputs: OutputListResponse | None,
    errors: dict[str, str | None],
) -> dict[str, Any]:
    job_data = _to_dict(job) or {}
    drift_data = _to_dict(drift) if drift is not None else {}
    summary_data = _to_dict(summary) if summary is not None else {}
    outputs_data = _to_dict(outputs) if outputs is not None else {}

    steps = job_data.get("steps") or []
    features = drift_data.get("features") or []
    feature_counts = _count_feature_severity(features)
    step_counts = _count_step_statuses(steps)
    metadata_text = (
        job_data.get("experiment_name"),
        job_data.get("job_name"),
        job_data.get("display_name"),
    )
    task_type = (
        drift_data.get("task_type")
        or summary_data.get("task_type")
        or _tag(job_data, "task_type")
        or _infer_task_type(*metadata_text)
    )
    dataset_name = (
        drift_data.get("dataset_name")
        or _tag(job_data, "dataset_name")
        or _infer_dataset_name(*metadata_text)
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job": {
            "job_name": job_data.get("job_name"),
            "display_name": job_data.get("display_name"),
            "experiment_name": job_data.get("experiment_name"),
            "status": job_data.get("status"),
            "start_time": job_data.get("start_time"),
            "end_time": job_data.get("end_time"),
            "studio_url": job_data.get("studio_url"),
            "tags": job_data.get("tags") or {},
        },
        "task": {
            "task_type": task_type,
            "dataset_name": dataset_name,
        },
        "steps": {
            "counts": step_counts,
            "rows": steps,
        },
        "drift": {
            "available": bool(features),
            "overall_drift_detected": drift_data.get("overall_drift_detected"),
            "stability_score": drift_data.get("stability_score"),
            "drift_type": drift_data.get("drift_type"),
            "recommended_cadence": drift_data.get("recommended_cadence"),
            "recommended_days": drift_data.get("recommended_days"),
            "cadence_rationale": drift_data.get("cadence_rationale"),
            "comparison_available": drift_data.get("comparison_available"),
            "baseline_status": drift_data.get("baseline_status"),
            "baseline_metadata": drift_data.get("baseline_metadata") or {},
            "auto_retrain_decision": drift_data.get("auto_retrain_decision") or {},
            "auto_retrain_trigger": drift_data.get("auto_retrain_trigger") or {},
            "warnings": drift_data.get("warnings") or [],
            "feature_counts": feature_counts,
            "top_features": _top_features(features, limit=10),
            "features": features,
        },
        "model_summary": {
            "status": summary_data.get("status"),
            "champion_phase": summary_data.get("champion_phase"),
            "champion_score": summary_data.get("champion_score"),
            "available_outputs": summary_data.get("available_outputs") or [],
        },
        "outputs": {
            "available": [item.get("name") for item in outputs_data.get("outputs") or [] if isinstance(item, dict)],
        },
        "data_gaps": {key: value for key, value in errors.items() if value},
    }


def _build_subject(payload: dict[str, Any]) -> str:
    job = payload.get("job") or {}
    task = payload.get("task") or {}
    generated = _parse_utc(payload.get("generated_at_utc"))
    date_label = generated.strftime("%Y-%m-%d %H:%M UTC")
    experiment = job.get("experiment_name") or "unknown_experiment"
    status = job.get("status") or "Unknown"
    task_label = "/".join(
        part for part in (task.get("task_type"), task.get("dataset_name")) if part
    ) or "unknown_task"
    return f"MLOps V3 | {experiment} | {date_label} | {status} | {task_label}"


def _render_markdown(payload: dict[str, Any]) -> str:
    job = payload.get("job") or {}
    task = payload.get("task") or {}
    steps = payload.get("steps") or {}
    drift = payload.get("drift") or {}
    model_summary = payload.get("model_summary") or {}
    outputs = payload.get("outputs") or {}
    gaps = payload.get("data_gaps") or {}

    lines = [
        f"# MLOps V3 Job Notification: {job.get('display_name') or job.get('job_name')}",
        "",
        _table(
            ["Field", "Value"],
            [
                ["Generated at", payload.get("generated_at_utc") or "n/a"],
                ["Experiment", job.get("experiment_name") or "n/a"],
                ["Job", job.get("job_name") or "n/a"],
                ["Display name", job.get("display_name") or "n/a"],
                ["Status", job.get("status") or "n/a"],
                ["Task type", task.get("task_type") or "n/a"],
                ["Dataset", task.get("dataset_name") or "n/a"],
                ["Start time", job.get("start_time") or "n/a"],
                ["End time", job.get("end_time") or "n/a"],
                ["Azure ML Studio", job.get("studio_url") or "n/a"],
            ],
        ),
        "",
        "## Pipeline Stage Status",
        "",
        _table(
            ["Status", "Count"],
            [[key, value] for key, value in (steps.get("counts") or {}).items()],
        ),
        "",
        _table(
            ["Stage", "Name", "Status", "Start", "End"],
            [
                [
                    row.get("stage_key") or "n/a",
                    row.get("display_name") or row.get("name") or "n/a",
                    row.get("status") or "n/a",
                    row.get("start_time") or "n/a",
                    row.get("end_time") or "n/a",
                ]
                for row in steps.get("rows") or []
            ],
        ),
        "",
        "## Drift and Retrain Signals",
        "",
        _table(
            ["Signal", "Value"],
            [
                ["Drift report available", "yes" if drift.get("available") else "no"],
                ["Overall drift detected", drift.get("overall_drift_detected")],
                ["Stability score", drift.get("stability_score")],
                ["Drift type", drift.get("drift_type")],
                ["Baseline status", drift.get("baseline_status")],
                ["Baseline comparison", "ready" if drift.get("comparison_available") else "not available"],
                ["Recommended cadence", drift.get("recommended_cadence")],
                ["Recommended days", drift.get("recommended_days")],
            ],
        ),
        "",
        "## Auto-Retrain Decision",
        "",
        _render_decision(drift.get("auto_retrain_decision") or {}),
        "",
        "## Top Drifted Features",
        "",
        _table(
            ["Feature", "PSI", "Severity", "Drift detected"],
            [
                [
                    item.get("feature"),
                    _format_float(item.get("psi")),
                    item.get("severity"),
                    item.get("drift_detected"),
                ]
                for item in drift.get("top_features") or []
            ],
        ),
        "",
        "## Model Summary",
        "",
        _table(
            ["Field", "Value"],
            [
                ["Summary status", model_summary.get("status") or "n/a"],
                ["Champion phase", model_summary.get("champion_phase") or "n/a"],
                ["Champion score", _format_float(model_summary.get("champion_score"))],
                ["Available outputs", ", ".join(model_summary.get("available_outputs") or []) or "n/a"],
            ],
        ),
        "",
        "## Named Outputs",
        "",
        ", ".join(outputs.get("available") or []) or "No named outputs were available through the API.",
    ]

    warnings = drift.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    if gaps:
        lines.extend(["", "## Data Gaps", ""])
        lines.extend(f"- {key}: {value}" for key, value in gaps.items())

    return "\n".join(lines).rstrip() + "\n"


def _render_decision(decision: dict[str, Any]) -> str:
    if not decision:
        return "No auto-retrain decision was present in the drift report."
    lines = [
        _table(
            ["Field", "Value"],
            [
                ["Outcome", decision.get("outcome") or "n/a"],
                ["Severity", decision.get("severity") or "n/a"],
                ["Submit candidate", decision.get("should_submit")],
                ["Eligible for promotion", decision.get("eligible_for_promotion")],
            ],
        )
    ]
    reasons = decision.get("reasons") or []
    if reasons:
        lines.extend(["", "Reasons:"])
        lines.extend(f"- {reason}" for reason in reasons)
    return "\n".join(lines)


def _write_report_artifacts(
    report_dir: Path,
    job_name: str,
    markdown_body: str,
    payload: dict[str, Any],
) -> list[NotificationArtifact]:
    report_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(job_name)
    markdown_path = report_dir / f"{slug}_notification.md"
    json_path = report_dir / f"{slug}_notification.json"
    drift_csv_path = report_dir / f"{slug}_drift_features.csv"
    steps_csv_path = report_dir / f"{slug}_steps.csv"

    markdown_path.write_text(markdown_body, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_drift_csv(drift_csv_path, payload)
    _write_steps_csv(steps_csv_path, payload)

    return [
        _artifact(markdown_path, "text/markdown"),
        _artifact(json_path, "application/json"),
        _artifact(drift_csv_path, "text/csv"),
        _artifact(steps_csv_path, "text/csv"),
    ]


def _write_drift_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = ((payload.get("drift") or {}).get("features") or [])
    fieldnames = ["feature", "psi", "severity", "drift_detected"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_steps_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = ((payload.get("steps") or {}).get("rows") or [])
    fieldnames = ["stage_key", "name", "display_name", "status", "start_time", "end_time", "is_inferred"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _send_smtp_email(package: JobNotificationPackage) -> None:
    sender = settings.notification_sender_email.strip() or settings.notification_smtp_username.strip()
    recipient = _recipient_address()
    message = EmailMessage()
    message["Subject"] = package.subject
    message["From"] = sender
    message["To"] = recipient
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = make_msgid(domain="mlops-v3.local")
    message.set_content(package.markdown_body)

    for artifact in package.artifacts:
        if not artifact.included_in_email:
            continue
        path = Path(artifact.path)
        data = path.read_bytes()
        maintype, subtype = artifact.mime_type.split("/", 1)
        message.add_attachment(data, maintype=maintype, subtype=subtype, filename=artifact.name)

    if settings.notification_smtp_ssl:
        with smtplib.SMTP_SSL(
            settings.notification_smtp_host,
            settings.notification_smtp_port,
            timeout=settings.notification_smtp_timeout_seconds,
        ) as smtp:
            _authenticate_and_send(smtp, sender, recipient, message)
    else:
        with smtplib.SMTP(
            settings.notification_smtp_host,
            settings.notification_smtp_port,
            timeout=settings.notification_smtp_timeout_seconds,
        ) as smtp:
            if settings.notification_smtp_starttls:
                smtp.starttls()
            _authenticate_and_send(smtp, sender, recipient, message)


def _authenticate_and_send(
    smtp: smtplib.SMTP,
    sender: str,
    recipient: str,
    message: EmailMessage,
) -> None:
    username = settings.notification_smtp_username.strip()
    password = settings.notification_smtp_password
    if username and password:
        smtp.login(username, password)
    smtp.send_message(message, from_addr=sender, to_addrs=[recipient])


def _missing_smtp_settings() -> list[str]:
    missing: list[str] = []
    smtp_host = settings.notification_smtp_host.strip().lower()
    smtp_username = settings.notification_smtp_username.strip()
    smtp_password = settings.notification_smtp_password.strip()
    if len(_recipient_list()) != 1:
        missing.append("NOTIFICATION_RECIPIENT_EMAIL (exactly one address)")
    if not smtp_host:
        missing.append("NOTIFICATION_SMTP_HOST")
    if not settings.notification_sender_email.strip() and not smtp_username:
        missing.append("NOTIFICATION_SENDER_EMAIL or NOTIFICATION_SMTP_USERNAME")
    if smtp_host == "smtp.gmail.com":
        if not smtp_username:
            missing.append("NOTIFICATION_SMTP_USERNAME")
        if not smtp_password:
            missing.append("NOTIFICATION_SMTP_PASSWORD (Gmail app password)")
    elif smtp_username and not smtp_password:
        missing.append("NOTIFICATION_SMTP_PASSWORD")
    return missing


def _configured_smtp_host() -> str | None:
    return settings.notification_smtp_host.strip() or None


def _recipient_address() -> str:
    recipients = _recipient_list()
    if not recipients:
        return ""
    if len(recipients) != 1:
        return settings.notification_recipient_email.strip()
    return recipients[0]


def _recipient_list() -> list[str]:
    return [addr.strip() for addr in settings.notification_recipient_email.split(",") if addr.strip()]


def _infer_task_type(*texts: str | None) -> str | None:
    for text in texts:
        tokens = _metadata_tokens(text)
        for task_type in _TASK_TYPES:
            if task_type in tokens:
                return task_type
    return None


def _infer_dataset_name(*texts: str | None) -> str | None:
    for text in texts:
        tokens = _metadata_tokens(text)
        for task_type in _TASK_TYPES:
            if task_type not in tokens:
                continue
            start = tokens.index(task_type) + 1
            dataset_tokens: list[str] = []
            for token in tokens[start:]:
                if token in _DATASET_STOP_TOKENS or re.fullmatch(r"\d{8,}|[a-f0-9]{10,}", token):
                    break
                dataset_tokens.append(token)
            if dataset_tokens:
                return "_".join(dataset_tokens)
    return None


def _metadata_tokens(text: str | None) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return [token for token in normalized.split("_") if token]


def _make_report_dir(job_name: str) -> Path:
    root = Path(settings.notification_report_dir).expanduser()
    if not root.is_absolute():
        root = _REPO_ROOT / root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / _slug(job_name) / stamp


def _artifact(path: Path, mime_type: str) -> NotificationArtifact:
    size = path.stat().st_size
    return NotificationArtifact(
        name=path.name,
        path=str(path),
        size_bytes=size,
        mime_type=mime_type,
        included_in_email=size <= settings.notification_max_attachment_bytes,
    )


def _to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return None


def _count_step_statuses(steps: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"total": len(steps)}
    for row in steps:
        status = str(row.get("status") or "Unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _count_feature_severity(features: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"total": len(features), "none": 0, "moderate": 0, "severe": 0}
    for row in features:
        severity = str(row.get("severity") or "none")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _top_features(features: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> float:
        try:
            return float(row.get("psi") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return sorted(features, key=key, reverse=True)[:limit]


def _tag(job_data: dict[str, Any], key: str) -> str | None:
    tags = job_data.get("tags") or {}
    value = tags.get(key) if isinstance(tags, dict) else None
    return str(value) if value else None


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["n/a" for _ in headers]]
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    rendered_rows = ["| " + " | ".join(_markdown_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *rendered_rows])


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _parse_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _slug(value: str) -> str:
    slug = _SAFE_SLUG.sub("_", value.strip())[:128].strip("._-")
    return slug or "job"