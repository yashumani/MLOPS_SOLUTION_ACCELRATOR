from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace


def test_notification_subject_contains_experiment_date_status_and_task():
    from api.services.notification_service import _build_subject

    subject = _build_subject(
        {
            "generated_at_utc": "2026-05-22T12:34:00+00:00",
            "job": {
                "experiment_name": "classification_telecom_churn_v3",
                "status": "Completed",
            },
            "task": {
                "task_type": "classification",
                "dataset_name": "telecom_churn",
            },
        }
    )

    assert subject == (
        "MLOps V3 | classification_telecom_churn_v3 | "
        "2026-05-22 12:34 UTC | Completed | classification/telecom_churn"
    )


def test_notification_package_writes_expected_artifacts(tmp_path, monkeypatch):
    from api.core.config import settings
    from api.schemas.pipeline import DriftResponse, DriftResultItem, JobStatus, StepStatus
    from api.services import notification_service

    monkeypatch.setattr(settings, "notification_report_dir", str(tmp_path))
    monkeypatch.setattr(settings, "notification_recipient_email", "mlops-oncall@example.com")
    monkeypatch.setattr(
        notification_service,
        "pipeline_service",
        SimpleNamespace(
            get_pipeline_summary=lambda _job: None,
            list_outputs=lambda _job: None,
            get_job=lambda _job: JobStatus(
                job_name="job-1",
                experiment_name="classification_telecom_churn_v3",
                display_name="telecom run",
                status="Completed",
                start_time=datetime(2026, 5, 22, 12, 0, 0),
                steps=[
                    StepStatus(
                        name="s13",
                        display_name="S13 Drift",
                        stage_key="s13",
                        status="Completed",
                    )
                ],
            ),
            get_job_drift=lambda _job: DriftResponse(
                job_name="job-1",
                task_type="classification",
                dataset_name="telecom_churn",
                features=[
                    DriftResultItem(
                        feature="tenure",
                        psi=0.31,
                        drift_detected=True,
                        severity="severe",
                    )
                ],
            ),
        ),
    )

    package = notification_service.build_job_notification_package("job-1")

    assert package.recipient == "mlops-oncall@example.com"
    assert package.report_dir.exists()
    assert {artifact.mime_type for artifact in package.artifacts} == {
        "text/markdown",
        "application/json",
        "text/csv",
    }
    assert all((package.report_dir / artifact.name).exists() for artifact in package.artifacts)


def test_gmail_smtp_requires_app_password(monkeypatch):
    from api.core.config import settings
    from api.services.notification_service import _missing_smtp_settings

    monkeypatch.setattr(settings, "notification_recipient_email", "mlops-oncall@example.com")
    monkeypatch.setattr(settings, "notification_sender_email", "mlops-notifications@example.com")
    monkeypatch.setattr(settings, "notification_smtp_host", "smtp.gmail.com")
    monkeypatch.setattr(settings, "notification_smtp_username", "mlops-notifications@example.com")
    monkeypatch.setattr(settings, "notification_smtp_password", "")

    assert "NOTIFICATION_SMTP_PASSWORD (Gmail app password)" in _missing_smtp_settings()


def test_notification_payload_infers_task_and_dataset_from_auto_retrain_job():
    from api.schemas.pipeline import JobStatus
    from api.services.notification_service import _build_payload

    payload = _build_payload(
        job=JobStatus(
            job_name="auto-retrain-classification-telecom-churn-daily-12345",
            experiment_name="classification_telecom_churn_auto_retrain",
            display_name="auto-retrain-classification-telecom-churn-daily-20260522T020033Z",
            status="Completed",
            steps=[],
        ),
        drift=None,
        summary=None,
        outputs=None,
        errors={"drift": None, "summary": None, "outputs": None},
    )

    assert payload["task"] == {
        "task_type": "classification",
        "dataset_name": "telecom_churn",
    }
