"""Pydantic models for pipeline endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Submission ────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    config_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_]+$",
        description="Config filename stem, e.g. config_classification_telecom_churn_azureml",
    )
    compute: str | None = Field(
        None,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Override compute target",
    )
    force_rerun: bool = Field(False, description="Disable component caching")
    baseline_job: str | None = Field(
        None,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Previous job name for drift baseline comparison",
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Extra job tags")


class SubmitResponse(BaseModel):
    job_name: str
    experiment_name: str
    display_name: str
    status: str
    studio_url: str


# ── Job status ────────────────────────────────────────────────

class StepStatus(BaseModel):
    name: str
    display_name: str | None = None
    stage_key: str | None = None
    is_inferred: bool = False
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None


class JobStatus(BaseModel):
    job_name: str
    experiment_name: str | None = None
    display_name: str | None = None
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    studio_url: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    steps: list[StepStatus] = Field(default_factory=list)


# ── Job list ──────────────────────────────────────────────────

class JobSummary(BaseModel):
    job_name: str
    experiment_name: str | None = None
    display_name: str | None = None
    status: str
    start_time: datetime | None = None
    studio_url: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobSummary]
    total: int


# ── Experiment tree (UI hierarchy: experiment → jobs) ─────────

class ExperimentNode(BaseModel):
    """One experiment with its child jobs, ordered most-recent first."""
    experiment_name: str
    job_count: int
    last_activity: datetime | None = None
    jobs: list[JobSummary] = Field(default_factory=list)


class ExperimentTreeResponse(BaseModel):
    """Experiment-grouped job listing for hierarchical pickers."""
    experiments: list[ExperimentNode] = Field(default_factory=list)
    total_experiments: int = 0
    total_jobs: int = 0


# ── Outputs ───────────────────────────────────────────────────

class OutputInfo(BaseModel):
    name: str
    type: str | None = None


class OutputListResponse(BaseModel):
    job_name: str
    outputs: list[OutputInfo]


class OutputFileInfo(BaseModel):
    name: str
    relative_path: str
    size_bytes: int
    kind: str  # json | csv | text | image | binary | html | yaml | markdown


class OutputContentResponse(BaseModel):
    """Parsed content of a named output, designed for UI rendering."""
    job_name: str
    output_name: str
    files: list[OutputFileInfo] = Field(default_factory=list)
    json_content: Any | None = None      # parsed JSON (dict or list)
    text_preview: str | None = None      # for non-JSON text files (truncated)
    csv_preview: list[dict] | None = None  # first N rows of any CSV
    primary_file: str | None = None
    truncated: bool = False


class LocalOutputFileInfo(BaseModel):
    relative_path: str
    name: str
    is_dir: bool = False
    size_bytes: int | None = None
    modified_time: datetime | None = None
    kind: str | None = None
    depth: int = 0


class LocalOutputsResponse(BaseModel):
    root: str = "outputs"
    files: list[LocalOutputFileInfo] = Field(default_factory=list)
    total: int = 0
    truncated: bool = False


# ── Pipeline summary (aggregated reports) ─────────────────────

class PipelineSummaryResponse(BaseModel):
    """Combined view of the four aggregate JSON reports for a job."""
    job_name: str
    task_type: str | None = None
    status: str | None = None
    champion_phase: str | None = None
    champion_score: float | None = None
    baseline_aggregate: Any | None = None
    phaseb_aggregate: Any | None = None
    phasec_aggregate: Any | None = None
    final_report: Any | None = None
    available_outputs: list[str] = Field(default_factory=list)


# ── Metrics ───────────────────────────────────────────────────

class ModelMetric(BaseModel):
    model_name: str
    engine: str | None = None
    phase: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    is_champion: bool = False


class MetricsResponse(BaseModel):
    job_name: str
    task_type: str | None = None
    models: list[ModelMetric] = Field(default_factory=list)


# ── Drift ─────────────────────────────────────────────────────

class DriftResultItem(BaseModel):
    feature: str
    psi: float
    drift_detected: bool
    severity: str = "none"  # none | moderate | severe


class DriftResponse(BaseModel):
    job_name: str
    task_type: str | None = None
    dataset_name: str | None = None
    overall_drift_detected: bool = False
    stability_score: float | None = None
    drift_type: str | None = None
    recommended_cadence: str | None = None
    recommended_days: int | None = None
    cadence_rationale: str | None = None
    comparison_available: bool = False
    baseline_status: str | None = None
    baseline_metadata: dict[str, Any] = Field(default_factory=dict)
    auto_retrain_decision: dict[str, Any] = Field(default_factory=dict)
    auto_retrain_trigger: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    drifted_columns: list[str] = Field(default_factory=list)
    features: list[DriftResultItem] = Field(default_factory=list)
    evidently_report_path: str | None = None
    studio_url: str | None = None


# -- Notifications ---------------------------------------------------------

class NotificationEmailRequest(BaseModel):
    dry_run: bool = Field(
        False,
        description="Generate report files without sending SMTP email.",
    )


class NotificationArtifact(BaseModel):
    name: str
    path: str
    size_bytes: int
    mime_type: str
    included_in_email: bool = True


class NotificationEmailResponse(BaseModel):
    job_name: str
    recipient: str
    subject: str
    status: str
    sent: bool = False
    report_dir: str
    artifacts: list[NotificationArtifact] = Field(default_factory=list)
    message: str | None = None
    smtp_host: str | None = None


# ── Resubmit ──────────────────────────────────────────────────

class ResubmitRequest(BaseModel):
    job_name: str = Field(..., description="Name of the job to resubmit")
    force_rerun: bool = Field(True, description="Force re-run ignoring cache")
    revision_mode: Literal["exact_replay", "new_revision"] = Field(
        "exact_replay",
        description=(
            "exact_replay requires the current config/source to match the original "
            "immutable execution identity; new_revision explicitly submits current inputs"
        ),
    )
    revision_reason: str | None = Field(
        None,
        min_length=1,
        max_length=256,
        description="Required operator reason when revision_mode is new_revision",
    )


# ── Baseline capture ──────────────────────────────────────────

class BaselineCaptureRequest(BaseModel):
    job_name: str = Field(
        ..., description="Completed job to extract drift baseline from",
    )


class BaselineCaptureResponse(BaseModel):
    job_name: str
    baseline_path: str | None = None
    output_present: bool = False
    status: str = "captured"
    studio_url: str | None = None


# -- Auto-retrain operations -----------------------------------------------

class AutoRetrainScheduleRow(BaseModel):
    task_type: str
    dataset_name: str
    config_name: str
    schedule_name: str
    cadence: str
    cadence_days: int
    decision_mode: str = "candidate_retrain"
    promotion_mode: str = "manual"
    enabled_expected: bool = True
    live_state: str = "unverified"
    actual_enabled: bool | None = None
    provisioning_status: str | None = None
    source: str = "planned_only"


class AutoRetrainScheduleResponse(BaseModel):
    schedules: list[AutoRetrainScheduleRow] = Field(default_factory=list)
    total: int = 0
    ledger_path: str
    latest_records: list[dict[str, Any]] = Field(default_factory=list)
    azure_checked_at: str | None = None
    azure_error: str | None = None


class AutoRetrainDecisionListResponse(BaseModel):
    ledger_path: str
    records: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0


class AutoRetrainControllerPlanRequest(BaseModel):
    config_name: str = Field(
        ...,
        min_length=1,
        max_length=160,
        description="Config filename or stem, e.g. config_classification_telecom_churn_azureml.yml",
    )
    ledger_path: str | None = Field(None, description="Optional JSONL ledger override")
    decision_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Relative path to the explicit S14 retrain_decision.json artifact "
            "under the configured auto-retrain ledger root"
        ),
    )
    trigger: str = Field("manual_ui", max_length=80)
    schedule_name: str | None = Field(None, max_length=160)
    experiment_name: str | None = Field(None, max_length=160)
    display_name: str | None = Field(None, max_length=200)
    force_submit: bool = False
    force_reason: str | None = Field(None, max_length=500)


class AutoRetrainControllerPlanResponse(BaseModel):
    config_name: str
    task_type: str
    dataset_name: str
    baseline_uri: str
    experiment_name: str
    display_name: str
    command: str
    ledger_path: str
    decision_path: str
    pending_decision_record: dict[str, Any]


class AutoRetrainBaselineApprovalRequest(BaseModel):
    config_name: str = Field(..., min_length=1, max_length=160)
    baseline_job_name: str | None = Field(
        None,
        max_length=160,
        description="Required producing Azure ML job used to verify baseline ownership and identity.",
    )
    output_baseline_uri: str | None = Field(
        None,
        description="Explicit drift_baseline URI. If omitted, baseline_job_name is inspected.",
    )
    ledger_path: str | None = Field(None, description="Optional JSONL ledger override")
    schedule_name: str | None = Field(None, max_length=160)
    reason: str = Field("Operator approved drift baseline for future auto-retrain.")


class AutoRetrainBaselineApprovalResponse(BaseModel):
    status: str
    ledger_path: str
    record: dict[str, Any]
    baseline_uri: str
    studio_url: str | None = None


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    azure_ml_connected: bool = False
    workspace: str | None = None
    timestamp: str | None = None
