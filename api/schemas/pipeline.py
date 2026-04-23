"""Pydantic models for pipeline endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Submission ────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    config_name: str = Field(
        ...,
        description="Config filename stem, e.g. config_classification_telecom_churn_azureml",
    )
    compute: str | None = Field(None, description="Override compute target")
    force_rerun: bool = Field(False, description="Disable component caching")
    baseline_job: str | None = Field(
        None, description="Previous job name for drift baseline comparison"
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
    overall_drift_detected: bool = False
    stability_score: float | None = None
    drift_type: str | None = None
    drifted_columns: list[str] = Field(default_factory=list)
    features: list[DriftResultItem] = Field(default_factory=list)
    evidently_report_path: str | None = None
    studio_url: str | None = None


# ── Resubmit ──────────────────────────────────────────────────

class ResubmitRequest(BaseModel):
    job_name: str = Field(..., description="Name of the job to resubmit")
    force_rerun: bool = Field(True, description="Force re-run ignoring cache")


# ── Baseline capture ──────────────────────────────────────────

class BaselineCaptureRequest(BaseModel):
    job_name: str = Field(
        ..., description="Completed job to extract drift baseline from",
    )


class BaselineCaptureResponse(BaseModel):
    job_name: str
    baseline_path: str | None = None
    status: str = "captured"
    studio_url: str | None = None


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    azure_ml_connected: bool = False
    workspace: str | None = None
    timestamp: str | None = None
