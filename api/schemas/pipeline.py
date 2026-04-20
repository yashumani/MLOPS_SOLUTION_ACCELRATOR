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
    tags: dict[str, str] = Field(default_factory=dict)
    steps: list[StepStatus] = Field(default_factory=list)


# ── Job list ──────────────────────────────────────────────────

class JobSummary(BaseModel):
    job_name: str
    experiment_name: str | None = None
    display_name: str | None = None
    status: str
    start_time: datetime | None = None


class JobListResponse(BaseModel):
    jobs: list[JobSummary]
    total: int


# ── Outputs ───────────────────────────────────────────────────

class OutputInfo(BaseModel):
    name: str
    type: str | None = None


class OutputListResponse(BaseModel):
    job_name: str
    outputs: list[OutputInfo]


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


# ── Resubmit ──────────────────────────────────────────────────

class ResubmitRequest(BaseModel):
    job_name: str = Field(..., description="Name of the job to resubmit")
    force_rerun: bool = Field(True, description="Force re-run ignoring cache")
