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
