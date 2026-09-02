"""Pydantic models for config endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConfigSummary(BaseModel):
    config_name: str
    task_type: str | None = None
    dataset_name: str | None = None
    target_column: str | None = None


class ConfigListResponse(BaseModel):
    configs: list[ConfigSummary]
    total: int


class ConfigDetail(ConfigSummary):
    content: dict[str, Any] = Field(default_factory=dict, description="Full YAML content")


class ConfigValidationIssue(BaseModel):
    path: str = Field("$", description="JSONPath-like location of the issue")
    message: str
    level: str = Field("error", description="error or warning")


class ConfigValidationRequest(BaseModel):
    content: dict[str, Any] = Field(..., description="YAML content parsed as a mapping")


class ConfigValidationResponse(BaseModel):
    valid: bool
    errors: list[ConfigValidationIssue] = Field(default_factory=list)
    warnings: list[ConfigValidationIssue] = Field(default_factory=list)


class ConfigStagePreview(BaseModel):
    stage_id: str
    label: str
    enabled: bool = True
    summary: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ConfigPreviewRequest(BaseModel):
    content: dict[str, Any] = Field(..., description="YAML content parsed as a mapping")
    config_name: str | None = Field(None, description="Optional config name for naming preview")


class ConfigPreviewResponse(BaseModel):
    valid: bool
    validation: ConfigValidationResponse
    config_name: str | None = None
    experiment_name: str | None = None
    task_type: str | None = None
    dataset_name: str | None = None
    target_column: str | None = None
    dataset_uri_preview: str | None = None
    compute_target: str | None = None
    baseline_engines: list[str] = Field(default_factory=list)
    phase_b_engines: list[str] = Field(default_factory=list)
    phase_b_variant_budget: int | None = None
    phase_c_trials: int | None = None
    phase_c_timeout_seconds: int | None = None
    stage_plan: list[ConfigStagePreview] = Field(default_factory=list)
