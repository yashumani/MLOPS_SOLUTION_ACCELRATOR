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
