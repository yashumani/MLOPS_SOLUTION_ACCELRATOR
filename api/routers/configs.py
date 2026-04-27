"""Config listing, detail, and CRUD endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.core.security import verify_api_key
from api.schemas.config import ConfigDetail, ConfigListResponse
from api.services import config_service, pipeline_service

router = APIRouter(prefix="/api/v1/configs", tags=["configs"], dependencies=[Depends(verify_api_key)])


# Non-terminal Azure ML job states. Mutating a config that an in-flight job
# is using would create attribution drift between the running pipeline and the
# YAML on disk, so we refuse the mutation.
_NON_TERMINAL = {
    "NotStarted",
    "Starting",
    "Provisioning",
    "Preparing",
    "Queued",
    "Running",
    "Finalizing",
    "CancelRequested",
}


class ConfigWriteRequest(BaseModel):
    content: dict[str, Any] = Field(..., description="Full YAML content as a mapping")


def _guard_no_running_jobs(config_name: str) -> None:
    """Raise 409 if any non-terminal job exists for the experiment derived from this config."""
    try:
        experiment_name = pipeline_service._derive_experiment_name(config_name)
        jobs = pipeline_service.list_jobs(
            experiment_name=experiment_name,
            status_filter=None,
            max_results=20,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed on guard uncertainty
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot verify running jobs for config '{config_name}'. "
                "Refusing mutation until Azure ML status can be checked."
            ),
        ) from exc
    items = getattr(jobs, "jobs", None) or []
    for j in items:
        status = getattr(j, "status", None) or ""
        if status in _NON_TERMINAL:
            job_name = getattr(j, "job_name", "?")
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Cannot modify config '{config_name}': job '{job_name}' "
                    f"(status={status}) is still using it."
                ),
            )


@router.get("", response_model=ConfigListResponse)
async def list_configs():
    """List all available Azure ML pipeline configs."""
    configs = config_service.list_configs()
    return ConfigListResponse(configs=configs, total=len(configs))


@router.get("/{config_name}", response_model=ConfigDetail)
async def get_config(config_name: str):
    """Get full details of a specific config."""
    try:
        return config_service.get_config(config_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{config_name}", response_model=ConfigDetail, status_code=201)
async def create_config(config_name: str, body: ConfigWriteRequest):
    """Create a new config. Fails with 409 if it already exists."""
    try:
        return config_service.create_config(config_name, body.content)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/{config_name}", response_model=ConfigDetail)
async def update_config(config_name: str, body: ConfigWriteRequest):
    """Overwrite an existing config. Refused if a non-terminal job is using it."""
    _guard_no_running_jobs(config_name)
    try:
        return config_service.update_config(config_name, body.content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{config_name}")
async def delete_config(config_name: str):
    """Delete a config. Refused if a non-terminal job is using it."""
    _guard_no_running_jobs(config_name)
    try:
        return config_service.delete_config(config_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
