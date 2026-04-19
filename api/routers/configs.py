"""Config listing and detail endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.core.security import verify_api_key
from api.schemas.config import ConfigDetail, ConfigListResponse
from api.services import config_service

router = APIRouter(prefix="/api/v1/configs", tags=["configs"], dependencies=[Depends(verify_api_key)])


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
