"""API key authentication dependency."""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from api.core.config import settings

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(_header)) -> str:
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server",
        )
    if not api_key or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


async def require_config_mutation_enabled() -> None:
    """Refuse checkout mutation unless the deployment explicitly enables it."""
    if not settings.api_config_mutation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Config mutation is disabled for this deployment; use the "
                "reviewed Git configuration workflow"
            ),
        )
