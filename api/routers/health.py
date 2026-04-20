"""Health check endpoint (no auth required)."""

from datetime import datetime, timezone

from fastapi import APIRouter

from api.schemas.pipeline import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=HealthResponse)
async def health():
    connected = False
    workspace: str | None = None
    try:
        from api.core.azure_ml import get_ml_client

        ml_client = get_ml_client()
        workspace = ml_client.workspace_name
        connected = True
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        azure_ml_connected=connected,
        workspace=workspace,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
