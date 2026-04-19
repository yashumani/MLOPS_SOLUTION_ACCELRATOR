"""Health check endpoint (no auth required)."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/v1/health")
async def health():
    return {"status": "ok"}
