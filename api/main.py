"""FastAPI application entrypoint for MLOps V3 Pipeline Management API."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.routers import configs, health, pipelines

logger = logging.getLogger(__name__)


async def _experiments_warm_loop(preload_count: int, ttl_seconds: int) -> None:
    """Background task: warm the experiments cache at startup, then refresh on a TTL."""
    from api.services.pipeline_service import refresh_experiments_cache

    loop = asyncio.get_running_loop()
    # First refresh runs immediately so the picker is hot before users hit it.
    while True:
        try:
            await loop.run_in_executor(None, refresh_experiments_cache, preload_count)
        except Exception:
            logger.exception("experiments cache refresh failed; will retry after TTL")
        try:
            await asyncio.sleep(ttl_seconds)
        except asyncio.CancelledError:
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: eagerly create the ML client so first request is fast
    from api.core.azure_ml import get_ml_client
    from api.core.config import settings

    try:
        get_ml_client()
    except Exception:
        logger.exception("failed to pre-create MLClient at startup")

    warmer_task: asyncio.Task | None = None
    if settings.experiment_cache_enabled:
        logger.info(
            "spawning experiments warmer (preload=%d, ttl=%ds)",
            settings.experiment_cache_preload_count,
            settings.experiment_cache_ttl_seconds,
        )
        warmer_task = asyncio.create_task(
            _experiments_warm_loop(
                settings.experiment_cache_preload_count,
                settings.experiment_cache_ttl_seconds,
            ),
            name="experiments-warmer",
        )
        app.state.experiments_warmer = warmer_task

    yield

    # Shutdown: cancel the warmer cleanly
    if warmer_task is not None:
        warmer_task.cancel()
        with suppress(asyncio.CancelledError):
            await warmer_task


app = FastAPI(
    title="MLOps V3 Pipeline Management API",
    description="REST API for submitting, monitoring, and managing Azure ML pipeline jobs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.get("/", tags=["root"])
async def root():
    """Root index. Points callers at docs and the most useful API routes."""
    return {
        "service": app.title,
        "status": "ok",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "health": "/api/v1/health",
        "liveness": "/healthz",
        "api_base": "/api/v1",
        "routes": {
            "health": "GET /api/v1/health",
            "configs_list": "GET /api/v1/configs",
            "config_detail": "GET /api/v1/configs/{config_name}",
            "pipelines_submit": "POST /api/v1/pipelines/submit",
            "pipelines_submit_async": "POST /api/v1/pipelines/submit/async",
            "pipelines_jobs": "GET /api/v1/pipelines/jobs",
            "pipelines_experiments": "GET /api/v1/pipelines/experiments",
            "pipelines_job_status": "GET /api/v1/pipelines/jobs/{job_name}",
            "pipelines_job_outputs": "GET /api/v1/pipelines/jobs/{job_name}/outputs",
            "pipelines_job_metrics": "GET /api/v1/pipelines/jobs/{job_name}/metrics",
            "pipelines_job_summary": "GET /api/v1/pipelines/jobs/{job_name}/summary",
        },
        "auth": {
            "header": "X-API-Key",
            "note": "Required for all /api/v1/* routes except /api/v1/health and /healthz.",
        },
    }


@app.get("/healthz", tags=["root"], include_in_schema=False)
async def healthz():
    """Lightweight liveness probe (no dependencies, no auth)."""
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Silence repeated 404 noise from browsers hitting /favicon.ico
    from fastapi import Response
    return Response(status_code=204)


# Routers
app.include_router(health.router)
app.include_router(configs.router)
app.include_router(pipelines.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
