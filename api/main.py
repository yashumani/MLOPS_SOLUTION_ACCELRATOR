"""FastAPI application entrypoint for MLOps V3 Pipeline Management API."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(configs.router)
app.include_router(pipelines.router)


if __name__ == "__main__":
    import uvicorn

    from api.core.config import settings

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
