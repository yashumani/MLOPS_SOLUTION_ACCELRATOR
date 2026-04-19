"""FastAPI application entrypoint for MLOps V3 Pipeline Management API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import configs, health, pipelines


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: eagerly create the ML client so first request is fast
    from api.core.azure_ml import get_ml_client

    try:
        get_ml_client()
    except Exception:
        pass  # log but don't block startup
    yield
    # Shutdown: nothing to clean up


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
