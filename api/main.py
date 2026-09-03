"""FastAPI application entrypoint for MLOps V3 Pipeline Management API."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from api.core.config import settings
from api.core.entra_auth import Principal
from api.core.security import complete_audit, verify_api_key
from api.routers import configs, health, pipelines, users
from orchestration import operational_state

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
    settings.validate_runtime_security()
    operational_state.configure_database(settings.mlops_operational_state_db)
    if settings.api_deployment_profile.strip().lower() == "multi_user":
        binding = {
            "tenant_id": settings.api_entra_tenant_id,
            "subscription_id": settings.azure_subscription_id,
            "resource_group": settings.azure_resource_group,
            "workspace_name": settings.azure_workspace_name,
        }
        with operational_state.transaction() as connection:
            existing = operational_state.get_document(connection, "configuration", "workspace")
            if existing is not None and existing != binding:
                raise RuntimeError("Operational state belongs to another tenant or workspace")
            operational_state.put_document(connection, "configuration", "workspace", binding)
        from api.services.user_access_service import initialize_users
        await run_in_threadpool(initialize_users, settings)

    # Startup: eagerly create the ML client so first request is fast
    from api.core.azure_ml import get_ml_client

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

    try:
        yield
    finally:
        if warmer_task is not None:
            warmer_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmer_task
        operational_state.configure_database("")


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
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def complete_actor_audit(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        audit_id = getattr(request.state, "audit_id", None)
        if audit_id is not None:
            try:
                await run_in_threadpool(complete_audit, audit_id, 500)
            except Exception:
                logger.exception("failed to record interrupted request audit %s", audit_id)
        raise
    audit_id = getattr(request.state, "audit_id", None)
    if audit_id is not None:
        try:
            await run_in_threadpool(complete_audit, audit_id, response.status_code)
        except Exception:
            logger.exception("failed to complete request audit %s", audit_id)
            return JSONResponse(status_code=503, content={"detail": "Request audit failed; inspect operation status before retrying"})
    return response


@app.get("/api/v1/auth/config", tags=["authentication"])
async def authentication_config():
    if settings.api_deployment_profile.strip().lower() != "multi_user":
        return {"mode": "api_key"}
    return {
        "mode": "entra",
        "tenant_id": settings.api_entra_tenant_id,
        "client_id": settings.api_entra_spa_client_id,
        "scope": f"api://{settings.api_entra_api_client_id}/{settings.api_entra_required_scope}",
        "redirect_uri": settings.api_entra_redirect_uri,
    }


@app.get("/api/v1/auth/me", tags=["authentication"])
async def authenticated_identity(principal=Depends(verify_api_key)):
    if isinstance(principal, Principal):
        return {"mode": "entra", **principal.as_dict()}
    return {"mode": "api_key", "roles": ["operator"]}


def _dashboard_url(request: Request) -> str:
    """Return the Streamlit dashboard URL for this deployment."""
    configured_url = settings.ui_base_url.strip().rstrip("/")
    if configured_url:
        return f"{configured_url}/"

    host = request.url.hostname or "localhost"
    scheme = (
        "https"
        if host.endswith(".instances.azureml.ms")
        else request.url.scheme or "http"
    )

    if host.endswith(".instances.azureml.ms"):
        labels = host.split(".")
        api_suffix = f"-{settings.api_port}"
        ui_suffix = f"-{settings.ui_port}"
        if labels and labels[0].endswith(api_suffix):
            labels[0] = f"{labels[0][:-len(api_suffix)]}{ui_suffix}"
        return f"{scheme}://{'.'.join(labels)}/"

    if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return f"http://{host}:{settings.ui_port}/"

    # Unknown external hosts must not control a browser redirect. Deployments
    # outside Azure ML must set UI_BASE_URL explicitly.
    return "/docs"


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept


@app.get("/", tags=["root"])
async def root(request: Request):
    """Root index. Redirect browsers to the UI; keep JSON for API clients."""
    dashboard_url = _dashboard_url(request)
    if _wants_html(request):
        return RedirectResponse(dashboard_url, status_code=307)

    return {
        "service": app.title,
        "status": "ok",
        "version": app.version,
        "dashboard": dashboard_url,
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
            "pipelines_job_email_notification": "POST /api/v1/pipelines/jobs/{job_name}/notifications/email",
        },
        "auth": {
            "header": "Authorization" if settings.api_deployment_profile.strip().lower() == "multi_user" else "X-API-Key",
            "deployment_profile": settings.api_deployment_profile,
            "note": "Required for operational routes; health and public authentication configuration are unauthenticated.",
        },
        "frontend": {
            "service": "Streamlit dashboard",
            "url": dashboard_url,
            "note": "Port 8000 is the FastAPI backend; use the dashboard URL for the frontend.",
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
app.include_router(users.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
