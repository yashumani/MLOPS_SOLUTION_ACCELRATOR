"""Pipeline submission, monitoring, cancel, and output endpoints."""

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse

from api.core.security import verify_api_key
from api.schemas.pipeline import (
    BaselineCaptureRequest,
    BaselineCaptureResponse,
    DriftResponse,
    ExperimentTreeResponse,
    JobListResponse,
    JobStatus,
    MetricsResponse,
    OutputContentResponse,
    OutputListResponse,
    PipelineSummaryResponse,
    ResubmitRequest,
    SubmitRequest,
    SubmitResponse,
)
from api.services import pipeline_service

router = APIRouter(
    prefix="/api/v1/pipelines",
    tags=["pipelines"],
    dependencies=[Depends(verify_api_key)],
)


# ── Submit ────────────────────────────────────────────────────

@router.post("/submit", response_model=SubmitResponse, status_code=202)
async def submit(req: SubmitRequest):
    """Submit a new pipeline job to Azure ML."""
    try:
        return pipeline_service.submit_pipeline(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Submit (async) ────────────────────────────────────────────

@router.post("/submit/async", status_code=202)
async def submit_async(req: SubmitRequest):
    """Enqueue a pipeline submission and return immediately with a request_id.

    Clients poll GET /submit/status/{request_id} until status='submitted',
    then switch to /jobs/{job_name} for live monitoring.
    """
    try:
        return pipeline_service.submit_pipeline_async(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/submit/status/{request_id}")
async def submit_status(request_id: str):
    """Return the current state of an async submit request."""
    record = pipeline_service.get_submit_request(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}")
    return record


# ── List ──────────────────────────────────────────────────────

@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    experiment_name: str | None = Query(None),
    status: str | None = Query(None),
    max_results: int = Query(50, ge=1, le=500),
):
    """List pipeline jobs, optionally filtered by experiment and status."""
    return pipeline_service.list_jobs(
        experiment_name=experiment_name,
        status_filter=status,
        max_results=max_results,
    )


# ── Experiment tree (hierarchical picker) ─────────────────────

@router.get("/experiments", response_model=ExperimentTreeResponse)
async def list_experiments(
    response: Response,
    max_results_per_experiment: int = Query(100, ge=1, le=500),
    force_refresh: bool = Query(False, description="Bypass warm cache and re-fetch from Azure ML"),
):
    """Return all jobs grouped by experiment for hierarchical pickers.

    Served from a startup-warmed in-memory cache when possible. The cache is
    refreshed periodically by a background task (see api.main.lifespan).
    Sets X-Cache, X-Cache-Age, and X-Cache-FetchedAt response headers.
    """
    cached, meta = pipeline_service.get_cached_experiments()

    cache_usable = (
        not force_refresh
        and cached is not None
        and meta["max_per_experiment"] == max_results_per_experiment
    )

    if cache_usable:
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Cache-Age"] = str(meta["age_seconds"] or 0)
        if meta["fetched_at"] is not None:
            response.headers["X-Cache-FetchedAt"] = meta["fetched_at"].isoformat() + "Z"
        return cached

    # Live fetch (cold cache, mismatched size, or force_refresh=true)
    data = pipeline_service.list_experiments(
        max_results_per_experiment=max_results_per_experiment,
    )
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Cache-Age"] = "0"
    return data


@router.post("/experiments/refresh", status_code=202)
async def refresh_experiments(
    max_results_per_experiment: int = Query(20, ge=1, le=500),
):
    """Trigger a background refresh of the experiments cache and return immediately."""
    _, meta = pipeline_service.get_cached_experiments()
    loop = asyncio.get_running_loop()
    # Fire-and-forget; the warm loop will continue on its own TTL afterward.
    loop.run_in_executor(
        None, pipeline_service.refresh_experiments_cache, max_results_per_experiment
    )
    return {
        "status": "refreshing",
        "previous_fetched_at": (
            meta["fetched_at"].isoformat() + "Z" if meta["fetched_at"] else None
        ),
        "previous_age_seconds": meta["age_seconds"],
    }


# ── Status ────────────────────────────────────────────────────

@router.get("/jobs/{job_name}", response_model=JobStatus)
async def get_job(job_name: str):
    """Get detailed status of a pipeline job and its child steps."""
    try:
        return pipeline_service.get_job(job_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {exc}")


# ── Cancel ────────────────────────────────────────────────────

@router.post("/jobs/{job_name}/cancel", response_model=JobStatus)
async def cancel_job(job_name: str):
    """Cancel a running pipeline job."""
    try:
        return pipeline_service.cancel_job(job_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Job not found or cannot cancel: {exc}")


# ── Outputs ───────────────────────────────────────────────────

@router.get("/jobs/{job_name}/outputs", response_model=OutputListResponse)
async def list_outputs(job_name: str):
    """List available outputs for a pipeline job."""
    try:
        return pipeline_service.list_outputs(job_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {exc}")


@router.get("/jobs/{job_name}/outputs/{output_name}/download")
async def download_output(job_name: str, output_name: str):
    """Download a specific output artifact from a pipeline job."""
    try:
        tmp = pipeline_service.download_output(job_name, output_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Output not available: {exc}")

    # Return first file found, or zip the directory
    files = list(tmp.rglob("*"))
    files = [f for f in files if f.is_file()]
    if not files:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=404, detail="No files in output")

    if len(files) == 1:
        return FileResponse(path=str(files[0]), filename=files[0].name)

    # Multiple files → zip
    zip_path = Path(f"{tmp}.zip")
    shutil.make_archive(str(tmp), "zip", str(tmp))
    return FileResponse(
        path=str(zip_path),
        filename=f"{job_name}_{output_name}.zip",
        media_type="application/zip",
    )


# ── Metrics (Phase 0a) ───────────────────────────────────────

@router.get("/jobs/{job_name}/metrics", response_model=MetricsResponse)
async def get_job_metrics(job_name: str):
    """Get per-model MLflow metrics for a pipeline job."""
    try:
        return pipeline_service.get_job_metrics(job_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Metrics not available: {exc}")


# ── Output content preview ───────────────────────────────────

@router.get(
    "/jobs/{job_name}/outputs/{output_name}/content",
    response_model=OutputContentResponse,
)
async def get_output_content(job_name: str, output_name: str):
    """Return parsed file content (JSON/CSV/text) of a named output for UI rendering."""
    try:
        return pipeline_service.get_output_content(job_name, output_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Output content not available: {exc}")


# ── Pipeline summary (combined aggregate reports) ────────────

@router.get("/jobs/{job_name}/summary", response_model=PipelineSummaryResponse)
async def get_pipeline_summary(job_name: str):
    """Return combined baseline / phaseB / phaseC / final reports for a job."""
    try:
        return pipeline_service.get_pipeline_summary(job_name)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Summary not available for '{job_name}'. The job may no longer "
                f"exist in this Azure ML workspace, or aggregate reports were not "
                f"produced. Underlying error: {exc}"
            ),
        )


# ── Drift (Phase 0b) ─────────────────────────────────────────

@router.get("/jobs/{job_name}/drift", response_model=DriftResponse)
async def get_job_drift(job_name: str):
    """Get drift detection results for a pipeline job."""
    try:
        return pipeline_service.get_job_drift(job_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Drift data not available: {exc}")


# ── Resubmit (Phase 0d) ──────────────────────────────────────

@router.post("/resubmit", response_model=SubmitResponse, status_code=202)
async def resubmit(req: ResubmitRequest):
    """Resubmit a pipeline job using the same configuration."""
    try:
        return pipeline_service.resubmit_pipeline(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Baseline capture (Phase 0c) ──────────────────────────────

@router.post("/baseline/capture", response_model=BaselineCaptureResponse)
async def capture_baseline(req: BaselineCaptureRequest):
    """Extract drift baseline artifacts from a completed job."""
    try:
        return pipeline_service.capture_baseline(req)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Baseline capture failed: {exc}")
