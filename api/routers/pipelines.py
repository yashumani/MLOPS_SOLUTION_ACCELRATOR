"""Pipeline submission, monitoring, cancel, and output endpoints."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.core.security import verify_api_key
from api.schemas.pipeline import (
    JobListResponse,
    JobStatus,
    OutputListResponse,
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
