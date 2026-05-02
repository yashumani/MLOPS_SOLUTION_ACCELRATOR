"""Pipeline submission, monitoring, and output retrieval service."""

import logging
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from azure.ai.ml import Input, MLClient
try:
    from azure.ai.ml.constants import ListViewType
except Exception:  # pragma: no cover - older SDKs may not expose this enum
    ListViewType = None  # type: ignore[assignment]

from api.core.azure_ml import get_ml_client
from api.core.config import settings
from api.schemas.pipeline import (
    BaselineCaptureRequest,
    BaselineCaptureResponse,
    DriftResponse,
    DriftResultItem,
    ExperimentNode,
    ExperimentTreeResponse,
    JobListResponse,
    JobStatus,
    JobSummary,
    LocalOutputFileInfo,
    LocalOutputsResponse,
    MetricsResponse,
    ModelMetric,
    OutputContentResponse,
    OutputFileInfo,
    OutputInfo,
    OutputListResponse,
    PipelineSummaryResponse,
    ResubmitRequest,
    StepStatus,
    SubmitRequest,
    SubmitResponse,
)
from api.utils.azure_links import build_studio_url

# Repo root so we can import pipeline_builder and read configs
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs"
_LOCAL_OUTPUTS_DIR = _REPO_ROOT / "outputs"
_SAFE_CONFIG_NAME = re.compile(r"^[A-Za-z0-9_]+$")

# In-memory cache for terminal-state job metrics/summaries.
# Outputs are immutable once a job is Completed/Failed/Canceled.
# Key: (function_name, job_name); Value: response object.
_TERMINAL_STATES = {"Completed", "Failed", "Canceled", "CancelRequested"}
_response_cache: dict[tuple[str, str], object] = {}

# Warm cache for the experiment→jobs tree, populated at startup and refreshed
# periodically by the background warmer in api.main.lifespan.
_logger = logging.getLogger(__name__)
_experiments_cache: dict = {
    "data": None,            # ExperimentTreeResponse | None
    "fetched_at": None,      # datetime | None (UTC)
    "max_per_experiment": None,  # int | None
    "duration_s": None,      # float | None
}
_experiments_lock = threading.Lock()

_STAGE_ALIASES = {
    "s01": "s1",
    "s02": "s2",
    "s03": "s3",
    "s04": "s4",
    "s05a": "s5a",
    "s05b": "s5b",
    "s05t": "s5t",
    "s05z": "s5z",
    "s6": "s06",
    "s08": "s08",
    "s8": "s08",
    "s09": "s09",
    "s9": "s09",
}

_STAGE_KEYWORDS = {
    "ingestion": "s1",
    "preparation": "s2",
    "preprocessing": "s3",
    "feature_engineering": "s4",
    "pycaret": "s5a",
    "flaml": "s5b",
    "timeseries": "s5t",
    "forecasting": "s5t",
    "aggregate_baseline": "s5z",
    "baseline_aggregate": "s5z",
    "variant_runner": "s06",
    "phaseb": "s06",
    "phasec_hpo": "s08",
    "optuna": "s08",
    "aggregate_phasec": "s09",
    "phasec_aggregate": "s09",
    "final_evaluation": "s10",
    "model_registration": "s12",
    "drift_monitor": "s13",
    "drift": "s13",
}

_CANONICAL_STAGE_LABELS = {
    "s1": "S01 Ingestion",
    "s2": "S02 Preparation",
    "s3": "S03 Preprocessing",
    "s4": "S04 Feature Engineering",
    "s5a": "S05a PyCaret Baseline",
    "s5b": "S05b FLAML Baseline",
    "s5t": "S05t Time-Series Baseline",
    "s5z": "S05z Baseline Aggregate",
    "s06": "S06 Phase B Variant Runner",
    "s08": "S08 Phase C Optuna HPO",
    "s09": "S09 Phase C Aggregate",
    "s10": "S10 Final Evaluation",
    "s12": "S12 Model Registration",
    "s13": "S13 Drift Monitor",
}

_OUTPUT_STAGE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "eda_report": ("s1",),
    "prep_report": ("s2",),
    "prep3_report": ("s3",),
    "fe_report": ("s4",),
    "baseline_pycaret_metrics": ("s5a",),
    "baseline_flaml_metrics": ("s5b",),
    "baseline_aggregate_report": ("s5z",),
    "baseline_champion_model": ("s5z",),
    "phaseb_leaderboard": ("s06",),
    "phaseb_all_results": ("s06",),
    "phaseb_champion_manifest": ("s06",),
    "phaseb_champion_model": ("s06",),
    # s09 depends on s08, so this output confirms both for historical jobs
    # where Azure ML child step metadata is missing.
    "phasec_aggregate_report": ("s08", "s09"),
    "phasec_champion_model": ("s09",),
    "final_report": ("s10",),
    "final_champion_model": ("s10",),
    "registry_info": ("s12",),
    "drift_report": ("s13",),
    "drift_baseline": ("s13",),
}


def _infer_stage_key(*names: str | None) -> str | None:
    """Infer the canonical DSL stage key from Azure child job identifiers."""
    for raw in names:
        text = (raw or "").lower()
        if not text:
            continue
        for token in re.findall(r"s\d{1,2}[a-z]?", text):
            normalized = _STAGE_ALIASES.get(token, token)
            if normalized in {
                "s1", "s2", "s3", "s4", "s5a", "s5b", "s5t",
                "s5z", "s06", "s08", "s09", "s10", "s12", "s13",
            }:
                return normalized
        compact = text.replace("-", "_").replace(" ", "_")
        for keyword, stage_key in _STAGE_KEYWORDS.items():
            if keyword in compact:
                return stage_key
    return None


def _stage_display_name(stage_key: str | None, fallback: str | None = None) -> str | None:
    if stage_key:
        return _CANONICAL_STAGE_LABELS.get(stage_key, fallback or stage_key)
    return fallback


def _append_inferred_steps_from_outputs(
    steps: list[StepStatus],
    outputs: dict[str, Any],
) -> None:
    """Backfill completed stages from parent named outputs for old job records."""
    seen = {step.stage_key for step in steps if step.stage_key}
    for output_name, stage_keys in _OUTPUT_STAGE_FALLBACKS.items():
        if output_name not in outputs:
            continue
        for stage_key in stage_keys:
            if stage_key in seen:
                continue
            steps.append(
                StepStatus(
                    name=f"{stage_key}_inferred_from_{output_name}",
                    display_name=(
                        f"{_stage_display_name(stage_key)} "
                        f"(inferred from {output_name})"
                    ),
                    stage_key=stage_key,
                    status="Completed",
                    is_inferred=True,
                )
            )
            seen.add(stage_key)


def refresh_experiments_cache(max_per_experiment: int) -> "ExperimentTreeResponse":
    """Re-fetch the experiment tree and atomically update the warm cache.

    Returns the freshly fetched ExperimentTreeResponse. Safe to call from
    a thread executor; uses an internal lock so concurrent refreshes coalesce.
    """
    t0 = time.monotonic()
    data = list_experiments(max_results_per_experiment=max_per_experiment)
    duration = time.monotonic() - t0
    with _experiments_lock:
        _experiments_cache["data"] = data
        _experiments_cache["fetched_at"] = datetime.utcnow()
        _experiments_cache["max_per_experiment"] = max_per_experiment
        _experiments_cache["duration_s"] = duration
    _logger.info(
        "warmed experiments cache: %d exp / %d jobs in %.1fs (max_per_experiment=%d)",
        data.total_experiments,
        data.total_jobs,
        duration,
        max_per_experiment,
    )
    return data


def get_cached_experiments() -> tuple["ExperimentTreeResponse | None", dict]:
    """Return (data, meta) from the warm cache without triggering a refresh.

    meta keys: fetched_at (datetime|None), age_seconds (int|None),
    max_per_experiment (int|None), duration_s (float|None).
    """
    with _experiments_lock:
        data = _experiments_cache["data"]
        fetched_at = _experiments_cache["fetched_at"]
        max_per_experiment = _experiments_cache["max_per_experiment"]
        duration_s = _experiments_cache["duration_s"]
    age = None
    if fetched_at is not None:
        age = int((datetime.utcnow() - fetched_at).total_seconds())
    return data, {
        "fetched_at": fetched_at,
        "age_seconds": age,
        "max_per_experiment": max_per_experiment,
        "duration_s": duration_s,
    }


def _derive_experiment_name(config_name: str) -> str:
    """Derive reusable experiment name from config filename.

    config_classification_telecom_churn_azureml → classification_telecom_churn_v3
    """
    normalized = config_name.replace("config_", "").replace("_azureml", "").replace("_local", "")
    return f"{normalized}_v3"


def _derive_display_name(experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{experiment_name}_{timestamp}_{unique_id}"


# _studio_url removed — use build_studio_url(ml_client, job_name) instead


def _load_config_yaml(config_name: str) -> dict:
    if not _SAFE_CONFIG_NAME.fullmatch(config_name):
        raise ValueError(f"Invalid config name: {config_name!r}")
    path = (_CONFIGS_DIR / f"{config_name}.yml").resolve()
    if _CONFIGS_DIR.resolve() not in path.parents:
        raise ValueError(f"Invalid config path for: {config_name!r}")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_name}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_name}")
    try:
        from src.orchestration.config_schema import validate_config

        validate_config(cfg)
    except Exception as exc:
        raise ValueError(f"Config validation failed for {config_name}: {exc}") from exc
    return cfg


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def submit_pipeline(req: SubmitRequest) -> SubmitResponse:
    """Submit an Azure ML pipeline job mirroring submit_pipeline.py logic."""
    ml_client = get_ml_client()
    cfg = _load_config_yaml(req.config_name)

    # Ensure pipeline_builder is importable
    pipelines_dir = str(_REPO_ROOT / "pipelines")
    if pipelines_dir not in sys.path:
        sys.path.insert(0, pipelines_dir)
    from pipeline_builder import full_pipeline  # type: ignore[import-untyped]

    datastore_name = (cfg.get("dataset") or {}).get("datastore_name", "mlops_blob")
    dataset_folder_uri = (
        f"azureml://subscriptions/{settings.azure_subscription_id}"
        f"/resourcegroups/{settings.azure_resource_group}"
        f"/workspaces/{settings.azure_workspace_name}"
        f"/datastores/{datastore_name}/paths/"
    )

    config_filename = f"{req.config_name}.yml"
    pipeline_kwargs: dict = dict(
        config_name=config_filename,
        dataset_folder=Input(path=dataset_folder_uri, type="uri_folder"),
    )

    # Optional drift baseline
    if req.baseline_job:
        try:
            baseline_job_obj = ml_client.jobs.get(req.baseline_job)
            outputs = baseline_job_obj.outputs or {}
            if "drift_baseline" in outputs:
                asset_id = getattr(outputs["drift_baseline"], "path", None)
                if asset_id:
                    pipeline_kwargs["drift_baseline_in"] = Input(path=asset_id, type="uri_folder")
        except Exception:
            pass  # proceed without baseline

    job = full_pipeline(**pipeline_kwargs)

    compute = req.compute or settings.compute_target
    job.settings.default_compute = compute
    if req.force_rerun:
        job.settings.force_rerun = True

    experiment_name = _derive_experiment_name(req.config_name)
    display_name = _derive_display_name(experiment_name)
    job.experiment_name = experiment_name
    job.display_name = display_name

    # Tags
    dataset_meta = cfg.get("dataset") or {}
    tags = {
        "dataset": dataset_meta.get("name", "unknown"),
        "task": cfg.get("task_type", "unknown"),
        "pipeline_version": "v3",
        "source": "api",
    }
    tags.update(req.tags)
    job.tags = tags

    submitted = ml_client.jobs.create_or_update(job)

    return SubmitResponse(
        job_name=submitted.name,
        experiment_name=experiment_name,
        display_name=display_name,
        status=submitted.status or "Submitted",
        studio_url=build_studio_url(ml_client, submitted.name),
    )


# ---------------------------------------------------------------------------
# Async submit (Phase 4) — fire-and-poll request table
# ---------------------------------------------------------------------------

# Azure ML job creation can take 5–60s wall time on cold compute. The async
# submit endpoint hands the work off to a background thread so the HTTP request
# returns in <100ms. Clients poll /submit/status/{request_id} until job_name
# appears, then switch to the standard /pipelines/jobs/{job_name} endpoints.

_submit_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="submit-async")
_submit_requests: dict[str, dict] = {}
_submit_lock = threading.Lock()


def _new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


def _submit_worker(request_id: str, req: SubmitRequest) -> None:
    """Background worker that performs the blocking Azure ML submission."""
    try:
        result = submit_pipeline(req)
        with _submit_lock:
            _submit_requests[request_id].update(
                {
                    "status": "submitted",
                    "job_name": result.job_name,
                    "experiment_name": result.experiment_name,
                    "display_name": result.display_name,
                    "studio_url": result.studio_url,
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                }
            )
    except Exception as exc:  # noqa: BLE001 — record any failure for the poller
        with _submit_lock:
            _submit_requests[request_id].update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                }
            )


def submit_pipeline_async(req: SubmitRequest) -> dict:
    """Enqueue a pipeline submission and return immediately with a request_id."""
    request_id = _new_request_id()
    record = {
        "request_id": request_id,
        "status": "pending",
        "config_name": req.config_name,
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "job_name": None,
        "experiment_name": None,
        "display_name": None,
        "studio_url": None,
        "error": None,
        "completed_at": None,
    }
    with _submit_lock:
        _submit_requests[request_id] = record
    _submit_executor.submit(_submit_worker, request_id, req)
    return record


def get_submit_request(request_id: str) -> dict | None:
    """Return a snapshot of a submit request's state, or None if unknown."""
    with _submit_lock:
        record = _submit_requests.get(request_id)
        return dict(record) if record else None


# ---------------------------------------------------------------------------
# List / Get / Cancel
# ---------------------------------------------------------------------------


def list_jobs(
    experiment_name: str | None = None,
    status_filter: str | None = None,
    max_results: int = 50,
) -> JobListResponse:
    ml_client = get_ml_client()

    jobs: list[JobSummary] = []
    for j in ml_client.jobs.list():
        if experiment_name and getattr(j, "experiment_name", None) != experiment_name:
            continue
        if status_filter and (j.status or "").lower() != status_filter.lower():
            continue
        jobs.append(
            JobSummary(
                job_name=j.name,
                experiment_name=getattr(j, "experiment_name", None),
                display_name=getattr(j, "display_name", None),
                status=j.status or "Unknown",
                start_time=getattr(j, "creation_context", None)
                and getattr(j.creation_context, "created_at", None),
                studio_url=build_studio_url(ml_client, j.name),
            )
        )
        if len(jobs) >= max_results:
            break

    return JobListResponse(jobs=jobs, total=len(jobs))


def list_experiments(max_results_per_experiment: int = 100) -> ExperimentTreeResponse:
    """Return all jobs grouped by experiment for hierarchical UI pickers.

    Each experiment node lists its jobs (most recent first) so the UI can
    render a two-level tree: experiment → display_name (job).
    """
    ml_client = get_ml_client()

    # Single pass over the workspace job list, group by experiment.
    grouped: dict[str, list[JobSummary]] = {}
    last_activity: dict[str, datetime] = {}

    for j in ml_client.jobs.list():
        exp = getattr(j, "experiment_name", None) or "(no experiment)"
        if len(grouped.get(exp, [])) >= max_results_per_experiment:
            continue
        created = (
            getattr(j, "creation_context", None)
            and getattr(j.creation_context, "created_at", None)
        )
        summary = JobSummary(
            job_name=j.name,
            experiment_name=exp,
            display_name=getattr(j, "display_name", None) or j.name,
            status=j.status or "Unknown",
            start_time=created,
            studio_url=build_studio_url(ml_client, j.name),
        )
        grouped.setdefault(exp, []).append(summary)
        if created and (exp not in last_activity or created > last_activity[exp]):
            last_activity[exp] = created

    # Sort jobs in each experiment by start_time desc (most recent first)
    nodes: list[ExperimentNode] = []
    for exp, jobs in grouped.items():
        jobs.sort(key=lambda x: x.start_time or datetime.min, reverse=True)
        nodes.append(
            ExperimentNode(
                experiment_name=exp,
                job_count=len(jobs),
                last_activity=last_activity.get(exp),
                jobs=jobs,
            )
        )
    nodes.sort(key=lambda n: n.last_activity or datetime.min, reverse=True)

    return ExperimentTreeResponse(
        experiments=nodes,
        total_experiments=len(nodes),
        total_jobs=sum(n.job_count for n in nodes),
    )


def get_job(job_name: str) -> JobStatus:
    ml_client = get_ml_client()
    j = ml_client.jobs.get(job_name)

    # Fetch child step statuses
    steps: list[StepStatus] = []
    try:
        list_kwargs: dict[str, Any] = {"parent_job_name": job_name}
        if ListViewType is not None:
            list_kwargs["list_view_type"] = ListViewType.ALL
        for child in ml_client.jobs.list(**list_kwargs):
            child_name = child.name
            azure_display_name = getattr(child, "display_name", None)
            properties = getattr(child, "properties", None) or {}
            module_name = properties.get("azureml.moduleName")
            stage_key = _infer_stage_key(azure_display_name, child_name, module_name)
            steps.append(
                StepStatus(
                    name=child_name,
                    display_name=_stage_display_name(stage_key, azure_display_name),
                    stage_key=stage_key,
                    status=child.status or "Unknown",
                    start_time=getattr(child, "creation_context", None)
                    and getattr(child.creation_context, "created_at", None),
                    end_time=None,
                )
            )
    except Exception:
        _logger.exception("failed to list child steps for job %s", job_name)

    _append_inferred_steps_from_outputs(steps, j.outputs or {})

    return JobStatus(
        job_name=j.name,
        experiment_name=getattr(j, "experiment_name", None),
        display_name=getattr(j, "display_name", None),
        status=j.status or "Unknown",
        start_time=getattr(j, "creation_context", None)
        and getattr(j.creation_context, "created_at", None),
        end_time=None,
        studio_url=build_studio_url(ml_client, j.name),
        tags=dict(j.tags) if j.tags else {},
        steps=steps,
    )


def cancel_job(job_name: str) -> JobStatus:
    ml_client = get_ml_client()
    ml_client.jobs.cancel(job_name)
    return get_job(job_name)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def list_outputs(job_name: str) -> OutputListResponse:
    ml_client = get_ml_client()
    j = ml_client.jobs.get(job_name)
    outputs_dict = j.outputs or {}
    items = [
        OutputInfo(name=k, type=getattr(v, "type", None))
        for k, v in outputs_dict.items()
    ]
    return OutputListResponse(job_name=job_name, outputs=items)


def download_output(job_name: str, output_name: str) -> Path:
    """Download a specific job output to a temp directory and return the path."""
    ml_client = get_ml_client()
    tmp = Path(tempfile.mkdtemp(prefix=f"mlops_output_{job_name}_"))
    ml_client.jobs.download(job_name, download_path=str(tmp), output_name=output_name)
    return tmp


# ---------------------------------------------------------------------------
# Output content preview (for UI rendering)
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {
    ".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md", ".html",
    ".yml", ".yaml", ".py", ".ipynb",
}
_BINARY_EXTENSIONS = {".pkl", ".joblib", ".bin", ".onnx", ".pt", ".pb", ".h5", ".parquet"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

_TEXT_PREVIEW_BYTES = 200_000  # ~200KB cap
_CSV_PREVIEW_ROWS = 200


def _classify_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".json" or ext == ".jsonl":
        return "json"
    if ext == ".csv" or ext == ".tsv":
        return "csv"
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in {".html", ".htm"}:
        return "html"
    if ext in {".md", ".markdown"}:
        return "markdown"
    if ext in {".yml", ".yaml"}:
        return "yaml"
    if ext in _BINARY_EXTENSIONS:
        return "binary"
    if ext in _TEXT_EXTENSIONS:
        return "text"
    # No / unknown extension — sniff the first bytes
    if not ext:
        try:
            with open(path, "rb") as fh:
                head = fh.read(2048)
            stripped = head.lstrip()
            if stripped[:1] in (b"{", b"["):
                return "json"
            # ASCII-printable test
            if head and sum(1 for b in head if 9 <= b <= 13 or 32 <= b <= 126) / len(head) > 0.85:
                return "text"
        except OSError:
            pass
    return "binary"


def get_output_content(job_name: str, output_name: str) -> OutputContentResponse:
    """Download an output and return its parsed content for UI rendering.

    Picks the most useful file in the output and returns:
      - json_content for JSON files
      - csv_preview (first N rows as list of dicts) for CSV
      - text_preview (truncated) for text/yaml/html/md
    """
    import csv
    import json

    ml_client = get_ml_client()
    tmp = Path(tempfile.mkdtemp(prefix=f"mlops_preview_{job_name}_"))
    files_info: list[OutputFileInfo] = []
    json_content: Any | None = None
    text_preview: str | None = None
    csv_preview: list[dict] | None = None
    primary_file: str | None = None
    truncated = False

    try:
        ml_client.jobs.download(
            job_name, download_path=str(tmp), output_name=output_name
        )
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Failed to download output '{output_name}': {exc}")

    try:
        # Collect file metadata
        all_files = sorted(
            [f for f in tmp.rglob("*") if f.is_file()],
            key=lambda p: (p.suffix.lower() != ".json", p.suffix.lower() != ".csv", str(p)),
        )

        for f in all_files:
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            files_info.append(OutputFileInfo(
                name=f.name,
                relative_path=str(f.relative_to(tmp)),
                size_bytes=size,
                kind=_classify_file(f),
            ))

        # Pick primary file: prefer JSON, then CSV, then text
        primary: Path | None = None
        for kind in ("json", "csv", "yaml", "markdown", "html", "text"):
            for f in all_files:
                if _classify_file(f) == kind:
                    primary = f
                    break
            if primary:
                break

        if primary:
            primary_file = str(primary.relative_to(tmp))
            kind = _classify_file(primary)
            try:
                if kind == "json":
                    if primary.suffix.lower() == ".jsonl":
                        rows = []
                        with open(primary) as fh:
                            for i, line in enumerate(fh):
                                if i >= _CSV_PREVIEW_ROWS:
                                    truncated = True
                                    break
                                line = line.strip()
                                if line:
                                    try:
                                        rows.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        pass
                        json_content = rows
                    else:
                        with open(primary) as fh:
                            json_content = json.load(fh)
                elif kind == "csv":
                    delim = "\t" if primary.suffix.lower() == ".tsv" else ","
                    rows: list[dict] = []
                    with open(primary, newline="") as fh:
                        reader = csv.DictReader(fh, delimiter=delim)
                        for i, row in enumerate(reader):
                            if i >= _CSV_PREVIEW_ROWS:
                                truncated = True
                                break
                            rows.append(row)
                    csv_preview = rows
                elif kind in ("text", "yaml", "markdown", "html"):
                    raw = primary.read_bytes()
                    if len(raw) > _TEXT_PREVIEW_BYTES:
                        raw = raw[:_TEXT_PREVIEW_BYTES]
                        truncated = True
                    text_preview = raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — preview is best-effort
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return OutputContentResponse(
        job_name=job_name,
        output_name=output_name,
        files=files_info,
        json_content=json_content,
        text_preview=text_preview,
        csv_preview=csv_preview,
        primary_file=primary_file,
        truncated=truncated,
    )


def list_local_outputs(max_depth: int = 4, max_files: int = 500) -> LocalOutputsResponse:
    """Return a read-only inventory of the repo-local outputs/ folder."""
    max_depth = max(1, min(max_depth, 10))
    max_files = max(1, min(max_files, 2_000))

    if not _LOCAL_OUTPUTS_DIR.exists():
        return LocalOutputsResponse(files=[], total=0, truncated=False)

    items: list[LocalOutputFileInfo] = []
    truncated = False
    base = _LOCAL_OUTPUTS_DIR.resolve()

    for path in sorted(base.rglob("*"), key=lambda p: str(p).lower()):
        try:
            rel = path.relative_to(base)
        except ValueError:
            continue
        if any(part.startswith("__pycache__") for part in rel.parts):
            continue
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        if len(items) >= max_files:
            truncated = True
            break
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            LocalOutputFileInfo(
                relative_path=str(rel),
                name=path.name,
                is_dir=path.is_dir(),
                size_bytes=None if path.is_dir() else stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                kind="directory" if path.is_dir() else _classify_file(path),
                depth=depth,
            )
        )

    return LocalOutputsResponse(
        files=items,
        total=len(items),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Pipeline summary (combined aggregate reports)
# ---------------------------------------------------------------------------


def get_pipeline_summary(job_name: str) -> PipelineSummaryResponse:
    """Return all four aggregate JSON reports for a job in one structured payload."""
    ml_client = get_ml_client()
    parent = ml_client.jobs.get(job_name)
    task_type = (parent.tags or {}).get("task")
    status = getattr(parent, "status", None)
    available = list((parent.outputs or {}).keys())

    cache_key = ("summary", job_name)
    if str(status or "") in _TERMINAL_STATES and cache_key in _response_cache:
        return _response_cache[cache_key]  # type: ignore[return-value]

    report_names = [
        "baseline_aggregate_report",
        "phaseb_aggregate_report",
        "phasec_aggregate_report",
        "final_report",
    ]

    reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        def _fetch(output_name: str) -> tuple[str, Any]:
            if output_name not in available:
                return output_name, None
            target = tmp_root / output_name
            try:
                ml_client.jobs.download(
                    name=job_name, output_name=output_name, download_path=str(target)
                )
                return output_name, _read_json_report(target)
            except Exception:
                return output_name, None

        with ThreadPoolExecutor(max_workers=4) as pool:
            for name, data in pool.map(_fetch, report_names):
                reports[name] = data

    final = reports.get("final_report") or {}
    selection = final.get("selection") or {}

    response = PipelineSummaryResponse(
        job_name=job_name,
        task_type=task_type,
        status=status,
        champion_phase=selection.get("key"),
        champion_score=selection.get("score"),
        baseline_aggregate=reports.get("baseline_aggregate_report"),
        phaseb_aggregate=reports.get("phaseb_aggregate_report"),
        phasec_aggregate=reports.get("phasec_aggregate_report"),
        final_report=final or None,
        available_outputs=available,
    )
    if str(status or "") in _TERMINAL_STATES:
        _response_cache[cache_key] = response
    return response


# ---------------------------------------------------------------------------
# Metrics (Phase 0a)
# ---------------------------------------------------------------------------


def _read_json_report(download_dir: Path) -> dict | None:
    """Find and load the first JSON file under download_dir."""
    import json

    if not download_dir.exists():
        return None
    for f in download_dir.rglob("*"):
        if f.is_file():
            try:
                with open(f) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                continue
    return None


def get_job_metrics(job_name: str) -> MetricsResponse:
    """Retrieve per-phase metrics from the pipeline's aggregate JSON reports.

    The V3 pipeline writes per-phase scores to named outputs:
      - baseline_aggregate_report (phase A)
      - phaseb_aggregate_report (phase B)
      - phasec_aggregate_report (phase C)
      - final_report (holdout metrics per phase + champion selection)
    """
    ml_client = get_ml_client()
    parent = ml_client.jobs.get(job_name)
    task_type = (parent.tags or {}).get("task", None)
    job_status = str(getattr(parent, "status", "") or "")

    cache_key = ("metrics", job_name)
    if job_status in _TERMINAL_STATES and cache_key in _response_cache:
        return _response_cache[cache_key]  # type: ignore[return-value]

    models: list[ModelMetric] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        def _download(output_name: str) -> dict | None:
            target = tmp_root / output_name
            try:
                ml_client.jobs.download(
                    name=job_name, output_name=output_name, download_path=str(target)
                )
            except Exception:
                return None
            return _read_json_report(target)

        # Parallel download: 4 named outputs concurrently (~4x faster)
        names = [
            "baseline_aggregate_report",
            "phaseb_aggregate_report",
            "phasec_aggregate_report",
            "final_report",
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_download, names))
        baseline_agg = results[0] or {}
        phaseb_agg = results[1] or {}
        phasec_agg = results[2] or {}
        final_rep = results[3] or {}

        champion_phase = (final_rep.get("selection") or {}).get("key")

        # Phase A (baseline)
        baseline_metrics = final_rep.get("baseline_metrics") or {}
        baseline_engine = (baseline_agg.get("selection") or {}).get("source")
        if baseline_metrics or baseline_agg:
            score = (baseline_agg.get("selection") or {}).get("score")
            if score is not None and "score" not in baseline_metrics:
                baseline_metrics = {**baseline_metrics, "score": float(score)}
            models.append(
                ModelMetric(
                    model_name=f"baseline ({baseline_engine})" if baseline_engine else "baseline",
                    engine=baseline_engine,
                    phase="baseline",
                    metrics={k: float(v) for k, v in baseline_metrics.items()
                             if isinstance(v, (int, float))},
                    is_champion=(champion_phase == "baseline"),
                )
            )

        # Phase B
        phaseb_metrics = final_rep.get("phaseb_metrics") or {}
        phaseb_sel = (phaseb_agg.get("selection") or {})
        phaseb_key = phaseb_sel.get("key", "")
        phaseb_engine = None
        if "_" in phaseb_key:
            phaseb_engine = phaseb_key.rsplit("_", 1)[-1]
        if phaseb_metrics or phaseb_agg:
            if phaseb_sel.get("score") is not None and "score" not in phaseb_metrics:
                phaseb_metrics = {**phaseb_metrics, "score": float(phaseb_sel["score"])}
            models.append(
                ModelMetric(
                    model_name=f"phaseB ({phaseb_key})" if phaseb_key else "phaseB",
                    engine=phaseb_engine,
                    phase="phaseB",
                    metrics={k: float(v) for k, v in phaseb_metrics.items()
                             if isinstance(v, (int, float))},
                    is_champion=(champion_phase == "phaseb"),
                )
            )

        # Phase C
        phasec_metrics = final_rep.get("phasec_metrics") or {}
        if phasec_metrics or phasec_agg:
            phasec_sel = (phasec_agg.get("selection") or {})
            if phasec_sel.get("score") is not None and "score" not in phasec_metrics:
                phasec_metrics = {**phasec_metrics, "score": float(phasec_sel["score"])}
            models.append(
                ModelMetric(
                    model_name=f"phaseC ({phasec_sel.get('key','hpo')})",
                    engine=None,
                    phase="phaseC",
                    metrics={k: float(v) for k, v in phasec_metrics.items()
                             if isinstance(v, (int, float))},
                    is_champion=(champion_phase == "phasec"),
                )
            )

    response = MetricsResponse(job_name=job_name, task_type=task_type, models=models)
    if job_status in _TERMINAL_STATES:
        _response_cache[cache_key] = response
    return response


# ---------------------------------------------------------------------------
# Drift (Phase 0b)
# ---------------------------------------------------------------------------


def get_job_drift(job_name: str) -> DriftResponse:
    """Retrieve drift detection results from a completed pipeline job.

    Parses the actual ``s13_drift_monitor`` JSON schema:

      {
        "self_check": {"overall_psi": ..., "drifted_features": [...], "status": "PASS|WARN"},
        "feature_psi_scores": {feature: psi},
        "stability_assessment": {"stability_score": ..., "recommended_cadence": ...},
        "comparison_drift": {...}                # only when baseline_in provided
      }
    """
    import json

    ml_client = get_ml_client()
    parent = ml_client.jobs.get(job_name)
    available = list((parent.outputs or {}).keys())

    if "drift_report" not in available:
        # Output not yet produced (job still running, failed before s13, or
        # this pipeline doesn't include drift).
        return DriftResponse(
            job_name=job_name,
            studio_url=build_studio_url(ml_client, job_name),
        )

    cache_key = ("drift", job_name)
    status = str(getattr(parent, "status", "") or "")
    if status in _TERMINAL_STATES and cache_key in _response_cache:
        return _response_cache[cache_key]  # type: ignore[return-value]

    tmp = Path(tempfile.mkdtemp(prefix=f"mlops_drift_{job_name}_"))
    try:
        try:
            ml_client.jobs.download(
                job_name, download_path=str(tmp), output_name="drift_report"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download drift_report: {exc}")

        # Producer writes to args.drift_report path directly (no extension);
        # try *.json first, then any file with parseable JSON content.
        report: dict | None = None
        candidates = list(tmp.rglob("*.json")) + [
            p for p in tmp.rglob("*") if p.is_file() and p.suffix == ""
        ]
        for f in candidates:
            try:
                report = json.loads(f.read_text())
                if isinstance(report, dict):
                    break
                report = None
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue

        if not report:
            _logger.warning(
                "drift_report download had no parseable JSON; files=%s",
                [str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file()],
            )
            return DriftResponse(
                job_name=job_name,
                studio_url=build_studio_url(ml_client, job_name),
            )

        # ── Parse actual s13 schema (see src/steps/s13_drift_monitor.py) ──
        self_check = report.get("self_check") or {}
        # Fallback chain in case key naming evolves.
        psi_scores: dict = (
            report.get("feature_psi_scores")
            or report.get("psi_scores")
            or report.get("psi_per_feature")
            or {}
        )
        stability = report.get("stability_assessment") or {}
        comparison = report.get("comparison_drift") or {}

        if not psi_scores:
            _logger.warning(
                "drift_report missing per-feature PSI; top-level keys=%s",
                list(report.keys()),
            )

        features: list[DriftResultItem] = []
        for feat_name, raw in psi_scores.items():
            # Per-feature value may be a float OR a nested dict like
            # {"psi": 0.05, "n_bins": 10, ...}; handle both.
            if isinstance(raw, dict):
                raw = raw.get("psi") or raw.get("score") or raw.get("value")
            try:
                psi_v = float(raw)
            except (TypeError, ValueError):
                continue
            severity = (
                "severe" if psi_v >= 0.25
                else "moderate" if psi_v >= 0.10
                else "none"
            )
            features.append(
                DriftResultItem(
                    feature=str(feat_name),
                    psi=psi_v,
                    drift_detected=psi_v >= 0.10,
                    severity=severity,
                )
            )

        # Sort by PSI descending (worst drift first)
        features.sort(key=lambda x: x.psi, reverse=True)

        # Comparison drift block exists in every report with at least
        # {"available": false}; treat it as present only when "available" is True.
        comparison_available = bool(comparison.get("available"))
        evidently = comparison.get("evidently") or {}
        concept = comparison.get("concept_drift") or {}

        # Overall drift indicator: prefer comparison drift signals when a
        # previous baseline was supplied; otherwise fall back to self-check
        # status, with per-feature PSI threshold as a final fallback.
        if comparison_available:
            overall = bool(
                evidently.get("dataset_drift")
                or concept.get("detected")
            )
            drift_type = "comparison"
        elif self_check:
            overall = self_check.get("status") == "WARN"
            drift_type = "self_check"
        else:
            overall = any(f.drift_detected for f in features)
            drift_type = "psi"

        # drifted_columns must be list[str] per DriftResponse schema. Try
        # several shapes: Evidently's drift_by_columns list (dicts with
        # "column"), self_check.drifted_features (dicts with "feature"), or
        # simply derive from per-feature PSI.
        drifted_cols: list[str] = []
        ev_cols = evidently.get("drifted_columns") or []
        if isinstance(ev_cols, list) and ev_cols:
            for item in ev_cols:
                if isinstance(item, dict):
                    name = item.get("column") or item.get("feature")
                    if name:
                        drifted_cols.append(str(name))
                elif isinstance(item, str):
                    drifted_cols.append(item)
        if not drifted_cols:
            sc_cols = self_check.get("drifted_features") or []
            if isinstance(sc_cols, list):
                for item in sc_cols:
                    if isinstance(item, dict):
                        name = item.get("feature") or item.get("column")
                        if name:
                            drifted_cols.append(str(name))
                    elif isinstance(item, str):
                        drifted_cols.append(item)
        if not drifted_cols:
            drifted_cols = [f.feature for f in features if f.drift_detected]

        response = DriftResponse(
            job_name=job_name,
            overall_drift_detected=overall,
            stability_score=stability.get("stability_score"),
            drift_type=drift_type,
            drifted_columns=drifted_cols,
            features=features,
            evidently_report_path=report.get("evidently_report_path"),
            studio_url=build_studio_url(ml_client, job_name),
        )
        if status in _TERMINAL_STATES:
            _response_cache[cache_key] = response
        return response
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Resubmit (Phase 0d)
# ---------------------------------------------------------------------------


def resubmit_pipeline(req: ResubmitRequest) -> SubmitResponse:
    """Resubmit a pipeline job using the same configuration as the original."""
    ml_client = get_ml_client()
    original = ml_client.jobs.get(req.job_name)
    tags = dict(original.tags) if original.tags else {}

    config_name = tags.get("config_name", "")
    if not config_name:
        # Infer from experiment name
        exp = getattr(original, "experiment_name", "") or ""
        config_name = f"config_{exp.replace('_v3', '')}_azureml"

    submit_req = SubmitRequest(
        config_name=config_name,
        force_rerun=req.force_rerun,
        tags={"resubmit_from": req.job_name},
    )
    return submit_pipeline(submit_req)


# ---------------------------------------------------------------------------
# Baseline capture (Phase 0c)
# ---------------------------------------------------------------------------


def capture_baseline(req: BaselineCaptureRequest) -> BaselineCaptureResponse:
    """Extract drift baseline artifacts from a completed pipeline job."""
    ml_client = get_ml_client()
    j = ml_client.jobs.get(req.job_name)

    baseline_path: str | None = None
    outputs = j.outputs or {}
    if "drift_baseline" in outputs:
        baseline_path = getattr(outputs["drift_baseline"], "path", None)

    return BaselineCaptureResponse(
        job_name=req.job_name,
        baseline_path=baseline_path,
        status="captured" if baseline_path else "no_baseline_output",
        studio_url=build_studio_url(ml_client, req.job_name),
    )
