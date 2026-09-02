"""Config listing and parsing service."""

import re
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator

from api.schemas.config import (
    ConfigDetail,
    ConfigPreviewResponse,
    ConfigStagePreview,
    ConfigSummary,
    ConfigValidationIssue,
    ConfigValidationResponse,
)

# Repo root: api/ is one level below the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs"

_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_]+$")


def _sanitize(name: str) -> str:
    """Ensure config name contains only safe characters (path traversal prevention)."""
    if not _SAFE_NAME.match(name):
        raise ValueError(f"Invalid config name: {name!r}")
    return name


def _config_path(name: str) -> Path:
    return _CONFIGS_DIR / f"{_sanitize(name)}.yml"


def _validate_payload(content: Any) -> dict:
    """Run jsonschema validation. Raises ValueError on failure."""
    if not isinstance(content, dict):
        raise ValueError("Config content must be a YAML mapping (dict).")
    try:
        # Local import: avoid hard-loading jsonschema at module import time.
        from src.orchestration.config_schema import validate_config

        compiled = validate_config(content)
    except Exception as exc:  # noqa: BLE001 — surface as ValueError to router
        raise ValueError(f"Config validation failed: {exc}") from exc
    return compiled


def _issue_path(parts: Any) -> str:
    items = [str(part) for part in parts]
    return "$" if not items else "$" + "".join(f"[{part}]" if part.isdigit() else f".{part}" for part in items)


def _warning(path: str, message: str) -> ConfigValidationIssue:
    return ConfigValidationIssue(path=path, message=message, level="warning")


def _config_warnings(content: dict[str, Any]) -> list[ConfigValidationIssue]:
    warnings: list[ConfigValidationIssue] = []
    dataset = content.get("dataset") or {}
    azure = content.get("azureml") or content.get("azure_ml") or {}
    phases = content.get("phases") or {}
    task_type = content.get("task_type")

    if task_type in {"classification", "regression"} and not dataset.get("target_column"):
        warnings.append(_warning("$.dataset.target_column", "Supervised tasks require a target column."))
    if not (dataset.get("blob_path") or dataset.get("azureml_uri") or dataset.get("path")):
        warnings.append(_warning("$.dataset", "Dataset path is not explicit; submission may rely on datastore defaults."))
    if str(azure.get("compute_target") or "").startswith("<"):
        warnings.append(_warning("$.azureml.compute_target", "Compute target is still a placeholder."))

    phase_b = phases.get("phase_b") or phases.get("phase_b_recipes") or {}
    if isinstance(phase_b, dict):
        budget = phase_b.get("max_variants") or phase_b.get("max_recipes")
        if isinstance(budget, int) and budget > 40:
            warnings.append(_warning("$.phases.phase_b", "Phase B variant budget is high; confirm cost and runtime."))

    phase_c = phases.get("phase_c_hpo") or phases.get("phase_c") or {}
    if isinstance(phase_c, dict):
        trials = phase_c.get("n_trials") or phase_c.get("trials")
        if isinstance(trials, int) and trials > 100:
            warnings.append(_warning("$.phases.phase_c_hpo.n_trials", "HPO trial count is high; confirm budget."))
    return warnings


def validate_content(content: Any) -> ConfigValidationResponse:
    """Validate a parsed config payload without writing it to disk."""
    errors: list[ConfigValidationIssue] = []
    if not isinstance(content, dict):
        return ConfigValidationResponse(
            valid=False,
            errors=[ConfigValidationIssue(path="$", message="Config content must be a YAML mapping (dict).")],
            warnings=[],
        )

    try:
        from src.orchestration.config_schema import CONFIG_SCHEMA, validate_config

        compiled = validate_config(content)
        validator = Draft7Validator(CONFIG_SCHEMA)
        for err in sorted(
            validator.iter_errors(compiled),
            key=lambda item: list(item.absolute_path),
        ):
            errors.append(ConfigValidationIssue(path=_issue_path(err.absolute_path), message=err.message))
    except Exception as exc:  # noqa: BLE001 - validation infrastructure failure should be explicit
        errors.append(ConfigValidationIssue(path="$", message=str(exc)))

    warnings = _config_warnings(content)
    return ConfigValidationResponse(valid=not errors, errors=errors, warnings=warnings)


def get_config_schema() -> dict[str, Any]:
    """Return the JSON schema used by the backend config validator."""
    from src.orchestration.config_schema import CONFIG_SCHEMA

    return CONFIG_SCHEMA


def _as_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return default


def _phase_b_budget(phases: dict[str, Any], recipes: Any) -> int | None:
    phase_b = phases.get("phase_b") or phases.get("phase_b_recipes") or {}
    if isinstance(phase_b, dict):
        for key in ("max_variants", "max_recipes"):
            if isinstance(phase_b.get(key), int):
                return phase_b[key]
        variants = phase_b.get("variants") or phase_b.get("recipes")
        if isinstance(variants, list):
            return len(variants)
    if isinstance(recipes, list):
        return len(recipes)
    return None


def _phase_c_trials(phases: dict[str, Any]) -> tuple[int | None, int | None]:
    phase_c = phases.get("phase_c_hpo") or phases.get("phase_c") or {}
    if not isinstance(phase_c, dict):
        return None, None
    trials = phase_c.get("n_trials") or phase_c.get("trials") or phase_c.get("budget")
    timeout = phase_c.get("timeout_seconds") or phase_c.get("timeout")
    return (trials if isinstance(trials, int) else None, timeout if isinstance(timeout, int) else None)


def _dataset_uri_preview(dataset: dict[str, Any]) -> str | None:
    for key in ("azureml_uri", "path", "uri"):
        if dataset.get(key):
            return str(dataset[key])
    datastore = dataset.get("datastore_name") or "mlops_blob"
    blob_path = dataset.get("blob_path")
    if blob_path:
        return f"azureml://datastores/{datastore}/paths/{blob_path}"
    return None


def _stage_plan(content: dict[str, Any]) -> list[ConfigStagePreview]:
    task_type = str(content.get("task_type") or "unknown")
    stages = content.get("stages") or {}
    stage1 = content.get("stage1") or stages.get("stage1") or {}
    stage2 = content.get("stage2") or stages.get("stage2") or {}
    stage3 = content.get("stage3") or stages.get("stage3") or {}
    stage4 = content.get("stage4") or stages.get("stage4") or {}
    phases = content.get("phases") or {}
    baseline = phases.get("phase_a_baseline") or {}
    phase_b = phases.get("phase_b") or phases.get("phase_b_recipes") or {}
    phase_c = phases.get("phase_c_hpo") or phases.get("phase_c") or {}
    baseline_engines = _as_list(baseline.get("engines") if isinstance(baseline, dict) else None, ["pycaret", "flaml"])

    return [
        ConfigStagePreview(stage_id="S01", label="Data Ingestion and Profiling", summary=f"min_rows={stage1.get('min_rows', 'default')}", warnings=[]),
        ConfigStagePreview(stage_id="S02", label="Data Cleaning and Preparation", summary=f"numeric_imputation={stage2.get('imputation_numeric', stage2.get('imputation_strategy', 'default'))}", warnings=[]),
        ConfigStagePreview(stage_id="S03", label="Feature Preprocessing", summary=f"encoding={stage3.get('encoding', 'default')}; scaling={stage3.get('scaling', 'default')}", warnings=[]),
        ConfigStagePreview(stage_id="S04", label="Feature Engineering and Holdout Split", summary=f"selection={stage4.get('feature_selection', stage4.get('selection_method', 'default'))}", warnings=[]),
        ConfigStagePreview(stage_id="S05a", label="PyCaret Baseline", enabled="pycaret" in baseline_engines, summary="required baseline engine when enabled"),
        ConfigStagePreview(stage_id="S05b", label="FLAML Baseline", enabled="flaml" in baseline_engines, summary="required baseline engine when enabled"),
        ConfigStagePreview(stage_id="S05t", label="Time-Series Baseline", enabled=task_type in {"forecasting", "timeseries"}, summary="enabled only for user-configured time-series/forecasting tasks"),
        ConfigStagePreview(stage_id="S05z", label="Aggregate Baseline", summary="preserve all baseline candidate records"),
        ConfigStagePreview(stage_id="S06", label="Phase B Variant Runner", summary=f"variant_budget={_phase_b_budget(phases, content.get('recipes')) or 'default'}"),
        ConfigStagePreview(stage_id="S08", label="Phase C HPO", summary=f"optimizer={(phase_c or {}).get('optimizer', 'optuna') if isinstance(phase_c, dict) else 'optuna'}"),
        ConfigStagePreview(stage_id="S09", label="Aggregate Phase C", summary="preserve all HPO trial and aggregate records"),
    ]


def preview_config(content: Any, config_name: str | None = None) -> ConfigPreviewResponse:
    """Build a no-side-effect workbench preview for a config draft."""
    validation = validate_content(content)
    cfg = content if isinstance(content, dict) else {}
    dataset = cfg.get("dataset") or {}
    azure = cfg.get("azureml") or cfg.get("azure_ml") or {}
    phases = cfg.get("phases") or {}
    baseline = phases.get("phase_a_baseline") or {}
    phase_b = phases.get("phase_b") or phases.get("phase_b_recipes") or {}
    baseline_engines = _as_list(baseline.get("engines") if isinstance(baseline, dict) else None, ["pycaret", "flaml"])
    phase_b_engines = _as_list(phase_b.get("engines") if isinstance(phase_b, dict) else None, baseline_engines)
    phase_c_trials, phase_c_timeout = _phase_c_trials(phases)

    return ConfigPreviewResponse(
        valid=validation.valid,
        validation=validation,
        config_name=config_name,
        experiment_name=cfg.get("experiment_name"),
        task_type=cfg.get("task_type"),
        dataset_name=dataset.get("name"),
        target_column=dataset.get("target_column"),
        dataset_uri_preview=_dataset_uri_preview(dataset),
        compute_target=azure.get("compute_target") or azure.get("compute"),
        baseline_engines=baseline_engines,
        phase_b_engines=phase_b_engines,
        phase_b_variant_budget=_phase_b_budget(phases, cfg.get("recipes")),
        phase_c_trials=phase_c_trials,
        phase_c_timeout_seconds=phase_c_timeout,
        stage_plan=_stage_plan(cfg),
    )


def _build_detail(name: str, cfg: dict) -> ConfigDetail:
    dataset = cfg.get("dataset") or {}
    return ConfigDetail(
        config_name=name,
        task_type=cfg.get("task_type"),
        dataset_name=dataset.get("name"),
        target_column=dataset.get("target_column"),
        content=cfg,
    )


def list_configs() -> list[ConfigSummary]:
    """Return summaries of all *_azureml.yml configs."""
    results: list[ConfigSummary] = []
    for path in sorted(_CONFIGS_DIR.glob("config_*_azureml.yml")):
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            dataset = cfg.get("dataset") or {}
            results.append(
                ConfigSummary(
                    config_name=path.stem,
                    task_type=cfg.get("task_type"),
                    dataset_name=dataset.get("name"),
                    target_column=dataset.get("target_column"),
                )
            )
        except Exception:
            results.append(ConfigSummary(config_name=path.stem))
    return results


def get_config(config_name: str) -> ConfigDetail:
    """Load full YAML content for a given config name."""
    safe = _sanitize(config_name)
    path = _CONFIGS_DIR / f"{safe}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {safe}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _build_detail(safe, cfg)


# ── CRUD (Phase 4) ─────────────────────────────────────────────

def create_config(config_name: str, content: Any) -> ConfigDetail:
    """Create a new config file. Refuses if it already exists."""
    safe = _sanitize(config_name)
    path = _config_path(safe)
    if path.exists():
        raise FileExistsError(f"Config already exists: {safe}")
    cfg = _validate_payload(content)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return _build_detail(safe, cfg)


def update_config(config_name: str, content: Any) -> ConfigDetail:
    """Overwrite an existing config file. Refuses if it does not exist."""
    safe = _sanitize(config_name)
    path = _config_path(safe)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {safe}")
    cfg = _validate_payload(content)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return _build_detail(safe, cfg)


def delete_config(config_name: str) -> dict:
    """Delete an existing config file."""
    safe = _sanitize(config_name)
    path = _config_path(safe)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {safe}")
    path.unlink()
    return {"status": "deleted", "config_name": safe}
