"""Config listing and parsing service."""

import re
from pathlib import Path
from typing import Any

import yaml

from api.schemas.config import ConfigDetail, ConfigSummary

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

        validate_config(content)
    except Exception as exc:  # noqa: BLE001 — surface as ValueError to router
        raise ValueError(f"Config validation failed: {exc}") from exc
    return content


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

