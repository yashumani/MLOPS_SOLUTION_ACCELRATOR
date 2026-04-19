"""Config listing and parsing service."""

import re
from pathlib import Path

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
    dataset = cfg.get("dataset") or {}
    return ConfigDetail(
        config_name=safe,
        task_type=cfg.get("task_type"),
        dataset_name=dataset.get("name"),
        target_column=dataset.get("target_column"),
        content=cfg,
    )
