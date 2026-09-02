from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ExecutionManifest


def load_execution_manifest(path: str | Path) -> ExecutionManifest:
    """Load and cryptographically validate an immutable execution manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise RuntimeError(
            f"ExecutionManifest input does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"Invalid ExecutionManifest JSON: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("ExecutionManifest JSON root must be an object")
    return ExecutionManifest.from_dict(payload)


def validate_execution_manifest_binding(
    path: str | Path,
    compiled_config: Mapping[str, Any],
) -> ExecutionManifest:
    """Bind a downstream component to the submitter's exact config identity."""
    manifest = load_execution_manifest(path)
    config_hash = str(compiled_config.get("compiled_config_hash") or "")
    task_type = str(compiled_config.get("task_type") or "")
    if manifest.config_hash != config_hash:
        raise RuntimeError(
            "ExecutionManifest config_hash does not match the compiled config"
        )
    if manifest.task_type != task_type:
        raise RuntimeError(
            "ExecutionManifest task_type does not match the compiled config"
        )
    return manifest
