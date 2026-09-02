from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import validate_azure_environment as validator


def test_load_expected_versions_requires_exact_pins(tmp_path: Path):
    conda = tmp_path / "conda.yml"
    conda.write_text(
        "dependencies:\n"
        "  - python=3.10\n"
        "  - pip:\n"
        "      - mlflow==3.15.0\n"
        "      - azure-core>=1.41.0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not exactly pinned: azure-core>=1.41.0"):
        validator._load_expected_versions(conda)


def test_validate_expected_versions_rejects_runtime_drift(monkeypatch):
    observed = {"mlflow": "3.15.0", "azure-core": "1.40.0"}
    monkeypatch.setattr(
        validator.importlib.metadata,
        "version",
        lambda name: observed[name],
    )

    with pytest.raises(RuntimeError, match='"observed": "1.40.0"'):
        validator._validate_expected_versions(
            {"mlflow": "3.15.0", "azure-core": "1.41.0"}
        )


def test_mlflow_smoke_rejects_package_version_mismatch(monkeypatch):
    versions = {
        "mlflow": "2.14.3",
        "mlflow-skinny": "3.15.2",
    }
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace())
    monkeypatch.setattr(
        validator.importlib.metadata,
        "version",
        lambda name: versions[name],
    )

    with pytest.raises(
        RuntimeError,
        match=r"mlflow=2\.14\.3, mlflow-skinny=3\.15\.2",
    ):
        validator._validate_mlflow()
