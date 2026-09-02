from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scripts import validate_azure_environment as validator


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
