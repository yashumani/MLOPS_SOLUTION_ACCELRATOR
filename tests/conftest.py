"""Shared pytest fixtures for V3 backend tests.

Mocks Azure ML and MLflow so unit tests run offline. Live Azure
integration tests must be marked with @pytest.mark.integration and
explicitly opted-in via -m integration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _no_azureml_env(monkeypatch):
    """Strip Azure ML env vars so steps don't accidentally hit live services."""
    for var in ("MLFLOW_TRACKING_URI", "AZUREML_RUN_ID", "AZUREML_RUN_TOKEN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_ml_client():
    """Return a mocked azure.ai.ml.MLClient. Use with `with mock_ml_client():`."""
    with patch("azure.ai.ml.MLClient") as mocked:
        instance = MagicMock()
        mocked.return_value = instance
        yield instance


@pytest.fixture
def mock_mlflow():
    """Patch mlflow logging so tests do not write to a tracking server."""
    with patch("mlflow.log_metric"), patch("mlflow.log_param"), \
         patch("mlflow.log_dict"), patch("mlflow.start_run"), \
         patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
        yield


@pytest.fixture
def tmp_csv(tmp_path):
    """Build a small classification CSV and return its path."""
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "f1": rng.normal(size=200),
        "f2": rng.normal(size=200),
        "target": rng.integers(0, 2, size=200),
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p
