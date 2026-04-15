"""Tests for BaselineCapture — capture, save, load round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.drift_detection.baseline_capture import BaselineCapture
from src.drift_detection.drift_config import DriftConfig


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Simple numeric + categorical DataFrame for testing."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "feature_a": rng.normal(10, 2, n),
        "feature_b": rng.normal(50, 5, n),
        "category": rng.choice(["A", "B", "C"], n),
        "prediction": rng.choice([0, 1], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def config() -> DriftConfig:
    """Config with target_column set to 'target'."""
    cfg = DriftConfig()
    cfg.column_mapping.target_column = "target"
    cfg.column_mapping.prediction_column = "prediction"
    return cfg


# ── Tests ───────────────────────────────────────────────────────

class TestBaselineCapture:
    """Test suite for BaselineCapture."""

    def test_capture_stores_reference(self, config: DriftConfig, sample_df: pd.DataFrame) -> None:
        bc = BaselineCapture(config)
        bc.capture(sample_df)
        assert bc.reference_df is not None
        assert len(bc.reference_df) == len(sample_df)

    def test_capture_computes_feature_stats(self, config: DriftConfig, sample_df: pd.DataFrame) -> None:
        bc = BaselineCapture(config)
        bc.capture(sample_df)
        assert "feature_a" in bc._feature_stats
        assert "mean" in bc._feature_stats["feature_a"]

    def test_capture_empty_raises(self, config: DriftConfig) -> None:
        bc = BaselineCapture(config)
        with pytest.raises(ValueError, match="empty"):
            bc.capture(pd.DataFrame())

    def test_save_and_load_round_trip(
        self, config: DriftConfig, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        bc = BaselineCapture(config)
        bc.capture(sample_df)
        out_dir = bc.save(str(tmp_path / "baseline"))

        assert (out_dir / "reference_data.parquet").exists()
        assert (out_dir / "baseline_stats.json").exists()

        loaded = BaselineCapture.load(config, str(out_dir))
        pd.testing.assert_frame_equal(loaded.reference_df, sample_df)

    def test_load_missing_raises(self, config: DriftConfig, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BaselineCapture.load(config, str(tmp_path / "nonexistent"))

    def test_feature_columns_excludes_special(self, config: DriftConfig, sample_df: pd.DataFrame) -> None:
        bc = BaselineCapture(config)
        bc.capture(sample_df)
        feat_cols = bc.feature_columns
        assert "prediction" not in feat_cols
        assert "target" not in feat_cols
        assert "feature_a" in feat_cols

    def test_save_before_capture_raises(self, config: DriftConfig) -> None:
        bc = BaselineCapture(config)
        with pytest.raises(RuntimeError, match="capture"):
            bc.save()

    def test_prediction_stats_populated(self, config: DriftConfig, sample_df: pd.DataFrame) -> None:
        bc = BaselineCapture(config)
        bc.capture(sample_df)
        assert "mean" in bc._prediction_stats

    def test_label_stats_populated(self, config: DriftConfig, sample_df: pd.DataFrame) -> None:
        bc = BaselineCapture(config)
        bc.capture(sample_df)
        assert "value_counts" in bc._label_stats
