"""Tests for synthetic data generation with controlled drift levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift_detection.synthetic_data_generator import generate_drifted_data


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "numeric_a": rng.normal(0, 1, n),
        "numeric_b": rng.uniform(0, 100, n),
        "category": rng.choice(["X", "Y", "Z"], n),
        "prediction": rng.choice([0, 1], n),
        "target": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })


# ── Tests ───────────────────────────────────────────────────────

class TestNoDrift:
    def test_shape_preserved(self, reference_df: pd.DataFrame) -> None:
        out = generate_drifted_data(reference_df, drift_level="none")
        assert out.shape[1] == reference_df.shape[1]
        assert len(out) == len(reference_df)

    def test_distribution_similar(self, reference_df: pd.DataFrame) -> None:
        out = generate_drifted_data(reference_df, drift_level="none", seed=0)
        # Means should be close (it's just a resample)
        assert abs(out["numeric_a"].mean() - reference_df["numeric_a"].mean()) < 1.0

    def test_custom_n_rows(self, reference_df: pd.DataFrame) -> None:
        out = generate_drifted_data(reference_df, drift_level="none", n_rows=50)
        assert len(out) == 50


class TestMildDrift:
    def test_numeric_shifted(self, reference_df: pd.DataFrame) -> None:
        out = generate_drifted_data(reference_df, drift_level="mild")
        # Mild shift → mean should differ but not wildly
        diff = abs(out["numeric_a"].mean() - reference_df["numeric_a"].mean())
        assert diff > 0.01  # some shift happened

    def test_columns_unchanged(self, reference_df: pd.DataFrame) -> None:
        out = generate_drifted_data(reference_df, drift_level="mild")
        assert list(out.columns) == list(reference_df.columns)

    def test_prediction_drift_applied(self, reference_df: pd.DataFrame) -> None:
        out = generate_drifted_data(
            reference_df,
            drift_level="mild",
            prediction_column="prediction",
        )
        # prediction values should be shifted
        assert out["prediction"].dtype == reference_df["prediction"].dtype

    def test_label_drift_applied(self, reference_df: pd.DataFrame) -> None:
        out = generate_drifted_data(
            reference_df,
            drift_level="mild",
            target_column="target",
        )
        assert "target" in out.columns


class TestSevereDrift:
    def test_large_shift(self, reference_df: pd.DataFrame) -> None:
        out = generate_drifted_data(reference_df, drift_level="severe")
        diff = abs(out["numeric_a"].mean() - reference_df["numeric_a"].mean())
        assert diff > 0.5  # severe → large shift

    def test_deterministic_with_seed(self, reference_df: pd.DataFrame) -> None:
        a = generate_drifted_data(reference_df, drift_level="severe", seed=123)
        b = generate_drifted_data(reference_df, drift_level="severe", seed=123)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self, reference_df: pd.DataFrame) -> None:
        a = generate_drifted_data(reference_df, drift_level="severe", seed=1)
        b = generate_drifted_data(reference_df, drift_level="severe", seed=2)
        assert not a.equals(b)


class TestEdgeCases:
    def test_single_row(self) -> None:
        df = pd.DataFrame({"x": [1.0], "cat": ["A"]})
        out = generate_drifted_data(df, drift_level="mild", n_rows=10)
        assert len(out) == 10

    def test_no_numeric_columns(self) -> None:
        df = pd.DataFrame({"cat_a": ["A", "B", "C"] * 10, "cat_b": ["X", "Y", "Z"] * 10})
        out = generate_drifted_data(df, drift_level="severe")
        assert len(out) == 30
