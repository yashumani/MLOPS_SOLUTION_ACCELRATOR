"""Tests for DriftChecker — all four drift types plus the aggregate runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift_detection.baseline_capture import BaselineCapture
from src.drift_detection.drift_checker import DriftChecker, DriftResult
from src.drift_detection.drift_config import DriftConfig


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def config() -> DriftConfig:
    cfg = DriftConfig()
    cfg.column_mapping.prediction_column = "prediction"
    cfg.column_mapping.target_column = "target"
    return cfg


@pytest.fixture
def reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    target = rng.choice([0, 1], n, p=[0.5, 0.5])
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(100, 10, n),
        "prediction": target,          # perfect predictions for concept tests
        "target": target,
    })


@pytest.fixture
def baseline(config: DriftConfig, reference_df: pd.DataFrame) -> BaselineCapture:
    bc = BaselineCapture(config)
    bc.capture(reference_df)
    return bc


# ── DriftResult dataclass ──────────────────────────────────────

class TestDriftResult:
    def test_defaults(self) -> None:
        r = DriftResult()
        assert r.drift_detected is False
        assert r.drift_type == ""
        assert r.drifted_columns == []

    def test_custom_values(self) -> None:
        r = DriftResult(
            drift_detected=True,
            drift_type="feature",
            drift_score=0.42,
            drifted_columns=["col_x"],
            timestamp="2025-01-01T00:00:00",
        )
        assert r.drift_score == 0.42
        assert r.drifted_columns == ["col_x"]


# ── Feature drift ──────────────────────────────────────────────

class TestFeatureDrift:
    def test_no_drift_on_identical_data(
        self, config: DriftConfig, baseline: BaselineCapture, reference_df: pd.DataFrame
    ) -> None:
        checker = DriftChecker(config, baseline)
        result = checker.check_feature_drift(reference_df)
        assert result.drift_type == "feature"
        # identical data should yield very low score
        assert result.drift_score < config.get_threshold("feature")

    def test_detects_severe_shift(
        self, config: DriftConfig, baseline: BaselineCapture, reference_df: pd.DataFrame
    ) -> None:
        shifted = reference_df.copy()
        shifted["feature_a"] = shifted["feature_a"] + 100  # massive shift
        checker = DriftChecker(config, baseline)
        result = checker.check_feature_drift(shifted)
        assert result.drift_detected is True
        assert len(result.drifted_columns) > 0


# ── Prediction drift ───────────────────────────────────────────

class TestPredictionDrift:
    def test_no_drift_on_identical(
        self, config: DriftConfig, baseline: BaselineCapture, reference_df: pd.DataFrame
    ) -> None:
        checker = DriftChecker(config, baseline)
        result = checker.check_prediction_drift(reference_df)
        assert result.drift_type == "prediction"

    def test_missing_column_returns_empty(
        self, config: DriftConfig, baseline: BaselineCapture
    ) -> None:
        checker = DriftChecker(config, baseline)
        df_no_pred = pd.DataFrame({"feature_a": [1, 2, 3]})
        result = checker.check_prediction_drift(df_no_pred)
        assert result.drift_detected is False


# ── Concept drift ──────────────────────────────────────────────

class TestConceptDrift:
    def test_no_concept_drift_perfect_match(
        self, config: DriftConfig, baseline: BaselineCapture, reference_df: pd.DataFrame
    ) -> None:
        checker = DriftChecker(config, baseline)
        result = checker.check_concept_drift(reference_df)
        assert result.drift_type == "concept"
        assert result.drift_score <= 0.0  # no accuracy drop

    def test_concept_drift_triggered(
        self, config: DriftConfig, baseline: BaselineCapture, reference_df: pd.DataFrame
    ) -> None:
        bad = reference_df.copy()
        # flip 30% of predictions → big accuracy drop
        rng = np.random.default_rng(99)
        mask = rng.random(len(bad)) < 0.3
        bad.loc[mask, "prediction"] = 1 - bad.loc[mask, "prediction"]

        checker = DriftChecker(config, baseline)
        result = checker.check_concept_drift(bad)
        assert result.drift_detected is True
        assert result.details["accuracy_drop"] > 0

    def test_missing_target_skips(self, config: DriftConfig, baseline: BaselineCapture) -> None:
        checker = DriftChecker(config, baseline)
        df = pd.DataFrame({"prediction": [0, 1]})
        result = checker.check_concept_drift(df)
        assert result.drift_detected is False


# ── Label drift ────────────────────────────────────────────────

class TestLabelDrift:
    def test_no_drift_same_distribution(
        self, config: DriftConfig, baseline: BaselineCapture, reference_df: pd.DataFrame
    ) -> None:
        checker = DriftChecker(config, baseline)
        result = checker.check_label_drift(reference_df)
        assert result.drift_type == "label"

    def test_missing_target_skips(self, config: DriftConfig, baseline: BaselineCapture) -> None:
        checker = DriftChecker(config, baseline)
        df = pd.DataFrame({"feature_a": [1, 2]})
        result = checker.check_label_drift(df)
        assert result.drift_detected is False


# ── Aggregate runner ───────────────────────────────────────────

class TestRunAllChecks:
    def test_returns_four_results(
        self, config: DriftConfig, baseline: BaselineCapture, reference_df: pd.DataFrame
    ) -> None:
        checker = DriftChecker(config, baseline)
        results = checker.run_all_checks(reference_df)
        assert len(results) == 4
        types = {r.drift_type for r in results}
        assert types == {"feature", "prediction", "concept", "label"}
