"""Test drift detector: PSI math + injection function."""

import numpy as np
import pandas as pd
import pytest

from utils.drift_detector import (
    compute_feature_psi,
    inject_synthetic_drift,
    PSI_GREEN,
    PSI_YELLOW,
)


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.normal(loc=0, scale=1, size=2000),
        "f2": rng.normal(loc=5, scale=2, size=2000),
    })


def test_psi_identical_data_is_near_zero(reference_df):
    scores = compute_feature_psi(reference_df, reference_df.copy(), n_bins=10)
    assert all(v < PSI_GREEN for v in scores.values()), scores


def test_inject_synthetic_drift_shifts_mean(reference_df):
    out = inject_synthetic_drift(reference_df, "f1", shift_sigma=2.0)
    delta = out["f1"].mean() - reference_df["f1"].mean()
    expected = 2.0 * reference_df["f1"].std()
    assert abs(delta - expected) < 0.01


def test_inject_synthetic_drift_does_not_mutate_input(reference_df):
    before = reference_df["f1"].mean()
    _ = inject_synthetic_drift(reference_df, "f1", shift_sigma=3.0)
    after = reference_df["f1"].mean()
    assert before == after


def test_inject_synthetic_drift_raises_on_missing_column(reference_df):
    with pytest.raises(Exception):
        inject_synthetic_drift(reference_df, "doesnotexist", shift_sigma=1.0)


def test_psi_detects_injected_drift(reference_df):
    """The core honesty test: a 2σ shift MUST trigger PSI > PSI_YELLOW."""
    drifted = inject_synthetic_drift(reference_df, "f1", shift_sigma=2.0)
    scores = compute_feature_psi(reference_df[["f1"]], drifted[["f1"]], n_bins=10)
    assert scores["f1"] > PSI_YELLOW, f"detector did not catch 2σ shift: {scores}"
