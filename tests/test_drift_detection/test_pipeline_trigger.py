"""Tests for PipelineTrigger — evaluation, dry-run, and logging."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from src.drift_detection.drift_checker import DriftResult
from src.drift_detection.drift_config import DriftConfig
from src.drift_detection.pipeline_trigger import PipelineTrigger


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def config() -> DriftConfig:
    cfg = DriftConfig()
    cfg.column_mapping.prediction_column = "prediction"
    cfg.column_mapping.target_column = "target"
    return cfg


def _make_results(detected: bool = False) -> List[DriftResult]:
    """Helper to create a set of DriftResults."""
    return [
        DriftResult(drift_detected=detected, drift_type="feature", drift_score=0.30 if detected else 0.01),
        DriftResult(drift_detected=False, drift_type="prediction", drift_score=0.02),
        DriftResult(drift_detected=False, drift_type="concept", drift_score=0.0),
        DriftResult(drift_detected=False, drift_type="label", drift_score=0.03),
    ]


# ── Tests ───────────────────────────────────────────────────────

class TestPipelineTrigger:

    def test_no_trigger_when_no_drift(self, config: DriftConfig) -> None:
        trigger = PipelineTrigger(config, dry_run=True)
        summary = trigger.evaluate(_make_results(detected=False))
        assert summary["should_trigger"] is False
        assert summary["triggered_by"] == []
        assert summary["action"] == "none"

    def test_trigger_when_drift_detected(self, config: DriftConfig) -> None:
        trigger = PipelineTrigger(config, dry_run=True)
        summary = trigger.evaluate(_make_results(detected=True))
        assert summary["should_trigger"] is True
        assert "feature" in summary["triggered_by"]
        assert summary["action"] == "trigger_full_pipeline"

    def test_dry_run_flag_propagated(self, config: DriftConfig) -> None:
        trigger = PipelineTrigger(config, dry_run=True)
        summary = trigger.evaluate(_make_results(detected=True))
        assert summary["dry_run"] is True

    def test_trigger_history_accumulated(self, config: DriftConfig) -> None:
        trigger = PipelineTrigger(config, dry_run=True)
        trigger.evaluate(_make_results(detected=False))
        trigger.evaluate(_make_results(detected=True))
        assert len(trigger.trigger_history) == 2

    def test_save_trigger_log(self, config: DriftConfig, tmp_path: Path) -> None:
        trigger = PipelineTrigger(config, dry_run=True)
        trigger.evaluate(_make_results(detected=True))
        log_path = trigger.save_trigger_log(str(tmp_path))
        assert log_path.exists()
        assert log_path.suffix == ".json"

    def test_details_contain_score_and_threshold(self, config: DriftConfig) -> None:
        trigger = PipelineTrigger(config, dry_run=True)
        summary = trigger.evaluate(_make_results(detected=True))
        assert "feature" in summary["details"]
        info = summary["details"]["feature"]
        assert "score" in info
        assert "threshold" in info

    def test_multiple_drift_types(self, config: DriftConfig) -> None:
        results = [
            DriftResult(drift_detected=True, drift_type="feature", drift_score=0.5),
            DriftResult(drift_detected=True, drift_type="label", drift_score=0.3),
            DriftResult(drift_detected=False, drift_type="prediction", drift_score=0.01),
            DriftResult(drift_detected=False, drift_type="concept", drift_score=0.0),
        ]
        trigger = PipelineTrigger(config, dry_run=True)
        summary = trigger.evaluate(results)
        assert summary["should_trigger"] is True
        assert set(summary["triggered_by"]) == {"feature", "label"}
