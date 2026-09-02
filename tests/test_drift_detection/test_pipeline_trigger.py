"""Tests for PipelineTrigger — evaluation, dry-run, and logging."""

from __future__ import annotations

import ast
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
        assert summary["execution"]["status"] == "dry_run"

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

    def test_non_dry_run_does_not_submit_when_auto_retrain_disabled(
        self,
        config: DriftConfig,
    ) -> None:
        trigger = PipelineTrigger(config, dry_run=False)
        summary = trigger.evaluate(_make_results(detected=True))
        assert summary["execution"]["status"] == "disabled"

    def test_enabled_trigger_delegates_to_s14_without_submitting(
        self,
        config: DriftConfig,
    ) -> None:
        config.auto_retrain.enabled = True
        config.auto_retrain.mode = "submit"

        trigger = PipelineTrigger(config, dry_run=False)
        summary = trigger.evaluate(_make_results(detected=True))
        execution = summary["execution"]

        assert execution["status"] == "delegated"
        assert execution["submitted"] is False
        assert execution["next_stage"] == "s14_retrain_decision"
        assert execution["submission_owner"] == "external_controller"
        assert execution["required_artifact"] == "retrain_decision.json"

    def test_pipeline_trigger_source_contains_no_child_submission_path(self) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "drift_detection"
            / "pipeline_trigger.py"
        ).read_text(encoding="utf-8")

        assert "subprocess" not in source
        assert "submit_pipeline.py" not in source
        assert "_build_submit_command" not in source


def test_drift_config_loads_auto_retrain_section(tmp_path: Path) -> None:
    config_path = tmp_path / "drift_config.yaml"
    config_path.write_text(
        "auto_retrain:\n"
        "  enabled: true\n"
        "  mode: submit\n"
        "  config_path: configs/config_classification_telecom_churn_azureml.yml\n"
        "  compute: mlops-cluster\n"
        "  extra_args:\n"
        "    - --use_phase1\n"
    )
    config = DriftConfig.from_yaml(str(config_path))
    assert config.auto_retrain.enabled is True
    assert config.auto_retrain.mode == "submit"
    assert config.auto_retrain.compute == "mlops-cluster"
    assert config.auto_retrain.extra_args == ["--use_phase1"]


def test_legacy_schedule_setup_cannot_construct_or_submit_training_dag() -> None:
    schedule_source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "setup_drift_schedule.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(schedule_source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "azure.ai.ml" not in imported_names
    assert "pipelines.pipeline_builder" not in imported_names
    assert "full_pipeline" not in schedule_source
    assert "begin_create_or_update" not in schedule_source
    assert "run_auto_retrain_controller.py" in schedule_source
