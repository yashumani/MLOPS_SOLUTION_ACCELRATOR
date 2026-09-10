import csv
import json
import shutil
import sys

import pytest

from src.steps import aggregate_phasec


@pytest.fixture
def aggregate_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yml"
    config.write_text("task_type: regression\n", encoding="utf-8")
    metrics_path = tmp_path / "hpo.json"
    model = tmp_path / "optimized"
    model.mkdir()
    (model / "model_bundle.pkl").write_bytes(b"contract-test-artifact")
    (model / "model_bundle_manifest.json").write_text(
        json.dumps({"candidate_id": "phaseb-1:tuned:123"}), encoding="utf-8",
    )
    report = tmp_path / "named-report.json"
    champion = tmp_path / "champion"
    monkeypatch.setattr(sys, "argv", [
        "aggregate_phasec", "--config", str(config),
        "--hpo_metrics", str(metrics_path), "--optimized_model", str(model),
        "--report_out", str(report), "--champion_out", str(champion),
    ])
    metrics = {
        "schema_version": 2,
        "status": "success",
        "candidate_id": "phaseb-1:tuned:123",
        "execution_id": "execution-123",
        "config_hash": "config-123",
        "best_score": 0.0,
        "best_params": {"n_estimators": 50},
        "algorithm": "randomforest",
        "selection_metric": "r2",
        "n_trials_completed": 3,
    }
    return tmp_path, metrics_path, model, report, champion, metrics


def run_case(case):
    _, metrics_path, _, _, _, metrics = case
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    aggregate_phasec.main()


def read_evidence(case):
    root, _, _, report, _, _ = case
    evidence = json.loads(report.read_text(encoding="utf-8"))
    visible = json.loads((root / "outputs/phasec_aggregate_report.json").read_text())
    assert visible == evidence
    signal = json.loads((root / "outputs/phasec_stage_signal.json").read_text())
    summary = json.loads((root / "outputs/phasec_champion_summary.json").read_text())
    with (root / "outputs/s09_candidates.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return evidence, signal, summary, row


@pytest.mark.parametrize("score", [0.0, -0.25, 0.75])
def test_success_requires_copied_model_and_keeps_exact_candidate(aggregate_case, score):
    aggregate_case[-1]["best_score"] = score
    run_case(aggregate_case)
    evidence, signal, summary, row = read_evidence(aggregate_case)
    assert evidence["candidate_usable"] is True
    assert evidence["model_copied"] is True
    assert evidence["hpo_evidence"] == aggregate_case[-1]
    assert signal["candidate_count_out"] == 1
    assert signal["recommendation"] == "proceed"
    assert signal["recommendation_reason"] == "HPO optimised model ready"
    assert signal["best_score"] == score
    assert signal["best_metric_name"] == "r2"
    assert summary["model_copied"] is True
    assert summary["best_score"] == score
    assert row["status"] == "ok"
    assert row["candidate_id"] == "phaseb-1:tuned:123"
    assert row["primary_metric_name"] == "r2"
    assert float(row["primary_metric_value"]) == score
    assert row["is_stage_best"] == "True"
    assert (aggregate_case[4] / "model_bundle.pkl").read_bytes() == b"contract-test-artifact"


@pytest.mark.parametrize("reason", [
    "unsupported_phaseb_algorithm_family",
    "same_family_hpo_failed: Optuna study finished with zero successful trials",
])
def test_skipped_hpo_preserves_reason_without_publishing_candidate(aggregate_case, reason):
    metrics = aggregate_case[-1]
    metrics.update(status="skipped_unsupported", reason=reason, preserve_phaseb=True,
                   phaseb_candidate_id="phaseb-1", phaseb_algorithm="ExtraTreesEstimator")
    del metrics["candidate_id"]
    run_case(aggregate_case)
    evidence, signal, summary, row = read_evidence(aggregate_case)
    assert evidence["status"] == "skipped_unsupported"
    assert evidence["reason"] == reason
    assert evidence["hpo_evidence"] == metrics
    assert evidence["selection"]["score"] is None
    assert evidence["candidate_usable"] is False
    assert summary["best_score"] is None
    assert summary["reason"] == reason
    assert signal["candidate_count_out"] == 0
    assert signal["best_score"] is None
    assert signal["recommendation_reason"] == reason
    assert signal["extra"]["preserve_phaseb"] is True
    assert row["status"] == "skipped_unsupported"
    assert row["failure_reason"] == reason
    assert row["is_stage_best"] == "False"
    assert (aggregate_case[4] / ".no_model").read_text() == reason
    assert not (aggregate_case[4] / "model_bundle.pkl").exists()


@pytest.mark.parametrize("missing", ["model_bundle.pkl", "model_bundle_manifest.json"])
def test_success_without_bundle_fails_and_never_advertises_score(aggregate_case, missing):
    (aggregate_case[2] / missing).unlink()
    with pytest.raises(RuntimeError, match="missing its exact ModelBundle"):
        run_case(aggregate_case)
    evidence, signal, summary, row = read_evidence(aggregate_case)
    assert evidence["status"] == "failed"
    assert evidence["model_copied"] is False
    assert evidence["hpo_evidence"]["status"] == "success"
    assert signal["candidate_count_out"] == 0
    assert summary["candidate_usable"] is False
    assert row["status"] == "failed"
    assert row["is_stage_best"] == "False"


def test_copy_failure_is_not_a_successful_pipeline_stage(aggregate_case, monkeypatch):
    original = shutil.copy2

    def fail_model_copy(source, target, *args, **kwargs):
        if str(source).endswith("model_bundle.pkl"):
            raise OSError("simulated destination failure")
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_model_copy)
    with pytest.raises(RuntimeError, match="simulated destination failure"):
        run_case(aggregate_case)
    evidence, signal, summary, row = read_evidence(aggregate_case)
    assert evidence["model_copied"] is False
    assert evidence["selection"]["score"] is None
    assert summary["status"] == "failed"
    assert signal["recommendation"] == "stop"
    assert row["failure_reason"].endswith("simulated destination failure")


@pytest.mark.parametrize("score", [None, True, "0.8"])
def test_invalid_success_score_fails_closed(aggregate_case, score):
    aggregate_case[-1]["best_score"] = score
    with pytest.raises(RuntimeError, match="finite numeric best_score"):
        run_case(aggregate_case)
    evidence, signal, summary, row = read_evidence(aggregate_case)
    assert evidence["candidate_usable"] is False
    assert signal["best_score"] is None
    assert summary["best_score"] is None
    assert row["is_stage_best"] == "False"


def test_success_without_identity_fails_closed(aggregate_case):
    del aggregate_case[-1]["candidate_id"]
    with pytest.raises(RuntimeError, match="exact candidate identity"):
        run_case(aggregate_case)


@pytest.mark.parametrize("contents", [
    "{", "[]", "null", "{}", '{"status":"failed"}',
    '{"status":"skipped_unsupported","reason":" "}',
    '{"status":"success","best_score":NaN}',
    '{"status":"success","best_score":Infinity}',
])
def test_invalid_metrics_are_not_converted_to_an_empty_success(aggregate_case, contents):
    aggregate_case[1].write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        aggregate_phasec.main()
    assert not aggregate_case[3].exists()


def test_missing_metrics_fail_instead_of_silently_skipping(aggregate_case):
    with pytest.raises(FileNotFoundError):
        aggregate_phasec.main()
    assert not aggregate_case[3].exists()
