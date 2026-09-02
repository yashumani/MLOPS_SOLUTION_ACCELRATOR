import json
import inspect

import pandas as pd
import pytest
from orchestration.contracts import SplitManifest, canonical_hash
from src.steps import final_evaluation
from src.steps.final_evaluation import (
    bind_selected_candidate_to_source_bundle,
    load_selection_evidence,
    select_champion_from_selection_evidence,
    validate_locked_holdout_identity,
    validate_selection_comparability,
    validate_selection_lineage,
)
from src.utils.model_bundle import ModelBundle


def test_champion_is_selected_from_cv_evidence_before_locked_test(tmp_path):
    baseline = tmp_path / "baseline"
    phaseb = tmp_path / "phaseb"
    phasec = tmp_path / "phasec"
    for path in (baseline, phaseb, phasec):
        path.mkdir()
    (baseline / "selection_manifest.json").write_text(
        json.dumps({"status": "success", "selection_score": 0.72})
    )
    (phaseb / "champion_manifest.json").write_text(
        json.dumps({"status": "success", "primary_metric_value": 0.81})
    )
    (phasec / "selection_manifest.json").write_text(
        json.dumps({"status": "success", "selection_score": 0.79})
    )

    evidence = {
        "baseline": load_selection_evidence(baseline, "baseline"),
        "phaseb": load_selection_evidence(phaseb, "phaseb"),
        "phasec": load_selection_evidence(phasec, "phasec"),
    }
    assert select_champion_from_selection_evidence(evidence) == ("phaseb", 0.81)


def test_phaseb_manifest_exposes_common_evaluator_comparison_contract(
    tmp_path,
):
    phaseb = tmp_path / "phaseb"
    phaseb.mkdir()
    (phaseb / "champion_manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "candidate_id": "phaseb-1",
                "primary_metric_name": "balanced_accuracy",
                "primary_metric_value": 0.81,
                "metrics": {
                    "common_evaluator": {
                        "split_fingerprint": "folds-1",
                        "total_folds": 5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = {
        "phaseb": load_selection_evidence(phaseb, "phaseb"),
    }

    validate_selection_comparability(
        evidence,
        expected_metric="balanced_accuracy",
        expected_folds=5,
        minimum_candidates=1,
    )


def _comparable_selection_evidence(
    *,
    metric: str = "balanced_accuracy",
    folds: int = 5,
    fingerprint: str = "locked-folds",
) -> dict:
    return {
        "status": "success",
        "selection_score": 0.75,
        "metric_name": metric,
        "total_folds": folds,
        "split_fingerprint": fingerprint,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"metric": "accuracy"}, "does not match compiled metric"),
        ({"folds": 3}, "do not match compiled fold count"),
        ({"fingerprint": ""}, "lacks split_fingerprint"),
    ),
)
def test_selection_comparability_rejects_mismatched_cv_contract(
    overrides,
    message,
):
    evidence = {
        "phaseb": _comparable_selection_evidence(),
        "phasec": _comparable_selection_evidence(**overrides),
    }

    with pytest.raises(RuntimeError, match=message):
        validate_selection_comparability(
            evidence,
            expected_metric="balanced_accuracy",
            expected_folds=5,
        )


def test_selection_comparability_rejects_different_fold_assignments():
    evidence = {
        "phaseb": _comparable_selection_evidence(
            fingerprint="phaseb-folds"
        ),
        "phasec": _comparable_selection_evidence(
            fingerprint="phasec-folds"
        ),
    }

    with pytest.raises(RuntimeError, match="different deterministic fold"):
        validate_selection_comparability(
            evidence,
            expected_metric="balanced_accuracy",
            expected_folds=5,
        )


def test_selection_comparability_requires_multiple_candidates():
    with pytest.raises(RuntimeError, match="at least 2"):
        validate_selection_comparability(
            {"phaseb": _comparable_selection_evidence()},
            expected_metric="balanced_accuracy",
            expected_folds=5,
        )


def test_skipped_phasec_cannot_replace_phaseb(tmp_path):
    phasec = tmp_path / "phasec"
    phasec.mkdir()
    (phasec / "selection_manifest.json").write_text(
        json.dumps(
            {
                "status": "skipped_unsupported",
                "selection_score": 0.99,
                "reason": "unsupported family",
            }
        )
    )
    evidence = {
        "phaseb": {
            "status": "success",
            "selection_score": 0.7,
        },
        "phasec": load_selection_evidence(phasec, "phasec"),
    }
    assert select_champion_from_selection_evidence(evidence) == ("phaseb", 0.7)


def test_selection_lineage_fails_closed_on_missing_or_mixed_runs():
    valid = {
        "status": "success",
        "selection_score": 0.8,
        "lineage": {
            "execution_id": "execution-1",
            "parent_run_id": "parent-1",
            "candidate_run_id": "child-1",
        },
    }
    missing = {
        "status": "success",
        "selection_score": 0.7,
        "lineage": {
            "execution_id": "execution-1",
            "parent_run_id": "parent-1",
        },
    }
    with pytest.raises(RuntimeError, match="lacks exact lineage"):
        validate_selection_lineage({"baseline": valid, "phaseb": missing})

    mixed = {
        **valid,
        "lineage": {
            **valid["lineage"],
            "execution_id": "execution-2",
            "candidate_run_id": "child-2",
        },
    }
    with pytest.raises(RuntimeError, match="mixes execution IDs"):
        validate_selection_lineage({"baseline": valid, "phasec": mixed})


def test_s10_has_one_locked_metric_call_and_no_locked_reference_reuse():
    source = inspect.getsource(final_evaluation.main)

    assert source.count("locked_test_metrics = eval_model(") == 1
    assert "phaseb_eval_data.csv" not in source
    assert "_shap_sample = X_train.sample(" in source
    assert "capture_input_schema(X_test)" not in source
    assert "X_test.head(1)" not in source


def _stage2_split_manifest(holdout_ids: list[str]) -> SplitManifest:
    return SplitManifest(
        task_type="classification",
        strategy="stratified",
        random_seed=42,
        train_count=8,
        validation_count=0,
        test_count=len(holdout_ids),
        train_ids_hash=canonical_hash([f"train-{index}" for index in range(8)]),
        validation_ids_hash=canonical_hash([]),
        test_ids_hash=canonical_hash(holdout_ids),
        data_version="sample@1:data/sample.csv",
    )


def test_s10_binds_locked_holdout_to_stage2_split_manifest():
    holdout_ids = ["row-2", "row-9"]
    manifest = _stage2_split_manifest(holdout_ids)

    binding = validate_locked_holdout_identity(
        manifest,
        pd.Series(holdout_ids),
        task_type="classification",
    )

    assert binding == {
        "split_id": manifest.split_id,
        "data_version": manifest.data_version,
        "test_count": 2,
        "test_ids_hash": manifest.test_ids_hash,
    }


def test_s10_rejects_holdout_that_does_not_match_stage2_split_manifest():
    manifest = _stage2_split_manifest(["row-2", "row-9"])

    with pytest.raises(ValueError, match="identity hash"):
        validate_locked_holdout_identity(
            manifest,
            pd.Series(["row-9", "row-2"]),
            task_type="classification",
        )


def test_s10_rejects_selected_candidate_manifest_relabeling_source_bundle():
    source_bundle = ModelBundle(
        estimator={"model": "source"},
        task_type="classification",
        candidate_id="source-candidate",
    )

    with pytest.raises(RuntimeError, match="candidate_id"):
        bind_selected_candidate_to_source_bundle(
            {
                "candidate_id": "relabeled-candidate",
                "model_bundle_id": source_bundle.bundle_id,
            },
            source_bundle,
        )


def test_s10_rejects_selected_manifest_bound_to_different_bundle_identity():
    source_bundle = ModelBundle(
        estimator={"model": "source"},
        task_type="classification",
        candidate_id="source-candidate",
    )

    with pytest.raises(RuntimeError, match="bundle identity"):
        bind_selected_candidate_to_source_bundle(
            {
                "candidate_id": source_bundle.candidate_id,
                "model_bundle_id": "different-bundle-id",
            },
            source_bundle,
        )
