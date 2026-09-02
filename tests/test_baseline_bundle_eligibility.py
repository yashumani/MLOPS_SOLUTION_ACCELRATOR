from src.steps.aggregate_baseline import (
    resolve_selected_model_source,
    select_champion,
)


def _evidence(candidate_id: str, score: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "status": "success",
        "primary_metric": "balanced_accuracy",
        "selection_score": score,
        "split_fingerprint": "same-split",
    }


def test_partial_baseline_bundles_are_ineligible_even_with_higher_score():
    partial = {
        "engine": "pycaret",
        "raw_input_bundle_eligible": False,
        "evaluation": _evidence("partial", 0.99),
    }
    complete = {
        "engine": "flaml",
        "raw_input_bundle_eligible": True,
        "model_bundle": {"bundle_id": "bundle-1"},
        "evaluation": _evidence("complete", 0.75),
    }

    selected = select_champion(partial, complete, task="classification")

    assert selected["source"] == "flaml"
    assert selected["candidate_id"] == "complete"


def test_no_baseline_is_selected_without_complete_raw_bundle():
    partial = {
        "engine": "pycaret",
        "raw_input_bundle_eligible": False,
        "evaluation": _evidence("partial", 0.99),
    }

    selected = select_champion(partial, None, task="classification")

    assert selected["source"] is None
    assert selected["score"] is None


def test_schema_v2_ineligible_baseline_never_falls_back_to_direct_score():
    partial = {
        "schema_version": 2,
        "engine": "pycaret",
        "raw_input_bundle_eligible": False,
        "balanced_accuracy": 0.99,
        "model_bundle": {"bundle_id": "stale-partial"},
        "evaluation": _evidence("partial", 0.99),
    }

    selected = select_champion(partial, None, task="classification")

    assert selected == {
        "source": None,
        "score": None,
        "reason": "No eligible schema-v2 raw-input baseline bundle",
    }


def test_missing_selected_artifact_never_substitutes_other_engine(tmp_path):
    pycaret = tmp_path / "pycaret"
    flaml = tmp_path / "flaml"
    pycaret.mkdir()
    flaml.mkdir()
    (flaml / "model_bundle.pkl").write_bytes(b"bundle")
    (flaml / "model_bundle_manifest.json").write_text("{}")

    selected = resolve_selected_model_source(
        "pycaret",
        {"pycaret": pycaret, "flaml": flaml},
    )

    assert selected is None
