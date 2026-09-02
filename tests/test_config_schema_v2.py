from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from orchestration.config_compiler import compile_config


CONFIGS = (
    "configs/config_classification_telecom_churn_azureml.yml",
    "configs/config_regression_insurance_azureml.yml",
    "configs/config_clustering_online_retail_azureml.yml",
)


def _minimal(task_type: str = "classification") -> dict:
    dataset = {"name": "sample", "version": "1"}
    if task_type != "clustering":
        dataset["target_column"] = "target"
    return {
        "schema_version": "2.0",
        "experiment_name": "schema-v2-test",
        "preset": "production",
        "task_type": task_type,
        "dataset": dataset,
    }


@pytest.mark.parametrize("path", CONFIGS)
def test_canonical_configs_compile_to_explicit_schema_v2(path: str) -> None:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    compiled = compile_config(raw, source_name=Path(path).name)

    assert compiled["schema_version"] == "2.0"
    assert compiled["migration"]["source_schema_version"] in {
        "2.0",
        "unversioned",
    }
    assert compiled["source_config"] == Path(path).name
    assert len(compiled["compiled_config_hash"]) == 64
    assert compiled["metrics"]["selection_protocol"] == "cross_validation"
    assert compiled["metrics"]["locked_test_once"] is True
    assert compiled["metrics"]["min_comparable_candidates"] == 2
    assert compiled["split"]["locked_test"] is True
    assert compiled["azureml"]["default_datastore"] == compiled["dataset"]["datastore_name"]
    assert compiled["phases"]["phase_b"]["planner"]["round1_max_variants"] <= 40
    assert compiled["phases"]["phase_b"]["planner"]["round2_max_variants"] <= 8
    assert compiled["phases"]["phase_b"]["time_budget_per_variant"] <= 600
    assert compiled["phases"]["phase_b"]["phase_timeout_seconds"] <= 10800
    assert compiled["phases"]["phase_c_hpo"]["n_trials"] <= 50
    assert compiled["phases"]["phase_c_hpo"]["timeout_seconds"] <= 3600


def test_explicit_pipeline_output_datastore_is_preserved() -> None:
    raw = _minimal()
    raw["dataset"]["datastore_name"] = "input-data"
    raw["azureml"] = {"default_datastore": "pipeline-outputs"}

    compiled = compile_config(raw)

    assert compiled["dataset"]["datastore_name"] == "input-data"
    assert compiled["azureml"]["default_datastore"] == "pipeline-outputs"


def test_v2_rejects_budget_expansion_instead_of_clamping() -> None:
    raw = _minimal()
    raw["phases"] = {
        "phase_b": {"planner": {"round1_max_variants": 41}}
    }
    with pytest.raises(ValidationError, match="round1_max_variants"):
        compile_config(raw)


@pytest.mark.parametrize("minimum", (1, 4))
def test_v2_rejects_invalid_minimum_comparable_candidates(minimum: int) -> None:
    raw = _minimal()
    raw["metrics"] = {"min_comparable_candidates": minimum}

    with pytest.raises(ValidationError, match="min_comparable_candidates"):
        compile_config(raw)


def test_legacy_migration_clamps_historical_caps_explicitly() -> None:
    raw = _minimal()
    raw.pop("schema_version")
    raw["phases"] = {
        "phase_b": {
            "time_budget_per_variant": 900,
            "phase_timeout_seconds": 20000,
            "planner": {
                "round1_max_variants": 50,
                "round2_max_variants": 10,
            },
        },
        "phase_c_hpo": {"n_trials": 100, "timeout_seconds": 7200},
    }
    compiled = compile_config(raw)

    assert compiled["phases"]["phase_b"]["planner"]["round1_max_variants"] == 40
    assert compiled["phases"]["phase_b"]["planner"]["round2_max_variants"] == 8
    assert compiled["phases"]["phase_b"]["time_budget_per_variant"] == 600
    assert compiled["phases"]["phase_b"]["phase_timeout_seconds"] == 10800
    assert compiled["phases"]["phase_c_hpo"]["n_trials"] == 50
    assert compiled["phases"]["phase_c_hpo"]["timeout_seconds"] == 3600
    rewritten_paths = {
        rewrite["path"] for rewrite in compiled["migration"]["rewrites"]
    }
    assert compiled["migration"]["applied"] is True
    assert "phases.phase_b.planner.round1_max_variants" in rewritten_paths
    assert "phases.phase_b.planner.round2_max_variants" in rewritten_paths
    assert "phases.phase_b.time_budget_per_variant" in rewritten_paths
    assert "phases.phase_b.phase_timeout_seconds" in rewritten_paths
    assert "phases.phase_c_hpo.n_trials" in rewritten_paths
    assert "phases.phase_c_hpo.timeout_seconds" in rewritten_paths


def test_only_approved_tasks_and_clustering_engine_are_allowed() -> None:
    with pytest.raises(ValidationError, match="task_type"):
        compile_config(_minimal("forecasting"))

    clustering = compile_config(_minimal("clustering"))
    assert clustering["phases"]["phase_b"]["engines"] == ["pycaret"]


def test_explicit_v2_regression_rejects_stratified_split() -> None:
    raw = _minimal("regression")
    raw["split"] = {"strategy": "stratified"}

    with pytest.raises(ValidationError, match="classification"):
        compile_config(raw)


def test_explicit_v2_clustering_rejects_flaml_instead_of_rewriting() -> None:
    raw = _minimal("clustering")
    raw["phases"] = {
        "phase_b": {"engines": ["pycaret", "flaml"]}
    }

    with pytest.raises(ValidationError, match="PyCaret only"):
        compile_config(raw)


def test_explicit_empty_pass_aliases_disable_promotion() -> None:
    raw = _minimal()
    raw["registry"] = {"pass_aliases": []}

    compiled = compile_config(raw)

    assert compiled["registry"]["pass_aliases"] == []


def test_omitted_pass_aliases_keep_champion_default() -> None:
    compiled = compile_config(_minimal())

    assert compiled["registry"]["pass_aliases"] == ["champion"]


def test_phase_a_canary_limits_are_preserved() -> None:
    raw = _minimal()
    raw["phases"] = {
        "phase_a_baseline": {
            "cv_folds": 3,
            "candidate_engine_timeout_seconds": 120,
            "flaml_config": {"time_budget": 90},
        }
    }

    compiled = compile_config(raw)
    phase_a = compiled["phases"]["phase_a_baseline"]

    assert phase_a["cv_folds"] == 3
    assert phase_a["candidate_engine_timeout_seconds"] == 120
    assert phase_a["flaml_config"]["time_budget"] == 90
    assert compiled["stage5"]["pycaret_fold"] == 3
    assert compiled["stage5"]["flaml_time_budget"] == 90


def test_phase_a_engines_are_independent_from_phase_b() -> None:
    raw = _minimal()
    raw["phases"] = {
        "phase_a_baseline": {"engines": ["pycaret"]},
        "phase_b": {"engines": ["pycaret", "flaml"]},
    }

    compiled = compile_config(raw)

    assert compiled["phases"]["phase_a_baseline"]["engines"] == ["pycaret"]
    assert compiled["phases"]["phase_b"]["engines"] == ["pycaret", "flaml"]


@pytest.mark.parametrize(
    ("phase_b", "message"),
    (
        ({"enable_profiling": False}, "enable_profiling"),
        ({"selection_strategy": "alphabetical"}, "selection_strategy"),
        ({"planner": {"enabled": False}}, "planner.enabled"),
    ),
)
def test_v2_rejects_unsupported_funnel_controls(
    phase_b: dict,
    message: str,
) -> None:
    raw = _minimal()
    raw["phases"] = {"phase_b": phase_b}

    with pytest.raises(ValidationError, match=message):
        compile_config(raw)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("min_relevance_score", 30.0),
        ("diversity_boost", True),
        ("imputation_preset", "statistical"),
    ),
)
def test_v2_rejects_retired_phase_b_selection_controls(
    field_name: str,
    value: object,
) -> None:
    raw = _minimal()
    raw["phases"] = {"phase_b": {field_name: value}}

    with pytest.raises(ValidationError, match=field_name):
        compile_config(raw)


def test_legacy_selection_controls_are_migrated_out_of_runtime_config() -> None:
    raw = _minimal()
    raw.pop("schema_version")
    raw["phases"] = {
        "phase_b": {
            "min_relevance_score": 30.0,
            "diversity_boost": True,
            "imputation_preset": "statistical",
        }
    }

    compiled = compile_config(raw)

    assert "min_relevance_score" not in compiled["phases"]["phase_b"]
    assert "diversity_boost" not in compiled["phases"]["phase_b"]
    assert "imputation_preset" not in compiled["phases"]["phase_b"]
    rewrites = {item["path"] for item in compiled["migration"]["rewrites"]}
    assert {
        "phases.phase_b.min_relevance_score",
        "phases.phase_b.diversity_boost",
        "phases.phase_b.imputation_preset",
    }.issubset(rewrites)


def test_legacy_funnel_rewrites_are_explicit() -> None:
    raw = _minimal()
    raw.pop("schema_version")
    raw["phases"] = {
        "phase_b": {
            "enable_profiling": False,
            "selection_strategy": "alphabetical",
            "planner": {"enabled": False},
        }
    }

    compiled = compile_config(raw)
    rewrites = {item["path"] for item in compiled["migration"]["rewrites"]}

    assert compiled["phases"]["phase_b"]["enable_profiling"] is True
    assert compiled["phases"]["phase_b"]["selection_strategy"] == "profile_scored"
    assert compiled["phases"]["phase_b"]["planner"]["enabled"] is True
    assert {
        "phases.phase_b.enable_profiling",
        "phases.phase_b.selection_strategy",
        "phases.phase_b.planner.enabled",
    }.issubset(rewrites)


def test_v2_rejects_unversioned_dataset() -> None:
    raw = _minimal()
    raw["dataset"].pop("version")

    with pytest.raises(ValidationError, match="dataset.version"):
        compile_config(raw)


def test_dataset_content_digest_is_validated_and_materialized() -> None:
    raw = _minimal()
    raw["dataset"]["content_sha256"] = "not-a-digest"
    with pytest.raises(ValidationError, match="content_sha256"):
        compile_config(raw)

    raw["dataset"]["content_sha256"] = "a" * 64
    assert compile_config(raw)["dataset"]["content_sha256"] == "a" * 64


def test_dataset_excluded_columns_are_materialized() -> None:
    raw = _minimal()
    raw["dataset"]["id_columns"] = ["customer_id"]
    raw["dataset"]["excluded_columns"] = ["post_outcome_status"]

    compiled = compile_config(raw)

    assert compiled["dataset"]["id_columns"] == ["customer_id"]
    assert compiled["dataset"]["excluded_columns"] == ["post_outcome_status"]


@pytest.mark.parametrize("strategy", ("group", "time", "preassigned"))
def test_unimplemented_split_policies_fail_before_submission(strategy: str) -> None:
    raw = _minimal()
    raw["split"] = {"strategy": strategy}

    with pytest.raises(ValidationError, match="not implemented end to end"):
        compile_config(raw)


def test_cross_validation_contract_has_no_ignored_validation_partition() -> None:
    raw = _minimal()
    raw["split"] = {"validation_fraction": 0.2}

    with pytest.raises(ValidationError, match="validation_fraction must be 0.0"):
        compile_config(raw)

    raw["split"]["validation_fraction"] = 0.0
    assert compile_config(raw)["split"]["validation_fraction"] == 0.0


def test_phase_a_canary_timeout_cannot_exceed_cap() -> None:
    raw = _minimal()
    raw["phases"] = {
        "phase_a_baseline": {
            "candidate_engine_timeout_seconds": 601,
        }
    }

    with pytest.raises(ValidationError, match="candidate_engine_timeout_seconds"):
        compile_config(raw)


def test_legacy_semantic_rewrites_are_recorded() -> None:
    raw = _minimal("regression")
    raw.pop("schema_version")
    raw["split"] = {"strategy": "stratified"}
    compiled = compile_config(raw)

    assert compiled["split"]["strategy"] == "random"
    assert {
        rewrite["path"] for rewrite in compiled["migration"]["rewrites"]
    } == {"split.strategy"}
