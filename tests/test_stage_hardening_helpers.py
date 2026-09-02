from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.orchestration.config_schema import validate_config
from src.steps.phasec_optuna_hpo import _compute_hpo_cost
from src.steps.stage2_preparation import (
    build_partitioned_stage2_inputs,
    drop_excluded_feature_columns,
    prep_dataframe,
    resolve_excluded_feature_columns,
    resolve_protected_columns,
    resolve_raw_model_exclusions,
)
from src.steps.stage3_preprocessing import (
    build_preprocessing_anomaly_report,
    resolve_recipe_path,
)
from src.steps.stage4_feature_engineering import resolve_recipe_path as resolve_stage4_recipe_path
from src.utils.variant_schema import validate_variant_yaml


def test_stage2_resolves_target_and_id_protected_columns() -> None:
    cfg = {
        "dataset": {
            "target_column": "churn",
            "id_columns": ["customer_id"],
        },
        "stage2": {"protected_columns": "account_id"},
    }

    assert resolve_protected_columns(cfg, "churn") == [
        "account_id",
        "churn",
        "customer_id",
    ]


def test_stage2_drops_ids_and_explicit_leakage_columns() -> None:
    cfg = {
        "dataset": {
            "target_column": "churn",
            "id_columns": ["customer_id"],
            "excluded_columns": ["post_outcome_status"],
        }
    }
    frame = pd.DataFrame(
        {
            "customer_id": ["a", "b"],
            "feature": [1, 2],
            "post_outcome_status": [1, 0],
            "churn": [1, 0],
        }
    )

    excluded = resolve_excluded_feature_columns(cfg, "churn")
    result = drop_excluded_feature_columns(frame, excluded)

    assert excluded == ["customer_id", "post_outcome_status"]
    assert result.columns.tolist() == ["feature", "churn"]


def test_stage2_rejects_target_as_excluded_column() -> None:
    cfg = {"dataset": {"excluded_columns": ["churn"]}}

    with pytest.raises(ValueError, match="Target column"):
        resolve_excluded_feature_columns(cfg, "churn")


def test_stage2_rejects_missing_excluded_column() -> None:
    with pytest.raises(ValueError, match="absent: missing"):
        drop_excluded_feature_columns(pd.DataFrame({"feature": [1]}), ["missing"])


def test_stage2_excludes_columns_before_learned_preparation() -> None:
    frame = pd.DataFrame(
        {
            "invoice_id": [f"invoice-{index}" for index in range(12)],
            "description": [f"item-{index}" for index in range(12)],
            "quantity": list(range(12)),
        }
    )

    raw_partitioned, model_partitioned = build_partitioned_stage2_inputs(
        frame,
        target_col=None,
        task_type="clustering",
        excluded_columns=["description", "invoice_id"],
        holdout_fraction=0.25,
        random_seed=42,
        split_strategy="random",
        time_column=None,
    )
    prepared, dropped, _ = prep_dataframe(
        model_partitioned,
        None,
        {"imputation_numeric": "median", "high_cardinality_max": 2},
        "clustering",
    )

    assert {"description", "invoice_id"}.issubset(raw_partitioned.columns)
    assert {"description", "invoice_id"}.isdisjoint(model_partitioned.columns)
    assert {"description", "invoice_id"}.isdisjoint(prepared.columns)
    assert dropped == []


def test_stage2_propagates_learned_schema_drops_to_raw_model_inputs() -> None:
    exclusions = resolve_raw_model_exclusions(
        ["customer_id"],
        ["invoice_date", "customer_id"],
    )
    frame = pd.DataFrame(
        {
            "customer_id": ["a", "b"],
            "invoice_date": ["2026-01-01", "2026-01-02"],
            "quantity": [1, 2],
        }
    )

    filtered = drop_excluded_feature_columns(frame, exclusions)

    assert exclusions == ["customer_id", "invoice_date"]
    assert filtered.columns.tolist() == ["quantity"]


def test_stage3_resolves_task_specific_baseline_alias() -> None:
    root = Path(__file__).resolve().parents[1]

    path = resolve_recipe_path(root, "recipe_baseline.yml", "classification")

    assert path is not None
    assert path.as_posix().endswith("configs/recipes/classification/baseline_recipe.yml")


def test_stage3_missing_explicit_recipe_returns_none() -> None:
    root = Path(__file__).resolve().parents[1]

    assert resolve_recipe_path(root, "missing_recipe.yml", "classification") is None


def test_stage4_resolves_task_specific_baseline_alias() -> None:
    root = Path(__file__).resolve().parents[1]

    path = resolve_stage4_recipe_path(root, "recipe_baseline.yml", "classification")

    assert path is not None
    assert path.as_posix().endswith("configs/recipes/classification/baseline_recipe.yml")


def test_stage3_anomaly_report_excludes_target_from_numeric_gate() -> None:
    input_df = pd.DataFrame({"feature": [1.0, 2.0], "target": ["yes", "no"]})
    output_df = pd.DataFrame({"feature": [0.1, 0.2], "target": ["yes", "no"]})

    report = build_preprocessing_anomaly_report(
        input_df,
        output_df,
        target_col="target",
        recipe_name="recipe_baseline.yml",
        recipe_path="configs/recipes/classification/baseline_recipe.yml",
        recipe_found=True,
    )

    assert report["status"] == "pass"
    assert report["non_numeric_feature_count"] == 0


def test_stage3_anomaly_report_flags_non_numeric_features() -> None:
    input_df = pd.DataFrame({"feature": ["a", "b"], "target": [1, 0]})
    output_df = pd.DataFrame({"feature": ["a", "b"], "target": [1, 0]})

    report = build_preprocessing_anomaly_report(
        input_df,
        output_df,
        target_col="target",
        recipe_name=None,
        recipe_path=None,
        recipe_found=False,
    )

    assert report["status"] == "fail"
    assert report["anomalies"][0]["code"] == "non_numeric_features_after_s03"


def test_stage3_anomaly_report_accepts_bool_indicator_features() -> None:
    input_df = pd.DataFrame({"category": ["a", "b"], "target": [1, 0]})
    output_df = pd.DataFrame({"category_b": [True, False], "target": [1, 0]})

    report = build_preprocessing_anomaly_report(
        input_df,
        output_df,
        target_col="target",
        recipe_name="recipe_baseline.yml",
        recipe_path="configs/recipes/classification/baseline_recipe.yml",
        recipe_found=True,
    )

    assert report["status"] == "pass"
    assert report["non_numeric_feature_count"] == 0


def test_stage3_anomaly_report_includes_warning_summaries() -> None:
    input_df = pd.DataFrame({"feature": [1.0, 2.0, 100.0], "target": [1, 0, 1]})
    output_df = pd.DataFrame({"feature": [0.1, 0.2, 8.0], "target": [1, 0, 1]})

    report = build_preprocessing_anomaly_report(
        input_df,
        output_df,
        target_col="target",
        recipe_name="recipe_baseline.yml",
        recipe_path="configs/recipes/classification/baseline_recipe.yml",
        recipe_found=True,
        multicollinearity={"vif_scores": {"feature": 12.5}},
        test_results={
            "outlier_analysis": {"feature": {"has_outliers": True, "outlier_percentage": 33.3}},
            "normality_tests": {"feature": {"is_normal": False}},
        },
    )

    assert report["status"] == "warn"
    assert report["multicollinearity_summary"]["high_vif_count"] == 1
    assert report["outlier_summary"]["features_with_outliers"] == 1
    assert report["distribution_summary"]["non_normal_or_skewed_count"] == 1


def test_config_schema_accepts_hardening_properties() -> None:
    validate_config(
        {
            "experiment_name": "hardening_test",
            "preset": "production",
            "task_type": "classification",
            "dataset": {"name": "demo", "target_column": "target"},
            "stage2": {"protected_columns": ["customer_id"], "high_cardinality_max": 200},
            "phases": {
                "phase_b": {"safety_net_review_required": True},
                "phase_c_hpo": {"optimizer": "optuna", "n_trials": 2, "compute_rate_usd_per_hour": 1.5},
            },
        }
    )


def test_config_schema_rejects_bad_protected_columns() -> None:
    with pytest.raises(Exception, match="protected_columns"):
        validate_config(
            {
                "experiment_name": "hardening_test",
                "preset": "production",
                "task_type": "classification",
                "dataset": {"name": "demo", "target_column": "target"},
                "stage2": {"protected_columns": ["customer_id", ""]},
            }
        )


def test_variant_yaml_validation_reports_task_mismatch() -> None:
    root = Path(__file__).resolve().parents[1]
    variant_path = root / "configs/recipes/classification/variant_search/variant_01ace0cb3ddd.yml"

    result = validate_variant_yaml(str(variant_path), task_type="regression")

    assert result["valid"] is False
    assert any("doesn't match expected" in error for error in result["errors"])


def test_variant_yaml_validation_reports_missing_file() -> None:
    result = validate_variant_yaml("/tmp/does-not-exist-variant.yml", task_type="classification")

    assert result["valid"] is False
    assert "not found" in result["errors"][0]


def test_phasec_cost_estimation_uses_wall_clock_when_trials_missing() -> None:
    class Study:
        trials = []

    cost = _compute_hpo_cost(Study(), started_at=0.0, compute_rate_usd_per_hour=2.0)

    assert cost["compute_rate_usd_per_hour"] == 2.0
    assert cost["estimated_cost_usd"] >= 0.0
    assert cost["estimation_basis"] == "wall_clock_elapsed"
