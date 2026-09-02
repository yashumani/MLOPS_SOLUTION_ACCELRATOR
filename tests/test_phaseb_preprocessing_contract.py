"""Regression tests for the shared Phase B train/holdout transformer contract."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.steps.s06_phaseb_variant_runner import (
    VariantResult,
    count_distinct_phaseb_candidates,
    fit_round1_proxy_preprocessor,
    is_usable_phaseb_result,
    require_valid_phaseb_results,
)
from src.utils.common_evaluator import build_fold_local_pipeline
from src.utils.fitted_variant_preprocessor import FittedVariantPreprocessor


def _variant(
    *,
    encoding: str = "none",
    scaling: str = "none",
    task_type: str = "classification",
):
    return {
        "recipe_name": "unit-test",
        "task_type": task_type,
        "stage3_preprocessing": {
            "imputation": {"method": "median"},
            "outlier_handling": {"method": "none"},
            "encoding": {"categorical_method": encoding},
            "scaling": {"method": scaling},
            "imbalance_handling": {"method": "none"},
        },
        "stage4_feature_engineering": {
            "feature_selection": {"method": "none", "threshold": 0.01},
        },
    }


class _VariantStub:
    variant_id = "unit-test"

    def __init__(self, recipe):
        self.recipe = recipe

    def to_dict(self):
        return self.recipe


def _transform_pair(train, holdout, recipe):
    preprocessor = FittedVariantPreprocessor(recipe, random_seed=42)
    transformed_train = preprocessor.fit_transform(
        train.drop(columns=["target"]),
        train["target"],
    )
    transformed_holdout = preprocessor.transform(
        holdout.drop(columns=["target"])
    )
    transformed_train["target"] = train["target"].values
    transformed_holdout["target"] = holdout["target"].values
    return transformed_train, transformed_holdout


def test_label_encoder_and_scaler_are_shared():
    train = pd.DataFrame(
        {
            "category": ["a", "a", "b", "b"],
            "numeric": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 0, 1, 1],
        }
    )
    holdout = pd.DataFrame(
        {
            "category": ["a", "b"],
            "numeric": [1.5, 3.5],
            "target": [0, 1],
        }
    )

    transformed_train, transformed_holdout = _transform_pair(
        train,
        holdout,
        _variant(encoding="label", scaling="standard"),
    )

    assert transformed_holdout["category"].iloc[0] == pytest.approx(
        transformed_train.loc[train["category"].eq("a"), "category"].iloc[0]
    )
    assert transformed_holdout["category"].iloc[1] == pytest.approx(
        transformed_train.loc[train["category"].eq("b"), "category"].iloc[0]
    )


def test_onehot_reference_category_is_shared():
    train = pd.DataFrame(
        {
            "category": ["a", "b", "c"],
            "numeric": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )
    holdout = pd.DataFrame(
        {
            "category": ["b", "c"],
            "numeric": [4.0, 5.0],
            "target": [1, 0],
        }
    )

    transformed_train, transformed_holdout = _transform_pair(
        train,
        holdout,
        _variant(encoding="onehot"),
    )

    assert list(transformed_train.columns) == list(transformed_holdout.columns)
    assert transformed_holdout["category_b"].tolist() == [True, False]
    assert transformed_holdout["category_c"].tolist() == [False, True]


def test_preprocessor_detaches_read_only_input_buffers():
    backing = np.arange(24, dtype=float).reshape(8, 3)
    backing.flags.writeable = False
    frame = pd.DataFrame(
        backing,
        columns=["first", "second", "third"],
        copy=False,
    )

    transformed = FittedVariantPreprocessor(
        _variant(scaling="standard"),
        random_seed=42,
    ).fit_transform(frame, pd.Series([0, 1] * 4))

    assert transformed.to_numpy(copy=False).flags.writeable is True
    transformed.iloc[0, 0] = -999.0
    assert backing[0, 0] == 0.0


@pytest.mark.parametrize("scaling", ("yeo_johnson", "quantile"))
def test_declared_scaler_is_fitted_on_training_and_applied_to_holdout(scaling):
    train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 0, 1, 1],
        }
    )
    holdout = pd.DataFrame({"numeric": [10.0], "target": [1]})

    transformed_train, transformed_holdout = _transform_pair(
        train,
        holdout,
        _variant(scaling=scaling),
    )

    if scaling == "yeo_johnson":
        from sklearn.preprocessing import PowerTransformer

        expected = PowerTransformer(
            method="yeo-johnson",
            standardize=True,
        ).fit(train[["numeric"]])
    else:
        from sklearn.preprocessing import QuantileTransformer

        expected = QuantileTransformer(
            output_distribution="normal",
            random_state=42,
        ).fit(train[["numeric"]])

    assert transformed_holdout["numeric"].iloc[0] == pytest.approx(
        expected.transform(holdout[["numeric"]])[0, 0]
    )
    assert np.isfinite(transformed_train["numeric"]).all()


def test_no_valid_results_fails_instead_of_publishing_placeholders():
    with pytest.raises(
        RuntimeError,
        match="cross-validation selection evidence is mandatory",
    ):
        require_valid_phaseb_results([])


def test_phaseb_requires_multiple_distinct_comparable_candidates():
    first = VariantResult(
        variant_id="recipe-a",
        engine="pycaret",
        algorithm="model-a",
        metrics={"primary_metric": 0.75},
        runtime_sec=1.0,
        timed_out=False,
        failed=False,
        candidate_id="candidate-a",
    )
    duplicate = VariantResult(
        variant_id="recipe-a",
        engine="pycaret",
        algorithm="model-a",
        metrics={"primary_metric": 0.75},
        runtime_sec=1.0,
        timed_out=False,
        failed=False,
        candidate_id="candidate-a",
    )
    second = VariantResult(
        variant_id="recipe-b",
        engine="flaml",
        algorithm="model-b",
        metrics={"primary_metric": 0.74},
        runtime_sec=1.0,
        timed_out=False,
        failed=False,
        candidate_id="candidate-b",
    )

    assert count_distinct_phaseb_candidates([first, duplicate]) == 1
    with pytest.raises(RuntimeError, match="at least 2"):
        require_valid_phaseb_results(
            [first, duplicate],
            minimum_candidates=2,
        )
    require_valid_phaseb_results([first, second], minimum_candidates=2)


def test_component_does_not_mask_missing_phaseb_evidence():
    component = Path("components/s06_phaseb_variant_runner.yml").read_text(
        encoding="utf-8"
    )

    assert yaml.safe_load(component)["version"] == 14
    assert 'echo "[]"' not in component
    assert 'echo "{}"' not in component
    assert "--leaderboard_out ${{outputs.leaderboard_csv}}" in component
    assert "--all_results_out ${{outputs.all_results_json}}" in component
    assert "&& cp " not in component


def test_phaseb_main_does_not_read_cache_hit_before_assignment():
    source = Path("src/steps/s06_phaseb_variant_runner.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    cache_hit_loads = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Name)
        and node.id == "cache_hit"
        and isinstance(node.ctx, ast.Load)
    ]
    cache_hit_stores = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Name)
        and node.id == "cache_hit"
        and isinstance(node.ctx, ast.Store)
    ]

    assert not cache_hit_loads or (
        cache_hit_stores
        and min(cache_hit_stores) < min(cache_hit_loads)
    )


def test_s06_common_evaluator_pipeline_applies_recipe_sampler_fold_locally():
    from sklearn.linear_model import LogisticRegression

    recipe = _variant()
    recipe["stage3_preprocessing"]["imbalance_handling"] = {
        "method": "smote"
    }
    pipeline = build_fold_local_pipeline(
        FittedVariantPreprocessor(recipe, random_seed=73),
        LogisticRegression(random_state=73),
        recipe=recipe,
        task_type="classification",
        random_seed=73,
    )
    source = Path("src/steps/s06_phaseb_variant_runner.py").read_text(
        encoding="utf-8"
    )

    assert pipeline.__class__.__module__ == "imblearn.pipeline"
    assert list(pipeline.named_steps) == [
        "preprocessing",
        "resampler",
        "estimator",
    ]
    assert pipeline.named_steps["resampler"].random_state == 73
    assert "evaluation_pipeline = build_fold_local_pipeline(" in source


@pytest.mark.parametrize("score", (-0.42, 0.0, 0.005))
def test_weak_finite_phaseb_results_remain_eligible_for_s10(score):
    result = VariantResult(
        variant_id="weak-but-valid",
        engine="pycaret",
        algorithm="candidate",
        metrics={"primary_metric": score},
        runtime_sec=1.0,
        timed_out=False,
        failed=False,
    )

    assert is_usable_phaseb_result(result) is True


def test_non_finite_phaseb_result_is_not_eligible():
    result = VariantResult(
        variant_id="invalid",
        engine="pycaret",
        algorithm="candidate",
        metrics={"primary_metric": float("nan")},
        runtime_sec=1.0,
        timed_out=False,
        failed=False,
    )

    assert is_usable_phaseb_result(result) is False


def test_targetless_variance_selector_executes_for_clustering():
    recipe = _variant()
    recipe["stage4_feature_engineering"]["feature_selection"] = {
        "method": "variance",
        "threshold": 0.0,
    }
    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "varying": [0.0, 1.0, 2.0, 3.0],
        }
    )

    transformed = FittedVariantPreprocessor(
        recipe,
        random_seed=42,
    ).fit_transform(frame, None)

    assert list(transformed.columns) == ["varying"]


def test_targetless_correlation_selector_fails_closed():
    recipe = _variant()
    recipe["stage4_feature_engineering"]["feature_selection"] = {
        "method": "correlation",
        "threshold": 0.1,
    }

    with pytest.raises(ValueError, match="requires a target"):
        FittedVariantPreprocessor(recipe, random_seed=42).fit(
            pd.DataFrame({"feature": [1.0, 2.0, 3.0]}),
            None,
        )


def test_round1_proxy_preprocessor_fits_only_proxy_training_rows():
    recipe = _variant(scaling="standard")
    train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    validation = pd.DataFrame({"feature": [100.0, 200.0]})
    changed_validation = pd.DataFrame({"feature": [1_000_000.0, 2_000_000.0]})
    target = pd.Series([0, 0, 1, 1])

    _, _, first = fit_round1_proxy_preprocessor(
        train,
        validation,
        target,
        _VariantStub(recipe),
        random_seed=42,
    )
    _, _, second = fit_round1_proxy_preprocessor(
        train,
        changed_validation,
        target,
        _VariantStub(recipe),
        random_seed=42,
    )

    assert first.scaler_.mean_.tolist() == pytest.approx([1.5])
    assert second.scaler_.mean_.tolist() == pytest.approx([1.5])
    assert first.selected_columns_ == second.selected_columns_


@pytest.mark.parametrize(
    ("task_type", "target", "expected_scorer"),
    [
        ("regression", pd.Series([0.0, 1.0, 2.0, 3.0]), "regression"),
        ("classification", pd.Series(range(40)), "classification"),
    ],
)
def test_mutual_information_uses_recipe_task_type(
    monkeypatch,
    task_type,
    target,
    expected_scorer,
):
    calls = []

    def classification_scorer(features, labels, random_state):
        calls.append("classification")
        return np.ones(features.shape[1])

    def regression_scorer(features, labels, random_state):
        calls.append("regression")
        return np.ones(features.shape[1])

    monkeypatch.setattr(
        "sklearn.feature_selection.mutual_info_classif",
        classification_scorer,
    )
    monkeypatch.setattr(
        "sklearn.feature_selection.mutual_info_regression",
        regression_scorer,
    )
    recipe = _variant(task_type=task_type)
    recipe["stage4_feature_engineering"]["feature_selection"] = {
        "method": "mutual_info",
        "threshold": 0.0,
    }
    frame = pd.DataFrame(
        {
            "feature": np.linspace(0.0, 1.0, len(target)),
        }
    )

    FittedVariantPreprocessor(recipe, random_seed=42).fit(frame, target)

    assert calls == [expected_scorer]


def test_zero_feature_selection_threshold_is_preserved():
    recipe = _variant(task_type="clustering")
    recipe["stage4_feature_engineering"]["feature_selection"] = {
        "method": "variance",
        "threshold": 0.0,
    }
    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "small_variance": [0.0, 0.0, 0.0, 0.1],
        }
    )

    transformed = FittedVariantPreprocessor(
        recipe,
        random_seed=42,
    ).fit_transform(frame, None)

    assert list(transformed.columns) == ["small_variance"]


def test_empty_feature_selection_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "sklearn.feature_selection.mutual_info_classif",
        lambda features, labels, random_state: np.zeros(features.shape[1]),
    )
    recipe = _variant(task_type="classification")
    recipe["stage4_feature_engineering"]["feature_selection"] = {
        "method": "mutual_info",
        "threshold": 0.1,
    }

    with pytest.raises(ValueError, match="removed all features"):
        FittedVariantPreprocessor(recipe, random_seed=42).fit(
            pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}),
            pd.Series([0, 0, 1, 1]),
        )
