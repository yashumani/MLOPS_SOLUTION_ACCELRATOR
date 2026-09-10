import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from src.steps.phasec_optuna_hpo import (
    _write_skipped_unsupported,
    complete_phaseb_recipe,
    final_estimator_params,
    fit_final_model_with_hard_timeout,
    normalize_phaseb_algorithm,
    phasec_candidate_id,
    seeded_optuna_sampler,
)
from src.utils.common_evaluator import build_training_resampler


class SlowFinalEstimator:
    def fit(self, _X, _y):
        time.sleep(10)
        return self


def test_supported_phaseb_families_are_normalized_without_substitution():
    assert normalize_phaseb_algorithm("XGBClassifier") == "xgboost"
    assert normalize_phaseb_algorithm("Random Forest Classifier") == "randomforest"
    assert normalize_phaseb_algorithm("Logistic Regression") == "logisticregression"
    assert normalize_phaseb_algorithm("K-Means Clustering") == "kmeans"


def test_unknown_family_is_not_mapped_to_xgboost():
    assert normalize_phaseb_algorithm("ExtraTreesClassifier") is None
    assert normalize_phaseb_algorithm("unsupported") is None


def test_skipped_unsupported_preserves_phaseb_and_writes_no_model(tmp_path):
    args = SimpleNamespace(
        metrics_out=str(tmp_path / "metrics.json"),
        study_out=str(tmp_path / "study"),
        model_out=str(tmp_path / "model"),
    )
    _write_skipped_unsupported(
        args,
        "unsupported_phaseb_algorithm_family",
        {"candidate_id": "phaseb-1", "algorithm": "extra_trees"},
    )
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["status"] == "skipped_unsupported"
    assert payload["preserve_phaseb"] is True
    assert payload["phaseb_candidate_id"] == "phaseb-1"
    assert not (tmp_path / "model" / "model.pkl").exists()


def test_tuned_candidate_identity_is_distinct_and_parameter_bound():
    first = phasec_candidate_id(
        "phaseb-1",
        "randomforest",
        {"max_depth": 4},
    )
    same = phasec_candidate_id(
        "phaseb-1",
        "randomforest",
        {"max_depth": 4},
    )
    other = phasec_candidate_id(
        "phaseb-1",
        "randomforest",
        {"max_depth": 8},
    )

    assert first == same
    assert first != "phaseb-1"
    assert first != other
    assert ":tuned:" in first


def test_phasec_requires_complete_recipe_mapping():
    recipe = {
        "stage3_preprocessing": {
            "imputation": {"method": "median"},
            "encoding": {"categorical_method": "onehot"},
            "scaling": {"method": "standard"},
            "imbalance_handling": {"method": "none"},
        },
        "stage4_feature_engineering": {
            "feature_selection": {"method": "none"},
        },
    }
    assert complete_phaseb_recipe({"recipe": recipe}) == recipe
    assert complete_phaseb_recipe({"recipe": "recipe.yml"}) is None

    partial = json.loads(json.dumps(recipe))
    del partial["stage3_preprocessing"]["imbalance_handling"]
    assert complete_phaseb_recipe({"recipe": partial}) is None

    partial = json.loads(json.dumps(recipe))
    del partial["stage4_feature_engineering"]["feature_selection"]
    assert complete_phaseb_recipe({"recipe": partial}) is None


def test_phasec_sampler_and_final_estimators_reapply_locked_seed():
    assert seeded_optuna_sampler(73).__class__.__name__ == "TPESampler"
    assert final_estimator_params(
        "xgboost",
        {"max_depth": 4},
        73,
    )["random_state"] == 73
    assert final_estimator_params(
        "catboost",
        {"depth": 4},
        73,
    )["random_seed"] == 73
    assert final_estimator_params(
        "logisticregression",
        {"C": 1.0},
        73,
    )["random_state"] == 73


def test_phasec_final_fit_uses_killable_deadline():
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="hard deadline"):
        fit_final_model_with_hard_timeout(
            SlowFinalEstimator(),
            np.asarray([[0.0], [1.0]]),
            np.asarray([0.0, 1.0]),
            resampler=None,
            timeout_seconds=0.1,
        )
    assert time.monotonic() - started < 6.0


def test_phasec_final_fit_returns_fitted_estimator():
    fitted = fit_final_model_with_hard_timeout(
        DummyRegressor(strategy="mean"),
        np.asarray([[0.0], [1.0]]),
        np.asarray([0.0, 2.0]),
        resampler=None,
        timeout_seconds=10.0,
    )
    assert fitted.predict([[3.0]]).tolist() == [1.0]


@pytest.mark.parametrize("task", ["regression", "clustering"])
@pytest.mark.parametrize("imbalance", [None, "absent", {"method": "none"}])
def test_nonclassification_recipe_without_resampling_is_complete(task, imbalance):
    recipe = {
        "task_type": task,
        "stage3_preprocessing": {
            "imputation": {"method": "mean"},
            "encoding": {"categorical_method": "label"},
            "scaling": {"method": "standard"},
            "outlier_handling": None,
        },
        "stage4_feature_engineering": {"feature_selection": {"method": "none"}},
    }
    if imbalance != "absent":
        recipe["stage3_preprocessing"]["imbalance_handling"] = imbalance
    before = json.dumps(recipe, sort_keys=True)
    actual = complete_phaseb_recipe({"full_recipe": recipe, "task_type": task}, task_type=task)
    assert actual == recipe
    assert actual is not recipe
    assert json.dumps(recipe, sort_keys=True) == before
    assert build_training_resampler(actual, 42) is None


@pytest.mark.parametrize("task", ["regression", "clustering"])
@pytest.mark.parametrize("imbalance", [{}, {"method": "smote"}, "none", False])
def test_nonclassification_recipe_rejects_invalid_or_active_resampling(task, imbalance):
    recipe = {
        "task_type": task,
        "stage3_preprocessing": {
            "imputation": {"method": "mean"},
            "encoding": {"categorical_method": "label"},
            "scaling": {"method": "standard"},
            "imbalance_handling": imbalance,
        },
        "stage4_feature_engineering": {"feature_selection": {"method": "none"}},
    }
    assert complete_phaseb_recipe({"full_recipe": recipe}, task_type=task) is None


@pytest.mark.parametrize("task", ["classification", "unknown"])
def test_recipe_task_cannot_be_reinterpreted_to_bypass_resampling_contract(task):
    recipe = {
        "task_type": "regression",
        "stage3_preprocessing": {
            "imputation": {"method": "mean"},
            "encoding": {"categorical_method": "label"},
            "scaling": {"method": "standard"},
            "imbalance_handling": None,
        },
        "stage4_feature_engineering": {"feature_selection": {"method": "none"}},
    }
    assert complete_phaseb_recipe({"recipe": recipe}, task_type=task) is None


@pytest.mark.parametrize("task", ["regression", "clustering"])
def test_nonclassification_still_requires_learned_transform_configuration(task):
    recipe = {
        "task_type": task,
        "stage3_preprocessing": {
            "imputation": {"method": "mean"},
            "encoding": {"categorical_method": "label"},
            "scaling": {"method": "standard"},
            "imbalance_handling": None,
        },
        "stage4_feature_engineering": {"feature_selection": {"method": "none"}},
    }
    del recipe["stage3_preprocessing"]["encoding"]
    assert complete_phaseb_recipe({"recipe": recipe}, task_type=task) is None
