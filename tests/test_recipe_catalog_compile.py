from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from utils.recipe_catalog import (
    compile_recipe_catalog,
    select_catalog_entries,
)
from utils.fitted_variant_preprocessor import SUPPORTED_RECIPE_METHODS


def _recipe(method: str = "mean") -> dict:
    return {
        "recipe_name": "recipe",
        "version": "1.0",
        "description": "test recipe",
        "task_type": "classification",
        "stage3_preprocessing": {
            "imputation": {"method": method},
            "encoding": {"categorical_method": "onehot"},
            "scaling": {"method": "none"},
            "imbalance_handling": {"method": "none"},
        },
        "stage4_feature_engineering": {
            "feature_selection": {"method": "correlation", "threshold": 0.85}
        },
        "variant_metadata": {
            "variant_id": "recipe",
            "estimated_runtime_sec": 30,
        },
    }


def test_catalog_validates_deduplicates_and_logically_quarantines(tmp_path) -> None:
    task_root = tmp_path / "classification" / "variant_search"
    task_root.mkdir(parents=True)
    first = _recipe()
    duplicate = _recipe()
    duplicate["recipe_name"] = "duplicate-name"
    duplicate["description"] = "different prose, same semantics"
    duplicate["variant_metadata"]["variant_id"] = "duplicate"
    invalid = _recipe()
    invalid["stage3_preprocessing"]["encoding"]["categorical_method"] = (
        "unsupported_encoder"
    )
    for name, payload in (
        ("a.yml", first),
        ("b.yml", duplicate),
        ("invalid.yml", invalid),
    ):
        (task_root / name).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    catalog = compile_recipe_catalog(tmp_path, "classification")

    assert catalog.checked_count == 3
    assert catalog.valid_count == 2
    assert catalog.unique_count == 1
    assert len(catalog.duplicates) == 1
    assert len(catalog.quarantined) == 1
    assert (task_root / "invalid.yml").exists()
    assert len(catalog.catalog_hash) == 64


def test_real_catalog_selection_is_deterministic_safe_and_bounded() -> None:
    catalog = compile_recipe_catalog("configs/recipes", "classification")
    first = select_catalog_entries(
        catalog, library="variant_search", tier="progressive", max_variants=40
    )
    second = select_catalog_entries(
        catalog, library="variant_search", tier="progressive", max_variants=40
    )

    assert 1 <= len(first) <= 40
    assert [entry.semantic_hash for entry in first] == [
        entry.semantic_hash for entry in second
    ]
    assert len({entry.semantic_hash for entry in first}) == len(first)
    quarantined_paths = {entry.path for entry in catalog.quarantined}
    assert not quarantined_paths.intersection(entry.path for entry in first)


@pytest.mark.parametrize(
    ("section", "field", "method"),
    (
        ("encoding", "categorical_method", "ordinal"),
        ("encoding", "categorical_method", "frequency"),
        ("encoding", "categorical_method", "target"),
        ("scaling", "method", "maxabs"),
        ("scaling", "method", "power"),
    ),
)
def test_catalog_quarantines_methods_outside_fitted_inference_contract(
    tmp_path,
    section,
    field,
    method,
) -> None:
    task_root = tmp_path / "classification" / "variant_search"
    task_root.mkdir(parents=True)
    recipe = _recipe()
    recipe["stage3_preprocessing"][section][field] = method
    (task_root / "unsupported.yml").write_text(
        yaml.safe_dump(recipe, sort_keys=False),
        encoding="utf-8",
    )

    catalog = compile_recipe_catalog(tmp_path, "classification")

    assert catalog.valid_count == 0
    assert len(catalog.quarantined) == 1
    dimension = "encoding" if section == "encoding" else "scaling"
    assert method not in SUPPORTED_RECIPE_METHODS[dimension]


@pytest.mark.parametrize("selector", ("correlation", "mutual_info"))
def test_catalog_quarantines_target_dependent_clustering_selectors(
    tmp_path,
    selector,
) -> None:
    task_root = tmp_path / "clustering" / "variant_search"
    task_root.mkdir(parents=True)
    recipe = _recipe()
    recipe["task_type"] = "clustering"
    recipe["stage4_feature_engineering"]["feature_selection"] = {
        "method": selector,
        "threshold": 0.01,
    }
    (task_root / f"{selector}.yml").write_text(
        yaml.safe_dump(recipe, sort_keys=False),
        encoding="utf-8",
    )

    catalog = compile_recipe_catalog(tmp_path, "clustering")

    assert catalog.valid_count == 0
    assert len(catalog.quarantined) == 1
    assert "target-dependent feature selection" in " ".join(
        catalog.quarantined[0].errors
    )


def test_catalog_accepts_targetless_variance_selection_for_clustering(
    tmp_path,
) -> None:
    task_root = tmp_path / "clustering" / "variant_search"
    task_root.mkdir(parents=True)
    recipe = _recipe()
    recipe["task_type"] = "clustering"
    recipe["stage4_feature_engineering"]["feature_selection"] = {
        "method": "variance",
        "threshold": 0.0,
    }
    (task_root / "variance.yml").write_text(
        yaml.safe_dump(recipe, sort_keys=False),
        encoding="utf-8",
    )

    catalog = compile_recipe_catalog(tmp_path, "clustering")

    assert catalog.valid_count == 1
    assert catalog.unique_count == 1
