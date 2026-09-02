from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from steps.s06_phaseb_variant_runner import (
    LEGACY_VARIANTS_LIST_MAX_CHARS,
    load_candidate_catalog,
    validate_candidate_catalog_binding,
)


def test_execution_manifest_and_candidate_catalog_are_wired_end_to_end() -> None:
    submit = Path("pipelines/submit_pipeline.py").read_text(encoding="utf-8")
    builder = Path("pipelines/pipeline_builder.py").read_text(encoding="utf-8")
    component = yaml.safe_load(
        Path("components/s06_phaseb_variant_runner.yml").read_text(
            encoding="utf-8"
        )
    )
    runner = Path("src/steps/s06_phaseb_variant_runner.py").read_text(
        encoding="utf-8"
    )

    assert "candidate_catalog=candidate_catalog_input" in submit
    assert 'variants_list=""' in submit
    assert "execution_manifest=execution_manifest_input" in submit
    assert submit.count("_sdk_local_file_input(") == 5
    assert submit.count("datastore=default_datastore") >= 4
    assert builder.count('execution_manifest: Input(type="uri_file"') == 2
    assert builder.count('candidate_catalog: Input(type="uri_file"') == 2
    assert builder.count('"execution_manifest": s06.outputs.execution_manifest_out') == 2
    assert component["inputs"]["execution_manifest"]["type"] == "uri_file"
    assert component["inputs"]["candidate_catalog"]["type"] == "uri_file"
    assert component["inputs"]["variants_list"]["type"] == "string"
    assert component["inputs"]["variants_list"]["optional"] is True
    assert component["outputs"]["execution_manifest_out"]["type"] == "uri_file"
    assert "--execution_manifest" in component["command"]
    assert "--candidate_catalog" in component["command"]
    assert "--variants_list" in component["command"]
    assert "--candidate_catalog" in runner
    assert "validate_candidate_catalog_binding(" in runner
    assert "LEGACY_VARIANTS_LIST_MAX_CHARS" in runner
    assert "validate_execution_manifest_for_run(" in runner
    assert "Recipe funnel produced no eligible Round 2 variants" in runner


def test_candidate_catalog_file_has_no_string_transport_limit(tmp_path) -> None:
    recipe_paths = [
        f"classification/variant_{index:04d}_{'x' * 40}.yml"
        for index in range(151)
    ]
    payload = {
        "execution_id": "execution",
        "recipe_catalog_hash": "catalog",
        "recipe_paths": recipe_paths,
        "recipe_ids": [f"recipe-{index}" for index in range(151)],
        "candidate_ids": ["candidate"],
        "candidate_records": [{"candidate_id": "candidate"}],
    }
    path = tmp_path / "candidate_catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, loaded_paths = load_candidate_catalog(path)

    assert len(",".join(loaded_paths)) > LEGACY_VARIANTS_LIST_MAX_CHARS
    assert loaded_paths == recipe_paths
    validate_candidate_catalog_binding(dict(payload), loaded)


def test_candidate_catalog_substitution_fails_closed() -> None:
    payload = {
        "execution_id": "execution",
        "recipe_catalog_hash": "catalog",
        "recipe_paths": ["classification/variant.yml"],
        "recipe_ids": ["recipe"],
        "candidate_ids": ["candidate"],
        "candidate_records": [{"candidate_id": "candidate"}],
    }
    substituted = dict(payload)
    substituted["candidate_ids"] = ["other"]

    with pytest.raises(ValueError, match="candidate_ids"):
        validate_candidate_catalog_binding(payload, substituted)


def test_drift_baseline_uses_training_partition_in_both_graphs() -> None:
    builder = Path("pipelines/pipeline_builder.py").read_text(encoding="utf-8")
    tree = ast.parse(builder)
    drift_inputs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "s13_kwargs"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        drift_inputs.append(
            {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in node.value.keywords
                if keyword.arg
            }
        )

    assert len(drift_inputs) == 2
    assert all(
        inputs["dataset_in"] == "s4.outputs.train_out"
        for inputs in drift_inputs
    )
    assert "dataset_in=s4.outputs.dataset_out" not in builder


def test_s06_component_budget_defaults_match_locked_caps() -> None:
    component = yaml.safe_load(
        Path("components/s06_phaseb_variant_runner.yml").read_text(
            encoding="utf-8"
        )
    )
    inputs = component["inputs"]

    assert inputs["time_budget_per_variant"]["default"] <= 600
    assert inputs["phaseb_time_budget_sec"]["default"] == 10800
    assert inputs["round1_max_variants"]["default"] == 40
    assert inputs["round2_max_variants"]["default"] == 8
