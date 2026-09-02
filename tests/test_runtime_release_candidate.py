from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "config" / "mlops_v3_unified_environment"
EXPECTED_ENVIRONMENT = "mlops-v3-unified:33"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _canonical_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_active_runtime_matches_versioned_v33_and_declared_hash() -> None:
    active_conda = RUNTIME_ROOT / "conda.yml"
    versioned_conda = RUNTIME_ROOT / "conda_v33.yml"
    active_text = _canonical_text(active_conda)

    assert active_text == _canonical_text(versioned_conda)
    conda_sha256 = hashlib.sha256(active_text.encode("utf-8")).hexdigest()

    for filename, expected_conda_file in (
        ("environment.yml", "conda.yml"),
        ("environment_v33.yml", "conda_v33.yml"),
    ):
        environment = _load_yaml(RUNTIME_ROOT / filename)
        assert environment["name"] == "mlops-v3-unified"
        assert environment["version"] == 33
        assert environment["conda_file"] == expected_conda_file
        assert environment["tags"]["conda_sha256"] == conda_sha256


def test_industry_matrix_has_five_v33_scenarios_per_task_type() -> None:
    catalog = _load_yaml(
        REPO_ROOT
        / "configs"
        / "qualification"
        / "industry_matrix_execution_catalog.yml"
    )
    scenarios = catalog["configs"]

    assert catalog["environment"] == EXPECTED_ENVIRONMENT
    assert catalog["scenario_count"] == 15
    assert Counter(item["task_type"] for item in scenarios) == {
        "classification": 5,
        "regression": 5,
        "clustering": 5,
    }

    for scenario in scenarios:
        config = _load_yaml(REPO_ROOT / scenario["config_path"])
        assert config["task_type"] == scenario["task_type"]
        assert config["azureml"]["environment"] == EXPECTED_ENVIRONMENT
