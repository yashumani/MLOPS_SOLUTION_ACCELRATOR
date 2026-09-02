from __future__ import annotations

from scripts.generate_industry_qualification_configs import _build_config


def test_build_config_binds_qualified_identity_and_exclusions() -> None:
    scenario = {
        "id": "classification-manufacturing-machine-failure",
        "task_type": "classification",
        "industry": "manufacturing",
        "blob_path": "qualification/source-data/ai4i.csv",
        "target_column": "Machine failure",
        "exclude_columns": ["UDI", "Product ID"],
    }
    profile = {
        "id": scenario["id"],
        "qualification_status": "schema_pass",
        "privacy_review_status": "approved_for_nonproduction_qualification",
        "content_sha256": "a" * 64,
        "schema_sha256": "b" * 64,
    }

    config = _build_config(
        scenario,
        profile,
        environment="mlops-v3-unified:29",
        profile_run_id="profile-run",
    )

    assert config["dataset"]["content_sha256"] == "a" * 64
    assert config["dataset"]["excluded_columns"] == ["UDI", "Product ID"]
    assert config["azureml"]["environment"] == "mlops-v3-unified:29"
    assert config["registry"]["pass_aliases"] == []
    assert config["registry"]["warning_aliases"] == []


def test_build_clustering_config_uses_pycaret_only() -> None:
    scenario = {
        "id": "clustering-retail-segments",
        "task_type": "clustering",
        "industry": "retail",
        "blob_path": "qualification/source-data/retail.csv",
        "exclude_columns": ["CustomerID"],
    }
    profile = {
        "id": scenario["id"],
        "qualification_status": "schema_pass",
        "privacy_review_status": "approved_for_nonproduction_qualification",
        "content_sha256": "a" * 64,
        "schema_sha256": "b" * 64,
    }

    config = _build_config(
        scenario,
        profile,
        environment="mlops-v3-unified:29",
        profile_run_id="profile-run",
    )

    assert config["phases"]["phase_b"]["engines"] == ["pycaret"]
    assert "target_column" not in config["dataset"]
