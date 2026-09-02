from __future__ import annotations

from scripts.register_qualification_data_assets import build_asset_plan


def test_build_asset_plan_deduplicates_shared_source() -> None:
    scenarios = []
    profiles = []
    for index in range(15):
        task_type = ("classification", "regression", "clustering")[index % 3]
        scenario_id = f"{task_type}-scenario-{index}"
        blob_path = "qualification/shared.csv" if index < 2 else f"qualification/{index}.csv"
        source_id = "shared-source" if index < 2 else f"source-{index}"
        scenarios.append(
            {
                "id": scenario_id,
                "task_type": task_type,
                "industry": f"industry-{index}",
                "blob_path": blob_path,
                "provenance": {
                    "source_dataset_id": source_id,
                    "license": "CC0",
                },
            }
        )
        profiles.append(
            {
                "id": scenario_id,
                "qualification_status": "schema_pass",
                "privacy_review_status": "approved_for_nonproduction_qualification",
                "content_sha256": ("a" if index < 2 else "b") * 64,
                "schema_sha256": ("c" if index < 2 else "d") * 64,
            }
        )

    plan = build_asset_plan(
        {"scenarios": scenarios},
        {"scenarios": profiles},
        datastore="mlops_blob",
        version="20260902.1",
        profile_run_id="profile-run",
    )

    assert len(plan) == 14
    shared = next(item for item in plan if item["name"].endswith("shared-source"))
    assert shared["path"] == "azureml://datastores/mlops_blob/paths/qualification/shared.csv"
    assert shared["tags"]["profile_run_id"] == "profile-run"
    assert shared["tags"]["scenario_ids"].count(",") == 1
