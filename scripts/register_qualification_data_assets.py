#!/usr/bin/env python3
"""Register the qualified release datasets as versioned Azure ML data assets."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def build_asset_plan(
    manifest: dict[str, Any],
    report: dict[str, Any],
    *,
    datastore: str,
    version: str,
    profile_run_id: str,
) -> list[dict[str, Any]]:
    scenarios = manifest.get("scenarios") or []
    profiles = {str(item["id"]): item for item in report.get("scenarios") or []}
    if len(scenarios) != 15 or len(profiles) != 15:
        raise ValueError("Asset registration requires exactly 15 scenarios and profiles")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        profile = profiles.get(scenario_id)
        if not profile:
            raise ValueError(f"Profile missing scenario {scenario_id}")
        if profile.get("qualification_status") != "schema_pass":
            raise ValueError(f"Scenario {scenario_id} is not qualified")
        if profile.get("privacy_review_status") != (
            "approved_for_nonproduction_qualification"
        ):
            raise ValueError(f"Scenario {scenario_id} lacks privacy approval")
        grouped[str(scenario["blob_path"])].append((scenario, profile))

    plan: list[dict[str, Any]] = []
    for blob_path, items in sorted(grouped.items()):
        scenario, profile = items[0]
        provenance = scenario.get("provenance") or {}
        source_dataset_id = str(provenance["source_dataset_id"])
        content_sha256 = str(profile["content_sha256"])
        schema_sha256 = str(profile["schema_sha256"])
        for other_scenario, other_profile in items[1:]:
            other_provenance = other_scenario.get("provenance") or {}
            if (
                str(other_provenance.get("source_dataset_id")) != source_dataset_id
                or str(other_profile.get("content_sha256")) != content_sha256
                or str(other_profile.get("schema_sha256")) != schema_sha256
            ):
                raise ValueError(f"Conflicting identities for shared blob path {blob_path}")

        scenario_ids = sorted(str(item[0]["id"]) for item in items)
        task_types = sorted({str(item[0]["task_type"]) for item in items})
        industries = sorted({str(item[0]["industry"]) for item in items})
        plan.append(
            {
                "name": f"mlops-v3-qualification-{_slug(source_dataset_id)}",
                "version": version,
                "path": f"azureml://datastores/{datastore}/paths/{blob_path}",
                "description": (
                    "Non-production MLOps V3 release qualification source for "
                    + ", ".join(scenario_ids)
                ),
                "tags": {
                    "evidence_scope": "qualification",
                    "profile_run_id": profile_run_id,
                    "source_dataset_id": source_dataset_id,
                    "license": str(provenance["license"]),
                    "content_sha256": content_sha256,
                    "schema_sha256": schema_sha256,
                    "scenario_ids": ",".join(scenario_ids),
                    "task_types": ",".join(task_types),
                    "industries": ",".join(industries),
                    "privacy_scope": "approved-nonproduction-only",
                },
            }
        )
    return plan


def _register(ml_client: MLClient, plan: list[dict[str, Any]]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for item in plan:
        try:
            existing = ml_client.data.get(name=item["name"], version=item["version"])
        except ResourceNotFoundError:
            existing = None
        if existing is not None:
            existing_path = str(existing.path)
            if existing_path != item["path"] or dict(existing.tags or {}) != item["tags"]:
                raise RuntimeError(
                    f"Existing data asset identity differs: {item['name']}:{item['version']}"
                )
            registered = existing
            action = "verified_existing"
        else:
            registered = ml_client.data.create_or_update(
                Data(
                    name=item["name"],
                    version=item["version"],
                    type=AssetTypes.URI_FILE,
                    path=item["path"],
                    description=item["description"],
                    tags=item["tags"],
                )
            )
            action = "created"
        results.append(
            {
                "name": str(registered.name),
                "version": str(registered.version),
                "id": str(registered.id),
                "action": action,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--profile-run-id", required=True)
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument("--datastore", default="mlops_blob")
    parser.add_argument("--version", default="20260902.1")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8")) or {}
    report = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    plan = build_asset_plan(
        manifest,
        report,
        datastore=args.datastore,
        version=args.version,
        profile_run_id=args.profile_run_id,
    )
    if not args.apply:
        print(json.dumps({"mode": "dry_run", "asset_count": len(plan), "plan": plan}, indent=2))
        return 0

    ml_client = MLClient(
        AzureCliCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )
    results = _register(ml_client, plan)
    print(json.dumps({"mode": "apply", "asset_count": len(results), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
