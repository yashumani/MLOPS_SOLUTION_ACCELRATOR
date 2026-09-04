#!/usr/bin/env python3
"""Audit the governed qualification data assets without modifying Azure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT / "src", Path(__file__).resolve().parent):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)

from _azure_ctx import load_azure_context  # noqa: E402
from register_qualification_data_assets import build_asset_plan  # noqa: E402
from utils.azure_helper import get_ml_client  # noqa: E402


DEFAULT_CANDIDATES = (
    ROOT / "configs" / "qualification" / "industry_matrix_candidates.yml"
)
DEFAULT_CATALOG = (
    ROOT / "configs" / "qualification" / "industry_matrix_execution_catalog.yml"
)
DEFAULT_VERSION = "20260902.1"


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return payload


def build_profile_report(
    candidates: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    catalog_records = catalog.get("configs")
    candidate_records = candidates.get("scenarios")
    if not isinstance(catalog_records, list) or not isinstance(candidate_records, list):
        raise ValueError("Qualification catalog and candidates must contain lists")
    catalog_ids = [str(item.get("scenario_id") or "") for item in catalog_records]
    candidate_ids = [str(item.get("id") or "") for item in candidate_records]
    if (
        len(catalog_ids) != 15
        or len(candidate_ids) != 15
        or len(set(catalog_ids)) != 15
        or len(set(candidate_ids)) != 15
        or set(catalog_ids) != set(candidate_ids)
    ):
        raise ValueError(
            "Data asset audit requires the same 15 unique catalog and candidate scenarios"
        )
    expected = {
        str(item["scenario_id"]): item for item in catalog_records
    }
    dispositions = (
        (candidates.get("privacy_review") or {}).get("scenario_dispositions") or {}
    )
    if not isinstance(dispositions, dict) or set(dispositions) != set(candidate_ids):
        raise ValueError("Privacy dispositions must cover exactly the 15 scenarios")
    profiles: list[dict[str, Any]] = []
    for scenario in candidate_records:
        scenario_id = str(scenario.get("id") or "")
        record = expected.get(scenario_id)
        disposition = dispositions.get(scenario_id) or {}
        if any(
            str(scenario.get(field) or "") != str(record.get(field) or "")
            for field in ("task_type", "industry")
        ):
            raise ValueError(
                f"Catalog task or industry differs for scenario {scenario_id!r}"
            )
        if disposition.get("status") != "approved_for_nonproduction_qualification":
            raise ValueError(f"Scenario {scenario_id!r} lacks privacy approval")
        for field in ("dataset_content_sha256", "dataset_schema_sha256"):
            digest = str(record.get(field) or "").lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Scenario {scenario_id!r} has invalid {field}")
        profiles.append(
            {
                "id": scenario_id,
                "qualification_status": "schema_pass",
                "privacy_review_status": disposition.get("status"),
                "content_sha256": record.get("dataset_content_sha256"),
                "schema_sha256": record.get("dataset_schema_sha256"),
            }
        )
    return {"scenarios": profiles}


def _path_matches(actual: object, expected: str) -> bool:
    normalized_actual = str(actual or "").replace("\\", "/").rstrip("/").lower()
    normalized_expected = expected.replace("\\", "/").rstrip("/").lower()
    if normalized_actual == normalized_expected:
        return True
    suffix = normalized_expected.removeprefix("azureml://")
    return normalized_actual.startswith("azureml://") and normalized_actual.endswith(
        "/" + suffix
    )


def audit_assets(client: Any, plan: list[dict[str, Any]]) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    errors: list[str] = []
    for expected in plan:
        name = str(expected["name"])
        version = str(expected["version"])
        try:
            observed = client.data.get(name=name, version=version)
            tags = {str(key): str(value) for key, value in (observed.tags or {}).items()}
            expected_tags = {
                str(key): str(value) for key, value in expected["tags"].items()
            }
            checks = {
                "version": str(observed.version) == version,
                "asset_type": str(getattr(observed, "type", "")).lower()
                == "uri_file",
                "datastore_path": _path_matches(observed.path, expected["path"]),
                "scope": tags.get("evidence_scope") == "qualification"
                and tags.get("privacy_scope") == "approved-nonproduction-only",
                "governance_tags": all(
                    tags.get(field) == expected_tags[field]
                    for field in ("profile_run_id", "source_dataset_id", "license")
                ),
                "hashes": tags.get("content_sha256")
                == expected_tags["content_sha256"]
                and tags.get("schema_sha256") == expected_tags["schema_sha256"],
                "scenario_tags": all(
                    tags.get(field) == expected_tags[field]
                    for field in ("scenario_ids", "task_types", "industries")
                ),
            }
            failed = [field for field, passed in checks.items() if not passed]
            if failed:
                errors.append(f"{name}:{version} failed {', '.join(failed)}")
            assets.append(
                {
                    "name": name,
                    "version": str(observed.version),
                    "path": str(observed.path),
                    "scenario_ids": expected_tags["scenario_ids"].split(","),
                    "task_types": expected_tags["task_types"].split(","),
                    "industries": expected_tags["industries"].split(","),
                    "content_sha256": expected_tags["content_sha256"],
                    "schema_sha256": expected_tags["schema_sha256"],
                    "checks": checks,
                }
            )
        except Exception as exc:  # noqa: BLE001 - every failed live read is evidence
            errors.append(f"{name}:{version} read failed: {type(exc).__name__}: {exc}")
    scenario_count = sum(len(item["scenario_ids"]) for item in assets)
    return {
        "schema_version": "1.0",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_version": plan[0]["version"] if plan else None,
        "asset_count": len(assets),
        "scenario_count": scenario_count,
        "all_passed": not errors and len(assets) == len(plan),
        "errors": errors,
        "assets": assets,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--datastore", default="mlops_blob")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        candidates = _read_yaml(args.candidates.resolve())
        catalog = _read_yaml(args.catalog.resolve())
        profile_run_id = str(catalog.get("profile_run_id") or "").strip()
        if not profile_run_id:
            raise ValueError("Qualification catalog profile_run_id is required")
        profile = build_profile_report(candidates, catalog)
        plan = build_asset_plan(
            candidates,
            profile,
            datastore=args.datastore,
            version=args.version,
            profile_run_id=profile_run_id,
        )
        context = load_azure_context()
        client = get_ml_client(
            context.subscription_id,
            context.resource_group,
            context.workspace_name,
        )
        report = audit_assets(client, plan)
    except Exception as exc:  # noqa: BLE001 - report construction must fail closed
        report = {
            "schema_version": "1.0",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "asset_count": 0,
            "scenario_count": 0,
            "all_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "assets": [],
        }
    _write_json_atomic(args.output_json.resolve(), report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
