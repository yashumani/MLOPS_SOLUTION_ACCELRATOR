#!/usr/bin/env python3
"""Profile the 15-scenario release data matrix on Azure ML compute."""

from __future__ import annotations

import argparse
import base64
import csv
import gc
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.data_identity import canonical_dataframe_sha256  # noqa: E402


TASK_TYPES = {"classification", "regression", "clustering"}
PRIVACY_REVIEW_STATUSES = {
    "approved_for_nonproduction_qualification",
    "blocked",
}
DIRECT_IDENTIFIER_PATTERN = re.compile(
    r"(^|_)(ssn|social_security|email|e_mail|phone|mobile|address|"
    r"first_name|last_name|full_name|patient_name|customer_name)($|_)",
    re.IGNORECASE,
)
QUASI_IDENTIFIER_PATTERN = re.compile(
    r"(^|_)(age|birth|dob|gender|sex|zip|postal|patient_id|customer_id|"
    r"account_id|claimnumber|claim_number)($|_)",
    re.IGNORECASE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(frame: pd.DataFrame) -> str:
    schema = [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(frame.columns, frame.dtypes)
    ]
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_dataset_path(dataset_root: Path, blob_path: str) -> Path:
    candidate = Path(blob_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe blob_path: {blob_path!r}")
    resolved = (dataset_root / candidate).resolve()
    if not resolved.is_relative_to(dataset_root):
        raise ValueError(f"blob_path escapes dataset root: {blob_path!r}")
    return resolved


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenarios = payload.get("scenarios") or []
    if len(scenarios) != 15:
        raise ValueError(f"Release matrix must contain exactly 15 scenarios, got {len(scenarios)}")

    identifiers = [str(item.get("id") or "") for item in scenarios]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every scenario requires a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Scenario ids must be unique")

    counts = Counter(str(item.get("task_type") or "") for item in scenarios)
    if set(counts) != TASK_TYPES or any(counts[task] != 5 for task in TASK_TYPES):
        raise ValueError(f"Release matrix requires five scenarios per task type, got {dict(counts)}")

    for task_type in TASK_TYPES:
        industries = {
            str(item.get("industry") or "").strip().casefold()
            for item in scenarios
            if item.get("task_type") == task_type
        }
        if "" in industries or len(industries) != 5:
            raise ValueError(f"{task_type} requires five distinct non-empty industries")
    for scenario in scenarios:
        provenance = scenario.get("provenance") or {}
        missing_provenance = [
            field
            for field in ("source_url", "license", "source_dataset_id")
            if not str(provenance.get(field) or "").strip()
        ]
        if missing_provenance:
            raise ValueError(
                f"{scenario['id']} is missing provenance fields: "
                + ", ".join(missing_provenance)
            )

    privacy_review = payload.get("privacy_review") or {}
    missing_review_fields = [
        field
        for field in ("scope", "reviewed_at", "restrictions")
        if not privacy_review.get(field)
    ]
    if missing_review_fields:
        raise ValueError(
            "privacy_review is missing fields: " + ", ".join(missing_review_fields)
        )
    dispositions = privacy_review.get("scenario_dispositions") or {}
    if set(dispositions) != set(identifiers):
        missing = sorted(set(identifiers) - set(dispositions))
        unexpected = sorted(set(dispositions) - set(identifiers))
        raise ValueError(
            "privacy_review.scenario_dispositions must exactly cover scenario ids; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for scenario in scenarios:
        disposition = dispositions[scenario["id"]] or {}
        status = str(disposition.get("status") or "")
        rationale = str(disposition.get("rationale") or "").strip()
        if status not in PRIVACY_REVIEW_STATUSES or not rationale:
            raise ValueError(
                f"{scenario['id']} has an invalid privacy disposition: {disposition}"
            )
        scenario["privacy_review"] = disposition
    return payload, scenarios


def _privacy_profile(
    columns: list[str],
    *,
    excluded_columns: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    normalized = {
        column: re.sub(
            r"[^a-z0-9]+",
            "_",
            re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column).casefold(),
        ).strip("_")
        for column in columns
    }
    raw_direct = sorted(
        column
        for column, normalized_column in normalized.items()
        if DIRECT_IDENTIFIER_PATTERN.search(normalized_column)
    )
    raw_quasi = sorted(
        column
        for column, normalized_column in normalized.items()
        if QUASI_IDENTIFIER_PATTERN.search(normalized_column)
    )
    excluded = set(map(str, excluded_columns))
    direct = sorted(set(raw_direct) - excluded)
    quasi = sorted(set(raw_quasi) - excluded)
    risk = "high" if direct else "medium" if quasi else "low"
    return {
        "heuristic_risk": risk,
        "direct_identifier_columns": direct,
        "quasi_identifier_columns": quasi,
        "raw_direct_identifier_columns": raw_direct,
        "raw_quasi_identifier_columns": raw_quasi,
        "excluded_identifier_columns": sorted(
            excluded.intersection(set(raw_direct) | set(raw_quasi))
        ),
        "note": (
            "Column-name heuristic over effective model features; "
            "manual privacy review remains authoritative."
        ),
    }


def _target_profile(
    frame: pd.DataFrame,
    *,
    task_type: str,
    target_column: str | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if task_type == "clustering":
        return {"required": False, "column": None}, errors
    if not target_column:
        return {"required": True, "column": None}, ["target_column is required"]
    if target_column not in frame.columns:
        return {
            "required": True,
            "column": target_column,
            "available_columns": [str(column) for column in frame.columns],
        }, [f"target column {target_column!r} is absent"]

    target = frame[target_column]
    non_null = target.dropna()
    profile: dict[str, Any] = {
        "required": True,
        "column": target_column,
        "dtype": str(target.dtype),
        "missing_rate": float(target.isna().mean()),
        "unique_non_null": int(non_null.nunique()),
    }
    if non_null.empty:
        errors.append("target contains no non-null values")
        return profile, errors

    if task_type == "classification":
        counts = non_null.value_counts(dropna=False)
        class_count = int(len(counts))
        profile.update(
            {
                "class_count": class_count,
                "minimum_class_rows": int(counts.min()),
                "maximum_class_rows": int(counts.max()),
                "imbalance_ratio": float(counts.min() / counts.max()),
                "class_distribution": {
                    str(label): int(count) for label, count in counts.items()
                },
            }
        )
        if class_count < 2:
            errors.append("classification target has fewer than two classes")
        if class_count > min(100, max(20, int(len(non_null) ** 0.5))):
            errors.append("classification target has too many distinct values")
    else:
        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric_rate = float(numeric.notna().mean())
        profile["numeric_parse_rate"] = numeric_rate
        if numeric_rate < 0.98:
            errors.append("regression target is not at least 98% numeric")
        else:
            numeric = numeric.dropna()
            profile.update(
                {
                    "minimum": float(numeric.min()),
                    "maximum": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "standard_deviation": float(numeric.std(ddof=0)),
                }
            )
            if numeric.nunique() < 10:
                errors.append("regression target has fewer than ten distinct values")
    return profile, errors


def _profile_scenario(
    scenario: dict[str, Any],
    frame: pd.DataFrame,
    *,
    file_path: Path,
    raw_sha256: str,
    canonical_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    task_type = str(scenario["task_type"])
    target_column = scenario.get("target_column")
    excluded_columns = [str(value) for value in scenario.get("exclude_columns") or []]
    missing_exclusions = sorted(set(excluded_columns) - set(map(str, frame.columns)))
    usable_columns = [
        column
        for column in frame.columns
        if str(column) not in excluded_columns
        and str(column) != str(target_column or "")
        and frame[column].notna().any()
        and frame[column].nunique(dropna=True) > 1
    ]

    target, errors = _target_profile(
        frame,
        task_type=task_type,
        target_column=str(target_column) if target_column else None,
    )
    if missing_exclusions:
        errors.append(
            "configured exclusion columns are absent: " + ", ".join(missing_exclusions)
        )
    minimum_rows = int(scenario.get("minimum_rows", 100))
    if len(frame) < minimum_rows:
        errors.append(f"row count {len(frame)} is below minimum {minimum_rows}")
    if len(usable_columns) < 2:
        errors.append("fewer than two usable feature columns remain")

    license_name = str((scenario.get("provenance") or {}).get("license") or "").strip()
    provenance_status = "recorded" if license_name else "license_pending"
    manual_privacy_review = scenario.get("privacy_review") or {
        "status": "unreviewed",
        "rationale": "No manual privacy disposition was supplied.",
    }
    privacy = _privacy_profile(
        [str(column) for column in frame.columns],
        excluded_columns=[*excluded_columns, str(target_column or "")],
    )
    privacy["manual_review"] = manual_privacy_review
    privacy_status = str(manual_privacy_review.get("status") or "unreviewed")
    if errors:
        qualification_status = "schema_fail"
    elif provenance_status != "recorded":
        qualification_status = "schema_pass_license_pending"
    elif privacy_status == "blocked":
        qualification_status = "privacy_blocked"
    elif privacy_status != "approved_for_nonproduction_qualification":
        qualification_status = "schema_pass_privacy_pending"
    else:
        qualification_status = "schema_pass"
    return {
        "id": scenario["id"],
        "task_type": task_type,
        "industry": scenario["industry"],
        "objective": scenario.get("objective"),
        "blob_path": scenario["blob_path"],
        "file_name": file_path.name,
        "file_size_bytes": int(file_path.stat().st_size),
        "raw_file_sha256": raw_sha256,
        "content_sha256": canonical_sha256,
        "schema_sha256": schema_sha256,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "numeric_column_count": int(len(frame.select_dtypes(include="number").columns)),
        "categorical_column_count": int(
            len(frame.select_dtypes(include=["object", "category", "string"]).columns)
        ),
        "missing_cell_rate": float(frame.isna().to_numpy().mean()),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "excluded_columns": excluded_columns,
        "missing_excluded_columns": missing_exclusions,
        "usable_feature_count": len(usable_columns),
        "target": target,
        "privacy": privacy,
        "privacy_review_status": privacy_status,
        "provenance": scenario.get("provenance") or {},
        "provenance_status": provenance_status,
        "schema_task_status": "pass" if not errors else "fail",
        "qualification_status": qualification_status,
        "validation_errors": errors,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "task_type",
        "industry",
        "blob_path",
        "row_count",
        "column_count",
        "usable_feature_count",
        "content_sha256",
        "schema_sha256",
        "provenance_status",
        "privacy_review_status",
        "schema_task_status",
        "qualification_status",
        "validation_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            row["validation_errors"] = "; ".join(record.get("validation_errors") or [])
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/qualification")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, scenarios = _load_manifest(manifest_path)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        key = (
            str(scenario["blob_path"]),
            str(scenario.get("encoding") or "utf-8"),
            str(scenario.get("delimiter") or ","),
        )
        grouped[key].append(scenario)

    records: list[dict[str, Any]] = []
    for (blob_path, encoding, delimiter), grouped_scenarios in grouped.items():
        file_path = _safe_dataset_path(dataset_root, blob_path)
        print(f"Profiling {blob_path} for {len(grouped_scenarios)} scenario(s)", flush=True)
        try:
            raw_sha256 = _sha256_file(file_path)
            frame = pd.read_csv(file_path, sep=delimiter, encoding=encoding)
            canonical_sha256 = canonical_dataframe_sha256(frame)
            schema_sha256 = _schema_sha256(frame)
            for scenario in grouped_scenarios:
                records.append(
                    _profile_scenario(
                        scenario,
                        frame,
                        file_path=file_path,
                        raw_sha256=raw_sha256,
                        canonical_sha256=canonical_sha256,
                        schema_sha256=schema_sha256,
                    )
                )
            del frame
            gc.collect()
        except Exception as exc:
            for scenario in grouped_scenarios:
                records.append(
                    {
                        "id": scenario["id"],
                        "task_type": scenario["task_type"],
                        "industry": scenario["industry"],
                        "blob_path": blob_path,
                        "schema_task_status": "fail",
                        "qualification_status": "read_failed",
                        "provenance": scenario.get("provenance") or {},
                        "validation_errors": [f"{type(exc).__name__}: {exc}"],
                    }
                )

    records.sort(key=lambda item: str(item["id"]))
    status_counts = Counter(str(record["qualification_status"]) for record in records)
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_name": manifest.get("matrix_name"),
        "matrix_contract": {
            "scenario_count": len(records),
            "task_counts": dict(Counter(record["task_type"] for record in records)),
            "distinct_industries_by_task": {
                task: sorted(
                    {
                        str(record["industry"])
                        for record in records
                        if record["task_type"] == task
                    }
                )
                for task in sorted(TASK_TYPES)
            },
        },
        "status_counts": dict(status_counts),
        "privacy_review": {
            "scope": (manifest.get("privacy_review") or {}).get("scope"),
            "reviewed_at": (manifest.get("privacy_review") or {}).get("reviewed_at"),
            "restrictions": (manifest.get("privacy_review") or {}).get("restrictions"),
            "status_counts": dict(
                Counter(record.get("privacy_review_status") for record in records)
            ),
        },
        "scenarios": records,
    }
    report_json = json.dumps(report, indent=2, sort_keys=True)
    (output_dir / "industry_matrix_profile.json").write_text(
        report_json,
        encoding="utf-8",
    )
    _write_csv(output_dir / "industry_matrix_profile.csv", records)
    print(json.dumps(report["matrix_contract"], indent=2), flush=True)
    print(json.dumps(report["status_counts"], indent=2), flush=True)
    encoded_report = base64.b64encode(
        gzip.compress(report_json.encode("utf-8"), compresslevel=9)
    ).decode("ascii")
    print(f"INDUSTRY_MATRIX_PROFILE_GZIP_BASE64={encoded_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
