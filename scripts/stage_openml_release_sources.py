#!/usr/bin/env python3
"""Stage existing OpenML qualification sources under immutable release paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.data_identity import canonical_dataframe_sha256  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    source_dataset_id: str
    source_url: str
    input_arg: str
    output_name: str
    expected_rows: int
    expected_columns: tuple[str, ...]
    absent_identifier_columns: tuple[str, ...]
    expected_file_sha256: str
    expected_content_sha256: str


DATASETS = (
    DatasetSpec(
        key="openml-42477-credit-default",
        name="Default of Credit Card Clients",
        source_dataset_id="openml-42477-v1",
        source_url="https://www.openml.org/d/42477",
        input_arg="credit_input",
        output_name="default_of_credit_card_clients.csv",
        expected_rows=30000,
        expected_columns=tuple([f"x{index}" for index in range(1, 24)] + ["y"]),
        absent_identifier_columns=("id",),
        expected_file_sha256=(
            "39dbf34a3530f4fae130c35e28cf2021af417247c1bc4d8ca23f4febfb2173b4"
        ),
        expected_content_sha256=(
            "87b4c0cccf23e4366507e680368cd6b0ae7f26671498108b7929dc0887089617"
        ),
    ),
    DatasetSpec(
        key="openml-42876-workers-compensation",
        name="Workers Compensation",
        source_dataset_id="openml-42876-v1",
        source_url="https://www.openml.org/d/42876",
        input_arg="workers_comp_input",
        output_name="workers_compensation.csv",
        expected_rows=100000,
        expected_columns=(
            "DateTimeOfAccident",
            "DateReported",
            "Age",
            "Gender",
            "MaritalStatus",
            "DependentChildren",
            "DependentsOther",
            "WeeklyPay",
            "PartTimeFullTime",
            "HoursWorkedPerWeek",
            "DaysWorkedPerWeek",
            "ClaimDescription",
            "InitialCaseEstimate",
            "UltimateIncurredClaimCost",
        ),
        absent_identifier_columns=("ClaimNumber",),
        expected_file_sha256=(
            "a974d8945c27560a4ea62e46734fffc292f8f0654b8e502063fa9851cc122636"
        ),
        expected_content_sha256=(
            "7b341ae4ab23ddb43f584da2ff5e8c419bfb5c2b4f4e784529d836c8ddf9f5eb"
        ),
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(frame: pd.DataFrame) -> str:
    payload = [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(frame.columns, frame.dtypes)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_source(spec: DatasetSpec, frame: pd.DataFrame) -> None:
    if len(frame) != spec.expected_rows:
        raise RuntimeError(
            f"Unexpected {spec.key} row count: {len(frame)}; expected {spec.expected_rows}"
        )
    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != spec.expected_columns:
        raise RuntimeError(
            f"Unexpected {spec.key} columns: {actual_columns}; "
            f"expected {spec.expected_columns}"
        )
    present_identifiers = sorted(
        set(spec.absent_identifier_columns).intersection(actual_columns)
    )
    if present_identifiers:
        raise RuntimeError(
            f"Unexpected identifier columns in {spec.key}: {present_identifiers}"
        )


def _stage(
    spec: DatasetSpec,
    input_path: Path,
    output_root: Path,
    staged_at: str,
) -> dict[str, object]:
    frame = pd.read_csv(input_path)
    validate_source(spec, frame)

    source_file_sha256 = _sha256_file(input_path)
    content_sha256 = canonical_dataframe_sha256(frame)
    if source_file_sha256 != spec.expected_file_sha256:
        raise RuntimeError(
            f"Source file hash changed for {spec.key}: {source_file_sha256}"
        )
    if content_sha256 != spec.expected_content_sha256:
        raise RuntimeError(
            f"Canonical content hash changed for {spec.key}: {content_sha256}"
        )

    dataset_dir = output_root / spec.key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / spec.output_name
    shutil.copyfile(input_path, output_path)
    output_file_sha256 = _sha256_file(output_path)
    if output_file_sha256 != source_file_sha256:
        raise RuntimeError(f"Staged file hash mismatch for {spec.key}")

    provenance: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_name": spec.name,
        "source_repository": "OpenML",
        "source_dataset_id": spec.source_dataset_id,
        "source_url": spec.source_url,
        "license": "CC0",
        "staged_at": staged_at,
        "source_file_sha256": source_file_sha256,
        "staged_file_sha256": output_file_sha256,
        "content_sha256": content_sha256,
        "schema_sha256": _schema_sha256(frame),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "pre_normalized_identifier_columns": list(spec.absent_identifier_columns),
        "transforms": [
            "No transformation was performed by this staging job.",
            "The existing workspace CSV already omitted the listed identifier columns.",
        ],
    }
    (dataset_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credit-input", required=True)
    parser.add_argument("--workers-comp-input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    inputs = {
        "credit_input": Path(args.credit_input).resolve(),
        "workers_comp_input": Path(args.workers_comp_input).resolve(),
    }
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staged_at = datetime.now(timezone.utc).isoformat()
    records = [
        _stage(spec, inputs[spec.input_arg], output_root, staged_at)
        for spec in DATASETS
    ]
    manifest = {
        "schema_version": "1.0",
        "staged_at": staged_at,
        "dataset_count": len(records),
        "datasets": records,
    }
    (output_root / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
