#!/usr/bin/env python3
"""Acquire the licensed UCI datasets used by the release qualification matrix."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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
    uci_id: int
    dataset_page: str
    doi: str
    archive_urls: tuple[str, ...]
    member_suffix: str
    output_name: str
    expected_rows: int
    parser: Callable[[bytes], pd.DataFrame]
    target_column: str | None = None
    transforms: tuple[str, ...] = ()


HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def _read_heart(raw: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw), header=None, names=HEART_COLUMNS, na_values="?")
    frame["heart_disease"] = (pd.to_numeric(frame.pop("num"), errors="raise") > 0).astype(int)
    return frame


def _read_csv(raw: bytes, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), **kwargs)


def _read_real_estate(raw: bytes) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(raw))
    if len(frame.columns) != 8:
        raise RuntimeError(f"Unexpected real-estate column count: {len(frame.columns)}")
    frame.columns = [
        "record_id",
        "transaction_date",
        "house_age",
        "distance_to_mrt",
        "convenience_stores",
        "latitude",
        "longitude",
        "house_price_unit_area",
    ]
    return frame


def _read_energy_efficiency(raw: bytes) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(raw))
    if len(frame.columns) != 10:
        raise RuntimeError(f"Unexpected energy-efficiency column count: {len(frame.columns)}")
    frame.columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]
    return frame


def _read_airfoil(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(
        io.BytesIO(raw),
        sep=r"\s+",
        header=None,
        names=[
            "frequency",
            "attack_angle",
            "chord_length",
            "free_stream_velocity",
            "suction_side_displacement_thickness",
            "scaled_sound_pressure",
        ],
    )


def _read_excel(raw: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw))


def _archive_url(uci_id: int, slug: str) -> str:
    return f"https://archive.ics.uci.edu/static/public/{uci_id}/{slug}.zip"


DATASETS = (
    DatasetSpec(
        key="heart-disease",
        name="Heart Disease (Cleveland processed subset)",
        uci_id=45,
        dataset_page="https://archive.ics.uci.edu/dataset/45/heart+disease",
        doi="10.24432/C52P4X",
        archive_urls=(_archive_url(45, "heart%2Bdisease"),),
        member_suffix="processed.cleveland.data",
        output_name="heart_disease_cleveland.csv",
        expected_rows=303,
        parser=_read_heart,
        target_column="heart_disease",
        transforms=("Converted UCI num values greater than zero to binary heart_disease=1.",),
    ),
    DatasetSpec(
        key="iranian-churn",
        name="Iranian Churn",
        uci_id=563,
        dataset_page="https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        doi="10.24432/C5JW3Z",
        archive_urls=(
            _archive_url(563, "iranian%2Bchurn"),
            _archive_url(563, "iranian%2Bchurn%2Bdataset"),
        ),
        member_suffix="Customer Churn.csv",
        output_name="iranian_churn.csv",
        expected_rows=3150,
        parser=_read_csv,
        target_column="Churn",
    ),
    DatasetSpec(
        key="ai4i-predictive-maintenance",
        name="AI4I 2020 Predictive Maintenance Dataset",
        uci_id=601,
        dataset_page=(
            "https://archive.ics.uci.edu/dataset/601/"
            "ai4i+2020+predictive+maintenance+dataset"
        ),
        doi="10.24432/C5HS5C",
        archive_urls=(
            _archive_url(601, "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset"),
        ),
        member_suffix="ai4i2020.csv",
        output_name="ai4i2020.csv",
        expected_rows=10000,
        parser=_read_csv,
        target_column="Machine failure",
    ),
    DatasetSpec(
        key="student-performance",
        name="Student Performance (Portuguese language course)",
        uci_id=320,
        dataset_page="https://archive.ics.uci.edu/dataset/320/student+performance",
        doi="10.24432/C5TG7T",
        archive_urls=(_archive_url(320, "student%2Bperformance"),),
        member_suffix="student-por.csv",
        output_name="student_performance_portuguese.csv",
        expected_rows=649,
        parser=lambda raw: _read_csv(raw, sep=";"),
        target_column="G3",
    ),
    DatasetSpec(
        key="real-estate-valuation",
        name="Real Estate Valuation",
        uci_id=477,
        dataset_page=(
            "https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set"
        ),
        doi="10.24432/C5J30W",
        archive_urls=(_archive_url(477, "real%2Bestate%2Bvaluation%2Bdata%2Bset"),),
        member_suffix="Real estate valuation data set.xlsx",
        output_name="real_estate_valuation.csv",
        expected_rows=414,
        parser=_read_real_estate,
        target_column="house_price_unit_area",
        transforms=("Normalized the eight published column names to stable snake_case names.",),
    ),
    DatasetSpec(
        key="parkinsons-telemonitoring",
        name="Parkinsons Telemonitoring",
        uci_id=189,
        dataset_page="https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring",
        doi="10.24432/C5ZS3N",
        archive_urls=(_archive_url(189, "parkinsons%2Btelemonitoring"),),
        member_suffix="parkinsons_updrs.data",
        output_name="parkinsons_telemonitoring.csv",
        expected_rows=5875,
        parser=_read_csv,
        target_column="total_UPDRS",
    ),
    DatasetSpec(
        key="online-retail",
        name="Online Retail",
        uci_id=352,
        dataset_page="https://archive.ics.uci.edu/dataset/352/online+retail",
        doi="10.24432/C5BW33",
        archive_urls=(_archive_url(352, "online%2Bretail"),),
        member_suffix="Online Retail.xlsx",
        output_name="online_retail.csv",
        expected_rows=541909,
        parser=_read_excel,
    ),
    DatasetSpec(
        key="energy-efficiency",
        name="Energy Efficiency",
        uci_id=242,
        dataset_page="https://archive.ics.uci.edu/dataset/242/energy+efficiency",
        doi="10.24432/C51307",
        archive_urls=(_archive_url(242, "energy%2Befficiency"),),
        member_suffix="ENB2012_data.xlsx",
        output_name="energy_efficiency.csv",
        expected_rows=768,
        parser=_read_energy_efficiency,
        target_column="Y1",
        transforms=("Normalized published feature and response names to X1-X8 and Y1-Y2.",),
    ),
    DatasetSpec(
        key="airfoil-self-noise",
        name="Airfoil Self-Noise",
        uci_id=291,
        dataset_page="https://archive.ics.uci.edu/dataset/291/airfoil+self+noise",
        doi="10.24432/C5VW2C",
        archive_urls=(_archive_url(291, "airfoil%2Bself%2Bnoise"),),
        member_suffix="airfoil_self_noise.dat",
        output_name="airfoil_self_noise.csv",
        expected_rows=1503,
        parser=_read_airfoil,
        target_column="scaled_sound_pressure",
        transforms=("Assigned the six published UCI variable names to the headerless file.",),
    ),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _download(spec: DatasetSpec) -> tuple[bytes, str]:
    failures: list[str] = []
    for url in spec.archive_urls:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Savyminds-MLOps-Qualification/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read(), url
        except (urllib.error.URLError, TimeoutError) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(failures))


def _find_members(
    archive: bytes,
    suffix: str,
    *,
    prefix: str = "",
    depth: int = 0,
) -> list[tuple[bytes, str]]:
    if depth > 3:
        return []
    matches: list[tuple[bytes, str]] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        members = [name for name in bundle.namelist() if not name.endswith("/")]
        for name in members:
            display_name = f"{prefix}!{name}" if prefix else name
            if name.casefold().endswith(suffix.casefold()):
                matches.append((bundle.read(name), display_name))
            elif name.casefold().endswith(".zip"):
                matches.extend(
                    _find_members(
                        bundle.read(name),
                        suffix,
                        prefix=display_name,
                        depth=depth + 1,
                    )
                )
    return matches


def _extract_member(archive: bytes, suffix: str) -> tuple[bytes, str]:
    candidates = _find_members(archive, suffix)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one archive member ending {suffix!r}, "
            f"got {[name for _, name in candidates]}"
        )
    return candidates[0]


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    if len(set(normalized.columns)) != len(normalized.columns):
        raise RuntimeError("Dataset contains duplicate column names after normalization")
    for column in normalized.select_dtypes(include=["datetime", "datetimetz"]).columns:
        normalized[column] = normalized[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return normalized


def _validate(spec: DatasetSpec, frame: pd.DataFrame) -> None:
    if len(frame) != spec.expected_rows:
        raise RuntimeError(
            f"Unexpected {spec.key} row count: {len(frame)}; expected {spec.expected_rows}"
        )
    if len(frame.columns) < 3:
        raise RuntimeError(f"{spec.key} has fewer than three columns")
    if spec.target_column:
        if spec.target_column not in frame.columns:
            raise RuntimeError(
                f"{spec.key} target {spec.target_column!r} is absent: {list(frame.columns)}"
            )
        target = frame[spec.target_column]
        if target.isna().any() or target.nunique(dropna=True) < 2:
            raise RuntimeError(f"{spec.key} target is incomplete or constant")


def _acquire(spec: DatasetSpec, output_root: Path, retrieved_at: str) -> dict[str, object]:
    print(f"Acquiring UCI {spec.uci_id}: {spec.name}", flush=True)
    archive, archive_url = _download(spec)
    member, member_name = _extract_member(archive, spec.member_suffix)
    frame = _normalize_frame(spec.parser(member))
    _validate(spec, frame)

    dataset_dir = output_root / spec.key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / spec.output_name
    frame.to_csv(dataset_path, index=False)

    persisted = pd.read_csv(dataset_path)
    provenance: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_name": spec.name,
        "source_repository": "UCI Machine Learning Repository",
        "source_dataset_id": f"uci-{spec.uci_id}",
        "source_url": archive_url,
        "dataset_page": spec.dataset_page,
        "doi": spec.doi,
        "license": "CC BY 4.0",
        "retrieved_at": retrieved_at,
        "archive_member": member_name,
        "raw_archive_sha256": _sha256_bytes(archive),
        "source_member_sha256": _sha256_bytes(member),
        "persisted_file_sha256": _sha256_file(dataset_path),
        "content_sha256": canonical_dataframe_sha256(persisted),
        "schema_sha256": _schema_sha256(persisted),
        "rows": int(len(persisted)),
        "columns": int(len(persisted.columns)),
        "target_column": spec.target_column,
        "transforms": list(spec.transforms),
    }
    if spec.target_column:
        provenance["target_unique_non_null"] = int(
            persisted[spec.target_column].nunique(dropna=True)
        )
    (dataset_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    del persisted
    del frame
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-key",
        action="append",
        choices=sorted(spec.key for spec in DATASETS),
        help="Acquire only the selected dataset key; repeat for multiple datasets.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    requested = set(args.dataset_key or [])
    selected = [spec for spec in DATASETS if not requested or spec.key in requested]
    records = [_acquire(spec, output_root, retrieved_at) for spec in selected]
    report = {
        "schema_version": "1.0",
        "retrieved_at": retrieved_at,
        "dataset_count": len(records),
        "license": "CC BY 4.0",
        "datasets": records,
    }
    (output_root / "acquisition_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
