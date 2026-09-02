#!/usr/bin/env python3
"""Acquire and validate the UCI COIL 2000 training data on Azure compute."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.data_identity import canonical_dataframe_sha256  # noqa: E402


SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "tic-mld/ticdata2000.txt"
)
COLUMNS = [
    "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD", "MGODRK",
    "MGODPR", "MGODOV", "MGODGE", "MRELGE", "MRELSA", "MRELOV",
    "MFALLEEN", "MFGEKIND", "MFWEKIND", "MOPLHOOG", "MOPLMIDD",
    "MOPLLAAG", "MBERHOOG", "MBERZELF", "MBERBOER", "MBERMIDD",
    "MBERARBG", "MBERARBO", "MSKA", "MSKB1", "MSKB2", "MSKC", "MSKD",
    "MHHUUR", "MHKOOP", "MAUT1", "MAUT2", "MAUT0", "MZFONDS", "MZPART",
    "MINKM30", "MINK3045", "MINK4575", "MINK7512", "MINK123M",
    "MINKGEM", "MKOOPKLA", "PWAPART", "PWABEDR", "PWALAND", "PPERSAUT",
    "PBESAUT", "PMOTSCO", "PVRAAUT", "PAANHANG", "PTRACTOR", "PWERKT",
    "PBROM", "PLEVEN", "PPERSONG", "PGEZONG", "PWAOREG", "PBRAND",
    "PZEILPL", "PPLEZIER", "PFIETS", "PINBOED", "PBYSTAND", "AWAPART",
    "AWABEDR", "AWALAND", "APERSAUT", "ABESAUT", "AMOTSCO", "AVRAAUT",
    "AAANHANG", "ATRACTOR", "AWERKT", "ABROM", "ALEVEN", "APERSONG",
    "AGEZONG", "AWAOREG", "ABRAND", "AZEILPL", "APLEZIER", "AFIETS",
    "AINBOED", "ABYSTAND", "CARAVAN",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Savyminds-MLOps-Qualification/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    frame = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, names=COLUMNS)

    if frame.shape != (5822, 86):
        raise RuntimeError(f"Unexpected COIL 2000 training shape: {frame.shape}")
    if frame["CARAVAN"].isna().any() or set(frame["CARAVAN"].unique()) != {0, 1}:
        raise RuntimeError("COIL 2000 CARAVAN target is not complete and binary")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "coil2000.csv"
    frame.to_csv(dataset_path, index=False)
    content_sha256 = canonical_dataframe_sha256(frame)
    provenance = {
        "schema_version": "1.0",
        "dataset_name": "Insurance Company Benchmark (COIL 2000)",
        "source_repository": "UCI Machine Learning Repository",
        "source_url": SOURCE_URL,
        "dataset_page": "https://archive.ics.uci.edu/dataset/125",
        "doi": "10.24432/C5630S",
        "license": "CC BY 4.0",
        "citation": "Putten, P. (2000). Insurance Company Benchmark (COIL 2000).",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "raw_source_sha256": raw_sha256,
        "content_sha256": content_sha256,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "target_column": "CARAVAN",
        "target_distribution": {
            str(label): int(count)
            for label, count in frame["CARAVAN"].value_counts().sort_index().items()
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
