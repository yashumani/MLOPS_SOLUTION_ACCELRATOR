#!/usr/bin/env python3
"""Compute the canonical dataset digest used by production configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.data_identity import canonical_dataframe_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="CSV path or Azure-resolvable URI")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    frame = pd.read_csv(
        args.dataset,
        sep=args.delimiter,
        encoding=args.encoding,
    )
    print(canonical_dataframe_sha256(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
