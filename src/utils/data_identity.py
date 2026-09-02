"""Deterministic dataset-content identity shared by onboarding and Stage 1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def canonical_dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash ordered columns, dtypes, and every row value deterministically."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dataset fingerprint input must be a pandas DataFrame")
    digest = hashlib.sha256()
    schema: dict[str, Any] = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "row_count": int(len(frame)),
    }
    digest.update(
        json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        pd.util.hash_pandas_object(
            frame,
            index=False,
            categorize=True,
        ).values.tobytes()
    )
    return digest.hexdigest()
