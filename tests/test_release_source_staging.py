from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from scripts.stage_openml_release_sources import DATASETS, validate_source


def _small_spec():
    return replace(
        DATASETS[0],
        expected_rows=2,
        expected_columns=("feature", "target"),
        absent_identifier_columns=("id",),
    )


def test_validate_source_accepts_exact_pre_normalized_schema() -> None:
    frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})

    validate_source(_small_spec(), frame)


def test_validate_source_rejects_identifier_column() -> None:
    spec = replace(
        _small_spec(),
        expected_columns=("id", "feature", "target"),
    )
    frame = pd.DataFrame({"id": [1, 2], "feature": [1, 2], "target": [0, 1]})

    with pytest.raises(RuntimeError, match="Unexpected identifier columns"):
        validate_source(spec, frame)


def test_validate_source_rejects_schema_drift() -> None:
    frame = pd.DataFrame({"target": [0, 1], "feature": [1, 2]})

    with pytest.raises(RuntimeError, match="Unexpected .* columns"):
        validate_source(_small_spec(), frame)
