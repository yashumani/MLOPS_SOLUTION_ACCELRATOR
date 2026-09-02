from __future__ import annotations

import pandas as pd

from utils.data_identity import canonical_dataframe_sha256


def test_dataframe_identity_is_stable_and_content_sensitive() -> None:
    frame = pd.DataFrame(
        {
            "amount": [1.0, 2.5, None],
            "category": ["a", "b", "a"],
        }
    )

    first = canonical_dataframe_sha256(frame)
    second = canonical_dataframe_sha256(frame.copy())
    changed = frame.copy()
    changed.loc[1, "amount"] = 2.6

    assert first == second
    assert len(first) == 64
    assert canonical_dataframe_sha256(changed) != first


def test_dataframe_identity_binds_column_order() -> None:
    frame = pd.DataFrame({"a": [1], "b": [2]})

    assert canonical_dataframe_sha256(frame) != canonical_dataframe_sha256(
        frame[["b", "a"]]
    )
