import importlib.util
import io
from pathlib import Path
import sys
import zipfile

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "profile_industry_matrix.py"
SPEC = importlib.util.spec_from_file_location("profile_industry_matrix", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_privacy_profile = MODULE._privacy_profile
_profile_scenario = MODULE._profile_scenario

ACQUISITION_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "acquire_uci_release_matrix.py"
)
ACQUISITION_SPEC = importlib.util.spec_from_file_location(
    "acquire_uci_release_matrix",
    ACQUISITION_PATH,
)
assert ACQUISITION_SPEC and ACQUISITION_SPEC.loader
ACQUISITION_MODULE = importlib.util.module_from_spec(ACQUISITION_SPEC)
sys.modules[ACQUISITION_SPEC.name] = ACQUISITION_MODULE
ACQUISITION_SPEC.loader.exec_module(ACQUISITION_MODULE)
_extract_member = ACQUISITION_MODULE._extract_member


def test_privacy_profile_normalizes_spaces_and_camel_case() -> None:
    profile = _privacy_profile(
        ["Age Group", "CustomerID", "InvoiceNo", "feature"],
        excluded_columns=["CustomerID"],
    )

    assert profile["heuristic_risk"] == "medium"
    assert profile["quasi_identifier_columns"] == ["Age Group"]
    assert profile["raw_quasi_identifier_columns"] == ["Age Group", "CustomerID"]
    assert profile["excluded_identifier_columns"] == ["CustomerID"]
    assert profile["direct_identifier_columns"] == []


def test_privacy_profile_excludes_non_feature_direct_identifier() -> None:
    profile = _privacy_profile(
        ["address", "feature"],
        excluded_columns=["address"],
    )

    assert profile["heuristic_risk"] == "low"
    assert profile["direct_identifier_columns"] == []
    assert profile["raw_direct_identifier_columns"] == ["address"]


def test_uci_acquisition_extracts_member_from_nested_zip() -> None:
    nested_stream = io.BytesIO()
    with zipfile.ZipFile(nested_stream, "w") as nested:
        nested.writestr("student/student-por.csv", b"G3\n10\n")

    outer_stream = io.BytesIO()
    with zipfile.ZipFile(outer_stream, "w") as outer:
        outer.writestr("student.zip", nested_stream.getvalue())

    payload, member_name = _extract_member(outer_stream.getvalue(), "student-por.csv")

    assert payload == b"G3\n10\n"
    assert member_name == "student.zip!student/student-por.csv"


def test_matrix_profile_fails_when_configured_exclusion_is_absent(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
    frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})

    record = _profile_scenario(
        {
            "id": "classification-test",
            "task_type": "classification",
            "industry": "test",
            "blob_path": "dataset.csv",
            "target_column": "target",
            "exclude_columns": ["missing_leakage_column"],
            "provenance": {
                "source_url": "https://example.test/dataset",
                "license": "CC0",
                "source_dataset_id": "test-1",
            },
            "privacy_review": {
                "status": "approved_for_nonproduction_qualification",
                "rationale": "Synthetic test fixture.",
            },
        },
        frame,
        file_path=dataset_path,
        raw_sha256="a" * 64,
        canonical_sha256="b" * 64,
        schema_sha256="c" * 64,
    )

    assert record["schema_task_status"] == "fail"
    assert record["qualification_status"] == "schema_fail"
    assert (
        "configured exclusion columns are absent: missing_leakage_column"
        in record["validation_errors"]
    )


def test_matrix_profile_blocks_manual_privacy_rejection(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "feature_a,feature_b,target\n1,3,0\n2,4,1\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {"feature_a": [1, 2], "feature_b": [3, 4], "target": [0, 1]}
    )

    record = _profile_scenario(
        {
            "id": "classification-test",
            "task_type": "classification",
            "industry": "test",
            "blob_path": "dataset.csv",
            "target_column": "target",
            "minimum_rows": 2,
            "provenance": {
                "source_url": "https://example.test/dataset",
                "license": "CC0",
                "source_dataset_id": "test-1",
            },
            "privacy_review": {
                "status": "blocked",
                "rationale": "Fixture represents an unresolved privacy review.",
            },
        },
        frame,
        file_path=dataset_path,
        raw_sha256="a" * 64,
        canonical_sha256="b" * 64,
        schema_sha256="c" * 64,
    )

    assert record["schema_task_status"] == "pass"
    assert record["qualification_status"] == "privacy_blocked"
