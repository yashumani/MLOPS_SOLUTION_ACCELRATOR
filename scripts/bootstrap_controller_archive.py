#!/usr/bin/env python3
"""Bootstrap the controller preflight from a checksum-pinned Azure job input."""

import hashlib
import json
import os
import runpy
import sys
import zipfile
from pathlib import Path


def main() -> None:
    if not os.environ.get("AZUREML_RUN_ID"):
        raise RuntimeError("Archive bootstrap must execute inside an Azure ML job")
    if len(sys.argv) != 4:
        raise ValueError("Expected archive path, evidence path, and archive SHA-256")
    archive, output = map(Path, sys.argv[1:3])
    expected = sys.argv[3]
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("Expected a lowercase archive SHA-256")
    if archive.stat().st_size > 67108864:
        raise RuntimeError("Source archive exceeds the reviewed compressed size bound")
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("Uploaded source archive checksum mismatch")
    source = Path.cwd() / "validated-source"
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        if len(entries) > 10000 or sum(entry.file_size for entry in entries) > 536870912:
            raise RuntimeError("Source archive exceeds the reviewed size bounds")
        for entry in entries:
            path = Path(entry.filename)
            is_symlink = (entry.external_attr >> 16) & 0o170000 == 0o120000
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename or is_symlink:
                raise RuntimeError("Unsafe source archive entry")
        source.mkdir(exist_ok=False)
        package.extractall(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / "package_identity.json").write_text(
        json.dumps({"archive_sha256": actual, "entries": len(entries)}), encoding="utf-8",
    )
    sys.argv = [
        str(source / "scripts/validate_controller_release.py"),
        "--tests-dir", str(source / "tests"), "--output-dir", str(output),
        "--config", "config_qualification_classification_healthcare_heart_disease_azureml.yml",
        "--config", "config_qualification_regression_education_final_grade_azureml.yml",
        "--config", "config_qualification_clustering_retail_transaction_segments_azureml.yml",
        "--datastore-canary-job", "verify-workspace-datastores-70e5c60f-approved-20260903",
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
