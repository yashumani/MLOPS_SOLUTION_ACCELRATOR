#!/usr/bin/env python3
"""Run a qualification wave from a checksum-pinned source archive in Azure ML."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import zipfile


GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SCENARIO_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_verified_archive(archive: Path, source: Path) -> int:
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        if (
            len(entries) > MAX_ARCHIVE_ENTRIES
            or sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES
        ):
            raise RuntimeError("Source archive exceeds the reviewed size bounds")
        for entry in entries:
            path = Path(entry.filename)
            is_symlink = (entry.external_attr >> 16) & 0o170000 == 0o120000
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or ":" in entry.filename
                or is_symlink
            ):
                raise RuntimeError("Unsafe source archive entry")
        source.mkdir(exist_ok=False)
        package.extractall(source)
    return len(entries)


def main() -> None:
    if not os.environ.get("AZUREML_RUN_ID") or not os.environ.get("OBO_ENDPOINT"):
        raise RuntimeError(
            "Qualification archive bootstrap requires an Azure ML user-identity job"
        )
    if len(sys.argv) not in {8, 9}:
        raise ValueError(
            "Expected archive, output, archive SHA-256, Git commit, branch, "
            "datastore canary, and one or two scenarios"
        )
    archive = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    expected_archive = sys.argv[3].strip().lower()
    git_commit = sys.argv[4].strip().lower()
    git_branch = sys.argv[5].strip()
    datastore_canary = sys.argv[6].strip()
    scenarios = [value.strip() for value in sys.argv[7:]]
    if not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError("Source archive is unavailable or exceeds the size bound")
    if SHA256_PATTERN.fullmatch(expected_archive) is None:
        raise ValueError("Expected archive SHA-256 is invalid")
    actual_archive = _sha256(archive)
    if actual_archive != expected_archive:
        raise RuntimeError("Uploaded source archive checksum mismatch")
    if GIT_SHA_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("Release Git commit must be a full SHA")
    if (
        not git_branch
        or len(git_branch) > 255
        or any(char.isspace() for char in git_branch)
        or git_branch in {"main", "master"}
        or git_branch.startswith("release/")
    ):
        raise ValueError("Qualification requires an unprotected feature branch")
    if not datastore_canary or len(datastore_canary) > 255:
        raise ValueError("Datastore canary job name is invalid")
    if (
        len(set(scenarios)) != len(scenarios)
        or any(SCENARIO_PATTERN.fullmatch(value) is None for value in scenarios)
    ):
        raise ValueError("Qualification scenario identity is invalid")

    source = Path.cwd() / "validated-source"
    entry_count = _extract_verified_archive(archive, source)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "1.0",
        "archive_sha256": actual_archive,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "archive_entry_count": entry_count,
        "azureml_run_id": os.environ["AZUREML_RUN_ID"],
    }
    (output / "package_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    os.environ.update(
        {
            "MLOPS_AZURE_CREDENTIAL_MODE": "azureml_obo",
            "MLOPS_SOURCE_ROOT": str(source.resolve()),
            "MLOPS_SOURCE_ARCHIVE_PATH": str(archive),
            "MLOPS_SOURCE_ARCHIVE_SHA256": actual_archive,
            "MLOPS_SOURCE_GIT_COMMIT": git_commit,
            "MLOPS_SOURCE_GIT_BRANCH": git_branch,
            "MLOPS_STATE_DIR": str((output / "state").resolve()),
        }
    )
    arguments = [
        str(source / "scripts" / "run_qualification_wave.py"),
        "--datastore-canary-job",
        datastore_canary,
        "--output-dir",
        str(output / "wave"),
        "--max-hours",
        "20",
    ]
    for scenario in scenarios:
        arguments.extend(("--scenario", scenario))
    sys.argv = arguments
    runpy.run_path(arguments[0], run_name="__main__")


if __name__ == "__main__":
    main()
