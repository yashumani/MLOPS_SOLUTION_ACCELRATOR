"""Resolve source provenance from a clean Git tree or verified Azure archive."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any


GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


class SourceIdentityError(RuntimeError):
    """Raised when executable source cannot be tied to reviewed provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, Any] | None:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        top_level = Path(run("rev-parse", "--show-toplevel")).resolve()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    if top_level != root.resolve():
        raise SourceIdentityError(
            f"Git top-level {top_level} does not match source root {root.resolve()}"
        )
    commit = run("rev-parse", "HEAD").lower()
    branch = run("branch", "--show-current")
    if GIT_SHA_PATTERN.fullmatch(commit) is None or not branch:
        raise SourceIdentityError("Git source identity is incomplete")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(run("status", "--porcelain")),
        "provenance": "git_worktree",
        "archive_sha256": None,
    }


def _verified_archive_identity(root: Path) -> dict[str, Any]:
    if not os.environ.get("AZUREML_RUN_ID"):
        raise SourceIdentityError(
            "A verified archive identity is accepted only inside an Azure ML job"
        )
    source_root_text = os.environ.get("MLOPS_SOURCE_ROOT", "").strip()
    if not source_root_text:
        raise SourceIdentityError("MLOPS_SOURCE_ROOT is required")
    source_root = Path(source_root_text).resolve()
    if source_root != root.resolve():
        raise SourceIdentityError("MLOPS_SOURCE_ROOT does not match executable source")
    archive_text = os.environ.get("MLOPS_SOURCE_ARCHIVE_PATH", "").strip()
    archive = Path(archive_text).resolve() if archive_text else None
    if archive is None or not archive.is_file():
        raise SourceIdentityError("Verified source archive is unavailable")
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise SourceIdentityError("Verified source archive exceeds the size bound")
    expected_archive = os.environ.get("MLOPS_SOURCE_ARCHIVE_SHA256", "").lower()
    if SHA256_PATTERN.fullmatch(expected_archive) is None:
        raise SourceIdentityError("MLOPS_SOURCE_ARCHIVE_SHA256 is invalid")
    actual_archive = _sha256(archive)
    if actual_archive != expected_archive:
        raise SourceIdentityError("Verified source archive checksum mismatch")
    commit = os.environ.get("MLOPS_SOURCE_GIT_COMMIT", "").lower()
    branch = os.environ.get("MLOPS_SOURCE_GIT_BRANCH", "").strip()
    if GIT_SHA_PATTERN.fullmatch(commit) is None:
        raise SourceIdentityError("MLOPS_SOURCE_GIT_COMMIT is invalid")
    if not branch or len(branch) > 255 or any(char.isspace() for char in branch):
        raise SourceIdentityError("MLOPS_SOURCE_GIT_BRANCH is invalid")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": False,
        "provenance": "verified_azure_archive",
        "archive_sha256": actual_archive,
    }


def load_source_identity(root: Path) -> dict[str, Any]:
    """Return a fail-closed source identity for submission evidence."""

    resolved = root.resolve()
    git_identity = _git_identity(resolved)
    return git_identity if git_identity is not None else _verified_archive_identity(resolved)
