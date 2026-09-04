from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import _source_identity as source_identity


def _configure_archive(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    archive: Path,
    digest: str,
) -> None:
    monkeypatch.setattr(source_identity, "_git_identity", lambda _root: None)
    monkeypatch.setenv("AZUREML_RUN_ID", "qualification-orchestrator")
    monkeypatch.setenv("MLOPS_SOURCE_ROOT", str(root))
    monkeypatch.setenv("MLOPS_SOURCE_ARCHIVE_PATH", str(archive))
    monkeypatch.setenv("MLOPS_SOURCE_ARCHIVE_SHA256", digest)
    monkeypatch.setenv("MLOPS_SOURCE_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("MLOPS_SOURCE_GIT_BRANCH", "codex_ys/release-candidate")


def test_verified_azure_archive_supplies_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"checksum-pinned-source")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    _configure_archive(monkeypatch, root=root, archive=archive, digest=digest)

    identity = source_identity.load_source_identity(root)

    assert identity == {
        "commit": "a" * 40,
        "branch": "codex_ys/release-candidate",
        "dirty": False,
        "provenance": "verified_azure_archive",
        "archive_sha256": digest,
    }


def test_archive_identity_is_refused_outside_azure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(source_identity, "_git_identity", lambda _root: None)
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)

    with pytest.raises(source_identity.SourceIdentityError, match="inside an Azure ML job"):
        source_identity.load_source_identity(root)


def test_archive_identity_recomputes_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"actual-source")
    _configure_archive(monkeypatch, root=root, archive=archive, digest="0" * 64)

    with pytest.raises(source_identity.SourceIdentityError, match="checksum mismatch"):
        source_identity.load_source_identity(root)


def test_archive_identity_requires_exact_executable_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"source")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    _configure_archive(monkeypatch, root=root, archive=archive, digest=digest)
    monkeypatch.setenv("MLOPS_SOURCE_ROOT", str(tmp_path / "other"))

    with pytest.raises(source_identity.SourceIdentityError, match="source root"):
        source_identity.load_source_identity(root)
