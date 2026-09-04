from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import bootstrap_qualification_archive as bootstrap


def _archive(path: Path) -> str:
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("scripts/run_qualification_wave.py", "pass\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bootstrap_refuses_execution_outside_user_identity_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    monkeypatch.delenv("OBO_ENDPOINT", raising=False)

    with pytest.raises(RuntimeError, match="user-identity job"):
        bootstrap.main()


def test_bootstrap_binds_verified_archive_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "source.zip"
    digest = _archive(archive)
    output = tmp_path / "output"
    observed: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZUREML_RUN_ID", "qualification-wave")
    monkeypatch.setenv("OBO_ENDPOINT", "https://identity.test")
    monkeypatch.setattr(
        bootstrap.sys,
        "argv",
        [
            "bootstrap",
            str(archive),
            str(output),
            digest,
            "a" * 40,
            "codex_ys/release-candidate",
            "datastore-canary",
            "classification-healthcare-heart-disease",
        ],
    )

    def run_path(path: str, *, run_name: str) -> None:
        observed["path"] = path
        observed["run_name"] = run_name
        observed["argv"] = list(bootstrap.sys.argv)

    monkeypatch.setattr(bootstrap.runpy, "run_path", run_path)

    bootstrap.main()

    identity = json.loads(
        (output / "package_identity.json").read_text(encoding="utf-8")
    )
    assert identity["archive_sha256"] == digest
    assert identity["git_commit"] == "a" * 40
    assert bootstrap.os.environ["MLOPS_AZURE_CREDENTIAL_MODE"] == "azureml_obo"
    assert bootstrap.os.environ["MLOPS_SOURCE_ARCHIVE_SHA256"] == digest
    assert observed["run_name"] == "__main__"
    assert observed["argv"][-2:] == [
        "--scenario",
        "classification-healthcare-heart-disease",
    ]


def test_checksum_mismatch_cannot_extract_or_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "source.zip"
    _archive(archive)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZUREML_RUN_ID", "qualification-wave")
    monkeypatch.setenv("OBO_ENDPOINT", "https://identity.test")
    monkeypatch.setattr(
        bootstrap.sys,
        "argv",
        [
            "bootstrap",
            str(archive),
            str(tmp_path / "output"),
            "0" * 64,
            "a" * 40,
            "codex_ys/release-candidate",
            "datastore-canary",
            "classification-healthcare-heart-disease",
        ],
    )
    monkeypatch.setattr(
        bootstrap.runpy,
        "run_path",
        lambda *_args, **_kwargs: pytest.fail("unverified source executed"),
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bootstrap.main()
    assert not (tmp_path / "validated-source").exists()
