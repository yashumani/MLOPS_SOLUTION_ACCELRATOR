import ast
import base64
import hashlib
import shlex
from pathlib import Path

import pytest
import yaml

from scripts import bootstrap_controller_archive as bootstrap


ROOT = Path(__file__).resolve().parents[1]


def test_job_embeds_the_reviewed_bootstrap_without_changing_its_arguments():
    job = yaml.safe_load((ROOT / "configs/jobs/validate_controller_archive.yml").read_text(encoding="utf-8"))
    command = shlex.split(job["command"])
    assert command[:2] == ["python", "-c"]
    calls = [
        node for node in ast.walk(ast.parse(command[2]))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "base64"
        and node.func.attr == "b64decode"
    ]
    assert len(calls) == 1
    payload = base64.b64decode(ast.literal_eval(calls[0].args[0]), validate=True)
    reviewed = (ROOT / "scripts/bootstrap_controller_archive.py").read_text(encoding="utf-8").encode("utf-8")
    assert payload == reviewed
    assert hashlib.sha256(payload).hexdigest() == job["tags"]["launcher_sha256"]
    compile(payload, "bootstrap_controller_archive.py", "exec")
    assert command[3:] == [
        "${{inputs.source_archive}}", "${{outputs.evidence}}", job["tags"]["archive_sha256"],
    ]


def test_bootstrap_refuses_execution_outside_an_azure_job(monkeypatch):
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="inside an Azure ML job"):
        bootstrap.main()


def test_checksum_mismatch_cannot_extract_or_run_source(monkeypatch, tmp_path):
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"untrusted archive")
    output = tmp_path / "evidence"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZUREML_RUN_ID", "test-bootstrap")
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap", str(archive), str(output), "0" * 64])
    monkeypatch.setattr(bootstrap.runpy, "run_path", lambda *args, **kwargs: pytest.fail("unverified source executed"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bootstrap.main()
    assert not (tmp_path / "validated-source").exists()
    assert not output.exists()
