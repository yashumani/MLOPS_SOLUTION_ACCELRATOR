#!/usr/bin/env python3
"""Run drift and Phase B contracts only inside existing Azure ML compute."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


def main() -> int:
    if not os.environ.get("AZUREML_RUN_ID"):
        raise RuntimeError("Drift acceptance must execute inside Azure ML, never locally")
    if len(sys.argv) != 5:
        raise ValueError("Expected source archive, evidence directory, archive SHA256, Git SHA")
    archive, output = (Path(value).resolve() for value in sys.argv[1:3])
    expected, commit = sys.argv[3:]
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid immutable source identity")
    if archive.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Archive exceeds the existing source size limit")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Source archive checksum mismatch")
    source = Path.cwd() / "validated-source"
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        if len(entries) > 10000 or sum(item.file_size for item in entries) > 512 * 1024 * 1024:
            raise ValueError("Expanded source exceeds the existing size limit")
        for item in entries:
            path = Path(item.filename)
            symlink = (item.external_attr >> 16) & 0o170000 == 0o120000
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename or ":" in item.filename or symlink:
                raise ValueError("Unsafe archive entry")
        source.mkdir(exist_ok=False)
        package.extractall(source)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "azure_ml_drift_and_phaseb_contract_acceptance",
        "job_name": os.environ["AZUREML_RUN_ID"],
        "source_git_commit": commit,
        "source_archive_sha256": expected,
        "status": "running",
        "full_release_accepted": False,
        "production_change": False,
    }
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "src"))
    from scripts.validate_controller_release import TEST_TOOLS

    def run(command, filename, env=None):
        result = subprocess.run(
            command, cwd=source, env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        text = result.stdout + result.stderr
        (output / filename).write_text(text, encoding="utf-8")
        print(text[-16000:], flush=True)
        if result.returncode:
            raise RuntimeError(f"{filename} failed with exit code {result.returncode}")

    try:
        report["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "evidently", "mlflow")
        }
        with tempfile.TemporaryDirectory(prefix="mlops-drift-tests-") as directory:
            tools = Path(directory) / "tools"
            run([
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "--no-deps", "--target", str(tools), *TEST_TOOLS,
            ], "test-tools.log")
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(map(str, (tools, source, source / "src")))
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            run([
                sys.executable, "-m", "pytest", "-q", "-c", str(source / "pytest.ini"),
                "--junitxml", str(output / "drift-tests.xml"),
                str(source / "tests/test_drift_detection"),
                str(source / "tests/test_drift_detector.py"),
                str(source / "tests/test_orchestration/test_s14_retrain_decision.py"),
                str(source / "tests/test_phaseb_final_fit_contract.py"),
                str(source / "tests/test_phaseb_candidate_deadlines.py"),
            ], "drift-tests.log", env)
        report["status"] = "passed"
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return 2
    finally:
        (output / "drift-acceptance.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
