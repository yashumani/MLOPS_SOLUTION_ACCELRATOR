#!/usr/bin/env python3
"""Run bounded controller contracts and read-only live discovery in an Azure job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TEST_FILES = (
    "test_controller_archive_bootstrap.py",
    "test_controller_state_bootstrap.py",
    "test_automated_retrain_controller.py",
    "test_operational_state.py",
    "test_orchestration/test_auto_retrain_controller.py",
)
TEST_TOOLS = (
    "pytest==8.3.5", "iniconfig==2.1.0", "pluggy==1.6.0",
    "tomli==2.2.1", "exceptiongroup==1.3.0",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--datastore-canary-job", required=True)
    args = parser.parse_args()
    if not os.environ.get("AZUREML_RUN_ID"):
        raise RuntimeError("This validation entry point must execute inside an Azure ML job")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0", "scope": "controller-preflight",
        "job_name": os.environ["AZUREML_RUN_ID"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "checks": {},
        "controller_acceptance": False, "release_matrix_accepted": False,
    }

    def save() -> None:
        (output / "controller_preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    def run(command: list[str], log: str, *, env=None, timeout: int = 600) -> None:
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, check=False)
        (output / log).write_text(result.stdout + result.stderr, encoding="utf-8")
        print(result.stdout[-12000:] + result.stderr[-4000:], flush=True)
        if result.returncode:
            raise RuntimeError(f"{log} failed with exit code {result.returncode}")

    try:
        tests = args.tests_dir.resolve()
        for name in TEST_FILES:
            if not (tests / name).is_file():
                raise RuntimeError(f"Required uploaded test file is absent: {name}")
        report["test_sources"] = {
            path.relative_to(tests).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(tests.rglob("*.py"))
        }
        report["runtime_packages"] = {
            name: importlib.metadata.version(name)
            for name in ("azure-ai-ml", "azure-identity", "mlflow", "mlflow-skinny", "numpy", "scikit-learn")
        }
        if report["runtime_packages"]["mlflow"] != report["runtime_packages"]["mlflow-skinny"]:
            raise RuntimeError("MLflow package identities are inconsistent")
        from pipelines.submit_pipeline import _compute_upload_source_manifest
        source = _compute_upload_source_manifest()
        report["runtime_source_sha256"] = source["source_sha256"]
        (output / "source_manifest.json").write_text(json.dumps(source, indent=2, sort_keys=True), encoding="utf-8")
        save()
        # Test tools are isolated from the immutable training environment.
        with tempfile.TemporaryDirectory(prefix="mlops-controller-tests-") as directory:
            tools = Path(directory) / "tools"
            run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "--target", str(tools), *TEST_TOOLS], "test-tools.log", timeout=300)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(map(str, (tools, ROOT, ROOT / "src", tests)))
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            run([sys.executable, "-m", "pytest", "-q", "-c", str(ROOT / "pytest.ini"), "--junitxml", str(output / "controller-tests.xml"), *(str(tests / name) for name in TEST_FILES)], "controller-tests.log", env=env)
        report["checks"]["remote_contract_tests"] = "passed"
        save()

        from azure.ai.ml import MLClient
        from azure.identity import ManagedIdentityCredential
        from orchestration.auto_retrain_controller import AzureSubmissionContext
        from orchestration.automated_retrain_controller import discover_completed_runs
        from scripts.batch_submit_all import verify_live_release_gates

        context = AzureSubmissionContext(*(os.environ[name] for name in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME", "AZURE_COMPUTE")))
        client = MLClient(ManagedIdentityCredential(client_id=os.environ.get("AZURE_CLIENT_ID") or None), context.subscription_id, context.resource_group, context.workspace_name)
        client.workspaces.get(context.workspace_name)
        report["checks"]["managed_identity_workspace_read"] = "passed"
        report["release_gates"] = verify_live_release_gates(client, datastore_canary_job=args.datastore_canary_job, download_root=output / "datastore-evidence")
        report["checks"]["live_datastore_and_schedule_gates"] = "passed"
        save()

        report["discovery"] = []
        for name in args.config:
            config_path = (ROOT / "configs" / name).resolve()
            if config_path.parent != ROOT / "configs" or not config_path.is_file():
                raise ValueError("Discovery config must name an existing configs/ file")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            experiment = config["experiment_name"]
            names = discover_completed_runs(client, context, experiment, now=datetime.now(timezone.utc), max_age_seconds=86400, max_runs=200)
            report["discovery"].append({"config": name, "experiment": experiment, "completed_parents": names})
            save()
        report["checks"]["live_discovery_queries"] = "passed"
        report["completed_parent_observed"] = any(item["completed_parents"] for item in report["discovery"])
        report["status"] = "passed"
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"Controller preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
