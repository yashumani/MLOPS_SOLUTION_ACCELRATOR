#!/usr/bin/env python3
"""Validate the release-candidate Azure ML environment on cluster compute."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from packaging.requirements import Requirement


REQUIRED_PACKAGES = (
    "azure-ai-ml",
    "azureml-mlflow",
    "azureml-fsspec",
    "Boruta",
    "catboost",
    "category-encoders",
    "evidently",
    "flaml",
    "fsspec",
    "imbalanced-learn",
    "lightgbm",
    "marshmallow",
    "mlflow",
    "mlflow-skinny",
    "numpy",
    "optuna",
    "pandas",
    "pycaret",
    "scikit-learn",
    "setuptools",
    "xgboost",
)

IMPORT_MODULES = (
    "azure.ai.ml",
    "azure.core",
    "azure.identity",
    "azureml.mlflow",
    "azureml.fsspec",
    "boruta",
    "catboost",
    "category_encoders",
    "cryptography",
    "evidently",
    "flaml",
    "imblearn",
    "lightgbm",
    "mlflow",
    "optuna",
    "pycaret",
    "pyarrow",
    "pkg_resources",
    "google.protobuf",
    "sklearn",
    "xgboost",
)


def _load_expected_versions(conda_path: Path) -> dict[str, str]:
    payload = yaml.safe_load(conda_path.read_text(encoding="utf-8"))
    dependencies = payload.get("dependencies", [])
    pip_dependencies = next(
        (
            item["pip"]
            for item in dependencies
            if isinstance(item, dict) and isinstance(item.get("pip"), list)
        ),
        None,
    )
    if not pip_dependencies:
        raise RuntimeError(f"No pip dependencies found in {conda_path}")

    expected: dict[str, str] = {}
    for raw_requirement in pip_dependencies:
        requirement = Requirement(raw_requirement)
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise RuntimeError(
                f"Direct dependency is not exactly pinned: {raw_requirement}"
            )
        expected[requirement.name] = specifiers[0].version
    return expected


def _validate_expected_versions(expected: dict[str, str]) -> dict[str, str]:
    observed = {
        name: importlib.metadata.version(name)
        for name in sorted(expected)
    }
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in sorted(expected)
        if observed[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(
            "Azure ML environment package version mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return observed


def _run_pip_check() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode:
        raise RuntimeError(f"pip check failed ({result.returncode}): {output}")
    return output or "No broken requirements found."


def _validate_evidently() -> dict[str, object]:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    reference = pd.DataFrame({"feature": [0.0, 0.1, 0.2, 0.3, 0.4]})
    current = pd.DataFrame({"feature": [0.0, 0.1, 1.2, 1.3, 1.4]})
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    payload = report.as_dict()
    if not payload.get("metrics"):
        raise RuntimeError("Evidently smoke report did not produce metrics")
    return {
        "status": "pass",
        "metric_count": len(payload["metrics"]),
    }


def _validate_mlflow() -> dict[str, object]:
    import mlflow

    mlflow_version = importlib.metadata.version("mlflow")
    skinny_version = importlib.metadata.version("mlflow-skinny")
    if mlflow_version != skinny_version:
        raise RuntimeError(
            "MLflow package mismatch: "
            f"mlflow={mlflow_version}, mlflow-skinny={skinny_version}"
        )
    tracking_uri = mlflow.get_tracking_uri()
    if not tracking_uri.startswith("azureml://"):
        raise RuntimeError(f"Unexpected MLflow tracking URI: {tracking_uri}")
    environment_name = os.environ["MLOPS_ENVIRONMENT_UNDER_TEST"]
    run_name = environment_name.replace(":", "-") + "-smoke"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("environment", environment_name)
        mlflow.log_metric("environment_smoke_pass", 1.0)
        run_id = run.info.run_id
    return {
        "status": "pass",
        "mlflow_version": mlflow_version,
        "mlflow_skinny_version": skinny_version,
        "tracking_uri": tracking_uri,
        "run_id": run_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--expected-conda", type=Path)
    args = parser.parse_args()
    os.environ["MLOPS_ENVIRONMENT_UNDER_TEST"] = args.environment

    pip_check = _run_pip_check()
    imported = []
    for module_name in IMPORT_MODULES:
        importlib.import_module(module_name)
        imported.append(module_name)

    expected_versions = (
        _load_expected_versions(args.expected_conda.resolve())
        if args.expected_conda
        else {}
    )
    package_versions = {
        name: importlib.metadata.version(name)
        for name in REQUIRED_PACKAGES
    }
    if expected_versions:
        package_versions.update(_validate_expected_versions(expected_versions))
    evidence = {
        "schema_version": "1.0",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "azureml_environment": args.environment,
        "azureml_run_id": os.environ.get("AZUREML_RUN_ID", "unknown"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pip_check": pip_check,
        "package_versions": package_versions,
        "expected_package_versions": expected_versions,
        "package_version_check": "pass" if expected_versions else "not_requested",
        "imports": imported,
        "evidently": _validate_evidently(),
        "mlflow": _validate_mlflow(),
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment_smoke.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
