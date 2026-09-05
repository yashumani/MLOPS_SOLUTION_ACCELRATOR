#!/usr/bin/env python3
"""Generate the 15 qualified Azure ML execution configs from locked evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orchestration.config_compiler import compile_config  # noqa: E402


PRIMARY_METRICS = {
    "classification": "balanced_accuracy",
    "regression": "r2",
    "clustering": "silhouette",
}


def _candidate_timeout_seconds(profile: dict[str, Any]) -> int:
    """Size qualification candidate budgets from the immutable data profile."""

    row_count = int(profile.get("row_count") or 0)
    # Azure insurance qualification has 100k source rows (80k training rows).
    # Its 120s candidate ceiling did not support both search and common CV.
    return 300 if row_count >= 100_000 else 120


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _build_config(
    scenario: dict[str, Any],
    profile: dict[str, Any],
    *,
    environment: str,
    profile_run_id: str,
) -> dict[str, Any]:
    scenario_id = str(scenario["id"])
    if profile.get("id") != scenario_id:
        raise ValueError(f"Profile identity mismatch for {scenario_id}")
    if profile.get("qualification_status") != "schema_pass":
        raise ValueError(
            f"Scenario {scenario_id} is not qualified: "
            f"{profile.get('qualification_status')}"
        )
    if profile.get("privacy_review_status") != (
        "approved_for_nonproduction_qualification"
    ):
        raise ValueError(f"Scenario {scenario_id} lacks approved privacy evidence")

    task_type = str(scenario["task_type"])
    content_sha256 = str(profile["content_sha256"])
    dataset: dict[str, Any] = {
        "name": _slug(scenario_id),
        "version": f"20260902-{content_sha256[:12]}",
        "blob_path": str(scenario["blob_path"]),
        "datastore_name": "mlops_blob",
        "content_sha256": content_sha256,
    }
    target_column = scenario.get("target_column")
    if target_column:
        dataset["target_column"] = str(target_column)
    excluded_columns = [str(value) for value in scenario.get("exclude_columns") or []]
    if excluded_columns:
        dataset["excluded_columns"] = excluded_columns

    engines = ["pycaret"] if task_type == "clustering" else ["pycaret", "flaml"]
    round2_max_variants = 3 if task_type == "clustering" else 2
    split_strategy = "stratified" if task_type == "classification" else "random"
    candidate_timeout_seconds = _candidate_timeout_seconds(profile)
    config: dict[str, Any] = {
        "schema_version": "2.0",
        "experiment_name": f"qual-{_slug(scenario_id)}",
        "preset": "diagnostic",
        "task_type": task_type,
        "random_seed": 42,
        "holdout_fraction": 0.20,
        "holdout_split_strategy": split_strategy,
        "dataset": dataset,
        "azureml": {
            "subscription_id": "93044a08-5661-4f1b-b424-5eafe066a9d1",
            "resource_group": "mvpv1",
            "workspace_name": "mlops-accelerator",
            "compute_target": "mlopsv2computecluster",
            "default_datastore": "mlops_blob",
            "environment": environment,
        },
        "stage1": {
            "min_rows": 100,
            "max_missing_pct": 50,
            "generate_sweetviz": False,
            "eda_sample_size": 5000,
        },
        "split": {
            "strategy": split_strategy,
            "validation_fraction": 0.0,
            "test_fraction": 0.20,
            "cv_folds": 3,
            "locked_test": True,
        },
        "metrics": {
            "primary": PRIMARY_METRICS[task_type],
            "selection_protocol": "cross_validation",
            "cv_folds": 3,
            "locked_test_once": True,
            "min_comparable_candidates": 2,
        },
        "phases": {
            "phase_a_baseline": {
                "cv_folds": 3,
                "candidate_engine_timeout_seconds": candidate_timeout_seconds,
                "flaml_config": {"time_budget": candidate_timeout_seconds},
            },
            "phase_b": {
                "enable_profiling": True,
                "profiling_output_path": "outputs/dataset_profile.json",
                "library_dir": f"configs/recipes/{task_type}/variant_search",
                "library": "variant_search",
                "tier": "progressive",
                "max_variants": 4,
                "selection_strategy": "scored",
                "runtime_budget_sec": candidate_timeout_seconds,
                "time_budget_per_variant": candidate_timeout_seconds,
                "phase_timeout_seconds": 1800,
                "safety_net_review_required": True,
                "engines": engines,
                "planner": {
                    "enabled": True,
                    "round1_max_variants": 4,
                    "round2_max_variants": round2_max_variants,
                    "proxy_prune_threshold": 0.0,
                    "diversity_min_hamming_distance": 2,
                    "cache_enabled": True,
                },
            },
            "phase_c_hpo": {
                "optimizer": "optuna",
                "n_trials": 5,
                "timeout_seconds": 600,
            },
        },
        "registry": {
            "model_name": f"mlops-v3-qualification-{_slug(scenario_id)}",
            "quality_failure_policy": "warn",
            "warning_registration_allowed": True,
            "warning_tags": {
                "quality_decision": "warn",
                "validation_scope": "qualification",
                "industry_matrix_profile_run": profile_run_id,
            },
            "pass_aliases": [],
            "warning_aliases": [],
        },
        "recipes": [{"file": "recipes/baseline_recipe.yml"}],
    }
    compile_config(config, source_name=f"config_qualification_{_slug(scenario_id)}.yml")
    return config


def generate_configs(
    manifest: dict[str, Any],
    report: dict[str, Any],
    *,
    environment: str,
    profile_run_id: str,
    output_dir: Path,
    catalog_output: Path,
) -> list[dict[str, Any]]:
    scenarios = manifest.get("scenarios") or []
    profiles = {str(item["id"]): item for item in report.get("scenarios") or []}
    if len(scenarios) != 15 or len(profiles) != 15:
        raise ValueError("Config generation requires exactly 15 manifest and profile records")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        config = _build_config(
            scenario,
            profiles[scenario_id],
            environment=environment,
            profile_run_id=profile_run_id,
        )
        filename = f"config_qualification_{_slug(scenario_id).replace('-', '_')}_azureml.yml"
        output_path = output_dir / filename
        output_path.write_text(
            yaml.safe_dump(config, sort_keys=False, width=120),
            encoding="utf-8",
        )
        records.append(
            {
                "scenario_id": scenario_id,
                "task_type": scenario["task_type"],
                "industry": scenario["industry"],
                "config_path": output_path.relative_to(ROOT).as_posix(),
                "dataset_content_sha256": profiles[scenario_id]["content_sha256"],
                "dataset_schema_sha256": profiles[scenario_id]["schema_sha256"],
            }
        )

    catalog = {
        "schema_version": "1.0",
        "profile_run_id": profile_run_id,
        "environment": environment,
        "scenario_count": len(records),
        "task_counts": {
            task: sum(record["task_type"] == task for record in records)
            for task in PRIMARY_METRICS
        },
        "configs": records,
    }
    catalog_output.parent.mkdir(parents=True, exist_ok=True)
    catalog_output.write_text(
        yaml.safe_dump(catalog, sort_keys=False, width=120),
        encoding="utf-8",
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--profile-run-id", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "configs"))
    parser.add_argument(
        "--catalog-output",
        default=str(ROOT / "configs" / "qualification" / "industry_matrix_execution_catalog.yml"),
    )
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8")) or {}
    report = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    records = generate_configs(
        manifest,
        report,
        environment=args.environment,
        profile_run_id=args.profile_run_id,
        output_dir=Path(args.output_dir).resolve(),
        catalog_output=Path(args.catalog_output).resolve(),
    )
    print(json.dumps({"generated": len(records), "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
