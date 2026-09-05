#!/usr/bin/env python3
"""Fail-closed verification for Azure ML qualification evidence bundles."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "qualification" / "industry_matrix_execution_catalog.yml"
TASK_TYPES = ("classification", "regression", "clustering")
QUALITY_METRICS = {
    "classification": "balanced_accuracy",
    "regression": "r2",
    "clustering": "silhouette_score",
}
REQUIRED_OUTPUTS = (
    "execution_manifest",
    "split_manifest",
    "quality_decision",
    "final_report",
    "registry_info",
    "drift_report",
    "retrain_decision",
    "decision_ledger_record",
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class EvidenceInputError(ValueError):
    """Raised when the verifier input itself cannot be interpreted."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceInputError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise EvidenceInputError(f"{label} must contain a non-empty JSON object: {path}")
    return value


def _read_catalog(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceInputError(f"Cannot read qualification catalog {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("configs"), list):
        raise EvidenceInputError("Qualification catalog must contain a configs list")
    catalog: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(payload["configs"], start=1):
        if not isinstance(item, dict):
            raise EvidenceInputError(f"Catalog entry {index} must be an object")
        scenario_id = str(item.get("scenario_id") or "").strip()
        if not scenario_id:
            raise EvidenceInputError(f"Catalog entry {index} has no scenario_id")
        if scenario_id in catalog:
            raise EvidenceInputError(f"Duplicate catalog scenario_id: {scenario_id}")
        if item.get("task_type") not in TASK_TYPES or not str(item.get("industry") or "").strip():
            raise EvidenceInputError(f"Catalog scenario has invalid task or industry: {scenario_id}")
        for field in ("dataset_content_sha256", "dataset_schema_sha256"):
            if not SHA256_RE.fullmatch(str(item.get(field) or "")):
                raise EvidenceInputError(f"Catalog scenario has invalid {field}: {scenario_id}")
        catalog[scenario_id] = item
    return payload, catalog


def _resolve(base: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise EvidenceInputError(f"Missing path field: {field}")
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _single_payload(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_file():
        return _read_json(path, label=label)
    if not path.is_dir():
        raise EvidenceInputError(f"{label} path does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if len(files) != 1:
        raise EvidenceInputError(
            f"{label} directory must contain exactly one file; found {len(files)}: {path}"
        )
    return _read_json(files[0], label=label)


def _issue(issues: list[dict[str, str]], code: str, field: str, message: str) -> None:
    issues.append({"code": code, "field": field, "message": message})


def _expect(
    issues: list[dict[str, str]],
    condition: bool,
    code: str,
    field: str,
    message: str,
) -> None:
    if not condition:
        _issue(issues, code, field, message)


def _same(
    issues: list[dict[str, str]],
    actual: Any,
    expected: Any,
    code: str,
    field: str,
) -> None:
    _expect(
        issues,
        type(actual) is type(expected) and actual == expected,
        code,
        field,
        f"Expected {expected!r}; found {actual!r}",
    )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _load_monitor_jobs(
    paths: list[Path],
    issues: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    jobs_by_label: dict[str, dict[str, Any]] = {}
    job_ids: set[str] = set()
    for index, path in enumerate(paths, start=1):
        field = f"monitor_summaries[{index}]"
        try:
            summary = _read_json(path, label="monitor summary")
        except EvidenceInputError as exc:
            _issue(issues, "monitor_unreadable", field, str(exc))
            continue
        _same(issues, summary.get("schema_version"), "1.0", "monitor_schema", field)
        _same(issues, summary.get("state"), "passed", "monitor_state", field)
        jobs = summary.get("jobs")
        if not isinstance(jobs, list):
            _issue(issues, "monitor_jobs", field, "Monitor summary jobs must be a list")
            continue
        _same(
            issues,
            summary.get("expected_count"),
            len(jobs),
            "monitor_expected_count",
            field,
        )
        _same(
            issues,
            summary.get("accepted_submission_count"),
            len(jobs),
            "monitor_accepted_count",
            field,
        )
        _same(
            issues,
            summary.get("submission_failures"),
            [],
            "monitor_submission_failures",
            field,
        )
        for job_index, job in enumerate(jobs, start=1):
            job_field = f"{field}.jobs[{job_index}]"
            if not isinstance(job, dict):
                _issue(issues, "monitor_job", job_field, "Monitor job must be an object")
                continue
            label = str(job.get("label") or "").strip()
            job_id = str(job.get("job_id") or "").strip()
            _expect(issues, bool(label), "monitor_label", job_field, "Missing job label")
            _expect(issues, bool(job_id), "monitor_job_id", job_field, "Missing job ID")
            _same(issues, job.get("status"), "Completed", "monitor_job_status", job_field)
            _same(issues, job.get("query_error"), None, "monitor_query_error", job_field)
            if not label or not job_id:
                continue
            if label in jobs_by_label:
                _issue(issues, "duplicate_monitor_label", job_field, f"Duplicate label {label}")
            else:
                jobs_by_label[label] = job
            if job_id in job_ids:
                _issue(issues, "duplicate_monitor_job", job_field, f"Duplicate job ID {job_id}")
            job_ids.add(job_id)
    return jobs_by_label


def _load_asset_audit(
    path: Path,
    catalog: dict[str, dict[str, Any]],
    issues: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    try:
        audit = _read_json(path, label="data asset audit")
    except EvidenceInputError as exc:
        _issue(issues, "asset_audit_unreadable", "data_asset_audit", str(exc))
        return {}
    _same(issues, audit.get("schema_version"), "1.0", "asset_audit_schema", "data_asset_audit")
    _same(issues, audit.get("all_passed"), True, "asset_audit_state", "data_asset_audit")
    _same(issues, audit.get("errors"), [], "asset_audit_errors", "data_asset_audit")
    assets = audit.get("assets")
    if not isinstance(assets, list):
        _issue(issues, "asset_audit_assets", "data_asset_audit", "Assets must be a list")
        return {}
    by_scenario: dict[str, dict[str, Any]] = {}
    for index, asset in enumerate(assets, start=1):
        field = f"data_asset_audit.assets[{index}]"
        if not isinstance(asset, dict):
            _issue(issues, "asset_audit_asset", field, "Asset must be an object")
            continue
        checks = asset.get("checks")
        required_checks = {"version", "datastore_path", "scope", "hashes", "scenario_tags"}
        _expect(
            issues,
            isinstance(checks, dict)
            and required_checks.issubset(checks)
            and all(v is True for v in checks.values()),
            "asset_audit_checks",
            field,
            "Every data asset check must pass",
        )
        scenario_ids = asset.get("scenario_ids")
        if not isinstance(scenario_ids, list):
            _issue(issues, "asset_audit_scenarios", field, "scenario_ids must be a list")
            continue
        for scenario_id in scenario_ids:
            scenario_id = str(scenario_id)
            if scenario_id in by_scenario:
                _issue(
                    issues,
                    "duplicate_asset_scenario",
                    field,
                    f"Scenario {scenario_id} maps to more than one asset",
                )
            else:
                by_scenario[scenario_id] = asset
            expected = catalog.get(scenario_id)
            if expected:
                _same(
                    issues,
                    asset.get("content_sha256"),
                    expected.get("dataset_content_sha256"),
                    "asset_content_hash",
                    field,
                )
                _same(
                    issues,
                    asset.get("schema_sha256"),
                    expected.get("dataset_schema_sha256"),
                    "asset_schema_hash",
                    field,
                )
    _same(
        issues,
        audit.get("scenario_count"),
        len(catalog),
        "asset_audit_scenario_count",
        "data_asset_audit.scenario_count",
    )
    _same(
        issues,
        sorted(by_scenario),
        sorted(catalog),
        "asset_audit_coverage",
        "data_asset_audit.assets[].scenario_ids",
    )
    return by_scenario


def _load_named_outputs(
    pipeline_dir: Path,
    issues: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    root = pipeline_dir / "named-outputs"
    for name in REQUIRED_OUTPUTS:
        try:
            outputs[name] = _single_payload(root / name, label=f"named output {name}")
        except EvidenceInputError as exc:
            _issue(issues, "named_output", name, str(exc))
    return outputs


def _validate_execution_and_split(
    outputs: dict[str, dict[str, Any]],
    expected: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    execution = outputs.get("execution_manifest", {})
    split = outputs.get("split_manifest", {})
    task = expected["task_type"]
    if execution:
        _same(issues, execution.get("schema_version"), "2.0", "execution_schema", "execution_manifest")
        _same(issues, execution.get("task_type"), task, "execution_task", "execution_manifest.task_type")
        for name in ("execution_id", "config_hash", "code_sha"):
            _expect(
                issues,
                bool(SHA256_RE.fullmatch(str(execution.get(name) or ""))),
                f"execution_{name}",
                f"execution_manifest.{name}",
                f"{name} must be a SHA-256 value",
            )
        dataset = execution.get("dataset")
        _expect(issues, isinstance(dataset, dict), "execution_dataset", "execution_manifest.dataset", "Dataset identity is required")
        if isinstance(dataset, dict):
            _same(
                issues,
                dataset.get("content_sha256"),
                expected["dataset_content_sha256"],
                "execution_dataset_hash",
                "execution_manifest.dataset.content_sha256",
            )
    if split:
        _same(issues, split.get("schema_version"), "2.0", "split_schema", "split_manifest")
        _same(issues, split.get("task_type"), task, "split_task", "split_manifest.task_type")
        _same(issues, split.get("locked_test"), True, "split_locked", "split_manifest.locked_test")
        _expect(
            issues,
            bool(SHA256_RE.fullmatch(str(split.get("split_id") or ""))),
            "split_id",
            "split_manifest.split_id",
            "split_id must be a SHA-256 value",
        )
        _expect(issues, _is_positive_int(split.get("train_count")), "split_train_count", "split_manifest.train_count", "Training count must be positive")
        _expect(issues, _is_positive_int(split.get("test_count")), "split_test_count", "split_manifest.test_count", "Locked-test count must be positive")
    if execution and split:
        embedded = execution.get("split_manifest")
        _expect(issues, isinstance(embedded, dict), "execution_split", "execution_manifest.split_manifest", "Runtime split manifest is required")
        if isinstance(embedded, dict):
            _same(issues, embedded, split, "execution_split_binding", "execution_manifest.split_manifest")
        _same(issues, execution.get("runtime_split_id"), split.get("split_id"), "runtime_split_id", "execution_manifest.runtime_split_id")
    return execution


def _validate_stage6_quality(
    quality: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    if not quality:
        return
    _same(issues, quality.get("schema_version"), "2.0", "s06_quality_schema", "quality_decision")
    _same(issues, quality.get("decision"), "block", "s06_selection_only", "quality_decision.decision")
    _same(issues, quality.get("registration_allowed"), False, "s06_registration", "quality_decision.registration_allowed")
    _same(issues, quality.get("threshold"), None, "s06_threshold", "quality_decision.threshold")
    tags = quality.get("registration_tags")
    _expect(issues, isinstance(tags, dict), "s06_tags", "quality_decision.registration_tags", "Selection-only tags are required")
    if isinstance(tags, dict):
        _same(issues, tags.get("quality_stage"), "selection_only", "s06_stage", "quality_decision.registration_tags.quality_stage")
        _same(issues, tags.get("locked_test_evaluated"), "false", "s06_locked_test", "quality_decision.registration_tags.locked_test_evaluated")


def _validate_final_report(
    final: dict[str, Any],
    execution: dict[str, Any],
    split: dict[str, Any],
    expected: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    if not final:
        return {}
    task = expected["task_type"]
    _same(issues, final.get("schema_version"), 2, "final_schema", "final_report.schema_version")
    _same(issues, final.get("task"), task, "final_task", "final_report.task")
    _same(issues, final.get("holdout_source"), "stage2_split_manifest_bound_component_input", "holdout_source", "final_report.holdout_source")
    _same(issues, final.get("test_samples"), split.get("test_count"), "holdout_count", "final_report.test_samples")
    _same(issues, final.get("champion_valid"), True, "champion_valid", "final_report.champion_valid")
    validation = final.get("validation")
    _expect(issues, isinstance(validation, dict) and validation.get("valid") is True, "final_validation", "final_report.validation", "Final validation must pass")
    if isinstance(validation, dict):
        _same(issues, validation.get("errors"), [], "final_validation_errors", "final_report.validation.errors")
    output_validation = final.get("output_validation")
    _expect(issues, isinstance(output_validation, dict) and output_validation.get("valid") is True, "output_validation", "final_report.output_validation", "Bundle output validation must pass")
    if isinstance(output_validation, dict):
        _same(issues, output_validation.get("errors"), [], "output_validation_errors", "final_report.output_validation.errors")
        files = output_validation.get("files")
        names = {str(item.get("name")) for item in files if isinstance(item, dict)} if isinstance(files, list) else set()
        _expect(issues, {"model_bundle.pkl", "model_bundle_manifest.json"}.issubset(names), "bundle_files", "final_report.output_validation.files", "Both model bundle files are required")
        if isinstance(files, list):
            _expect(issues, all(isinstance(item, dict) and _is_positive_int(item.get("size")) for item in files), "bundle_file_size", "final_report.output_validation.files", "Bundle output files must have positive sizes")

    selection = final.get("selection")
    final_test = final.get("final_test")
    _expect(issues, isinstance(selection, dict), "selection", "final_report.selection", "Selection evidence is required")
    _expect(issues, isinstance(final_test, dict), "final_test", "final_report.final_test", "Locked final-test evidence is required")
    if isinstance(selection, dict):
        _same(issues, selection.get("source"), "cross_validation_or_validation", "selection_source", "final_report.selection.source")
        _same(issues, selection.get("locked_test_used_for_selection"), False, "selection_holdout_isolation", "final_report.selection.locked_test_used_for_selection")
        _expect(issues, bool(selection.get("candidate_id")), "selection_candidate", "final_report.selection.candidate_id", "Selected candidate ID is required")
        _expect(issues, _is_finite_number(selection.get("score")), "selection_score", "final_report.selection.score", "Selection score must be finite")
    if isinstance(final_test, dict):
        _same(issues, final_test.get("evaluated_once"), True, "final_test_once", "final_report.final_test.evaluated_once")
        _same(issues, final_test.get("row_count"), split.get("test_count"), "final_test_count", "final_report.final_test.row_count")
        if isinstance(selection, dict):
            _same(issues, final_test.get("candidate_phase"), selection.get("key"), "final_phase_binding", "final_report.final_test.candidate_phase")
            _same(issues, final_test.get("candidate_id"), selection.get("candidate_id"), "final_candidate_binding", "final_report.final_test.candidate_id")
            _same(issues, final_test.get("source_bundle_id"), selection.get("source_bundle_id"), "final_bundle_binding", "final_report.final_test.source_bundle_id")

    lineage = final.get("lineage")
    _expect(issues, isinstance(lineage, dict), "final_lineage", "final_report.lineage", "Lineage is required")
    if isinstance(lineage, dict):
        for field, expected_value in (
            ("execution_id", execution.get("execution_id")),
            ("config_hash", execution.get("config_hash")),
            ("code_sha", execution.get("code_sha")),
            ("split_id", split.get("split_id")),
        ):
            _same(issues, lineage.get(field), expected_value, f"lineage_{field}", f"final_report.lineage.{field}")
        if isinstance(selection, dict):
            _same(issues, lineage.get("source_candidate_id"), selection.get("candidate_id"), "lineage_candidate", "final_report.lineage.source_candidate_id")
            _same(issues, lineage.get("source_bundle_id"), selection.get("source_bundle_id"), "lineage_source_bundle", "final_report.lineage.source_bundle_id")
        for field in ("parent_run_id", "final_evaluation_run_id"):
            _expect(issues, bool(lineage.get(field)), "mlflow_lineage", f"final_report.lineage.{field}", f"{field} is required")
        candidate_lineage = lineage.get("candidate_lineage")
        _expect(issues, isinstance(candidate_lineage, dict), "candidate_lineage", "final_report.lineage.candidate_lineage", "Candidate MLflow lineage is required")
        if isinstance(candidate_lineage, dict):
            _same(issues, candidate_lineage.get("execution_id"), execution.get("execution_id"), "candidate_execution", "final_report.lineage.candidate_lineage.execution_id")
            _same(issues, candidate_lineage.get("parent_run_id"), lineage.get("parent_run_id"), "candidate_parent", "final_report.lineage.candidate_lineage.parent_run_id")
            _expect(issues, bool(candidate_lineage.get("candidate_run_id")), "candidate_run", "final_report.lineage.candidate_lineage.candidate_run_id", "Candidate run ID is required")

    embedded_execution = final.get("execution_manifest")
    _expect(issues, isinstance(embedded_execution, dict), "final_execution", "final_report.execution_manifest", "Embedded execution identity is required")
    if isinstance(embedded_execution, dict):
        for field in ("execution_id", "config_hash", "code_sha", "task_type"):
            _same(issues, embedded_execution.get(field), execution.get(field), f"final_execution_{field}", f"final_report.execution_manifest.{field}")
    embedded_split = final.get("split_manifest")
    _expect(issues, isinstance(embedded_split, dict), "final_split", "final_report.split_manifest", "Embedded locked-test identity is required")
    if isinstance(embedded_split, dict):
        for field in ("split_id", "data_version", "test_count", "test_ids_hash"):
            _same(issues, embedded_split.get(field), split.get(field), f"final_split_{field}", f"final_report.split_manifest.{field}")

    bundle = final.get("model_bundle")
    _expect(issues, isinstance(bundle, dict), "model_bundle", "final_report.model_bundle", "Raw-input model bundle manifest is required")
    if isinstance(bundle, dict):
        bundle_schema = bundle.get("bundle_schema_version")
        # Match ModelBundle's versioned state-hash compatibility without
        # importing the training runtime into this artifact-only verifier.
        _expect(
            issues,
            type(bundle_schema) is int and bundle_schema in (3, 4, 5),
            "bundle_schema",
            "final_report.model_bundle.bundle_schema_version",
            f"Expected integer ModelBundle schema 3, 4 or 5; found {bundle_schema!r}",
        )
        _same(issues, bundle.get("task_type"), task, "bundle_task", "final_report.model_bundle.task_type")
        if isinstance(selection, dict):
            _same(issues, bundle.get("candidate_id"), selection.get("candidate_id"), "bundle_candidate", "final_report.model_bundle.candidate_id")
        for field in ("bundle_id", "model_state_sha256", "artifact_sha256"):
            _expect(issues, bool(SHA256_RE.fullmatch(str(bundle.get(field) or ""))), f"bundle_{field}", f"final_report.model_bundle.{field}", f"{field} must be a SHA-256 value")
        for field in ("input_schema", "recipe", "signature"):
            _expect(issues, bool(bundle.get(field)), f"bundle_{field}", f"final_report.model_bundle.{field}", f"{field} is required")
        _expect(issues, isinstance(bundle.get("input_example"), list) and bool(bundle["input_example"]), "bundle_input_example", "final_report.model_bundle.input_example", "Raw input example is required")
        _expect(issues, isinstance(bundle.get("dependencies"), list) and bool(bundle["dependencies"]), "bundle_dependencies", "final_report.model_bundle.dependencies", "Dependencies are required")
        if isinstance(final_test, dict):
            _same(issues, bundle.get("final_test_metrics"), final_test.get("metrics"), "bundle_final_metrics", "final_report.model_bundle.final_test_metrics")
        selection_metrics = bundle.get("selection_metrics")
        _expect(issues, isinstance(selection_metrics, dict), "bundle_selection_metrics", "final_report.model_bundle.selection_metrics", "Bundle selection metrics are required")
        if isinstance(selection_metrics, dict) and isinstance(selection, dict):
            _same(issues, selection_metrics.get("selection_score"), selection.get("score"), "bundle_selection_score", "final_report.model_bundle.selection_metrics.selection_score")
            _same(issues, selection_metrics.get("source"), selection.get("source"), "bundle_selection_source", "final_report.model_bundle.selection_metrics.source")
        bundle_lineage = bundle.get("lineage")
        _expect(issues, isinstance(bundle_lineage, dict), "bundle_lineage", "final_report.model_bundle.lineage", "Bundle lineage is required")
        if isinstance(bundle_lineage, dict) and isinstance(lineage, dict):
            for field in ("execution_id", "config_hash", "code_sha", "split_id", "source_candidate_id", "source_bundle_id", "parent_run_id", "final_evaluation_run_id"):
                _same(issues, bundle_lineage.get(field), lineage.get(field), f"bundle_lineage_{field}", f"final_report.model_bundle.lineage.{field}")

    quality = final.get("quality_decision")
    _expect(issues, isinstance(quality, dict), "final_quality", "final_report.quality_decision", "Final quality decision is required")
    if isinstance(quality, dict):
        _same(issues, quality.get("schema_version"), "2.0", "final_quality_schema", "final_report.quality_decision.schema_version")
        _expect(issues, quality.get("decision") in ("pass", "warn"), "final_quality_decision", "final_report.quality_decision.decision", "Decision must be pass or warn")
        _same(issues, quality.get("metric_name"), QUALITY_METRICS[task], "final_quality_metric_name", "final_report.quality_decision.metric_name")
        _same(issues, quality.get("registration_allowed"), True, "final_registration", "final_report.quality_decision.registration_allowed")
        _expect(issues, _is_finite_number(quality.get("metric_value")), "final_quality_metric", "final_report.quality_decision.metric_value", "Final quality metric must be finite")
        metrics = final_test.get("metrics") if isinstance(final_test, dict) else None
        _expect(issues, isinstance(metrics, dict), "final_test_metrics", "final_report.final_test.metrics", "Locked-test metrics are required")
        if isinstance(metrics, dict):
            _same(issues, quality.get("metric_value"), metrics.get(QUALITY_METRICS[task]), "quality_metric_binding", "final_report.quality_decision.metric_value")
        _same(issues, quality.get("threshold"), final.get("quality_threshold"), "quality_threshold_binding", "final_report.quality_decision.threshold")
        if quality.get("decision") == "pass":
            _same(issues, final.get("quality_gate_passed"), True, "final_quality_gate", "final_report.quality_gate_passed")
        if isinstance(selection, dict):
            _same(issues, quality.get("candidate_id"), selection.get("candidate_id"), "quality_candidate", "final_report.quality_decision.candidate_id")
        if isinstance(bundle, dict):
            _same(issues, quality.get("evaluated_bundle_hash"), bundle.get("bundle_id"), "quality_bundle", "final_report.quality_decision.evaluated_bundle_hash")
        _expect(issues, bool(SHA256_RE.fullmatch(str(quality.get("decision_hash") or ""))), "quality_decision_hash", "final_report.quality_decision.decision_hash", "Decision hash must be SHA-256")
    return bundle if isinstance(bundle, dict) else {}


def _validate_registry_and_policy(
    outputs: dict[str, dict[str, Any]],
    final: dict[str, Any],
    execution: dict[str, Any],
    expected: dict[str, Any],
    scenario_id: str,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    registry = outputs.get("registry_info", {})
    quality = final.get("quality_decision", {}) if final else {}
    if not isinstance(quality, dict):
        quality = {}
    model_name = f"mlops-v3-qualification-{scenario_id}"
    if registry:
        _same(issues, registry.get("model_name"), model_name, "registry_model", "registry_info.model_name")
        version = registry.get("version")
        _expect(issues, bool(re.fullmatch(r"[1-9][0-9]*", str(version or ""))), "registry_version", "registry_info.version", "Exact positive model version is required")
        _same(issues, registry.get("model_uri"), f"models:/{model_name}/{version}", "registry_uri", "registry_info.model_uri")
        _same(issues, registry.get("task_type"), expected["task_type"], "registry_task", "registry_info.task_type")
        _same(issues, registry.get("execution_id"), execution.get("execution_id"), "registry_execution", "registry_info.execution_id")
        _same(issues, registry.get("config_hash"), execution.get("config_hash"), "registry_config", "registry_info.config_hash")
        _same(issues, registry.get("code_sha"), execution.get("code_sha"), "registry_code", "registry_info.code_sha")
        _same(issues, registry.get("dataset_content_sha256"), expected["dataset_content_sha256"], "registry_dataset", "registry_info.dataset_content_sha256")
        _same(issues, registry.get("quality_decision"), quality.get("decision"), "registry_quality", "registry_info.quality_decision")
        _same(issues, registry.get("registration_backend"), "mlflow", "registry_backend", "registry_info.registration_backend")
        _expect(issues, bool(registry.get("registration_run_id")), "registration_run", "registry_info.registration_run_id", "Registration run ID is required")
        _same(issues, registry.get("promotion_mode"), "manual", "registry_promotion_mode", "registry_info.promotion_mode")
        _same(issues, registry.get("promotion_performed"), False, "registry_promotion", "registry_info.promotion_performed")
        _same(issues, registry.get("requested_promotion_aliases"), [], "registry_aliases", "registry_info.requested_promotion_aliases")
        _same(issues, registry.get("stage"), "None", "registry_stage", "registry_info.stage")
        _same(issues, registry.get("lifecycle_stage"), "Unassigned", "registry_lifecycle", "registry_info.lifecycle_stage")

    drift = outputs.get("drift_report", {})
    if drift:
        _same(issues, drift.get("task_type"), expected["task_type"], "drift_task", "drift_report.task_type")
        identity = drift.get("identity")
        _expect(issues, isinstance(identity, dict), "drift_identity", "drift_report.identity", "Drift identity is required")
        if isinstance(identity, dict):
            _same(issues, identity.get("execution_id"), execution.get("execution_id"), "drift_execution", "drift_report.identity.execution_id")
            _same(issues, identity.get("config_hash"), execution.get("config_hash"), "drift_config", "drift_report.identity.config_hash")
            _same(issues, identity.get("model_name"), model_name, "drift_model", "drift_report.identity.model_name")

    retrain = outputs.get("retrain_decision", {})
    if retrain:
        _expect(issues, isinstance(retrain.get("decision_id"), str) and bool(retrain["decision_id"]), "retrain_decision_id", "retrain_decision.decision_id", "S14 decision ID is required")
        _same(issues, retrain.get("task_type"), expected["task_type"], "retrain_task", "retrain_decision.task_type")
        identity = retrain.get("identity")
        _expect(issues, isinstance(identity, dict), "retrain_identity", "retrain_decision.identity", "Retrain identity is required")
        if isinstance(identity, dict):
            for field, value in (("execution_id", execution.get("execution_id")), ("config_hash", execution.get("config_hash")), ("source_sha", execution.get("code_sha")), ("model_name", model_name)):
                _same(issues, identity.get(field), value, f"retrain_{field}", f"retrain_decision.identity.{field}")
        revision = retrain.get("revision_validation")
        _expect(issues, isinstance(revision, dict), "revision_validation", "retrain_decision.revision_validation", "Revision validation is required")
        if isinstance(revision, dict):
            _same(issues, revision.get("status"), "verified", "revision_status", "retrain_decision.revision_validation.status")
            _same(issues, revision.get("required_contract_present"), True, "revision_contract", "retrain_decision.revision_validation.required_contract_present")
            _same(issues, revision.get("missing_fields"), [], "revision_missing", "retrain_decision.revision_validation.missing_fields")
            _same(issues, revision.get("conflicts"), {}, "revision_conflicts", "retrain_decision.revision_validation.conflicts")
            _same(issues, revision.get("contract_error"), None, "revision_error", "retrain_decision.revision_validation.contract_error")
            _same(issues, revision.get("required_contract"), "final_report.execution_manifest", "revision_contract_name", "retrain_decision.revision_validation.required_contract")
        source_revision = retrain.get("source_revision")
        _expect(issues, isinstance(source_revision, dict), "source_revision", "retrain_decision.source_revision", "S14 source revision is required")
        if isinstance(source_revision, dict):
            for field, value in (("schema_version", "1.0"), ("execution_id", execution.get("execution_id")), ("config_hash", execution.get("config_hash")), ("source_sha", execution.get("code_sha"))):
                _same(issues, source_revision.get(field), value, f"source_revision_{field}", f"retrain_decision.source_revision.{field}")
        decision = retrain.get("decision")
        _expect(issues, isinstance(decision, dict) and isinstance(decision.get("should_submit"), bool), "retrain_policy", "retrain_decision.decision", "Policy must emit a boolean should_submit decision")

    ledger = outputs.get("decision_ledger_record", {})
    if ledger:
        _same(issues, ledger.get("task_type"), expected["task_type"], "ledger_task", "decision_ledger_record.task_type")
        _same(issues, ledger.get("promotion_mode"), "manual", "ledger_mode", "decision_ledger_record.promotion_mode")
        _same(
            issues,
            ledger.get("promotion_status"),
            "manual_pending",
            "ledger_promotion",
            "decision_ledger_record.promotion_status",
        )
        if retrain:
            _same(issues, ledger.get("decision_id"), retrain.get("decision_id"), "ledger_decision_id", "decision_ledger_record.decision_id")
            decision = retrain.get("decision")
            if isinstance(decision, dict):
                _same(issues, ledger.get("outcome"), decision.get("outcome"), "ledger_outcome", "decision_ledger_record.outcome")
            metadata = ledger.get("metadata")
            if isinstance(metadata, dict):
                _same(issues, metadata.get("source_revision"), retrain.get("source_revision"), "ledger_source_revision", "decision_ledger_record.metadata.source_revision")
                _same(issues, metadata.get("revision_validation"), retrain.get("revision_validation"), "ledger_revision_validation", "decision_ledger_record.metadata.revision_validation")
        metadata_identity = ledger.get("metadata", {}).get("identity") if isinstance(ledger.get("metadata"), dict) else None
        _expect(issues, isinstance(metadata_identity, dict), "ledger_identity", "decision_ledger_record.metadata.identity", "Ledger identity is required")
        if isinstance(metadata_identity, dict):
            _same(issues, metadata_identity.get("execution_id"), execution.get("execution_id"), "ledger_execution", "decision_ledger_record.metadata.identity.execution_id")
            _same(issues, metadata_identity.get("config_hash"), execution.get("config_hash"), "ledger_config", "decision_ledger_record.metadata.identity.config_hash")
            _same(issues, metadata_identity.get("source_sha"), execution.get("code_sha"), "ledger_code", "decision_ledger_record.metadata.identity.source_sha")
            _same(issues, metadata_identity.get("model_name"), model_name, "ledger_model", "decision_ledger_record.metadata.identity.model_name")
    return registry


def _validate_smoke(
    submission: dict[str, Any],
    smoke: dict[str, Any],
    registry: dict[str, Any],
    execution: dict[str, Any],
    expected: dict[str, Any],
    scenario_id: str,
    parent_job: str,
    issues: list[dict[str, str]],
) -> str:
    _same(issues, submission.get("schema_version"), "1.0", "smoke_submission_schema", "registered_model_smoke_submission")
    _same(issues, submission.get("status"), "Completed", "smoke_submission_status", "registered_model_smoke_submission.status")
    _same(issues, submission.get("scenario_id"), scenario_id, "smoke_submission_scenario", "registered_model_smoke_submission.scenario_id")
    _same(issues, submission.get("parent_job"), parent_job, "smoke_submission_parent", "registered_model_smoke_submission.parent_job")
    _same(issues, submission.get("model_uri"), registry.get("model_uri"), "smoke_submission_model", "registered_model_smoke_submission.model_uri")
    _same(issues, submission.get("execution_id"), execution.get("execution_id"), "smoke_submission_execution", "registered_model_smoke_submission.execution_id")
    _same(issues, submission.get("code_sha"), execution.get("code_sha"), "smoke_submission_code", "registered_model_smoke_submission.code_sha")
    git_commit = str(submission.get("source_git_commit") or "")
    _expect(issues, bool(GIT_SHA_RE.fullmatch(git_commit)), "smoke_submission_git", "registered_model_smoke_submission.source_git_commit", "Source Git commit must be a full SHA")
    smoke_job = str(submission.get("job_name") or "")
    _expect(issues, bool(smoke_job), "smoke_submission_job", "registered_model_smoke_submission.job_name", "Smoke job name is required")
    for field in ("environment", "compute", "output_datastore", "evidence_uri"):
        _expect(issues, bool(submission.get(field)), f"smoke_submission_{field}", f"registered_model_smoke_submission.{field}", f"{field} is required")

    _same(issues, smoke.get("schema_version"), "1.0", "smoke_schema", "registered_model_smoke_evidence")
    _same(issues, smoke.get("status"), "passed", "smoke_status", "registered_model_smoke_evidence.status")
    _same(issues, smoke.get("azureml_run_id"), smoke_job, "smoke_job_binding", "registered_model_smoke_evidence.azureml_run_id")
    for field, value in (
        ("model_name", registry.get("model_name")),
        ("model_version", registry.get("version")),
        ("model_uri", registry.get("model_uri")),
        ("execution_id", execution.get("execution_id")),
        ("code_sha", execution.get("code_sha")),
        ("dataset_content_sha256", expected["dataset_content_sha256"]),
        ("registration_run_id", registry.get("registration_run_id")),
    ):
        _same(issues, smoke.get(field), value, f"smoke_{field}", f"registered_model_smoke_evidence.{field}")
    _same(issues, smoke.get("lineage_tags_verified"), True, "smoke_lineage", "registered_model_smoke_evidence.lineage_tags_verified")
    _same(issues, smoke.get("input_type"), "DataFrame", "smoke_input_type", "registered_model_smoke_evidence.input_type")
    _expect(issues, _is_positive_int(smoke.get("input_rows")), "smoke_input_rows", "registered_model_smoke_evidence.input_rows", "Input row count must be positive")
    _same(issues, smoke.get("prediction_rows"), smoke.get("input_rows"), "smoke_prediction_rows", "registered_model_smoke_evidence.prediction_rows")
    _same(issues, smoke.get("aliases"), [], "smoke_aliases", "registered_model_smoke_evidence.aliases")
    _same(issues, smoke.get("current_stage"), "None", "smoke_stage", "registered_model_smoke_evidence.current_stage")
    _expect(issues, bool(smoke.get("signature")), "smoke_signature", "registered_model_smoke_evidence.signature", "Loaded MLflow signature is required")
    return git_commit


def _validate_scenario(
    entry: dict[str, Any],
    expected: dict[str, Any],
    base: Path,
    monitor_jobs: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    scenario_id = str(entry.get("scenario_id") or "").strip()
    parent_job = str(entry.get("parent_job") or "").strip()
    monitor_job = monitor_jobs.get(scenario_id)
    _expect(issues, monitor_job is not None, "parent_monitor_missing", "parent_job", "No terminal monitor record exists for this scenario")
    if monitor_job:
        _same(issues, parent_job, monitor_job.get("job_id"), "parent_monitor_binding", "parent_job")
    _expect(issues, scenario_id in assets, "asset_scenario_missing", "data_asset_audit", "Scenario is not covered by the verified data asset audit")

    try:
        pipeline_dir = _resolve(base, entry.get("pipeline_evidence_dir"), field="pipeline_evidence_dir")
        outputs = _load_named_outputs(pipeline_dir, issues)
    except EvidenceInputError as exc:
        _issue(issues, "pipeline_evidence", "pipeline_evidence_dir", str(exc))
        outputs = {}
    execution = _validate_execution_and_split(outputs, expected, issues)
    split = outputs.get("split_manifest", {})
    _validate_stage6_quality(outputs.get("quality_decision", {}), issues)
    final = outputs.get("final_report", {})
    bundle = _validate_final_report(final, execution, split, expected, issues)
    registry = _validate_registry_and_policy(outputs, final, execution, expected, scenario_id, issues)

    submission: dict[str, Any] = {}
    smoke: dict[str, Any] = {}
    for field, target in (
        ("registered_model_smoke_submission", "submission"),
        ("registered_model_smoke_evidence", "smoke"),
    ):
        try:
            payload = _single_payload(_resolve(base, entry.get(field), field=field), label=field)
            if target == "submission":
                submission = payload
            else:
                smoke = payload
        except EvidenceInputError as exc:
            _issue(issues, "smoke_evidence", field, str(exc))
    git_commit = ""
    if submission and smoke:
        git_commit = _validate_smoke(
            submission,
            smoke,
            registry,
            execution,
            expected,
            scenario_id,
            parent_job,
            issues,
        )
    return {
        "scenario_id": scenario_id,
        "task_type": expected["task_type"],
        "industry": expected["industry"],
        "parent_job": parent_job,
        "state": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": issues,
        "identity": {
            "execution_id": execution.get("execution_id"),
            "config_hash": execution.get("config_hash"),
            "runtime_source_sha256": execution.get("code_sha"),
            "source_git_commit": git_commit or None,
            "split_id": split.get("split_id"),
            "model_name": registry.get("model_name"),
            "model_version": registry.get("version"),
            "model_uri": registry.get("model_uri"),
            "model_bundle_id": bundle.get("bundle_id"),
            "registration_run_id": registry.get("registration_run_id"),
            "smoke_job": submission.get("job_name"),
        },
    }


def verify_qualification_evidence(
    manifest_path: Path,
    *,
    catalog_path: Path = DEFAULT_CATALOG,
    require_complete_matrix: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    catalog_path = catalog_path.resolve()
    manifest = _read_json(manifest_path, label="qualification evidence manifest")
    catalog_payload, catalog = _read_catalog(catalog_path)
    global_issues: list[dict[str, str]] = []
    _same(global_issues, manifest.get("schema_version"), "1.0", "manifest_schema", "schema_version")
    base = manifest_path.parent

    raw_monitors = manifest.get("monitor_summaries")
    if not isinstance(raw_monitors, list) or not raw_monitors:
        _issue(global_issues, "monitor_summaries", "monitor_summaries", "At least one monitor summary is required")
        monitor_paths: list[Path] = []
    else:
        monitor_paths = []
        for index, value in enumerate(raw_monitors, start=1):
            try:
                monitor_paths.append(_resolve(base, value, field=f"monitor_summaries[{index}]"))
            except EvidenceInputError as exc:
                _issue(global_issues, "monitor_path", f"monitor_summaries[{index}]", str(exc))
    monitor_jobs = _load_monitor_jobs(monitor_paths, global_issues)

    try:
        asset_path = _resolve(base, manifest.get("data_asset_audit"), field="data_asset_audit")
        assets = _load_asset_audit(asset_path, catalog, global_issues)
    except EvidenceInputError as exc:
        _issue(global_issues, "asset_audit_path", "data_asset_audit", str(exc))
        assets = {}

    entries = manifest.get("scenarios")
    if not isinstance(entries, list) or not entries:
        raise EvidenceInputError("Qualification evidence manifest must contain a non-empty scenarios list")
    normalized: list[dict[str, Any]] = []
    scenario_ids: list[str] = []
    parent_jobs: list[str] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            _issue(global_issues, "scenario_entry", f"scenarios[{index}]", "Scenario entry must be an object")
            continue
        scenario_id = str(entry.get("scenario_id") or "").strip()
        parent_job = str(entry.get("parent_job") or "").strip()
        _expect(global_issues, bool(scenario_id), "scenario_id", f"scenarios[{index}]", "scenario_id is required")
        _expect(global_issues, bool(parent_job), "parent_job", f"scenarios[{index}]", "parent_job is required")
        if not scenario_id or not parent_job:
            continue
        scenario_ids.append(scenario_id)
        parent_jobs.append(parent_job)
        if scenario_id not in catalog:
            _issue(global_issues, "unknown_scenario", f"scenarios[{index}]", f"Scenario is not in the catalog: {scenario_id}")
            continue
        normalized_entry = dict(entry)
        normalized_entry["scenario_id"] = scenario_id
        normalized_entry["parent_job"] = parent_job
        normalized.append(normalized_entry)
    for value, count in Counter(scenario_ids).items():
        if count > 1:
            _issue(global_issues, "duplicate_scenario", "scenarios", f"Duplicate scenario_id: {value}")
    for value, count in Counter(parent_jobs).items():
        if count > 1:
            _issue(global_issues, "duplicate_parent_job", "scenarios", f"Duplicate parent_job: {value}")

    scenario_reports = [
        _validate_scenario(entry, catalog[entry["scenario_id"]], base, monitor_jobs, assets)
        for entry in normalized
    ]
    for field in ("execution_id", "smoke_job"):
        identities = [report["identity"][field] for report in scenario_reports if isinstance(report["identity"][field], str) and report["identity"][field]]
        for value, count in Counter(identities).items():
            if count > 1:
                _issue(global_issues, f"duplicate_{field}", "scenarios", f"Duplicate {field}: {value}")
    selected = set(scenario_ids)
    catalog_ids = set(catalog)
    task_counts = Counter(report["task_type"] for report in scenario_reports)
    industries: dict[str, set[str]] = defaultdict(set)
    for report in scenario_reports:
        industries[report["task_type"]].add(report["industry"])

    release_candidate = manifest.get("release_candidate")
    expected_git = None
    expected_runtime = None
    if release_candidate is not None:
        if not isinstance(release_candidate, dict):
            _issue(global_issues, "release_candidate", "release_candidate", "Release candidate must be an object")
        else:
            expected_git = str(release_candidate.get("git_commit") or "")
            expected_runtime = str(release_candidate.get("runtime_source_sha256") or "")
            _expect(global_issues, bool(GIT_SHA_RE.fullmatch(expected_git)), "release_git", "release_candidate.git_commit", "Release Git commit must be a full SHA")
            _expect(global_issues, bool(SHA256_RE.fullmatch(expected_runtime)), "release_runtime", "release_candidate.runtime_source_sha256", "Runtime source identity must be SHA-256")
    if require_complete_matrix and not isinstance(release_candidate, dict):
        _issue(global_issues, "release_candidate_required", "release_candidate", "Complete-matrix verification requires an exact release candidate")

    runtime_hashes = sorted({str(report["identity"]["runtime_source_sha256"]) for report in scenario_reports if report["identity"]["runtime_source_sha256"]})
    git_commits = sorted({str(report["identity"]["source_git_commit"]) for report in scenario_reports if report["identity"]["source_git_commit"]})
    if expected_runtime:
        for report in scenario_reports:
            if report["identity"]["runtime_source_sha256"] != expected_runtime:
                _issue(report["issues"], "release_runtime_mismatch", "identity.runtime_source_sha256", f"Expected release runtime source {expected_runtime}")
                report["state"] = "failed"
                report["issue_count"] = len(report["issues"])
    if expected_git:
        for report in scenario_reports:
            if report["identity"]["source_git_commit"] != expected_git:
                _issue(report["issues"], "release_git_mismatch", "identity.source_git_commit", f"Expected release Git commit {expected_git}")
                report["state"] = "failed"
                report["issue_count"] = len(report["issues"])

    missing = sorted(catalog_ids - selected)
    unexpected = sorted(selected - catalog_ids)
    matrix_complete = not missing and not unexpected and len(selected) == len(catalog_ids)
    if require_complete_matrix:
        _expect(global_issues, matrix_complete, "matrix_coverage", "scenarios", f"Complete matrix required; missing={missing}, unexpected={unexpected}")
        for task in TASK_TYPES:
            _same(global_issues, task_counts.get(task, 0), 5, "matrix_task_count", f"matrix.task_counts.{task}")
            _same(global_issues, len(industries.get(task, set())), 5, "matrix_industry_count", f"matrix.industry_counts.{task}")
        _same(global_issues, len(runtime_hashes), 1, "matrix_runtime_identity", "runtime_source_sha256_values")
        _same(global_issues, len(git_commits), 1, "matrix_git_identity", "source_git_commit_values")

    accepted = sum(report["state"] == "passed" for report in scenario_reports)
    state = "passed" if not global_issues and accepted == len(scenario_reports) else "failed"
    return {
        "schema_version": "1.0",
        "scope": "azure_ml_qualification_artifact_acceptance",
        "verified_at_utc": utcnow(),
        "state": state,
        "complete_matrix_required": require_complete_matrix,
        "release_matrix_accepted": bool(require_complete_matrix and state == "passed" and matrix_complete),
        "manifest": str(manifest_path),
        "catalog": str(catalog_path),
        "catalog_profile_run_id": catalog_payload.get("profile_run_id"),
        "monitor_summary_count": len(monitor_paths),
        "scenario_count": len(scenario_reports),
        "accepted_scenario_count": accepted,
        "failed_scenario_count": len(scenario_reports) - accepted,
        "global_issue_count": len(global_issues),
        "global_issues": global_issues,
        "runtime_source_sha256_values": runtime_hashes,
        "source_git_commit_values": git_commits,
        "matrix": {
            "catalog_scenario_count": len(catalog),
            "complete": matrix_complete,
            "missing_scenarios": missing,
            "unexpected_scenarios": unexpected,
            "task_counts": {task: task_counts.get(task, 0) for task in TASK_TYPES},
            "industry_counts": {task: len(industries.get(task, set())) for task in TASK_TYPES},
            "industries": {task: sorted(industries.get(task, set())) for task in TASK_TYPES},
        },
        "scenarios": scenario_reports,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify terminal Azure ML qualification, lineage, registration, and inference evidence"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--require-complete-matrix", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_qualification_evidence(
            args.manifest,
            catalog_path=args.catalog,
            require_complete_matrix=args.require_complete_matrix,
        )
    except EvidenceInputError as exc:
        print(f"Qualification evidence input error: {exc}", file=sys.stderr)
        return 2
    _write_json_atomic(args.output_json.resolve(), report)
    print(
        f"state={report['state']} accepted={report['accepted_scenario_count']}/"
        f"{report['scenario_count']} complete_matrix={report['matrix']['complete']}"
    )
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
