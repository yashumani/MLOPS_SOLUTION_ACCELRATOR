"""Stage 14: auto-retrain decision gate.

Consumes s13 drift artifacts, applies the auto-retrain policy, and writes
operator-readable decision artifacts. This stage does not submit Azure ML jobs;
external controllers keep ownership of canonical pipeline submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestration.auto_retrain_decision_ledger import (  # noqa: E402
    AutoRetrainDecisionRecord,
    build_decision_record,
)
from orchestration import build_planned_schedules_table  # noqa: E402
from orchestration.auto_retrain_policy import (  # noqa: E402
    AutoRetrainPolicyConfig,
    evaluate_auto_retrain_policy,
)
from orchestration.contracts import (  # noqa: E402
    ContractValidationError,
    ExecutionManifest,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


STAGE_NAME = "s14_retrain_decision"
STAGE_ID = "S14"

_S14_POLICY_FIELDS = {
    "severe_feature_psi",
    "severe_drifted_share",
    "low_stability_score",
    "urgent_cadence_days",
    "minimum_promotion_delta",
    "allow_auto_promotion",
    "require_registered_candidate_for_promotion",
}

_REVISION_IDENTITY_ALIASES = {
    "execution_id": ("execution_id",),
    "config_hash": ("config_hash", "compiled_config_hash"),
    "source_sha": ("source_sha", "source_identity", "code_sha"),
}


def _safe_disable_autolog() -> None:
    try:
        mlflow.autolog(disable=True)
    except Exception as exc:  # noqa: BLE001 - telemetry must not fail the stage
        logger.debug("mlflow.autolog(disable=True) failed: %s", exc)


def _load_json_safe(path: str | None, label: str) -> dict[str, Any]:
    if not path:
        return {}
    candidate = Path(path)
    try:
        if candidate.is_dir():
            for name in ("retrain_decision.json", "drift_report.json", "final_report.json", "registry_info.json"):
                nested = candidate / name
                if nested.exists():
                    return json.loads(nested.read_text(encoding="utf-8"))
            json_files = sorted(candidate.glob("*.json"))
            if json_files:
                return json.loads(json_files[0].read_text(encoding="utf-8"))
            logger.warning("%s directory has no JSON files: %s", label, path)
            return {}
        if not candidate.exists():
            logger.warning("%s file not found: %s", label, path)
            return {}
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - bad upstream artifacts become blocked decisions
        logger.warning("%s failed to load %s: %s", label, path, exc)
        return {}


def _load_retrain_policy(
    config_path: str,
) -> tuple[AutoRetrainPolicyConfig, dict[str, Any]]:
    """Load and validate the effective S14 policy from drift configuration."""
    candidate = Path(config_path)
    if not candidate.is_file():
        raise FileNotFoundError(f"Drift policy config not found: {candidate}")

    raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Drift policy config must contain a YAML mapping")
    thresholds = raw.get("thresholds") or {}
    policy_raw = raw.get("retrain_policy") or {}
    if not isinstance(thresholds, dict) or not isinstance(policy_raw, dict):
        raise ValueError("thresholds and retrain_policy must be YAML mappings")

    required_thresholds = {"feature_drift", "concept_drift_accuracy_drop"}
    missing_thresholds = sorted(required_thresholds - set(thresholds))
    missing_policy = sorted(_S14_POLICY_FIELDS - set(policy_raw))
    unknown_policy = sorted(set(policy_raw) - _S14_POLICY_FIELDS)
    if missing_thresholds:
        raise ValueError(
            "Drift policy config is missing thresholds: "
            + ", ".join(missing_thresholds)
        )
    if missing_policy:
        raise ValueError(
            "Drift policy config is missing retrain_policy fields: "
            + ", ".join(missing_policy)
        )
    if unknown_policy:
        raise ValueError(
            "Drift policy config has unknown retrain_policy fields: "
            + ", ".join(unknown_policy)
        )

    bool_fields = {
        "allow_auto_promotion",
        "require_registered_candidate_for_promotion",
    }
    for field_name in bool_fields:
        if type(policy_raw[field_name]) is not bool:
            raise ValueError(f"retrain_policy.{field_name} must be a boolean")

    try:
        policy = AutoRetrainPolicyConfig(
            moderate_feature_psi=float(thresholds["feature_drift"]),
            severe_feature_psi=float(policy_raw["severe_feature_psi"]),
            severe_drifted_share=float(policy_raw["severe_drifted_share"]),
            concept_drift_drop=float(thresholds["concept_drift_accuracy_drop"]),
            low_stability_score=float(policy_raw["low_stability_score"]),
            urgent_cadence_days=int(policy_raw["urgent_cadence_days"]),
            minimum_promotion_delta=float(policy_raw["minimum_promotion_delta"]),
            allow_auto_promotion=policy_raw["allow_auto_promotion"],
            require_registered_candidate_for_promotion=policy_raw[
                "require_registered_candidate_for_promotion"
            ],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Drift policy config contains an invalid value: {exc}") from exc

    if policy.moderate_feature_psi < 0:
        raise ValueError("thresholds.feature_drift must be non-negative")
    if policy.severe_feature_psi <= policy.moderate_feature_psi:
        raise ValueError(
            "retrain_policy.severe_feature_psi must exceed thresholds.feature_drift"
        )
    if not 0 <= policy.severe_drifted_share <= 1:
        raise ValueError("retrain_policy.severe_drifted_share must be between 0 and 1")
    if policy.concept_drift_drop < 0:
        raise ValueError("thresholds.concept_drift_accuracy_drop must be non-negative")
    if not 0 <= policy.low_stability_score <= 100:
        raise ValueError("retrain_policy.low_stability_score must be between 0 and 100")
    if policy.urgent_cadence_days <= 0:
        raise ValueError("retrain_policy.urgent_cadence_days must be positive")
    if policy.minimum_promotion_delta < 0:
        raise ValueError("retrain_policy.minimum_promotion_delta must be non-negative")

    policy_metadata = {
        "source": "drift_policy_config",
        "config_path": str(candidate),
        "config_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "effective": asdict(policy),
    }
    return policy, policy_metadata


def _load_config_metadata(config_name: str) -> dict[str, str]:
    configs_dir = Path(__file__).resolve().parents[2] / "configs"
    config_path = configs_dir / config_name
    if not config_path.exists() and not config_name.endswith(".yml"):
        config_path = configs_dir / f"{config_name}.yml"
    if not config_path.exists():
        return {"task_type": "unknown", "dataset_name": config_name.replace(".yml", "")}
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.warning("Config metadata failed to parse from %s: %s", config_path, exc)
        return {"task_type": "unknown", "dataset_name": config_name.replace(".yml", "")}
    dataset = config.get("dataset") or {}
    return {
        "task_type": str(config.get("task_type") or "unknown"),
        "dataset_name": str(dataset.get("name") or config_name.replace(".yml", "")),
    }


def _first_present(mapping: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = mapping.get(name)
        if value:
            return str(value)
    return None


def _collect_identity(
    *artifacts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Collect identity and expose conflicting immutable revision values."""
    identity: dict[str, Any] = {}
    revision_values: dict[str, set[str]] = {
        field: set() for field in _REVISION_IDENTITY_ALIASES
    }
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        nested = artifact.get("identity")
        if isinstance(nested, dict):
            for key, value in nested.items():
                if value not in (None, ""):
                    identity.setdefault(key, value)

        sources = [
            nested,
            artifact.get("lineage"),
            artifact.get("execution_manifest"),
            artifact,
        ]
        for source in sources:
            if not isinstance(source, dict):
                continue
            for field, aliases in _REVISION_IDENTITY_ALIASES.items():
                for alias in aliases:
                    value = source.get(alias)
                    if value not in (None, ""):
                        revision_values[field].add(str(value))

    conflicts: dict[str, list[str]] = {}
    for field, aliases in _REVISION_IDENTITY_ALIASES.items():
        for alias in aliases:
            identity.pop(alias, None)
        values = sorted(revision_values[field])
        if len(values) == 1:
            identity[field] = values[0]
        elif len(values) > 1:
            conflicts[field] = values
    return identity, conflicts


def _build_source_revision(
    identity: dict[str, Any],
    conflicts: dict[str, list[str]],
    final_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution_manifest = final_report.get("execution_manifest")
    validated_manifest: ExecutionManifest | None = None
    contract_error: str | None = None
    if isinstance(execution_manifest, dict):
        try:
            validated_manifest = ExecutionManifest.from_dict(execution_manifest)
        except (ContractValidationError, KeyError, TypeError, ValueError) as exc:
            contract_error = f"{type(exc).__name__}: {exc}"
    source_revision = {
        "schema_version": "1.0",
        "execution_id": (
            validated_manifest.execution_id
            if validated_manifest is not None
            else identity.get("execution_id")
        ),
        "config_hash": (
            validated_manifest.config_hash
            if validated_manifest is not None
            else identity.get("config_hash")
        ),
        "source_sha": (
            validated_manifest.code_sha
            if validated_manifest is not None
            else identity.get("source_sha")
        ),
    }
    missing = [
        field
        for field in ("execution_id", "config_hash", "source_sha")
        if not source_revision.get(field)
    ]
    if conflicts:
        status = "conflict"
    elif contract_error:
        status = "invalid"
    elif missing or validated_manifest is None:
        status = "incomplete"
    else:
        status = "verified"
    return source_revision, {
        "status": status,
        "missing_fields": missing,
        "conflicts": conflicts,
        "required_contract": "final_report.execution_manifest",
        "required_contract_present": validated_manifest is not None,
        "contract_error": contract_error,
    }


def build_retrain_decision_payload(
    *,
    config_name: str,
    drift_report: dict[str, Any],
    final_report: dict[str, Any] | None = None,
    registry_info: dict[str, Any] | None = None,
    candidate_baseline_path: str | Path | None = None,
    trigger: str = "pipeline_s14",
    schedule_name: str | None = None,
    policy: AutoRetrainPolicyConfig | None = None,
    policy_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], AutoRetrainDecisionRecord]:
    """Build the stage output payload and ledger-shaped decision record."""
    final_report = final_report or {}
    registry_info = registry_info or {}

    metadata = _load_config_metadata(config_name)
    normalized_report = dict(drift_report or {})
    normalized_report.setdefault("config_name", config_name)
    normalized_report.setdefault("task_type", metadata["task_type"])
    normalized_report.setdefault("dataset_name", metadata["dataset_name"])
    identity, identity_conflicts = _collect_identity(
        normalized_report,
        final_report,
        registry_info,
    )
    source_revision, revision_validation = _build_source_revision(
        identity,
        identity_conflicts,
        final_report,
    )

    effective_policy = policy or AutoRetrainPolicyConfig()
    decision = evaluate_auto_retrain_policy(
        normalized_report,
        final_report=final_report,
        registry_info=registry_info,
        policy=effective_policy,
    )
    decision_dict = decision.as_dict()
    if decision_dict["should_submit"] and revision_validation["status"] != "verified":
        if revision_validation["status"] == "conflict":
            revision_reason = (
                "Immutable source revision identity conflicts across upstream artifacts."
            )
        elif revision_validation["status"] == "invalid":
            revision_reason = (
                "The final_report ExecutionManifest is invalid and cannot authorize "
                "a retrain submission."
            )
        else:
            revision_reason = (
                "Immutable source revision is incomplete; execution_id, config_hash, "
                "and source_sha are required before retraining can be submitted."
            )
        decision_dict = {
            **decision_dict,
            "outcome": "blocked",
            "should_submit": False,
            "eligible_for_promotion": False,
            "reasons": [*decision_dict.get("reasons", []), revision_reason],
            "signals": {
                **decision_dict.get("signals", {}),
                "revision_validation_status": revision_validation["status"],
            },
        }

    comparison = normalized_report.get("comparison_drift") or {}
    baseline_metadata = comparison.get("baseline_metadata") or {}
    input_baseline_uri = _first_present(
        comparison,
        ("baseline_uri", "baseline_input_uri", "input_baseline_uri", "baseline_path"),
    )
    candidate_baseline = str(candidate_baseline_path) if candidate_baseline_path else None
    timestamp = datetime.now(timezone.utc).isoformat()

    payload = {
        "stage": STAGE_NAME,
        "stage_id": STAGE_ID,
        "timestamp_utc": timestamp,
        "config_name": str(normalized_report.get("config_name") or config_name),
        "task_type": str(normalized_report.get("task_type") or metadata["task_type"]),
        "dataset_name": str(normalized_report.get("dataset_name") or metadata["dataset_name"]),
        "identity": identity,
        "source_revision": source_revision,
        "revision_validation": revision_validation,
        "policy": policy_metadata or {
            "source": "code_default",
            "effective": asdict(effective_policy),
        },
        "decision": decision_dict,
        "candidate_baseline_path": candidate_baseline,
        "comparison": {
            "available": bool(comparison.get("available")),
            "baseline_status": comparison.get("baseline_status") or "not_available",
            "input_baseline_uri": input_baseline_uri,
            "baseline_metadata": baseline_metadata,
        },
        "source": {
            "drift_execution_id": identity.get("execution_id"),
            "azureml_run_id": os.getenv("AZUREML_RUN_ID") or os.getenv("MLFLOW_RUN_ID"),
            "trigger": trigger,
            "schedule_name": schedule_name,
        },
    }

    planned_schedules_table = build_planned_schedules_table(
        current_task_type=payload["task_type"],
        current_dataset_name=payload["dataset_name"],
        current_config_name=payload["config_name"],
        current_schedule_name=schedule_name,
        decision=decision_dict,
        input_baseline_uri=input_baseline_uri,
        promotion_status="manual_pending",
    )
    payload["planned_schedules_table"] = planned_schedules_table

    record = build_decision_record(
        config_name=payload["config_name"],
        task_type=payload["task_type"],
        dataset_name=payload["dataset_name"],
        decision=decision_dict,
        trigger=trigger,
        schedule_name=schedule_name,
        input_baseline_uri=input_baseline_uri,
        output_baseline_uri=None,
        candidate_job_name=os.getenv("AZUREML_RUN_ID") or os.getenv("MLFLOW_RUN_ID"),
        promotion_status="manual_pending",
        approved_for_future_baseline=False,
        metadata={
            "source": STAGE_NAME,
            "stage_id": STAGE_ID,
            "candidate_baseline_path": candidate_baseline,
            "comparison_available": payload["comparison"]["available"],
            "baseline_status": payload["comparison"]["baseline_status"],
            "recommended_days": decision_dict.get("signals", {}).get("recommended_days"),
            "stability_score": decision_dict.get("signals", {}).get("stability_score"),
            "planned_schedules_table": planned_schedules_table,
            "identity": identity,
            "source_revision": source_revision,
            "revision_validation": revision_validation,
            "policy": payload["policy"],
        },
    )
    payload["decision_id"] = record.decision_id
    payload["retrain_decision"] = {
        "contract_type": "RetrainDecision",
        "schema_version": "2.0",
        "decision_id": record.decision_id,
        "source_revision": source_revision,
        **decision_dict,
    }
    return payload, record


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _log_decision(payload: dict[str, Any]) -> None:
    decision = payload.get("decision") or {}
    signals = decision.get("signals") or {}
    try:
        mlflow.log_param("stage", STAGE_NAME)
        mlflow.log_param("auto_retrain_outcome", decision.get("outcome"))
        mlflow.log_param("auto_retrain_severity", decision.get("severity"))
        mlflow.log_param("auto_retrain_should_submit", bool(decision.get("should_submit")))
        mlflow.log_param("auto_retrain_eligible_for_promotion", bool(decision.get("eligible_for_promotion")))
        mlflow.log_metric("comparison_drift_available", 1 if payload.get("comparison", {}).get("available") else 0)
        for metric_name in ("max_feature_psi", "mean_feature_psi", "stability_score", "recommended_days"):
            value = signals.get(metric_name)
            if value is not None:
                mlflow.log_metric(metric_name, float(value))
        mlflow.log_dict(payload, "retrain_decision.json")
    except Exception as exc:  # noqa: BLE001 - telemetry must not fail the stage
        logger.warning("MLflow logging failed (non-fatal): %s", exc)


def run_retrain_decision(args: argparse.Namespace) -> int:
    _safe_disable_autolog()
    policy, policy_metadata = _load_retrain_policy(args.drift_policy_config)
    drift_report = _load_json_safe(args.drift_report, "drift_report")
    final_report = _load_json_safe(args.final_report, "final_report")
    registry_info = _load_json_safe(args.registry_info, "registry_info")

    if not drift_report:
        logger.warning("No drift report could be loaded; writing blocked decision")
        drift_report = {
            "config_name": args.config_name,
            "comparison_drift": {"available": False, "baseline_status": "missing_drift_report"},
            "feature_psi_scores": {},
            "stability_assessment": {},
        }

    payload, record = build_retrain_decision_payload(
        config_name=args.config_name,
        drift_report=drift_report,
        final_report=final_report,
        registry_info=registry_info,
        candidate_baseline_path=args.candidate_baseline,
        trigger=args.trigger,
        schedule_name=args.schedule_name,
        policy=policy,
        policy_metadata=policy_metadata,
    )

    _write_json(args.retrain_decision, payload)
    _write_json(args.decision_ledger_record, record.as_dict())
    _log_decision(payload)

    logger.info("s14 retrain decision outcome: %s", payload["decision"].get("outcome"))
    logger.info("s14 retrain decision severity: %s", payload["decision"].get("severity"))
    logger.info("s14 should submit candidate: %s", payload["decision"].get("should_submit"))
    logger.info("Retrain decision written to %s", args.retrain_decision)
    logger.info("Ledger-shaped decision record written to %s", args.decision_ledger_record)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="s14 Auto-Retrain Decision Gate")
    parser.add_argument("--config_name", required=True, help="Config YAML filename")
    parser.add_argument("--drift_report", required=True, help="s13 drift report JSON")
    parser.add_argument("--candidate_baseline", required=True, help="s13 drift baseline folder")
    parser.add_argument("--final_report", default=None, help="s10 final report JSON")
    parser.add_argument("--registry_info", default=None, help="s12 registry info JSON")
    parser.add_argument(
        "--drift_policy_config",
        required=True,
        help="YAML file containing thresholds and retrain_policy",
    )
    parser.add_argument("--trigger", default="pipeline_s14", help="Decision trigger label")
    parser.add_argument("--schedule_name", default=None, help="Schedule name when invoked by a schedule")
    parser.add_argument("--retrain_decision", required=True, help="Output retrain decision JSON")
    parser.add_argument("--decision_ledger_record", required=True, help="Output ledger-shaped decision JSON")
    args = parser.parse_args()

    raise SystemExit(run_retrain_decision(args))


if __name__ == "__main__":
    main()
