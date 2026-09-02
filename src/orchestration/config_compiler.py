"""Compile legacy and schema-v2 YAML into one explicit pipeline contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from jsonschema import ValidationError

from .contracts import (
    SCHEMA_VERSION,
    SUPPORTED_ENGINES,
    SUPPORTED_TASKS,
    canonical_hash,
)


ROUND1_MAX_VARIANTS_CAP = 40
ROUND2_MAX_VARIANTS_CAP = 8
CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS = 600
PHASE_B_TIMEOUT_CAP_SECONDS = 3 * 60 * 60
HPO_TRIALS_CAP = 50
HPO_TIMEOUT_CAP_SECONDS = 60 * 60


_ROOT_KEYS = {
    "schema_version",
    "experiment_name",
    "task_type",
    "preset",
    "random_seed",
    "holdout_fraction",
    "holdout_split_strategy",
    "holdout_time_column",
    "dataset",
    "azureml",
    "azure_ml",
    "split",
    "metrics",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "phases",
    "registry",
    "recipes",
}

_NESTED_KEYS = {
    "dataset": {
        "name",
        "version",
        "content_sha256",
        "target_column",
        "blob_path",
        "local_path",
        "datastore_name",
        "delimiter",
        "encoding",
        "id_columns",
        "excluded_columns",
        "group_column",
        "time_column",
    },
    "azureml": {
        "subscription_id",
        "resource_group",
        "workspace_name",
        "compute_target",
        "default_datastore",
        "environment",
        "environment_name_preprocessing",
        "environment_name_training",
    },
    "split": {
        "strategy",
        "validation_fraction",
        "test_fraction",
        "cv_folds",
        "group_column",
        "time_column",
        "locked_test",
    },
    "metrics": {
        "primary",
        "selection_protocol",
        "cv_folds",
        "locked_test_once",
        "min_comparable_candidates",
    },
    "registry": {
        "model_name",
        "min_quality",
        "quality_failure_policy",
        "block_on_quality_fail",
        "warning_registration_allowed",
        "warning_tags",
        "pass_aliases",
        "warning_aliases",
    },
}

_PHASE_KEYS = {
    "phase_a_baseline",
    "phase_b_recipes",
    "phase_b",
    "phase_c_hpo",
}

_PHASE_A_KEYS = {
    "engines",
    "cv_folds",
    "candidate_engine_timeout_seconds",
    "flaml_config",
}

_FLAML_CONFIG_KEYS = {
    "time_budget",
}

_PHASE_B_KEYS = {
    "enable_profiling",
    "profiling_output_path",
    "library_dir",
    "library",
    "tier",
    "max_variants",
    "max_recipes",
    "selection_strategy",
    "min_relevance_score",
    "diversity_boost",
    "runtime_budget_sec",
    "time_budget_per_variant",
    "phase_timeout_seconds",
    "safety_net_review_required",
    "random_selection",
    "engines",
    "imputation_preset",
    "planner",
}

_PLANNER_KEYS = {
    "enabled",
    "round1_max_variants",
    "round2_max_variants",
    "proxy_prune_threshold",
    "diversity_min_hamming_distance",
    "cache_enabled",
}

_HPO_KEYS = {
    "optimizer",
    "n_trials",
    "timeout",
    "timeout_seconds",
    "compute_rate_usd_per_hour",
}

_STAGE_KEYS = {
    "stage1": {
        "min_rows",
        "max_missing_pct",
        "classification_min_samples_per_class",
        "regression_target_min_variance",
        "generate_sweetviz",
        "eda_sample_size",
    },
    "stage2": {
        "imputation_numeric",
        "imputation_categorical",
        "imputation_strategy",
        "statistical_tests_enabled",
        "high_cardinality_max",
        "protected_columns",
    },
    "stage3": {
        "encoding",
        "scaling",
        "imbalance_handling",
        "adaptive_scaling",
        "multicollinearity_check",
    },
    "stage4": {
        "feature_selection",
        "selection_method",
        "max_features",
        "apply_pca_threshold",
        "pca_variance_retained",
        "imbalance_detection",
    },
    "stage5": {
        "model_universe",
        "flaml_time_budget",
        "pycaret_fold",
        "pycaret_n_select",
    },
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise ValidationError(f"Unknown fields at {path}: {', '.join(unknown)}")


def _reject_unknown_fields(raw: Mapping[str, Any]) -> None:
    _reject_unknown(raw, _ROOT_KEYS, "<root>")
    for section, allowed in _NESTED_KEYS.items():
        current = raw.get(section)
        if current is not None:
            _reject_unknown(_mapping(current, section), allowed, section)
    azure_alias = raw.get("azure_ml")
    if azure_alias is not None:
        _reject_unknown(
            _mapping(azure_alias, "azure_ml"), _NESTED_KEYS["azureml"], "azure_ml"
        )
    for section, allowed in _STAGE_KEYS.items():
        current = raw.get(section)
        if current is not None:
            _reject_unknown(_mapping(current, section), allowed, section)

    phases = _mapping(raw.get("phases"), "phases")
    _reject_unknown(phases, _PHASE_KEYS, "phases")
    if "phase_a_baseline" in phases:
        phase_a = _mapping(
            phases["phase_a_baseline"], "phases.phase_a_baseline"
        )
        _reject_unknown(
            phase_a, _PHASE_A_KEYS, "phases.phase_a_baseline"
        )
        if "flaml_config" in phase_a:
            _reject_unknown(
                _mapping(
                    phase_a["flaml_config"],
                    "phases.phase_a_baseline.flaml_config",
                ),
                _FLAML_CONFIG_KEYS,
                "phases.phase_a_baseline.flaml_config",
            )
    for name in ("phase_b", "phase_b_recipes"):
        if name in phases:
            phase = _mapping(phases[name], f"phases.{name}")
            _reject_unknown(phase, _PHASE_B_KEYS, f"phases.{name}")
            if "planner" in phase:
                _reject_unknown(
                    _mapping(phase["planner"], f"phases.{name}.planner"),
                    _PLANNER_KEYS,
                    f"phases.{name}.planner",
                )
    if "phase_c_hpo" in phases:
        _reject_unknown(
            _mapping(phases["phase_c_hpo"], "phases.phase_c_hpo"),
            _HPO_KEYS,
            "phases.phase_c_hpo",
        )


def _merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(defaults))
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _default_primary_metric(task_type: str) -> str:
    return {
        "classification": "balanced_accuracy",
        "regression": "r2",
        "clustering": "silhouette",
    }[task_type]


def _default_engines(task_type: str) -> list[str]:
    return ["pycaret"] if task_type == "clustering" else ["pycaret", "flaml"]


def _validate_caps(config: Mapping[str, Any]) -> None:
    task_type = str(config["task_type"])
    if task_type not in SUPPORTED_TASKS:
        raise ValidationError(
            "task_type must be classification, regression, or clustering"
        )
    phase_b = config["phases"]["phase_b"]
    planner = phase_b["planner"]
    engines = list(phase_b["engines"])
    if not engines or len(engines) != len(set(engines)):
        raise ValidationError("phases.phase_b.engines must be non-empty and unique")
    unknown_engines = sorted(set(engines).difference(SUPPORTED_ENGINES))
    if unknown_engines:
        raise ValidationError(
            f"Unsupported Phase B engines: {', '.join(unknown_engines)}"
        )
    if task_type == "clustering" and engines != ["pycaret"]:
        raise ValidationError("Clustering must use PyCaret only")

    checks = (
        (
            "phases.phase_b.planner.round1_max_variants",
            int(planner["round1_max_variants"]),
            ROUND1_MAX_VARIANTS_CAP,
        ),
        (
            "phases.phase_b.planner.round2_max_variants",
            int(planner["round2_max_variants"]),
            ROUND2_MAX_VARIANTS_CAP,
        ),
        (
            "phases.phase_b.time_budget_per_variant",
            int(phase_b["time_budget_per_variant"]),
            CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS,
        ),
        (
            "phases.phase_b.phase_timeout_seconds",
            int(phase_b["phase_timeout_seconds"]),
            PHASE_B_TIMEOUT_CAP_SECONDS,
        ),
        (
            "phases.phase_c_hpo.n_trials",
            int(config["phases"]["phase_c_hpo"]["n_trials"]),
            HPO_TRIALS_CAP,
        ),
        (
            "phases.phase_c_hpo.timeout_seconds",
            int(config["phases"]["phase_c_hpo"]["timeout_seconds"]),
            HPO_TIMEOUT_CAP_SECONDS,
        ),
    )
    for name, value, cap in checks:
        if value < 1 or value > cap:
            raise ValidationError(f"{name} must be between 1 and {cap}")
    if int(planner["round2_max_variants"]) > int(planner["round1_max_variants"]):
        raise ValidationError(
            "Round 2 maximum cannot exceed the Round 1 maximum"
        )
    if int(phase_b["max_variants"]) > int(planner["round1_max_variants"]):
        raise ValidationError(
            "phases.phase_b.max_variants cannot exceed Round 1 maximum"
        )


def compile_config(
    raw_config: Mapping[str, Any],
    *,
    source_name: str = "",
) -> dict[str, Any]:
    """Return a strict schema-v2 config with every runtime default materialized."""

    if not isinstance(raw_config, Mapping):
        raise ValidationError("Pipeline configuration must be a mapping")
    raw = deepcopy(dict(raw_config))
    _reject_unknown_fields(raw)
    declared_version = raw.get("schema_version")
    if declared_version is not None and str(declared_version) not in {
        "1.0",
        SCHEMA_VERSION,
    }:
        raise ValidationError(
            "schema_version must be omitted for legacy configs or be one of "
            f"'1.0', {SCHEMA_VERSION!r}"
        )
    is_legacy = str(declared_version or "1.0") == "1.0"
    migration_rewrites: list[dict[str, Any]] = []

    def record_migration(
        path: str,
        before: Any,
        after: Any,
        reason: str,
    ) -> None:
        if before != after:
            migration_rewrites.append(
                {
                    "path": path,
                    "from": deepcopy(before),
                    "to": deepcopy(after),
                    "reason": reason,
                }
            )

    task_type = str(raw.get("task_type") or "").strip()
    if task_type not in SUPPORTED_TASKS:
        raise ValidationError(
            "task_type must be classification, regression, or clustering"
        )

    dataset_raw = _mapping(raw.get("dataset"), "dataset")
    dataset_version = str(dataset_raw.get("version") or "").strip()
    if not dataset_version and not is_legacy:
        raise ValidationError(
            "dataset.version is required for schema-v2 submissions; "
            "unversioned data cannot produce reproducible candidate identity"
        )
    content_sha256 = str(dataset_raw.get("content_sha256") or "").strip().lower()
    if content_sha256 and (
        len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
    ):
        raise ValidationError(
            "dataset.content_sha256 must be a 64-character lowercase hexadecimal digest"
        )
    azure_raw = _mapping(
        raw.get("azureml") or raw.get("azure_ml"), "azureml"
    )
    phases_raw = _mapping(raw.get("phases"), "phases")
    phase_a_raw = _mapping(
        phases_raw.get("phase_a_baseline"), "phases.phase_a_baseline"
    )
    phase_a_flaml_raw = _mapping(
        phase_a_raw.get("flaml_config"),
        "phases.phase_a_baseline.flaml_config",
    )
    phase_b_legacy = _mapping(
        phases_raw.get("phase_b_recipes"), "phases.phase_b_recipes"
    )
    phase_b_raw = _mapping(phases_raw.get("phase_b"), "phases.phase_b")
    planner_raw = _mapping(phase_b_raw.get("planner"), "phases.phase_b.planner")
    hpo_raw = _mapping(phases_raw.get("phase_c_hpo"), "phases.phase_c_hpo")
    split_raw = _mapping(raw.get("split"), "split")
    metrics_raw = _mapping(raw.get("metrics"), "metrics")
    registry_raw = _mapping(raw.get("registry"), "registry")

    random_seed = int(raw.get("random_seed", 42))
    test_fraction = float(
        split_raw.get("test_fraction", raw.get("holdout_fraction", 0.2))
    )
    split_strategy = str(
        split_raw.get(
            "strategy",
            raw.get(
                "holdout_split_strategy",
                "stratified" if task_type == "classification" else "random",
            ),
        )
    )
    if task_type != "classification" and split_strategy == "stratified":
        if not is_legacy:
            raise ValidationError(
                "split.strategy='stratified' is valid only for classification"
            )
        record_migration(
            "split.strategy",
            split_strategy,
            "random",
            "stratification requires classification labels",
        )
        split_strategy = "random"
    if split_strategy in {"group", "time", "preassigned"}:
        raise ValidationError(
            f"split.strategy={split_strategy!r} is not implemented end to end; "
            "use random/stratified or add the corresponding fold-local evaluator"
        )
    requested_validation_fraction = float(
        split_raw.get("validation_fraction", 0.0)
    )
    if requested_validation_fraction != 0.0:
        if not is_legacy:
            raise ValidationError(
                "split.validation_fraction must be 0.0 because candidate "
                "selection uses cross-validation and no dedicated validation "
                "partition is produced"
            )
        record_migration(
            "split.validation_fraction",
            requested_validation_fraction,
            0.0,
            "candidate selection uses cross-validation without a validation partition",
        )
    validation_fraction = 0.0
    cv_folds = int(split_raw.get("cv_folds", metrics_raw.get("cv_folds", 5)))
    phase_a_cv_folds = int(phase_a_raw.get("cv_folds", cv_folds))
    phase_a_candidate_timeout = int(
        phase_a_raw.get("candidate_engine_timeout_seconds", 600)
    )
    phase_a_flaml_timeout = int(
        phase_a_flaml_raw.get("time_budget", phase_a_candidate_timeout)
    )
    if phase_a_cv_folds < 2:
        raise ValidationError(
            "phases.phase_a_baseline.cv_folds must be at least 2"
        )
    for name, value in (
        (
            "phases.phase_a_baseline.candidate_engine_timeout_seconds",
            phase_a_candidate_timeout,
        ),
        (
            "phases.phase_a_baseline.flaml_config.time_budget",
            phase_a_flaml_timeout,
        ),
    ):
        if value < 1 or value > CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS:
            raise ValidationError(
                f"{name} must be between 1 and "
                f"{CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS}"
            )

    engines = list(
        phase_b_raw.get(
            "engines",
            phase_b_legacy.get("engines", _default_engines(task_type)),
        )
    )
    if task_type == "clustering" and engines != ["pycaret"]:
        if not is_legacy:
            raise ValidationError("Clustering must use PyCaret only")
        record_migration(
            "phases.phase_b.engines",
            engines,
            ["pycaret"],
            "FLAML does not support the clustering product contract",
        )
        engines = ["pycaret"]

    phase_a_engines = list(
        phase_a_raw.get("engines", _default_engines(task_type))
    )
    if not phase_a_engines or len(phase_a_engines) != len(set(phase_a_engines)):
        raise ValidationError(
            "phases.phase_a_baseline.engines must be non-empty and unique"
        )
    unsupported_phase_a_engines = sorted(
        set(phase_a_engines).difference(SUPPORTED_ENGINES)
    )
    if unsupported_phase_a_engines:
        raise ValidationError(
            "Unsupported Phase A engines: "
            + ", ".join(unsupported_phase_a_engines)
        )
    if task_type == "clustering" and phase_a_engines != ["pycaret"]:
        if not is_legacy:
            raise ValidationError("Clustering Phase A must use PyCaret only")
        record_migration(
            "phases.phase_a_baseline.engines",
            phase_a_engines,
            ["pycaret"],
            "FLAML does not support the clustering product contract",
        )
        phase_a_engines = ["pycaret"]

    round1_max = int(planner_raw.get("round1_max_variants", 40))
    round2_max = int(planner_raw.get("round2_max_variants", 8))
    if is_legacy:
        # Migration is explicit and safe: historical configs used larger
        # defaults, but a compiled v2 artifact can never exceed approved caps.
        migrated_round1 = min(round1_max, ROUND1_MAX_VARIANTS_CAP)
        migrated_round2 = min(round2_max, ROUND2_MAX_VARIANTS_CAP)
        record_migration(
            "phases.phase_b.planner.round1_max_variants",
            round1_max,
            migrated_round1,
            "legacy value exceeded the approved Round 1 cap",
        )
        record_migration(
            "phases.phase_b.planner.round2_max_variants",
            round2_max,
            migrated_round2,
            "legacy value exceeded the approved Round 2 cap",
        )
        round1_max = migrated_round1
        round2_max = migrated_round2
    configured_max = int(
        phase_b_raw.get(
            "max_variants",
            phase_b_legacy.get("max_recipes", round1_max),
        )
    )
    if is_legacy:
        max_variants = min(configured_max, round1_max)
        record_migration(
            "phases.phase_b.max_variants",
            configured_max,
            max_variants,
            "legacy value exceeded the effective Round 1 cap",
        )
    else:
        max_variants = configured_max

    requested_profiling = bool(phase_b_raw.get("enable_profiling", True))
    if not requested_profiling:
        if not is_legacy:
            raise ValidationError(
                "phases.phase_b.enable_profiling=false is unsupported; "
                "Azure-resolved profiling is required for deterministic ranking"
            )
        record_migration(
            "phases.phase_b.enable_profiling",
            False,
            True,
            "Azure-resolved profiling is required for deterministic ranking",
        )

    requested_selection = str(
        phase_b_raw.get("selection_strategy", "scored")
    )
    if requested_selection not in {"scored", "profile_scored"}:
        if not is_legacy:
            raise ValidationError(
                "phases.phase_b.selection_strategy must be 'scored'; "
                "alphabetical and random selection are not production-safe"
            )
        record_migration(
            "phases.phase_b.selection_strategy",
            requested_selection,
            "profile_scored",
            "production selection requires Azure-resolved profile scoring",
        )

    requested_planner = bool(planner_raw.get("enabled", True))
    if not requested_planner:
        if not is_legacy:
            raise ValidationError(
                "phases.phase_b.planner.enabled=false is unsupported; "
                "the bounded Round 0/1/2 funnel is mandatory"
            )
        record_migration(
            "phases.phase_b.planner.enabled",
            False,
            True,
            "the bounded Round 0/1/2 funnel is mandatory",
        )

    retired_selection_controls = {
        "min_relevance_score": (
            "S06 ranks every feasible recipe with data-aware relevance evidence"
        ),
        "diversity_boost": (
            "S06 enforces deterministic diversity through planner.diversity_min_hamming_distance"
        ),
        "imputation_preset": (
            "recipe capabilities are screened inside the canonical S06 funnel"
        ),
    }
    for field_name, reason in retired_selection_controls.items():
        if field_name not in phase_b_raw:
            continue
        if not is_legacy:
            raise ValidationError(
                f"phases.phase_b.{field_name} is unsupported; {reason}"
            )
        record_migration(
            f"phases.phase_b.{field_name}",
            phase_b_raw[field_name],
            "removed",
            reason,
        )

    phase_b = {
        "enable_profiling": True,
        "profiling_output_path": str(
            phase_b_raw.get(
                "profiling_output_path", "outputs/dataset_profile.json"
            )
        ),
        "library_dir": str(
            phase_b_raw.get(
                "library_dir",
                f"configs/recipes/{task_type}/variant_search",
            )
        ),
        "library": str(
            phase_b_raw.get(
                "library", phase_b_legacy.get("library", "variant_search")
            )
        ),
        "tier": str(
            phase_b_raw.get(
                "tier", phase_b_legacy.get("tier", "progressive")
            )
        ),
        "max_variants": max_variants,
        "selection_strategy": "profile_scored",
        "runtime_budget_sec": int(
            phase_b_raw.get(
                "runtime_budget_sec",
                phase_b_legacy.get("runtime_budget_sec", 600),
            )
        ),
        "time_budget_per_variant": int(
            phase_b_raw.get("time_budget_per_variant", 600)
        ),
        "phase_timeout_seconds": int(
            phase_b_raw.get("phase_timeout_seconds", 10800)
        ),
        "safety_net_review_required": bool(
            phase_b_raw.get(
                "safety_net_review_required",
                phase_b_legacy.get("safety_net_review_required", True),
            )
        ),
        "engines": engines,
        "planner": {
            "enabled": True,
            "round1_max_variants": round1_max,
            "round2_max_variants": round2_max,
            "proxy_prune_threshold": float(
                planner_raw.get("proxy_prune_threshold", 0.5)
            ),
            "diversity_min_hamming_distance": int(
                planner_raw.get("diversity_min_hamming_distance", 2)
            ),
            "cache_enabled": bool(planner_raw.get("cache_enabled", True)),
        },
    }
    if is_legacy:
        migrated_candidate_timeout = min(
            phase_b["time_budget_per_variant"],
            CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS,
        )
        migrated_phase_timeout = min(
            phase_b["phase_timeout_seconds"], PHASE_B_TIMEOUT_CAP_SECONDS
        )
        record_migration(
            "phases.phase_b.time_budget_per_variant",
            phase_b["time_budget_per_variant"],
            migrated_candidate_timeout,
            "legacy value exceeded the candidate-engine timeout cap",
        )
        record_migration(
            "phases.phase_b.phase_timeout_seconds",
            phase_b["phase_timeout_seconds"],
            migrated_phase_timeout,
            "legacy value exceeded the Phase B timeout cap",
        )
        phase_b["time_budget_per_variant"] = migrated_candidate_timeout
        phase_b["phase_timeout_seconds"] = migrated_phase_timeout

    requested_hpo_trials = int(hpo_raw.get("n_trials", 50))
    requested_hpo_timeout = int(
        hpo_raw.get("timeout_seconds", hpo_raw.get("timeout", 3600))
    )
    hpo_trials = (
        min(requested_hpo_trials, HPO_TRIALS_CAP)
        if is_legacy
        else requested_hpo_trials
    )
    hpo_timeout = (
        min(requested_hpo_timeout, HPO_TIMEOUT_CAP_SECONDS)
        if is_legacy
        else requested_hpo_timeout
    )
    if is_legacy:
        record_migration(
            "phases.phase_c_hpo.n_trials",
            requested_hpo_trials,
            hpo_trials,
            "legacy value exceeded the approved HPO trial cap",
        )
        record_migration(
            "phases.phase_c_hpo.timeout_seconds",
            requested_hpo_timeout,
            hpo_timeout,
            "legacy value exceeded the approved HPO timeout cap",
        )

    default_min_quality = {
        "classification": 0.5,
        "regression": 0.0,
        "clustering": 0.0,
    }
    quality_policy = str(
        registry_raw.get(
            "quality_failure_policy",
            "block" if registry_raw.get("block_on_quality_fail") else "warn",
        )
    )

    compiled: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "migration": {
            "source_schema_version": (
                str(declared_version) if declared_version is not None else "unversioned"
            ),
            "applied": bool(is_legacy),
            "rewrites": migration_rewrites,
        },
        "experiment_name": str(raw.get("experiment_name") or "").strip(),
        "task_type": task_type,
        "preset": str(raw.get("preset", "production")),
        "random_seed": random_seed,
        "holdout_fraction": test_fraction,
        "holdout_split_strategy": split_strategy,
        "holdout_time_column": split_raw.get(
            "time_column", raw.get("holdout_time_column")
        ),
        "dataset": {
            "name": str(dataset_raw.get("name") or "").strip(),
            "version": dataset_version or "unversioned",
            "content_sha256": content_sha256 or None,
            "target_column": dataset_raw.get("target_column"),
            "blob_path": str(dataset_raw.get("blob_path", "")),
            "local_path": str(dataset_raw.get("local_path", "")),
            "datastore_name": str(
                dataset_raw.get("datastore_name", "mlops_blob")
            ),
            "delimiter": str(dataset_raw.get("delimiter", ",")),
            "encoding": str(dataset_raw.get("encoding", "utf-8")),
            "id_columns": list(dataset_raw.get("id_columns") or []),
            "excluded_columns": list(dataset_raw.get("excluded_columns") or []),
            "group_column": dataset_raw.get("group_column"),
            "time_column": dataset_raw.get("time_column"),
        },
        "azureml": {
            "subscription_id": str(azure_raw.get("subscription_id", "")),
            "resource_group": str(azure_raw.get("resource_group", "")),
            "workspace_name": str(azure_raw.get("workspace_name", "")),
            "compute_target": str(azure_raw.get("compute_target", "")),
            "default_datastore": str(
                azure_raw.get(
                    "default_datastore",
                    dataset_raw.get("datastore_name", "mlops_blob"),
                )
            ),
            "environment": str(
                azure_raw.get("environment", "mlops-v3-unified:32")
            ),
            "environment_name_preprocessing": str(
                azure_raw.get(
                    "environment_name_preprocessing",
                    "mlops-v3-preprocessing",
                )
            ),
            "environment_name_training": str(
                azure_raw.get(
                    "environment_name_training", "mlops-v3-training"
                )
            ),
        },
        "split": {
            "strategy": split_strategy,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "cv_folds": cv_folds,
            "group_column": split_raw.get(
                "group_column", dataset_raw.get("group_column")
            ),
            "time_column": split_raw.get(
                "time_column", dataset_raw.get("time_column")
            ),
            "locked_test": True,
        },
        "metrics": {
            "primary": str(
                metrics_raw.get("primary", _default_primary_metric(task_type))
            ),
            "selection_protocol": "cross_validation",
            "cv_folds": cv_folds,
            "locked_test_once": True,
            "min_comparable_candidates": int(
                metrics_raw.get("min_comparable_candidates", 2)
            ),
        },
        "stage1": _merge(
            {
                "min_rows": 100,
                "max_missing_pct": 50.0,
                "classification_min_samples_per_class": 2,
                "regression_target_min_variance": 0.0,
                "generate_sweetviz": False,
                "eda_sample_size": 10000,
            },
            _mapping(raw.get("stage1"), "stage1"),
        ),
        "stage2": _merge(
            {
                "imputation_numeric": "median",
                "imputation_categorical": "most_frequent",
                "imputation_strategy": "from_stage1",
                "statistical_tests_enabled": True,
                "high_cardinality_max": 200,
                "protected_columns": [],
            },
            _mapping(raw.get("stage2"), "stage2"),
        ),
        "stage3": _merge(
            {
                "encoding": "onehot",
                "scaling": "adaptive",
                "imbalance_handling": "none",
                "adaptive_scaling": True,
                "multicollinearity_check": True,
            },
            _mapping(raw.get("stage3"), "stage3"),
        ),
        "stage4": _merge(
            {
                "feature_selection": "none",
                "selection_method": "none",
                "max_features": 100,
                "apply_pca_threshold": 100,
                "pca_variance_retained": 0.95,
                "imbalance_detection": task_type == "classification",
            },
            _mapping(raw.get("stage4"), "stage4"),
        ),
        "stage5": _merge(
            {
                "model_universe": [],
                "flaml_time_budget": phase_a_flaml_timeout,
                "pycaret_fold": phase_a_cv_folds,
                "pycaret_n_select": 1,
            },
            _mapping(raw.get("stage5"), "stage5"),
        ),
        "phases": {
            "phase_a_baseline": {
                "engines": phase_a_engines,
                "cv_folds": phase_a_cv_folds,
                "candidate_engine_timeout_seconds": phase_a_candidate_timeout,
                "flaml_config": {"time_budget": phase_a_flaml_timeout},
            },
            "phase_b_recipes": {
                "library": phase_b["library"],
                "tier": phase_b["tier"],
                "max_recipes": max_variants,
                "runtime_budget_sec": phase_b["runtime_budget_sec"],
                "random_selection": False,
                "safety_net_review_required": phase_b[
                    "safety_net_review_required"
                ],
                "engines": engines,
            },
            "phase_b": phase_b,
            "phase_c_hpo": {
                "optimizer": str(hpo_raw.get("optimizer", "optuna")),
                "n_trials": hpo_trials,
                "timeout_seconds": hpo_timeout,
                "compute_rate_usd_per_hour": float(
                    hpo_raw.get("compute_rate_usd_per_hour", 0.0)
                ),
            },
        },
        "registry": {
            "model_name": str(
                registry_raw.get(
                    "model_name",
                    f"{dataset_raw.get('name', 'model')}-{task_type}",
                )
            ),
            "min_quality": _merge(
                default_min_quality,
                _mapping(registry_raw.get("min_quality"), "registry.min_quality"),
            ),
            "quality_failure_policy": quality_policy,
            "warning_registration_allowed": bool(
                registry_raw.get("warning_registration_allowed", True)
            ),
            "warning_tags": _merge(
                {"quality_decision": "warn"},
                _mapping(registry_raw.get("warning_tags"), "registry.warning_tags"),
            ),
            "pass_aliases": list(
                registry_raw["pass_aliases"]
                if "pass_aliases" in registry_raw
                else ["champion"]
            ),
            "warning_aliases": list(
                registry_raw.get("warning_aliases") or []
            ),
            # Compatibility for current S10/S12 until they consume QualityDecision.
            "block_on_quality_fail": quality_policy == "block",
        },
        "recipes": deepcopy(list(raw.get("recipes") or [])),
        "source_config": source_name,
    }

    if not compiled["experiment_name"]:
        raise ValidationError("experiment_name must be a non-empty string")
    if not compiled["dataset"]["name"]:
        raise ValidationError("dataset.name must be a non-empty string")
    if task_type in {"classification", "regression"} and not compiled["dataset"][
        "target_column"
    ]:
        raise ValidationError(
            f"dataset.target_column is required for task_type={task_type!r}"
        )
    if task_type == "clustering":
        compiled["dataset"]["target_column"] = None
    if quality_policy not in {"warn", "block"}:
        raise ValidationError(
            "registry.quality_failure_policy must be 'warn' or 'block'"
        )
    warning_aliases = {
        str(alias).lower() for alias in compiled["registry"]["warning_aliases"]
    }
    if warning_aliases.intersection({"champion", "production"}):
        raise ValidationError(
            "Warning-quality models cannot receive champion/production aliases"
        )
    _validate_caps(compiled)

    # Validate the fully materialized artifact against the public schema.
    from .config_schema import validate_compiled_config

    validate_compiled_config(compiled)
    compiled["compiled_config_hash"] = canonical_hash(compiled)
    return compiled
