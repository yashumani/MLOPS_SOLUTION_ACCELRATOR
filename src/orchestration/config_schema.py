from typing import Any, Dict, Mapping

from jsonschema import validate, ValidationError

CONFIG_SCHEMA_LEGACY: Dict[str, Any] = {
    "type": "object",
    "required": ["experiment_name", "dataset", "task_type", "preset"],
    "properties": {
        "experiment_name": {"type": "string"},
        "task_type": {"type": "string", "enum": ["classification", "regression", "clustering", "forecasting"]},
        "preset": {"type": "string", "enum": ["diagnostic", "production"]},
        "holdout_fraction": {"type": "number", "minimum": 0.05, "maximum": 0.5},
        "random_seed": {"type": "integer", "minimum": 0},
        "dataset": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "target_column": {"type": "string"},
                "blob_path": {"type": "string"},
                "local_path": {"type": "string"},
                "datastore_name": {"type": "string"},
                "delimiter": {"type": "string", "maxLength": 1},
                "excluded_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            }
        },
        "azureml": {
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string"},
                "resource_group": {"type": "string"},
                "workspace_name": {"type": "string"},
                "compute_target": {"type": "string"},
                "environment_name_preprocessing": {"type": "string"},
                "environment_name_training": {"type": "string"}
            }
        },
        "stage1": {
            "type": "object",
            "properties": {
                "min_rows": {"type": "integer", "minimum": 1},
                "max_missing_pct": {"type": "number", "minimum": 0, "maximum": 100},
                "classification_min_samples_per_class": {"type": "integer", "minimum": 1},
                "regression_target_min_variance": {"type": "number", "minimum": 0},
                "generate_sweetviz": {"type": "boolean"},
                "eda_sample_size": {"type": "integer", "minimum": 1}
            }
        },
        "stage2": {
            "type": "object",
            "properties": {
                "imputation_numeric": {
                    "type": "string",
                    "enum": ["median", "knn", "iterative", "ffill", "group_median"]
                },
                "imputation_categorical": {
                    "type": "string",
                    "enum": ["most_frequent", "constant"]
                },
                "imputation_strategy": {"type": "string"},
                "statistical_tests_enabled": {"type": "boolean"},
                "high_cardinality_max": {"type": "integer", "minimum": 1},
                "protected_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": []
                }
            }
        },
        "stage3": {
            "type": "object",
            "properties": {
                "encoding": {
                    "type": "string",
                    "enum": ["onehot", "label", "target"]
                },
                "scaling": {
                    "type": "string",
                    "enum": ["standard", "robust", "quantile", "yeo_johnson", "adaptive"]
                },
                "imbalance_handling": {
                    "type": "string",
                    "enum": ["none", "smote", "adasyn"]
                },
                "adaptive_scaling": {"type": "boolean"},
                "multicollinearity_check": {"type": "boolean"}
            }
        },
        "stage4": {
            "type": "object",
            "properties": {
                "feature_selection": {
                    "type": "string",
                    "enum": ["none", "variance", "mutual_info", "boruta"]
                },
                "selection_method": {
                    "type": "string",
                    "enum": ["none", "variance", "mutual_info", "boruta"]
                },
                "max_features": {"type": "integer", "minimum": 1},
                "apply_pca_threshold": {"type": "integer", "minimum": 1},
                "pca_variance_retained": {"type": "number", "minimum": 0, "maximum": 1},
                "imbalance_detection": {"type": "boolean"}
            }
        },
        "stage5": {
            "type": "object",
            "properties": {
                "model_universe": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "flaml_time_budget": {"type": "integer", "minimum": 10},
                "pycaret_fold": {"type": "integer", "minimum": 2},
                "pycaret_n_select": {"type": "integer", "minimum": 1}
            }
        },
        "phases": {
            "type": "object",
            "properties": {
                "phase_a_baseline": {
                    "type": "object",
                    "properties": {
                        "engines": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["pycaret", "flaml"]}
                        },
                        "cv_folds": {"type": "integer", "minimum": 2},
                        "candidate_engine_timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 600
                        },
                        "flaml_config": {
                            "type": "object",
                            "properties": {
                                "time_budget": {"type": "integer", "minimum": 10}
                            }
                        }
                    }
                },
                "phase_b_recipes": {
                    "type": "object",
                    "properties": {
                        "max_variants": {"type": "integer", "minimum": 1},
                        "max_recipes": {"type": "integer", "minimum": 1},
                        "library": {"type": "string"},
                        "tier": {"type": "string"},
                        "runtime_budget_sec": {"type": "integer", "minimum": 1},
                        "random_selection": {"type": "boolean"},
                        "safety_net_review_required": {"type": "boolean"},
                        "engines": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["pycaret", "flaml"]}
                        }
                    }
                },
                "phase_b": {
                    "type": "object",
                    "properties": {
                        "enable_profiling": {"type": "boolean"},
                        "profiling_output_path": {"type": "string"},
                        "library_dir": {"type": "string"},
                        "max_variants": {"type": "integer", "minimum": 1},
                        "selection_strategy": {"type": "string", "enum": ["scored", "alphabetical", "random_seeded"]},
                        "min_relevance_score": {"type": "number", "minimum": 0, "maximum": 100},
                        "diversity_boost": {"type": "boolean"},
                        "runtime_budget_sec": {"type": "integer", "minimum": 1},
                        "time_budget_per_variant": {"type": "integer", "minimum": 1},
                        "safety_net_review_required": {"type": "boolean"},
                        "engines": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["pycaret", "flaml"]}
                        },
                        "imputation_preset": {"type": "string"},
                        "planner": {"type": "object"}
                    }
                },
                "phase_c_hpo": {
                    "type": "object",
                    "properties": {
                        "optimizer": {"type": "string", "enum": ["optuna"]},
                        "n_trials": {"type": "integer", "minimum": 1},
                        "timeout": {"type": "integer", "minimum": 10},
                        "timeout_seconds": {"type": "integer", "minimum": 10},
                        "compute_rate_usd_per_hour": {"type": "number", "minimum": 0}
                    }
                }
            }
        },
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"}
                }
            }
        }
    }
}


def _strict_object(
    properties: Dict[str, Any],
    *,
    required: tuple[str, ...] = (),
) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": properties,
    }


_string = {"type": "string"}
_non_empty_string = {"type": "string", "minLength": 1}
_nullable_string = {"type": ["string", "null"]}
_positive_integer = {"type": "integer", "minimum": 1}
_non_negative_integer = {"type": "integer", "minimum": 0}
_probability = {"type": "number", "minimum": 0.0, "maximum": 1.0}


CONFIG_SCHEMA_V2: Dict[str, Any] = _strict_object(
    {
        "schema_version": {"const": "2.0"},
        "migration": _strict_object(
            {
                "source_schema_version": _non_empty_string,
                "applied": {"type": "boolean"},
                "rewrites": {
                    "type": "array",
                    "items": _strict_object(
                        {
                            "path": _non_empty_string,
                            "from": {},
                            "to": {},
                            "reason": _non_empty_string,
                        },
                        required=("path", "from", "to", "reason"),
                    ),
                },
            },
            required=("source_schema_version", "applied", "rewrites"),
        ),
        "experiment_name": _non_empty_string,
        "task_type": {
            "type": "string",
            "enum": ["classification", "regression", "clustering"],
        },
        "preset": {"type": "string", "enum": ["diagnostic", "production"]},
        "random_seed": _non_negative_integer,
        # Compatibility fields remain explicit until every existing step consumes
        # the typed SplitManifest directly.
        "holdout_fraction": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "exclusiveMaximum": 0.5,
        },
        "holdout_split_strategy": {
            "type": "string",
            "enum": ["random", "stratified", "group", "time", "preassigned"],
        },
        "holdout_time_column": _nullable_string,
        "dataset": _strict_object(
            {
                "name": _non_empty_string,
                "version": _non_empty_string,
                "content_sha256": {
                    "anyOf": [
                        {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        {"type": "null"},
                    ]
                },
                "target_column": _nullable_string,
                "blob_path": _string,
                "local_path": _string,
                "datastore_name": _non_empty_string,
                "delimiter": {"type": "string", "minLength": 1, "maxLength": 1},
                "encoding": _non_empty_string,
                "id_columns": {
                    "type": "array",
                    "items": _non_empty_string,
                    "uniqueItems": True,
                },
                "excluded_columns": {
                    "type": "array",
                    "items": _non_empty_string,
                    "uniqueItems": True,
                },
                "group_column": _nullable_string,
                "time_column": _nullable_string,
            },
            required=(
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
            ),
        ),
        "azureml": _strict_object(
            {
                "subscription_id": _string,
                "resource_group": _string,
                "workspace_name": _string,
                "compute_target": _string,
                "default_datastore": _non_empty_string,
                "environment": _non_empty_string,
                "environment_name_preprocessing": _non_empty_string,
                "environment_name_training": _non_empty_string,
            },
            required=(
                "subscription_id",
                "resource_group",
                "workspace_name",
                "compute_target",
                "default_datastore",
                "environment",
                "environment_name_preprocessing",
                "environment_name_training",
            ),
        ),
        "split": _strict_object(
            {
                "strategy": {
                    "type": "string",
                    "enum": [
                        "random",
                        "stratified",
                        "group",
                        "time",
                        "preassigned",
                    ],
                },
                "validation_fraction": {"const": 0.0},
                "test_fraction": {
                    "type": "number",
                    "exclusiveMinimum": 0.0,
                    "exclusiveMaximum": 0.5,
                },
                "cv_folds": {"type": "integer", "minimum": 2},
                "group_column": _nullable_string,
                "time_column": _nullable_string,
                "locked_test": {"const": True},
            },
            required=(
                "strategy",
                "validation_fraction",
                "test_fraction",
                "cv_folds",
                "group_column",
                "time_column",
                "locked_test",
            ),
        ),
        "metrics": _strict_object(
            {
                "primary": _non_empty_string,
                "selection_protocol": {"const": "cross_validation"},
                "cv_folds": {"type": "integer", "minimum": 2},
                "locked_test_once": {"const": True},
                "min_comparable_candidates": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 3,
                },
            },
            required=(
                "primary",
                "selection_protocol",
                "cv_folds",
                "locked_test_once",
                "min_comparable_candidates",
            ),
        ),
        "stage1": _strict_object(
            {
                "min_rows": _positive_integer,
                "max_missing_pct": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
                "classification_min_samples_per_class": _positive_integer,
                "regression_target_min_variance": {
                    "type": "number",
                    "minimum": 0,
                },
                "generate_sweetviz": {"type": "boolean"},
                "eda_sample_size": _positive_integer,
            },
            required=(
                "min_rows",
                "max_missing_pct",
                "classification_min_samples_per_class",
                "regression_target_min_variance",
                "generate_sweetviz",
                "eda_sample_size",
            ),
        ),
        "stage2": _strict_object(
            {
                "imputation_numeric": _non_empty_string,
                "imputation_categorical": _non_empty_string,
                "imputation_strategy": _non_empty_string,
                "statistical_tests_enabled": {"type": "boolean"},
                "high_cardinality_max": _positive_integer,
                "protected_columns": {
                    "type": "array",
                    "items": _non_empty_string,
                    "uniqueItems": True,
                },
            },
            required=(
                "imputation_numeric",
                "imputation_categorical",
                "imputation_strategy",
                "statistical_tests_enabled",
                "high_cardinality_max",
                "protected_columns",
            ),
        ),
        "stage3": _strict_object(
            {
                "encoding": _non_empty_string,
                "scaling": _non_empty_string,
                "imbalance_handling": _non_empty_string,
                "adaptive_scaling": {"type": "boolean"},
                "multicollinearity_check": {"type": "boolean"},
            },
            required=(
                "encoding",
                "scaling",
                "imbalance_handling",
                "adaptive_scaling",
                "multicollinearity_check",
            ),
        ),
        "stage4": _strict_object(
            {
                "feature_selection": _non_empty_string,
                "selection_method": _non_empty_string,
                "max_features": _positive_integer,
                "apply_pca_threshold": _positive_integer,
                "pca_variance_retained": _probability,
                "imbalance_detection": {"type": "boolean"},
            },
            required=(
                "feature_selection",
                "selection_method",
                "max_features",
                "apply_pca_threshold",
                "pca_variance_retained",
                "imbalance_detection",
            ),
        ),
        "stage5": _strict_object(
            {
                "model_universe": {
                    "type": "array",
                    "items": _non_empty_string,
                    "uniqueItems": True,
                },
                "flaml_time_budget": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                },
                "pycaret_fold": {"type": "integer", "minimum": 2},
                "pycaret_n_select": _positive_integer,
            },
            required=(
                "model_universe",
                "flaml_time_budget",
                "pycaret_fold",
                "pycaret_n_select",
            ),
        ),
        "phases": _strict_object(
            {
                "phase_a_baseline": _strict_object(
                    {
                        "engines": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["pycaret", "flaml"],
                            },
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                        "cv_folds": {"type": "integer", "minimum": 2},
                        "candidate_engine_timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 600,
                        },
                        "flaml_config": _strict_object(
                            {
                                "time_budget": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 600,
                                }
                            },
                            required=("time_budget",),
                        ),
                    },
                    required=(
                        "engines",
                        "cv_folds",
                        "candidate_engine_timeout_seconds",
                        "flaml_config",
                    ),
                ),
                "phase_b_recipes": _strict_object(
                    {
                        "library": _non_empty_string,
                        "tier": _non_empty_string,
                        "max_recipes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 40,
                        },
                        "runtime_budget_sec": _positive_integer,
                        "random_selection": {"const": False},
                        "safety_net_review_required": {"type": "boolean"},
                        "engines": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["pycaret", "flaml"],
                            },
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                    },
                    required=(
                        "library",
                        "tier",
                        "max_recipes",
                        "runtime_budget_sec",
                        "random_selection",
                        "safety_net_review_required",
                        "engines",
                    ),
                ),
                "phase_b": _strict_object(
                    {
                        "enable_profiling": {"const": True},
                        "profiling_output_path": _non_empty_string,
                        "library_dir": _non_empty_string,
                        "library": _non_empty_string,
                        "tier": _non_empty_string,
                        "max_variants": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 40,
                        },
                        "selection_strategy": {"const": "profile_scored"},
                        "runtime_budget_sec": _positive_integer,
                        "time_budget_per_variant": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 600,
                        },
                        "phase_timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10800,
                        },
                        "safety_net_review_required": {"type": "boolean"},
                        "engines": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["pycaret", "flaml"],
                            },
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                        "planner": _strict_object(
                            {
                                "enabled": {"const": True},
                                "round1_max_variants": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 40,
                                },
                                "round2_max_variants": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 8,
                                },
                                "proxy_prune_threshold": {"type": "number"},
                                "diversity_min_hamming_distance": {
                                    "type": "integer",
                                    "minimum": 0,
                                },
                                "cache_enabled": {"type": "boolean"},
                            },
                            required=(
                                "enabled",
                                "round1_max_variants",
                                "round2_max_variants",
                                "proxy_prune_threshold",
                                "diversity_min_hamming_distance",
                                "cache_enabled",
                            ),
                        ),
                    },
                    required=(
                        "enable_profiling",
                        "profiling_output_path",
                        "library_dir",
                        "library",
                        "tier",
                        "max_variants",
                        "selection_strategy",
                        "runtime_budget_sec",
                        "time_budget_per_variant",
                        "phase_timeout_seconds",
                        "safety_net_review_required",
                        "engines",
                        "planner",
                    ),
                ),
                "phase_c_hpo": _strict_object(
                    {
                        "optimizer": {"const": "optuna"},
                        "n_trials": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 3600,
                        },
                        "compute_rate_usd_per_hour": {
                            "type": "number",
                            "minimum": 0,
                        },
                    },
                    required=(
                        "optimizer",
                        "n_trials",
                        "timeout_seconds",
                        "compute_rate_usd_per_hour",
                    ),
                ),
            },
            required=(
                "phase_a_baseline",
                "phase_b_recipes",
                "phase_b",
                "phase_c_hpo",
            ),
        ),
        "registry": _strict_object(
            {
                "model_name": _non_empty_string,
                "min_quality": _strict_object(
                    {
                        "classification": {"type": "number"},
                        "regression": {"type": "number"},
                        "clustering": {"type": "number"},
                    },
                    required=("classification", "regression", "clustering"),
                ),
                "quality_failure_policy": {
                    "type": "string",
                    "enum": ["warn", "block"],
                },
                "warning_registration_allowed": {"type": "boolean"},
                "warning_tags": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "pass_aliases": {
                    "type": "array",
                    "items": _non_empty_string,
                    "uniqueItems": True,
                },
                "warning_aliases": {
                    "type": "array",
                    "items": _non_empty_string,
                    "uniqueItems": True,
                },
                "block_on_quality_fail": {"type": "boolean"},
            },
            required=(
                "model_name",
                "min_quality",
                "quality_failure_policy",
                "warning_registration_allowed",
                "warning_tags",
                "pass_aliases",
                "warning_aliases",
                "block_on_quality_fail",
            ),
        ),
        "recipes": {
            "type": "array",
            "items": _strict_object(
                {"file": _non_empty_string},
                required=("file",),
            ),
        },
        "source_config": _string,
        "compiled_config_hash": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
    required=(
        "schema_version",
        "migration",
        "experiment_name",
        "task_type",
        "preset",
        "random_seed",
        "holdout_fraction",
        "holdout_split_strategy",
        "holdout_time_column",
        "dataset",
        "azureml",
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
        "source_config",
    ),
)

# API/UI schema consumers should see only the supported v2 product contract.
CONFIG_SCHEMA = CONFIG_SCHEMA_V2


def validate_compiled_config(config: Mapping[str, Any]) -> None:
    """Validate an already materialized schema-v2 configuration."""

    validate(instance=dict(config), schema=CONFIG_SCHEMA_V2)
    task_type = config.get("task_type")
    target_col = config.get("dataset", {}).get("target_column")
    if task_type in ("classification", "regression") and not target_col:
        raise ValidationError(
            f"dataset.target_column is required for task_type={task_type!r}"
        )
    if task_type == "clustering":
        engines = config.get("phases", {}).get("phase_b", {}).get("engines")
        if engines != ["pycaret"]:
            raise ValidationError("Clustering must use PyCaret only")
    planner = config.get("phases", {}).get("phase_b", {}).get("planner", {})
    if planner.get("round2_max_variants", 0) > planner.get(
        "round1_max_variants", 0
    ):
        raise ValidationError("Round 2 maximum cannot exceed Round 1 maximum")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compile legacy input or validate v2 input and return explicit v2."""

    if not isinstance(config, dict):
        raise ValidationError("Pipeline configuration must be a mapping")
    if config.get("schema_version") == "2.0" and "source_config" in config:
        validate_compiled_config(config)
        return dict(config)

    from .config_compiler import compile_config

    return compile_config(config)
