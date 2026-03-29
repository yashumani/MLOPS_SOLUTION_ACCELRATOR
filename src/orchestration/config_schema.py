from typing import Any, Dict

from jsonschema import validate, ValidationError

CONFIG_SCHEMA: Dict[str, Any] = {
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
                "delimiter": {"type": "string", "maxLength": 1}
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
                "high_cardinality_max": {"type": "integer", "minimum": 1}
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
                        "timeout_seconds": {"type": "integer", "minimum": 10}
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


def validate_config(config: Dict[str, Any]) -> None:
    """Validate pipeline config against schema with cross-field checks."""
    validate(instance=config, schema=CONFIG_SCHEMA)

    # Cross-field: target_column required for classification/regression
    task_type = config.get("task_type")
    target_col = config.get("dataset", {}).get("target_column")
    if task_type in ("classification", "regression") and not target_col:
        raise ValidationError(
            f"dataset.target_column is required for task_type='{task_type}'"
        )
