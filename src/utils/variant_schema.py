"""
Variant Schema - Data structures for pipeline variant configurations

Defines typed schemas for variant YAML files and validation logic.

Author: MLOps Solution Accelerator V3
Date: 2026-01-26
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from utils.fitted_variant_preprocessor import SUPPORTED_RECIPE_METHODS


def validate_variant_yaml(path: str, task_type: str | None = None) -> Dict[str, Any]:
    """Validate a variant YAML file and return a structured report.

    This is deliberately non-throwing so batch runners can record per-variant
    validation failures without losing the whole Phase B report.
    """
    path_obj = Path(path)
    report: Dict[str, Any] = {
        "path": str(path_obj),
        "variant_id": path_obj.stem,
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    if not path_obj.exists():
        report["valid"] = False
        report["errors"].append(f"Variant file not found: {path}")
        return report

    try:
        with open(path_obj, "r") as f:
            raw_yaml = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001 - report all parse/read failures
        report["valid"] = False
        report["errors"].append(f"YAML parse/read failed: {exc}")
        return report

    if not isinstance(raw_yaml, dict):
        report["valid"] = False
        report["errors"].append("Variant YAML must be a mapping")
        return report

    report["variant_id"] = (
        raw_yaml.get("variant_metadata", {}).get("variant_id")
        or raw_yaml.get("recipe_name")
        or path_obj.stem
    )

    version = raw_yaml.get("version")
    if version is None:
        report["errors"].append("Recipe version is required")
    elif str(version) != "1.0":
        report["errors"].append(
            f"Unsupported recipe version {version!r}; expected semantic version '1.0'"
        )
    has_legacy_metadata = "v1_metadata" in raw_yaml
    has_current_metadata = "variant_metadata" in raw_yaml
    if has_legacy_metadata and has_current_metadata:
        report["errors"].append(
            "Recipe cannot mix legacy v1_metadata with variant_metadata"
        )
    if has_legacy_metadata:
        legacy = raw_yaml.get("v1_metadata")
        if not isinstance(legacy, dict):
            report["errors"].append("v1_metadata must be a mapping")
        else:
            for field_name in (
                "compatibility_score",
                "expected_quality",
                "max_runtime_seconds",
                "level",
                "original_name",
            ):
                if field_name not in legacy:
                    report["errors"].append(
                        "Legacy v1_metadata missing required field: "
                        f"{field_name}"
                    )
    elif not has_current_metadata:
        report["errors"].append(
            "Recipe requires variant_metadata or complete legacy v1_metadata"
        )

    for key in ("recipe_name", "stage3_preprocessing", "stage4_feature_engineering"):
        if key not in raw_yaml:
            report["errors"].append(f"Missing required top-level key: {key}")

    stage3 = raw_yaml.get("stage3_preprocessing") or {}
    if not isinstance(stage3, dict):
        report["errors"].append("stage3_preprocessing must be a mapping")
        stage3 = {}
    for key in ("imputation", "encoding", "scaling"):
        if not isinstance(stage3.get(key), dict):
            report["errors"].append(f"stage3_preprocessing.{key} must be a mapping")

    stage4 = raw_yaml.get("stage4_feature_engineering") or {}
    if not isinstance(stage4, dict):
        report["errors"].append("stage4_feature_engineering must be a mapping")
        stage4 = {}
    if not isinstance(stage4.get("feature_selection"), dict):
        report["warnings"].append("stage4_feature_engineering.feature_selection missing; defaults will be used")

    if report["errors"]:
        report["valid"] = False
        return report

    try:
        variant = load_variant(str(path_obj))
        if task_type:
            validate_variant_for_task(variant, task_type)
    except Exception as exc:  # noqa: BLE001 - convert typed validation failures to report rows
        report["valid"] = False
        report["errors"].append(str(exc))

    return report


@dataclass
class ImputationConfig:
    method: str  # mean, median, knn, iterative, drop, forward_fill, backward_fill, etc.
    # Optional params (used by select methods, ignored by others)
    n_neighbors: Optional[int] = None       # knn (default: 5)
    max_iter: Optional[int] = None          # iterative (default: 10)
    fill_value: Optional[Any] = None        # constant / zero_fill
    trim_fraction: Optional[float] = None   # trimmed_mean / winsorized_mean (0-0.5)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodingConfig:
    categorical_method: str  # onehot, label, target, catboost
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingConfig:
    method: str  # none, standard, robust, minmax, quantile, yeo_johnson
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImbalanceHandlingConfig:
    method: str  # none, smote, adasyn, smoteenn, smotetomek
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlierHandlingConfig:
    method: str  # none, iqr_removal, iqr_capping, zscore, winsorize, isolation_forest
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage3PreprocessingConfig:
    imputation: ImputationConfig
    encoding: EncodingConfig
    scaling: ScalingConfig
    imbalance_handling: Optional[ImbalanceHandlingConfig] = None
    outlier_handling: Optional[OutlierHandlingConfig] = None


@dataclass
class FeatureSelectionConfig:
    method: str  # none, variance, mutual_info, chi2, correlation, rfe, boruta
    threshold: Optional[float] = None
    k_features: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage4FeatureEngineeringConfig:
    feature_selection: FeatureSelectionConfig


@dataclass
class VariantMetadata:
    variant_id: str
    leakage_risk: str = "unknown"  # none, low, medium, high, critical
    estimated_runtime_sec: int = 30
    generation_mode: str = "manual"


@dataclass
class VariantConfig:
    """Complete pipeline variant configuration.
    
    Represents one preprocessing + feature engineering strategy
    used to train and compare models.
    """
    recipe_name: str
    version: str
    description: str
    task_type: str  # classification, regression, clustering
    stage3_preprocessing: Stage3PreprocessingConfig
    stage4_feature_engineering: Stage4FeatureEngineeringConfig
    variant_metadata: VariantMetadata
    
    # Raw YAML for pass-through
    _raw_yaml: Optional[Dict[str, Any]] = field(default=None, repr=False)
    
    @property
    def variant_id(self) -> str:
        return self.variant_metadata.variant_id
    
    @property
    def leakage_risk(self) -> str:
        return self.variant_metadata.leakage_risk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "recipe_name": self.recipe_name,
            "version": self.version,
            "description": self.description,
            "task_type": self.task_type,
            "stage3_preprocessing": {
                "imputation": {
                    "method": self.stage3_preprocessing.imputation.method,
                    **self.stage3_preprocessing.imputation.parameters,
                },
                "encoding": {
                    "categorical_method": self.stage3_preprocessing.encoding.categorical_method,
                    **self.stage3_preprocessing.encoding.parameters,
                },
                "scaling": {
                    "method": self.stage3_preprocessing.scaling.method,
                    **self.stage3_preprocessing.scaling.parameters,
                },
                "imbalance_handling": (
                    {
                        "method": self.stage3_preprocessing.imbalance_handling.method,
                        **self.stage3_preprocessing.imbalance_handling.parameters,
                    }
                    if self.stage3_preprocessing.imbalance_handling
                    else None
                ),
                "outlier_handling": (
                    {
                        "method": self.stage3_preprocessing.outlier_handling.method,
                        **self.stage3_preprocessing.outlier_handling.parameters,
                    }
                    if self.stage3_preprocessing.outlier_handling
                    else None
                ),
            },
            "stage4_feature_engineering": {
                "feature_selection": {
                    "method": self.stage4_feature_engineering.feature_selection.method,
                    "threshold": self.stage4_feature_engineering.feature_selection.threshold,
                    "k_features": self.stage4_feature_engineering.feature_selection.k_features,
                    **self.stage4_feature_engineering.feature_selection.parameters,
                }
            },
            "variant_metadata": {
                "variant_id": self.variant_metadata.variant_id,
                "leakage_risk": self.variant_metadata.leakage_risk,
                "estimated_runtime_sec": self.variant_metadata.estimated_runtime_sec,
                "generation_mode": self.variant_metadata.generation_mode,
            }
        }


def load_variant(path: str) -> VariantConfig:
    """Load and parse a variant YAML file.
    
    Args:
        path: Path to variant YAML file
        
    Returns:
        VariantConfig instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML is malformed or missing required fields
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Variant file not found: {path}")
    
    with open(path_obj, 'r') as f:
        raw_yaml = yaml.safe_load(f)
    
    # Extract required fields with error handling
    try:
        # Stage 3 preprocessing
        stage3 = raw_yaml.get("stage3_preprocessing", {})
        imputation_data = dict(stage3.get("imputation") or {})
        imputation = ImputationConfig(
            method=imputation_data.get("method", "mean"),
            n_neighbors=imputation_data.get("n_neighbors"),
            max_iter=imputation_data.get("max_iter"),
            fill_value=imputation_data.get("fill_value"),
            trim_fraction=imputation_data.get("trim_fraction"),
            parameters={
                key: value
                for key, value in imputation_data.items()
                if key
                not in {
                    "method",
                    "n_neighbors",
                    "max_iter",
                    "fill_value",
                    "trim_fraction",
                    "research_note",
                }
            },
        )
        encoding_data = dict(stage3.get("encoding") or {})
        encoding = EncodingConfig(
            categorical_method=encoding_data.get("categorical_method", "onehot"),
            parameters={
                key: value
                for key, value in encoding_data.items()
                if key not in {"categorical_method", "research_note"}
            },
        )
        scaling_data = dict(stage3.get("scaling") or {})
        scaling = ScalingConfig(
            method=scaling_data.get("method", "none"),
            parameters={
                key: value
                for key, value in scaling_data.items()
                if key not in {"method", "research_note"}
            },
        )
        
        # Optional: imbalance handling
        imbalance_data = stage3.get("imbalance_handling")
        imbalance = None
        if imbalance_data:
            imbalance = ImbalanceHandlingConfig(
                method=imbalance_data.get("method", "none"),
                parameters={
                    key: value
                    for key, value in imbalance_data.items()
                    if key not in {"method", "research_note"}
                },
            )
        
        # Optional: outlier handling
        outlier_data = stage3.get("outlier_handling")
        outlier = None
        if outlier_data:
            outlier = OutlierHandlingConfig(
                method=outlier_data.get("method", "none"),
                parameters={
                    key: value
                    for key, value in outlier_data.items()
                    if key not in {"method", "research_note"}
                },
            )
        
        stage3_config = Stage3PreprocessingConfig(
            imputation=imputation,
            encoding=encoding,
            scaling=scaling,
            imbalance_handling=imbalance,
            outlier_handling=outlier
        )
        
        # Stage 4 feature engineering
        stage4 = raw_yaml.get("stage4_feature_engineering", {})
        fsel_data = stage4.get("feature_selection", {})
        feature_selection = FeatureSelectionConfig(
            method=fsel_data.get("method", "none"),
            threshold=fsel_data.get("threshold"),
            k_features=fsel_data.get("k_features"),
            parameters={
                key: value
                for key, value in fsel_data.items()
                if key
                not in {
                    "method",
                    "threshold",
                    "k_features",
                    "research_note",
                }
            },
        )
        stage4_config = Stage4FeatureEngineeringConfig(feature_selection=feature_selection)
        
        # Variant metadata
        meta_data = raw_yaml.get("variant_metadata", {})
        metadata = VariantMetadata(
            variant_id=meta_data.get("variant_id", raw_yaml.get("recipe_name", "unknown")),
            leakage_risk=meta_data.get("leakage_risk", "unknown"),
            estimated_runtime_sec=meta_data.get("estimated_runtime_sec", 30),
            generation_mode=meta_data.get("generation_mode", "manual")
        )
        
        # Create VariantConfig
        variant = VariantConfig(
            recipe_name=raw_yaml.get("recipe_name", "unknown"),
            version=raw_yaml.get("version", "1.0"),
            description=raw_yaml.get("description", ""),
            task_type=raw_yaml.get("task_type", "classification"),
            stage3_preprocessing=stage3_config,
            stage4_feature_engineering=stage4_config,
            variant_metadata=metadata,
            _raw_yaml=raw_yaml
        )
        
        return variant
        
    except KeyError as e:
        raise ValueError(f"Malformed variant YAML, missing field: {e}")


def validate_variant_for_task(variant: VariantConfig, task_type: str) -> None:
    """Validate that a variant is compatible with the task type.
    
    Args:
        variant: Variant configuration to validate
        task_type: Expected task type
        
    Raises:
        ValueError: If variant is incompatible with task type
    """
    errors = []

    supported_methods = SUPPORTED_RECIPE_METHODS

    method_values = {
        "imputation": variant.stage3_preprocessing.imputation.method,
        "encoding": variant.stage3_preprocessing.encoding.categorical_method,
        "scaling": variant.stage3_preprocessing.scaling.method,
        "imbalance": (
            variant.stage3_preprocessing.imbalance_handling.method
            if variant.stage3_preprocessing.imbalance_handling
            else "none"
        ),
        "outlier": (
            variant.stage3_preprocessing.outlier_handling.method
            if variant.stage3_preprocessing.outlier_handling
            else "none"
        ),
        "feature_selection": (
            variant.stage4_feature_engineering.feature_selection.method
        ),
    }
    for dimension, method in method_values.items():
        if method not in supported_methods[dimension]:
            errors.append(
                f"Unsupported {dimension} method for Phase B: {method!r}"
            )
    
    # Task type mismatch
    if variant.task_type != task_type:
        errors.append(
            f"Variant task_type '{variant.task_type}' doesn't match expected '{task_type}'"
        )
    
    # Classification-specific validation
    if task_type == "classification":
        # OK: can use all preprocessing methods
        pass
    
    # Regression-specific validation
    elif task_type == "regression":
        # DISALLOW: imbalance handling
        if variant.stage3_preprocessing.imbalance_handling:
            method = variant.stage3_preprocessing.imbalance_handling.method
            if method != "none":
                errors.append(
                    f"Regression cannot use imbalance handling (got '{method}')"
                )
        
        # DISALLOW: target encoding (requires target, causes leakage)
        if variant.stage3_preprocessing.encoding.categorical_method == "target":
            errors.append(
                "Regression with target encoding has high leakage risk (not supported in Phase-1)"
            )
    
    # Clustering-specific validation
    elif task_type == "clustering":
        # DISALLOW: target encoding
        if variant.stage3_preprocessing.encoding.categorical_method == "target":
            errors.append(
                "Clustering cannot use target encoding (no target available)"
            )
        
        # DISALLOW: imbalance handling
        if variant.stage3_preprocessing.imbalance_handling:
            method = variant.stage3_preprocessing.imbalance_handling.method
            if method != "none":
                errors.append(
                    f"Clustering cannot use imbalance handling (got '{method}')"
                )
        selector = variant.stage4_feature_engineering.feature_selection.method
        if selector in {"correlation", "mutual_info"}:
            errors.append(
                "Clustering cannot use target-dependent feature selection "
                f"(got {selector!r}); use 'variance' or 'none'"
            )
    
    # Leakage warnings (not errors, but flagged)
    if variant.stage3_preprocessing.encoding.categorical_method == "target":
        if variant.leakage_risk not in ["medium", "high", "critical"]:
            errors.append(
                f"Target encoding should have leakage_risk >= 'medium' (got '{variant.leakage_risk}')"
            )
    
    if errors:
        raise ValueError(f"Variant validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
