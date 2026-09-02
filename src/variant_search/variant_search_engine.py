"""
Pipeline Variant Search - Production-Grade Configuration Space Explorer

Terminology:
- Pipeline Template = fixed stage architecture (stages 0-11)
- Pipeline Variant = one configuration across Stages 2-4 (prep/preprocess/feature engineering)
- Search Space = allowed options + constraints
- Variant Search = process of exploring variants
- Winning Variant (Locked) = config frozen for production retraining

Industrial patterns:
- Progressive narrowing (cheap_eval → full_eval → HPO)
- Leakage-safe configuration tracking
- Task-specific constraint enforcement
- Bayesian/Optuna alternative support

Author: MLOps Solution Accelerator V3
Date: 2026-01-25
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
import json
import yaml
from enum import Enum
from datetime import datetime


class SearchMode(Enum):
    """Variant search execution modes."""
    GRID_SAMPLE = "grid_sample"          # Enumerate all valid combinations
    RANDOM_SAMPLE = "random_sample"      # Random sampling from valid space
    PROGRESSIVE = "progressive"          # Multi-fidelity (cheap → full → HPO)
    BAYESIAN = "bayesian"                # Optuna over pipeline knobs
    LOCKED = "locked"                    # Reuse frozen variant (production)


class LeakageRisk(Enum):
    """Data leakage risk levels."""
    NONE = "none"                        # No leakage concerns
    LOW = "low"                          # Minor leakage if misused
    MEDIUM = "medium"                    # Requires CV-safe implementation
    HIGH = "high"                        # Significant leakage without pipelines
    CRITICAL = "critical"                # Almost guaranteed leakage


@dataclass
class VariantSearchSpace:
    """
    Defines allowed options for each pipeline configuration dimension.
    
    Constraints are enforced based on task_type during generation.
    """
    # Stage 2-3: Preprocessing dimensions
    # Tier 0 (original 6) + Tier 1 expansion (10 new methods)
    imputation: List[str] = field(default_factory=lambda: [
        # Tier 0 — original methods
        "mean", "median", "mode", "drop", "knn", "iterative",
        # Tier 1 — pandas-native & lightweight sklearn
        "forward_fill", "backward_fill", "interpolate_linear",
        "constant",  # fill with sentinel value (e.g., 0 or -999)
        # Tier 1 — statistical variants
        "trimmed_mean",  # mean after removing outlier extremes
        "winsorized_mean",  # mean with capped extremes
        "random_sample",  # impute from observed distribution
        # Tier 1 — column-aware composites
        "numeric_mean_cat_mode",  # mean for numbers, mode for categories
        "numeric_median_cat_mode",  # median for numbers, mode for categories
        "zero_fill",  # fill everything with 0 (baseline reference)
    ])
    
    encoding: List[str] = field(default_factory=lambda: [
        "onehot", "label", "target", "catboost"
    ])
    
    scaling: List[str] = field(default_factory=lambda: [
        "none", "standard", "robust", "minmax", "quantile", "yeo_johnson"
    ])
    
    # Task-specific dimensions
    imbalance: List[str] = field(default_factory=lambda: [
        "none", "smote", "adasyn", "smoteenn", "smotetomek"
    ])  # Classification only
    
    outlier_handling: List[str] = field(default_factory=lambda: [
        "none", "iqr_removal", "iqr_capping", "zscore", "winsorize", "isolation_forest"
    ])  # Regression only
    
    # Stage 4: Feature engineering
    feature_selection: List[str] = field(default_factory=lambda: [
        "none", "variance", "mutual_info", "chi2", "correlation", "rfe", "boruta"
    ])
    
    # Feature selection params (conditional)
    k_features: List[int] = field(default_factory=lambda: [10, 15, 20, 30, 50])
    corr_threshold: List[float] = field(default_factory=lambda: [0.85, 0.90, 0.95])
    
    # Training configuration
    algorithm_family_classification: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "rf", "linear", "nb"
    ])
    
    algorithm_family_regression: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "rf", "linear", "svr"
    ])
    
    algorithm_family_clustering: List[str] = field(default_factory=lambda: [
        "kmeans", "gmm", "dbscan", "hclust", "birch"
    ])


@dataclass
class PipelineVariant:
    """
    Single pipeline configuration variant with metadata.
    
    Attributes:
        variant_id: SHA1 hash of normalized config (reproducible)
        task_type: classification, regression, or clustering
        config: Configuration dictionary for stages 2-4
        leakage_risk: Assessed leakage risk level
        estimated_runtime_sec: Expected runtime (used for progressive filtering)
        constraints_violated: List of constraint violations (empty if valid)
        metadata: Additional tracking info
    """
    variant_id: str
    task_type: str
    config: Dict[str, Any]
    leakage_risk: LeakageRisk
    estimated_runtime_sec: int
    constraints_violated: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if variant passed all constraints."""
        return len(self.constraints_violated) == 0
    
    def to_yaml_recipe(self, output_path: Path):
        """Save as V3 recipe YAML format."""
        recipe_dict = {
            "recipe_name": self.variant_id,
            "version": "1.0",
            "description": f"Pipeline Variant: {self.metadata.get('description', 'Auto-generated')}",
            "task_type": self.task_type,
            "stage3_preprocessing": {
                "imputation": {"method": self.config["imputation"]},
                "encoding": {"categorical_method": self.config["encoding"]},
                "scaling": {"method": self.config["scaling"]},
            },
            "stage4_feature_engineering": {
                "feature_selection": {"method": self.config["feature_selection"]}
            },
            "variant_metadata": {
                "variant_id": self.variant_id,
                "leakage_risk": self.leakage_risk.value,
                "estimated_runtime_sec": self.estimated_runtime_sec,
                "generation_mode": self.metadata.get("mode", "unknown"),
            }
        }
        
        # Add task-specific dimensions
        if self.task_type == "classification" and "imbalance" in self.config:
            recipe_dict["stage3_preprocessing"]["imbalance_handling"] = {
                "method": self.config["imbalance"]
            }
        elif self.task_type == "regression" and "outlier_handling" in self.config:
            recipe_dict["stage3_preprocessing"]["outlier_handling"] = {
                "method": self.config["outlier_handling"]
            }
        
        # Add feature selection params if applicable
        fs_method = self.config["feature_selection"]
        if fs_method in ["mutual_info", "rfe"] and "k_features" in self.config:
            recipe_dict["stage4_feature_engineering"]["feature_selection"]["k_features"] = self.config["k_features"]
        elif fs_method == "correlation" and "corr_threshold" in self.config:
            recipe_dict["stage4_feature_engineering"]["feature_selection"]["threshold"] = self.config["corr_threshold"]
        
        # Save to YAML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(recipe_dict, f, default_flow_style=False, sort_keys=False)


class VariantSearchEngine:
    """
    Production-grade pipeline variant search with constraint validation.
    
    Features:
    - Task-specific constraint enforcement
    - Leakage risk assessment
    - Progressive narrowing support
    - Deterministic variant IDs (SHA1 hash)
    - Export to locked manifest format
    """
    
    def __init__(self, task_type: str, search_space: Optional[VariantSearchSpace] = None):
        """
        Initialize search engine for specific task type.
        
        Args:
            task_type: One of 'classification', 'regression', 'clustering'
            search_space: Custom search space (defaults to full space)
        """
        if task_type not in ["classification", "regression", "clustering"]:
            raise ValueError(f"Invalid task_type: {task_type}")
        
        self.task_type = task_type
        self.search_space = search_space or VariantSearchSpace()
    
    def count_variants(self) -> Dict[str, Any]:
        """
        Count total variants (raw and valid) with breakdown.
        
        Returns:
            Dictionary with counts and exclusion reasons
        """
        # Calculate raw combinatorial count
        raw_counts = {
            "imputation": len(self.search_space.imputation),
            "encoding": len(self.search_space.encoding),
            "scaling": len(self.search_space.scaling),
            "feature_selection": len(self.search_space.feature_selection),
        }
        
        # Add task-specific dimensions
        if self.task_type == "classification":
            raw_counts["imbalance"] = len(self.search_space.imbalance)
            raw_counts["algorithm_family"] = len(self.search_space.algorithm_family_classification)
        elif self.task_type == "regression":
            raw_counts["outlier_handling"] = len(self.search_space.outlier_handling)
            raw_counts["algorithm_family"] = len(self.search_space.algorithm_family_regression)
        else:  # clustering
            raw_counts["algorithm_family"] = len(self.search_space.algorithm_family_clustering)
        
        # Calculate raw total (pure multiplication)
        total_raw = 1
        for count in raw_counts.values():
            total_raw *= count
        
        # Add feature selection parameter variations
        fs_param_variants = 0
        for fs_method in self.search_space.feature_selection:
            if fs_method in ["mutual_info", "rfe"]:
                fs_param_variants += len(self.search_space.k_features)
            elif fs_method == "correlation":
                fs_param_variants += len(self.search_space.corr_threshold)
            else:
                fs_param_variants += 1  # no params
        
        # Adjust for conditional params
        total_raw_with_params = (total_raw // len(self.search_space.feature_selection)) * fs_param_variants
        
        # Count valid variants (apply constraints)
        valid_variants = self._generate_all_valid_variants()
        total_valid = len(valid_variants)
        
        # Breakdown of excluded reasons
        excluded_reasons = {
            "task_incompatible": 0,
            "leakage_critical": 0,
            "runtime_infeasible": 0,
            "incompatible_combination": 0
        }
        
        # Estimate exclusions
        if self.task_type == "clustering":
            # Exclude target encoding + imbalance
            excluded_reasons["task_incompatible"] += (
                len([e for e in self.search_space.encoding if e in ["target", "catboost"]]) *
                (total_raw // len(self.search_space.encoding))
            )
        
        if self.task_type == "regression":
            # Exclude imbalance methods
            excluded_reasons["task_incompatible"] += len(self.search_space.imbalance) - 1  # keep "none"
        
        return {
            "task_type": self.task_type,
            "total_raw": total_raw_with_params,
            "total_valid": total_valid,
            "excluded_count": total_raw_with_params - total_valid,
            "dimension_counts": raw_counts,
            "excluded_reasons": excluded_reasons,
            "leakage_risk_distribution": self._count_by_leakage_risk(valid_variants)
        }
    
    def _count_by_leakage_risk(self, variants: List[PipelineVariant]) -> Dict[str, int]:
        """Count variants by leakage risk level."""
        counts = {risk.value: 0 for risk in LeakageRisk}
        for variant in variants:
            counts[variant.leakage_risk.value] += 1
        return counts
    
    def generate_variants(
        self,
        mode: SearchMode = SearchMode.GRID_SAMPLE,
        max_variants: int = 50,
        seed: int = 42,
        runtime_budget_sec: Optional[int] = None
    ) -> List[PipelineVariant]:
        """
        Generate pipeline variants with constraint enforcement.
        
        Args:
            mode: Search mode (grid_sample, random_sample, progressive)
            max_variants: Maximum number of variants to generate
            seed: Random seed for reproducibility
            runtime_budget_sec: Optional filter for cheap_eval phase
            
        Returns:
            List of valid PipelineVariant objects
        """
        if mode == SearchMode.GRID_SAMPLE:
            variants = self._generate_all_valid_variants()
            # Filter by runtime if specified
            if runtime_budget_sec:
                variants = [v for v in variants if v.estimated_runtime_sec <= runtime_budget_sec]
            # Sample up to max_variants
            return variants[:max_variants]
        
        elif mode == SearchMode.RANDOM_SAMPLE:
            import random
            rng = random.Random(seed)
            all_variants = sorted(
                self._generate_all_valid_variants(),
                key=lambda variant: variant.variant_id,
            )
            if runtime_budget_sec:
                all_variants = [v for v in all_variants if v.estimated_runtime_sec <= runtime_budget_sec]
            return rng.sample(all_variants, min(max_variants, len(all_variants)))
        
        elif mode == SearchMode.PROGRESSIVE:
            # Progressive narrowing: generate cheap variants first
            cheap_variants = self._generate_all_valid_variants()
            cheap_variants = [v for v in cheap_variants if v.estimated_runtime_sec <= 120]  # 2 min cutoff
            cheap_variants.sort(key=lambda v: v.estimated_runtime_sec)
            return cheap_variants[:max_variants]
        
        else:
            raise NotImplementedError(f"Mode {mode} not yet implemented")
    
    def _generate_all_valid_variants(self) -> List[PipelineVariant]:
        """Generate all valid variants (enforcing constraints)."""
        variants = []
        
        # Build dimension lists based on task type
        imputation_options = self.search_space.imputation
        encoding_options = self.search_space.encoding
        scaling_options = self.search_space.scaling
        fs_options = self.search_space.feature_selection
        
        # Task-specific dimensions
        if self.task_type == "classification":
            task_specific_dim = [("imbalance", opt) for opt in self.search_space.imbalance]
        elif self.task_type == "regression":
            task_specific_dim = [("outlier_handling", opt) for opt in self.search_space.outlier_handling]
        else:  # clustering
            task_specific_dim = [(None, None)]  # No task-specific dim
        
        # Generate combinations
        for imputation in imputation_options:
            for encoding in encoding_options:
                for scaling in scaling_options:
                    for fs_method in fs_options:
                        for task_dim_key, task_dim_val in task_specific_dim:
                            # Build config
                            config = {
                                "imputation": imputation,
                                "encoding": encoding,
                                "scaling": scaling,
                                "feature_selection": fs_method,
                            }
                            
                            if task_dim_key:
                                config[task_dim_key] = task_dim_val
                            
                            # Add feature selection params if needed
                            if fs_method in ["mutual_info", "rfe"]:
                                for k in self.search_space.k_features:
                                    config_with_k = config.copy()
                                    config_with_k["k_features"] = k
                                    variant = self._create_variant(config_with_k)
                                    if variant.is_valid():
                                        variants.append(variant)
                            elif fs_method == "correlation":
                                for thresh in self.search_space.corr_threshold:
                                    config_with_thresh = config.copy()
                                    config_with_thresh["corr_threshold"] = thresh
                                    variant = self._create_variant(config_with_thresh)
                                    if variant.is_valid():
                                        variants.append(variant)
                            else:
                                variant = self._create_variant(config)
                                if variant.is_valid():
                                    variants.append(variant)
        
        return variants
    
    def _create_variant(self, config: Dict[str, Any]) -> PipelineVariant:
        """Create variant with validation and metadata."""
        # Generate deterministic ID
        variant_id = self._generate_variant_id(config)
        
        # Assess leakage risk
        leakage_risk = self._assess_leakage_risk(config)
        
        # Estimate runtime
        runtime = self._estimate_runtime(config)
        
        # Check constraints
        violations = self._check_constraints(config)
        
        # Build metadata
        metadata = {
            "description": self._build_description(config),
            "mode": "grid_sample",
            "task_type": self.task_type
        }
        
        return PipelineVariant(
            variant_id=variant_id,
            task_type=self.task_type,
            config=config,
            leakage_risk=leakage_risk,
            estimated_runtime_sec=runtime,
            constraints_violated=violations,
            metadata=metadata
        )
    
    def _generate_variant_id(self, config: Dict[str, Any]) -> str:
        """Generate stable SHA-256 identity from the normalized configuration."""
        # Sort keys for determinism
        normalized = json.dumps(config, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]
    
    def _assess_leakage_risk(self, config: Dict[str, Any]) -> LeakageRisk:
        """Assess data leakage risk for this configuration."""
        encoding = config.get("encoding")
        fs_method = config.get("feature_selection")
        imbalance = config.get("imbalance", "none")
        
        # Target encoding without CV = HIGH risk
        if encoding in ["target", "catboost"]:
            return LeakageRisk.HIGH
        
        # SMOTE/resampling outside CV = MEDIUM risk
        if imbalance in ["smote", "adasyn", "smoteenn", "smotetomek"]:
            return LeakageRisk.MEDIUM
        
        # RFE/Boruta feature selection = LOW risk (uses model but usually safe)
        if fs_method in ["rfe", "boruta"]:
            return LeakageRisk.LOW
        
        return LeakageRisk.NONE
    
    def _estimate_runtime(self, config: Dict[str, Any]) -> int:
        """Estimate runtime in seconds based on config complexity."""
        base_runtime = 30  # baseline
        
        # Imputation penalties
        if "knn" in config["imputation"]:
            base_runtime += 30
        elif config["imputation"] == "iterative":
            base_runtime += 60
        
        # Feature selection penalties
        fs_method = config["feature_selection"]
        if fs_method == "rfe":
            base_runtime += 120
        elif fs_method == "boruta":
            base_runtime += 180
        elif fs_method in ["mutual_info", "chi2"]:
            base_runtime += 20
        
        # Imbalance handling
        if config.get("imbalance") in ["smote", "adasyn"]:
            base_runtime += 15
        
        return base_runtime
    
    def _check_constraints(self, config: Dict[str, Any]) -> List[str]:
        """Check task-specific constraints and return violations."""
        violations = []
        
        # Clustering constraints
        if self.task_type == "clustering":
            if config["encoding"] in ["target", "catboost"]:
                violations.append("Clustering cannot use target encoding (no target column)")
            if config.get("imbalance") and config["imbalance"] != "none":
                violations.append("Clustering cannot use imbalance handling")
        
        # Regression constraints
        if self.task_type == "regression":
            if config.get("imbalance") and config["imbalance"] != "none":
                violations.append("Regression cannot use SMOTE/imbalance methods")
            if config["feature_selection"] == "chi2":
                violations.append("Chi2 feature selection requires classification task")
        
        # Classification constraints
        if self.task_type == "classification":
            # No specific exclusions for classification (most flexible)
            pass
        
        return violations
    
    def _build_description(self, config: Dict[str, Any]) -> str:
        """Build human-readable description."""
        parts = [
            config["imputation"],
            config["encoding"],
            config["scaling"],
            config["feature_selection"]
        ]
        
        if "imbalance" in config and config["imbalance"] != "none":
            parts.insert(1, config["imbalance"])
        
        if "outlier_handling" in config and config["outlier_handling"] != "none":
            parts.insert(1, config["outlier_handling"])
        
        return "+".join(parts)
    
    def export_locked_manifest(
        self,
        winning_variant: PipelineVariant,
        performance_metrics: Dict[str, float],
        output_path: Path
    ):
        """
        Export winning variant as locked manifest for production retraining.
        
        Args:
            winning_variant: The selected champion variant
            performance_metrics: Evaluation metrics (CV mean/std + holdout)
            output_path: Where to save locked_variant_manifest.json
        """
        manifest = {
            "manifest_version": "1.0",
            "lock_timestamp": datetime.utcnow().isoformat() + "Z",
            "task_type": self.task_type,
            "variant_id": winning_variant.variant_id,
            "search_mode": "locked",  # Signals production reuse
            "configuration": winning_variant.config,
            "leakage_risk": winning_variant.leakage_risk.value,
            "leakage_safe": winning_variant.leakage_risk in [LeakageRisk.NONE, LeakageRisk.LOW],
            "performance": {
                "metrics": performance_metrics,
                "evaluation_mode": "5-fold CV + holdout",
            },
            "metadata": winning_variant.metadata,
            "retraining_instructions": {
                "search_mode": "locked",
                "reuse_variant_id": winning_variant.variant_id,
                "skip_search": True,
                "allow_drift_rescan_threshold": 0.05  # 5% metric drop triggers rescan
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Locked variant manifest saved: {output_path}")


# =============================================================================
# Example Usage & Testing
# =============================================================================

def example_usage():
    """Demonstrate variant search for all three task types."""
    
    print("=" * 80)
    print("🔍 PIPELINE VARIANT SEARCH - PRODUCTION DEMO")
    print("=" * 80)
    
    for task_type in ["classification", "regression", "clustering"]:
        print(f"\n📊 Task: {task_type.upper()}")
        print("-" * 80)
        
        # Initialize search engine
        engine = VariantSearchEngine(task_type=task_type)
        
        # Count variants
        counts = engine.count_variants()
        print(f"Total Raw Combinations: {counts['total_raw']:,}")
        print(f"Valid Variants: {counts['total_valid']:,}")
        print(f"Excluded: {counts['excluded_count']:,}")
        print(f"\nLeakage Risk Distribution:")
        for risk, count in counts['leakage_risk_distribution'].items():
            print(f"  {risk:10s}: {count:4d} variants")
        
        # Generate sample variants
        print(f"\n🎯 Generating 5 sample variants (progressive mode)...")
        variants = engine.generate_variants(
            mode=SearchMode.PROGRESSIVE,
            max_variants=5,
            seed=42
        )
        
        if variants:
            print(f"\nSample Variant (ID: {variants[0].variant_id}):")
            print(f"  Config: {variants[0].config}")
            print(f"  Leakage Risk: {variants[0].leakage_risk.value}")
            print(f"  Estimated Runtime: {variants[0].estimated_runtime_sec}s")
            print(f"  Description: {variants[0].metadata['description']}")


if __name__ == "__main__":
    example_usage()
