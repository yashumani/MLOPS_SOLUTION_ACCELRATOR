"""
V1 Recipe Converter: JSON → V3 YAML Recipe Format

Converts V1's 1000+ recipe JSON format to V3's YAML schema while preserving
compatibility scores, runtime estimates, and quality tier metadata.

Architecture:
- V1 JSON: Flat data_cleaning/feature_engineering/model_training structure
- V3 YAML: Nested stage3_preprocessing/stage4_feature_engineering structure

Key Transformations:
- V1 missing_value_strategy → V3 stage3_preprocessing.imputation
- V1 outlier_handling → V3 stage3_preprocessing.outlier_handling (regression only)
- V1 encoding → V3 stage3_preprocessing.encoding.categorical_method
- V1 scaling → V3 stage3_preprocessing.scaling.method
- V1 feature_selection → V3 stage4_feature_engineering.feature_selection
- V1 metadata → V3 v1_metadata (preserved for filtering/selection)

Limitations:
- V3 recipes don't include model selection (delegated to PyCaret compare_models())
- V3 recipes don't include HPO config (delegated to Phase C Optuna)
- V3 recipes don't include CV strategy (PyCaret defaults)
- V3 omits polynomial/interaction features (not supported)
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import re


class V1ToV3RecipeConverter:
    """Convert V1 JSON recipes to V3 YAML format with task-type awareness."""
    
    # Mapping tables for schema translation
    IMPUTATION_MAP = {
        "drop": {"method": "drop"},
        "mean": {"method": "mean"},
        "median": {"method": "median"},
        "mode": {"method": "mode"},
        "knn": {"method": "knn", "n_neighbors": 5},
        "knn_5": {"method": "knn", "n_neighbors": 5},
        "knn_10": {"method": "knn", "n_neighbors": 10},
        "iterative": {"method": "iterative", "max_iter": 10},
        "mice": {"method": "iterative", "max_iter": 10},  # MICE = iterative
        "missforest": {"method": "knn", "n_neighbors": 10},  # Approximate with KNN
        "datawig": {"method": "knn", "n_neighbors": 5},  # Fallback to KNN
        "automl_impute": {"method": "iterative", "max_iter": 10},  # Fallback
    }
    
    ENCODING_MAP = {
        "label": "label",
        "onehot": "onehot",
        "target": "target",
        "woe": "target",  # WOE approximated by target encoding
        "leave_one_out": "target",
        "james_stein": "target",
        "catboost": "target",
        "entity_embedding": "onehot",  # Fallback
    }
    
    SCALING_MAP = {
        "none": "none",
        "standard": "standard",
        "robust": "robust",
        "minmax": "minmax",
        "maxabs": "minmax",  # Approximate
        "quantile": "robust",  # Approximate
        "power": "standard",  # Fallback
        "rank": "minmax",  # Fallback
    }
    
    OUTLIER_MAP = {
        "none": {"method": "none"},
        "iqr_removal": {"method": "iqr", "action": "remove", "threshold": 1.5},
        "iqr_capping": {"method": "iqr", "action": "cap", "threshold": 1.5},
        "isolation_forest": {"method": "isolation_forest", "contamination": 0.1},
        "ensemble": {"method": "iqr", "action": "cap", "threshold": 1.5},
        "local_outlier_factor": {"method": "isolation_forest", "contamination": 0.1},
        "elliptic_envelope": {"method": "isolation_forest", "contamination": 0.1},
        "one_class_svm": {"method": "isolation_forest", "contamination": 0.1},
        "ensemble_voting": {"method": "iqr", "action": "cap", "threshold": 2.0},
    }
    
    FEATURE_SELECTION_MAP = {
        "none": {"method": "none"},
        "variance": {"method": "variance", "threshold": 0.01},
        "mutual_info": {"method": "mutual_info", "k_features": 20},
        "chi2": {"method": "chi2", "k_features": 20},
        "boruta": {"method": "mutual_info", "k_features": 30},  # Approximate
        "rfe": {"method": "rfe", "n_features": 20},
        "lasso": {"method": "mutual_info", "k_features": 20},  # Fallback
        "elastic_net": {"method": "mutual_info", "k_features": 20},
        "shap_based": {"method": "mutual_info", "k_features": 30},
        "lime_based": {"method": "mutual_info", "k_features": 30},
        "permutation_importance": {"method": "mutual_info", "k_features": 25},
        "ensemble_selection": {"method": "mutual_info", "k_features": 30},
    }
    
    IMBALANCE_MAP = {
        "none": {"method": "none"},
        "smote": {"method": "smote", "sampling_strategy": "auto", "k_neighbors": 5},
        "adasyn": {"method": "adasyn", "sampling_strategy": "auto", "n_neighbors": 5},
        "smoteenn": {"method": "smote", "sampling_strategy": "auto", "k_neighbors": 5},
        "smotetomek": {"method": "smote", "sampling_strategy": "auto", "k_neighbors": 5},
        "random_undersample": {"method": "none"},  # V3 doesn't support undersampling
        "cluster_centroids": {"method": "none"},
        "balanced_bagging": {"method": "smote", "sampling_strategy": "auto", "k_neighbors": 5},
    }
    
    def __init__(self, task_type: str = "classification"):
        """
        Initialize converter for specific task type.
        
        Args:
            task_type: One of 'classification', 'regression', 'clustering'
        """
        self.task_type = task_type
    
    def convert(self, v1_recipe: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a V1 recipe JSON to V3 YAML structure.
        
        Args:
            v1_recipe: V1 recipe dict with data_cleaning, feature_engineering, model_training
            
        Returns:
            V3-compatible recipe dict ready for YAML serialization
        """
        recipe_id = v1_recipe.get("recipe_id", "unknown")
        description = v1_recipe.get("description", "Converted from V1 recipe")
        
        # Build V3 recipe structure
        v3_recipe = {
            "recipe_name": recipe_id,
            "version": "1.0",
            "description": description,
            "task_type": self.task_type,
        }
        
        # Convert preprocessing (stage3)
        v3_recipe["stage3_preprocessing"] = self._convert_preprocessing(
            v1_recipe.get("data_cleaning", {}),
            v1_recipe.get("model_training", {})
        )
        
        # Convert feature engineering (stage4)
        v3_recipe["stage4_feature_engineering"] = self._convert_feature_engineering(
            v1_recipe.get("feature_engineering", {})
        )
        
        # Preserve V1 metadata for filtering/selection
        v3_recipe["v1_metadata"] = {
            "compatibility_score": v1_recipe.get("compatibility_score", 0.6),
            "expected_quality": v1_recipe.get("expected_quality", "Unknown"),
            "max_runtime_seconds": v1_recipe.get("max_runtime_seconds", 300),
            "level": v1_recipe.get("level", 0),
            "original_name": v1_recipe.get("name", recipe_id),
        }
        
        return v3_recipe
    
    def _convert_preprocessing(self, data_cleaning: Dict, model_training: Dict) -> Dict:
        """Convert V1 data_cleaning to V3 stage3_preprocessing."""
        preprocessing = {}
        
        # 1. Imputation
        missing_strategy = data_cleaning.get("missing_value_strategy", "mean")
        preprocessing["imputation"] = self.IMPUTATION_MAP.get(
            missing_strategy,
            {"method": "mean"}  # Default fallback
        )
        
        # 2. Outlier handling (regression only)
        if self.task_type == "regression":
            outlier_method = data_cleaning.get("outlier_handling", "none")
            preprocessing["outlier_handling"] = self.OUTLIER_MAP.get(
                outlier_method,
                {"method": "none"}
            )
        
        # 3. Imbalance handling (classification only)
        if self.task_type == "classification":
            # Try model_training first (dict format)
            imbalance_method = None
            model_imbalance = model_training.get("imbalance_handling", {})
            if isinstance(model_imbalance, dict):
                imbalance_method = model_imbalance.get("method", "none")
            
            # Fall back to data_cleaning (string format)
            if not imbalance_method or imbalance_method == "none":
                data_imbalance = data_cleaning.get("imbalance_handling", "none")
                if isinstance(data_imbalance, str):
                    imbalance_method = data_imbalance
                elif isinstance(data_imbalance, dict):
                    imbalance_method = data_imbalance.get("method", "none")
            
            preprocessing["imbalance_handling"] = self.IMBALANCE_MAP.get(
                imbalance_method,
                {"method": "none"}
            )
        
        # 4. Encoding
        encoding_method = data_cleaning.get("encoding", "onehot")
        preprocessing["encoding"] = {
            "categorical_method": self.ENCODING_MAP.get(encoding_method, "onehot"),
            "handle_unknown": "value" if self.ENCODING_MAP.get(encoding_method) == "target" else "error"
        }
        
        # 5. Scaling
        scaling_method = data_cleaning.get("scaling", "standard")
        preprocessing["scaling"] = {
            "method": self.SCALING_MAP.get(scaling_method, "standard")
        }
        
        return preprocessing
    
    def _convert_feature_engineering(self, feature_eng: Dict) -> Dict:
        """Convert V1 feature_engineering to V3 stage4_feature_engineering."""
        feat_eng = {}
        
        # Feature selection (only dimension V3 supports)
        selection_method = feature_eng.get("feature_selection", "none")
        feat_eng["feature_selection"] = self.FEATURE_SELECTION_MAP.get(
            selection_method,
            {"method": "none"}
        )
        
        # NOTE: V3 doesn't support these V1 features:
        # - polynomial_features → PyCaret handles internally
        # - interaction_features → PyCaret handles internally
        # - statistical_features → Not supported
        # - dimensionality_reduction → Not supported in recipes
        
        return feat_eng
    
    def validate(self, v3_recipe: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate V3 recipe for task-type compatibility.
        
        Args:
            v3_recipe: Converted V3 recipe dict
            
        Returns:
            (is_valid, warnings) tuple
        """
        warnings = []
        
        # Check SMOTE on non-classification
        if self.task_type != "classification":
            imbalance_method = v3_recipe.get("stage3_preprocessing", {}).get(
                "imbalance_handling", {}
            ).get("method", "none")
            
            if imbalance_method in ["smote", "adasyn", "smoteenn", "smotetomek"]:
                warnings.append(
                    f"⚠️  Recipe contains SMOTE ({imbalance_method}) which is classification-only. "
                    f"Will be skipped for {self.task_type} tasks."
                )
        
        # Check outlier handling on classification/clustering
        if self.task_type in ["classification", "clustering"]:
            outlier_method = v3_recipe.get("stage3_preprocessing", {}).get(
                "outlier_handling", {}
            ).get("method", "none")
            
            if outlier_method != "none":
                warnings.append(
                    f"⚠️  Recipe contains outlier handling ({outlier_method}) which is regression-specific. "
                    f"Will be skipped for {self.task_type} tasks."
                )
        
        # Check for target column requirement (clustering doesn't need it)
        if self.task_type == "clustering":
            if "target_column" in v3_recipe.get("dataset", {}):
                warnings.append(
                    "⚠️  Clustering doesn't require target_column. Will be ignored."
                )
        
        return len(warnings) == 0, warnings
    
    def save_yaml(self, v3_recipe: Dict[str, Any], output_path: Path) -> None:
        """
        Save V3 recipe to YAML file with validation.
        
        Args:
            v3_recipe: Converted V3 recipe dict
            output_path: Path to output YAML file
        """
        # Validate before saving
        is_valid, warnings = self.validate(v3_recipe)
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML with comments for warnings
        with open(output_path, 'w') as f:
            if warnings:
                f.write("# ⚠️  RECIPE WARNINGS:\n")
                for warning in warnings:
                    f.write(f"# {warning}\n")
                f.write("\n")
            
            yaml.dump(v3_recipe, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"✅ Saved: {output_path}")
        if warnings:
            for w in warnings:
                print(f"   {w}")


def convert_v1_recipe_batch(
    v1_recipes: list[Dict[str, Any]],
    task_type: str,
    output_dir: Path,
    tier: Optional[str] = None
) -> int:
    """
    Convert a batch of V1 recipes to V3 YAML format.
    
    Args:
        v1_recipes: List of V1 recipe dicts
        task_type: Target task type (classification/regression/clustering)
        output_dir: Base output directory (will create subdirs by tier)
        tier: Optional quality tier name for subdirectory organization
        
    Returns:
        Number of successfully converted recipes
    """
    converter = V1ToV3RecipeConverter(task_type=task_type)
    success_count = 0
    
    for v1_recipe in v1_recipes:
        try:
            # Convert to V3 format
            v3_recipe = converter.convert(v1_recipe)
            
            # Determine output path
            recipe_id = v3_recipe["recipe_name"]
            
            if tier:
                # Organize by tier: configs/recipes/classification/v1_generated/lightning_fast/recipe_001.yml
                tier_dir = output_dir / tier.lower().replace(" ", "_")
                output_path = tier_dir / f"{recipe_id}.yml"
            else:
                # Flat structure: configs/recipes/classification/v1_generated/recipe_001.yml
                output_path = output_dir / f"{recipe_id}.yml"
            
            # Save to YAML
            converter.save_yaml(v3_recipe, output_path)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to convert {v1_recipe.get('recipe_id', 'unknown')}: {e}")
    
    return success_count


if __name__ == "__main__":
    # Test conversion with sample V1 recipe
    sample_v1_recipe = {
        "recipe_id": "recipe_lightning_fast_0001",
        "name": "Lightning Fast: lr_l1 + prep_lightning + feat_minimal",
        "level": 5,
        "description": "Fast preprocessing, minimal features, logistic regression",
        "engine": "pycaret",
        "data_cleaning": {
            "missing_value_strategy": "drop",
            "outlier_handling": "none",
            "scaling": "none",
            "encoding": "label"
        },
        "feature_engineering": {
            "feature_selection": "none",
            "polynomial_features": False,
            "interaction_features": False
        },
        "model_training": {
            "pycaret": {
                "fold": 1,
                "include_models": ["lr"]
            },
            "imbalance_handling": {"method": "none"}
        },
        "max_runtime_seconds": 15,
        "compatibility_score": 0.85,
        "expected_quality": "Lightning Fast"
    }
    
    print("Testing V1→V3 Recipe Converter...")
    print("=" * 80)
    
    converter = V1ToV3RecipeConverter(task_type="classification")
    v3_recipe = converter.convert(sample_v1_recipe)
    
    print("\nV3 Recipe Output:")
    print(yaml.dump(v3_recipe, default_flow_style=False, sort_keys=False, indent=2))
    
    is_valid, warnings = converter.validate(v3_recipe)
    print(f"\nValidation: {'✅ PASS' if is_valid else '⚠️  WARNINGS'}")
    for w in warnings:
        print(f"  {w}")
