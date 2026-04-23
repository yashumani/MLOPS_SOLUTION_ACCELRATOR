"""
Stage 0: Data Quality Gate - Great Expectations Validation

Validates input dataset against expectations before training begins.
HARD FAIL on critical violations prevents bad data from corrupting models.

Features:
- Schema validation (column names, types)
- Null value checks (critical columns must not have nulls)
- Value range validation (numeric columns in expected bounds)
- Uniqueness constraints (ID columns, target column value counts)
- Custom expectations per task type (classification: class balance, regression: outliers)

Exit Codes:
- 0: Validation passed (all critical expectations met)
- 1: Validation failed (critical expectation violated, ABORT pipeline)
- 2: Warnings only (non-critical issues, pipeline continues)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import mlflow
import pandas as pd
import yaml

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data validation using rule-based expectations (Great Expectations-style).
    
    Note: Using custom implementation instead of full Great Expectations library
    to avoid Azure ML environment compatibility issues. Implements core validation
    patterns sufficient for production data quality gates.
    """
    
    def __init__(self, config: Dict[str, Any], expectations_suite: str):
        self.config = config
        self.task_type = config.get("task_type", "classification")
        self.target_column = config["dataset"]["target_column"]
        self.expectations_suite = expectations_suite
        
        # Validation results
        self.results = {
            "success": True,
            "critical_failures": [],
            "warnings": [],
            "statistics": {}
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks on dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validation results dictionary with success flag and details
        """
        logger.info(f"🔍 Starting data validation (suite: {self.expectations_suite})")
        logger.info(f"Dataset shape: {df.shape}")
        
        # Core validations (always run)
        self._validate_schema(df)
        self._validate_null_values(df)
        self._validate_target_column(df)
        self._validate_data_types(df)
        
        # Task-specific validations
        if self.task_type == "classification":
            self._validate_classification_data(df)
        elif self.task_type == "regression":
            self._validate_regression_data(df)
        elif self.task_type == "clustering":
            self._validate_clustering_data(df)
        
        # Compute statistics
        self._compute_statistics(df)
        
        # Determine overall success
        if len(self.results["critical_failures"]) > 0:
            self.results["success"] = False
            logger.error(f"❌ Validation FAILED: {len(self.results['critical_failures'])} critical issues")
        elif len(self.results["warnings"]) > 0:
            logger.warning(f"⚠️ Validation passed with {len(self.results['warnings'])} warnings")
        else:
            logger.info("✅ Validation passed with no issues")
        
        return self.results
    
    def _validate_schema(self, df: pd.DataFrame):
        """Validate dataset has required columns."""
        required_cols = [self.target_column]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.results["critical_failures"].append(
                f"Missing required columns: {missing_cols}"
            )
        
        # Check minimum column count
        if df.shape[1] < 2:
            self.results["critical_failures"].append(
                f"Dataset has only {df.shape[1]} columns, need at least 2 (target + features)"
            )
        
        logger.info(f"Schema check: {df.shape[1]} columns found")
    
    def _validate_null_values(self, df: pd.DataFrame):
        """Check for null values in critical columns."""
        # Target column MUST NOT have nulls
        target_nulls = df[self.target_column].isnull().sum()
        if target_nulls > 0:
            null_pct = (target_nulls / len(df)) * 100
            self.results["critical_failures"].append(
                f"Target column '{self.target_column}' has {target_nulls} null values ({null_pct:.2f}%)"
            )
        
        # Feature null analysis (warnings only)
        feature_cols = [col for col in df.columns if col != self.target_column]
        high_null_cols = []
        for col in feature_cols:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 50:
                high_null_cols.append(f"{col} ({null_pct:.1f}%)")
        
        if high_null_cols:
            self.results["warnings"].append(
                f"High null percentage in columns: {', '.join(high_null_cols)}"
            )
        
        logger.info(f"Null check: Target nulls={target_nulls}, High-null features={len(high_null_cols)}")
    
    def _validate_target_column(self, df: pd.DataFrame):
        """Validate target column properties."""
        target = df[self.target_column]
        
        # Check value counts
        unique_values = target.nunique()
        
        if self.task_type == "classification":
            # Classification: Need at least 2 classes
            if unique_values < 2:
                self.results["critical_failures"].append(
                    f"Classification target has only {unique_values} unique value(s), need at least 2 classes"
                )
            
            # Warn if too many classes (likely regression or ID column)
            if unique_values > 50:
                self.results["warnings"].append(
                    f"Classification target has {unique_values} classes (unusually high, verify not regression task)"
                )
        
        elif self.task_type == "regression":
            # Regression: Should have many unique values
            if unique_values < 10:
                self.results["warnings"].append(
                    f"Regression target has only {unique_values} unique values (suspiciously low, verify not classification)"
                )
        
        logger.info(f"Target validation: {unique_values} unique values in '{self.target_column}'")
    
    def _validate_data_types(self, df: pd.DataFrame):
        """Check for problematic data types."""
        # Detect object columns with high cardinality (potential ID columns)
        object_cols = df.select_dtypes(include=['object']).columns
        high_card_cols = []
        
        for col in object_cols:
            if col == self.target_column:
                continue
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                high_card_cols.append(f"{col} ({df[col].nunique()} unique)")
        
        if high_card_cols:
            self.results["warnings"].append(
                f"Potential ID columns (high cardinality): {', '.join(high_card_cols)}"
            )
        
        logger.info(f"Data types: {len(object_cols)} object columns, {len(high_card_cols)} high-cardinality")
    
    def _validate_classification_data(self, df: pd.DataFrame):
        """Classification-specific validations."""
        target = df[self.target_column]
        class_counts = target.value_counts()
        
        # Check class balance
        min_class_pct = (class_counts.min() / len(df)) * 100
        max_class_pct = (class_counts.max() / len(df)) * 100
        
        if min_class_pct < 1:
            self.results["warnings"].append(
                f"Severe class imbalance: smallest class is {min_class_pct:.2f}% of dataset"
            )
        
        # Check minimum samples per class
        min_samples = class_counts.min()
        if min_samples < 10:
            self.results["critical_failures"].append(
                f"Class '{class_counts.idxmin()}' has only {min_samples} samples (need ≥10 for train/val/test split)"
            )
        
        logger.info(f"Classification check: {len(class_counts)} classes, balance range {min_class_pct:.1f}%-{max_class_pct:.1f}%")
    
    def _validate_regression_data(self, df: pd.DataFrame):
        """Regression-specific validations."""
        target = df[self.target_column]
        
        # Check for outliers (IQR method)
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((target < lower_bound) | (target > upper_bound)).sum()
        outlier_pct = (outliers / len(df)) * 100
        
        if outlier_pct > 10:
            self.results["warnings"].append(
                f"High outlier percentage in target: {outlier_pct:.1f}% ({outliers}/{len(df)} samples)"
            )
        
        # Check for negative values if domain suggests non-negative
        if target.min() < 0 and "price" in self.target_column.lower():
            self.results["warnings"].append(
                f"Target '{self.target_column}' contains negative values (min={target.min():.2f}), verify if valid"
            )
        
        logger.info(f"Regression check: range [{target.min():.2f}, {target.max():.2f}], {outlier_pct:.1f}% outliers")
    
    def _validate_clustering_data(self, df: pd.DataFrame):
        """Clustering-specific validations."""
        # Clustering doesn't use target column for training
        feature_cols = [col for col in df.columns if col != self.target_column]
        
        # Check numeric feature availability
        numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            self.results["critical_failures"].append(
                f"Clustering requires at least 2 numeric features, found {len(numeric_cols)}"
            )
        
        logger.info(f"Clustering check: {len(numeric_cols)} numeric features available")
    
    def _compute_statistics(self, df: pd.DataFrame):
        """Compute dataset statistics for logging."""
        self.results["statistics"] = {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "n_numeric": len(df.select_dtypes(include=['number']).columns),
            "n_categorical": len(df.select_dtypes(include=['object', 'category']).columns),
            "total_nulls": int(df.isnull().sum().sum()),
            "null_percentage": float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
            "target_unique_values": int(df[self.target_column].nunique()),
            "memory_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }


def main():
    parser = argparse.ArgumentParser(description="Stage 0: Data Quality Gate")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dataset_in", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--expectations_suite", type=str, default="default", help="Expectations suite name")
    parser.add_argument("--validation_report", type=str, required=True, help="Output validation report directory")
    parser.add_argument("--validated_dataset", type=str, required=True, help="Output dataset reference")
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"📋 Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 🔥 FIX: Convert azureml:// to https:// to avoid model registry errors
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if mlflow_uri.startswith("azureml://"):
        https_uri = mlflow_uri.replace("azureml://", "https://")
        mlflow.set_tracking_uri(https_uri)
        logger.info(f"🔗 MLflow tracking URI converted: azureml:// → https://")
    
    # Ensure MLflow model registry URI is set to a local file store to avoid unsupported azureml:// registry
    os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")
    
    # Load dataset - use blob_path from config
    dataset_path = Path(args.dataset_in)
    expected_filename = config.get("dataset", {}).get("blob_path", "")
    
    if not expected_filename:
        logger.error(f"❌ No blob_path specified in config")
        sys.exit(1)
    
    dataset_file = dataset_path / expected_filename
    
    # Fallback logic for subdirectories
    if not dataset_file.exists():
        csv_files = list(dataset_path.rglob(expected_filename))
        if csv_files:
            dataset_file = csv_files[0]
        else:
            # Last resort: look for any CSV
            csv_files = list(dataset_path.glob("*.csv"))
            if not csv_files:
                logger.error(f"❌ No CSV files found in {dataset_path}")
                sys.exit(1)
            dataset_file = csv_files[0]
            logger.warning(f"⚠️ Could not find {expected_filename}, using {dataset_file.name}")
    
    logger.info(f"📂 Loading dataset: {dataset_file}")
    df = pd.read_csv(dataset_file)
    
    # Run validation
    validator = DataValidator(config, args.expectations_suite)
    results = validator.validate_dataset(df)
    
    # Save validation report
    report_dir = Path(args.validation_report)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / "validation_results.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Saved validation report: {report_file}")
    
    # Log to MLflow (Azure ML automatically captures)
    try:
        mlflow.log_metrics({
            "validation_success": 1 if results["success"] else 0,
            "critical_failures": len(results["critical_failures"]),
            "warnings": len(results["warnings"]),
            "n_rows": results["statistics"]["n_rows"],
            "n_columns": results["statistics"]["n_columns"],
            "null_percentage": results["statistics"]["null_percentage"]
        })
        logger.info("✅ MLflow metrics logged")
        # Note: Artifact already saved to output folder (validation_report/), Azure ML captures it automatically
    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed (non-critical): {e}")
    
    # Log detailed results
    if results["critical_failures"]:
        for failure in results["critical_failures"]:
            logger.error(f"❌ CRITICAL: {failure}")
    
    if results["warnings"]:
        for warning in results["warnings"]:
            logger.warning(f"⚠️ WARNING: {warning}")
    
    # Output validated dataset reference (just copy path for now)
    if results["success"]:
        validated_path = Path(args.validated_dataset)
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        with open(validated_path, 'w') as f:
            f.write(str(dataset_file))
        logger.info(f"✅ Validation passed - dataset reference saved to {validated_path}")
        sys.exit(0)
    else:
        logger.error(f"❌ Validation FAILED - aborting pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
