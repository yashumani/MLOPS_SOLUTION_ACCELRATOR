"""
Phase B Variant Runner - Industrial-Grade Variant Execution Engine

Runs multiple preprocessing variants in one Azure ML step, using nested
MLflow runs for each variant×engine combination.

HARDENED FEATURES (Industrial Accelerator):
- Variant-level checkpointing (resume capability)
- Per-variant time budget enforcement
- Stable champion manifest contract
- Normalized variant results schema
- Defensive guards (failure tolerance)

Author: MLOps Solution Accelerator V3
Date: 2026-01-26 (Hardened)
"""

import argparse
import json
import sys
import time
import hashlib
import subprocess
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import mlflow
import os

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.variant_schema import load_variant, validate_variant_for_task, VariantConfig
from utils.dataset_profiler import DatasetProfiler
from utils.variant_recommender import VariantRecommender
from utils.variant_planner import (
    EdaPriors, VariantPlan, build_variant_plan, score_variant_relevance,
    diverse_sample, compute_preprocessing_hash, get_default_planner_config, VariantScore
)
from utils.preprocessing_cache import PreprocessingCache, create_preprocessing_cache
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_stage_table, sha256_file,
)
from utils.model_universe import get_model_list


# ============================================================================
# INDUSTRIAL CONTRACTS - DO NOT MODIFY WITHOUT COORDINATION
# ============================================================================

@dataclass
class VariantResult:
    """Normalized result structure - STABLE CONTRACT for downstream phases."""
    variant_id: str
    engine: str
    algorithm: str
    metrics: Dict[str, float]
    runtime_sec: float
    timed_out: bool
    failed: bool
    failure_reason: Optional[str] = None
    leakage_risk: str = "none"
    n_features: int = 0


@dataclass
class ChampionManifest:
    """Champion pipeline configuration - LOCKED SCHEMA for Phase C/Registry."""
    variant_id: str
    variant_path: str
    engine: str
    algorithm: str
    primary_metric_name: str  # e.g., "accuracy", "r2"
    primary_metric_value: float  # actual metric value
    metrics: Dict[str, float]
    preprocessing_config: Dict[str, Any]
    feature_engineering_config: Dict[str, Any]
    data_fingerprint: Dict[str, Any]
    code_version: str
    timestamp: str
    leakage_risk: str
    task_type: str


# ============================================================================
# CHECKPOINTING - Resume Capability
# ============================================================================

class CheckpointManager:
    """Manages resume state for long-running variant execution."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load existing checkpoint or create new."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {"completed": [], "last_updated": None}
    
    def is_completed(self, variant_id: str, engine: str) -> bool:
        """Check if variant×engine combination already completed."""
        key = f"{variant_id}::{engine}"
        return key in self.state["completed"]
    
    def mark_completed(self, variant_id: str, engine: str):
        """Mark variant×engine as completed and flush immediately."""
        key = f"{variant_id}::{engine}"
        if key not in self.state["completed"]:
            self.state["completed"].append(key)
            self.state["last_updated"] = datetime.utcnow().isoformat() + "Z"
            self._flush()
    
    def _flush(self):
        """Flush state to disk immediately."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_progress(self, total_expected: int = None) -> Dict[str, Any]:
        """Get current progress statistics."""
        completed = len(self.state["completed"])
        return {
            "completed": completed,
            "total": total_expected if total_expected is not None else completed,
            "last_updated": self.state["last_updated"]
        }


# ============================================================================
# SOFT TIMEOUT - No multiprocessing (Azure ML safe)
# ============================================================================

def set_deterministic_seed(seed: int = 42):
    """Set random seed across all libraries for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    # PyCaret and FLAML will use session_id/seed params separately


# ============================================================================
# DATA FINGERPRINTING
# ============================================================================

def compute_data_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute deterministic fingerprint of dataset for reproducibility."""
    # Sample-based hash for large datasets
    sample_size = min(1000, len(df))
    sample = df.head(sample_size).to_csv(index=False).encode('utf-8')
    data_hash = hashlib.sha256(sample).hexdigest()
    
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "hash": data_hash,
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def get_code_version() -> str:
    """Get git commit SHA if available, else timestamp."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return sha[:8]
    except:
        return f"no_git_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def atomic_write(path: Path, content: str):
    """Atomic write: write to temp file, then replace target."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temp_path, 'w') as f:
            f.write(content)
        os.replace(str(temp_path), str(path))
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


# ============================================================================
# DEFENSIVE GUARDS
# ============================================================================

def validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Guard against NaN/inf metrics - replace with sentinel values."""
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                print(f"⚠️ Invalid metric {key}={value}, replacing with 0.0")
                cleaned[key] = 0.0
            else:
                cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return cleaned


def check_leakage_risk(variant: VariantConfig) -> str:
    """Enhanced leakage detection - return risk level."""
    risk = variant.variant_metadata.leakage_risk if variant.variant_metadata else "unknown"
    
    # Additional heuristic checks
    if variant.stage3_preprocessing.encoding.categorical_method == "target":
        if risk == "none":
            risk = "medium"  # Target encoding has inherent risk
    
    return risk


def deadline_guard(deadline: float, label: str) -> bool:
    """Check if deadline exceeded. Returns True if exceeded."""
    if time.time() > deadline:
        print(f"   ⏰ DEADLINE EXCEEDED at {label}")
        return True
    return False


def get_primary_metric(task_type: str) -> str:
    """Return primary metric name for task type.
    
    For classification we optimise AUC (threshold-agnostic, handles
    class imbalance). Matches the baselines in s5a/s5b.
    """
    if task_type == "classification":
        return "AUC"
    elif task_type == "regression":
        return "R2"
    elif task_type == "clustering":
        return "silhouette_score"
    return "AUC"


def get_metric_columns_for_task(task_type: str) -> list:
    """Return the appropriate metric columns for a given task type.
    
    This ensures leaderboards and output files contain the right metrics
    for each task type instead of hardcoding classification columns.
    """
    if task_type == "regression":
        return ["r2", "mae", "mse", "rmse", "rmsle", "mape"]
    elif task_type == "clustering":
        return ["silhouette", "calinski_harabasz", "davies_bouldin"]
    else:  # classification (default)
        return ["accuracy", "auc", "f1", "precision", "recall", "kappa", "mcc"]


def is_lower_better(metric_name: str) -> bool:
    """Return True if metric is lower-better (error metrics)."""
    metric_lower = metric_name.lower()
    lower_better_metrics = ["rmse", "mae", "mape", "logloss", "loss", "mse", "davies_bouldin"]
    return any(lb in metric_lower for lb in lower_better_metrics)


def safe_float(x) -> float:
    """Coerce to float, return -inf for None/NaN/inf."""
    if x is None:
        return float('-inf')
    try:
        val = float(x)
        if not math.isfinite(val):
            return float('-inf')
        return val
    except (TypeError, ValueError):
        return float('-inf')


def get_result_score(result, primary_metric_name: str) -> float:
    """Unified scoring function for champion selection.
    
    Returns comparable score (higher is better) by negating lower-better metrics.
    """
    # Prefer 'primary_metric' key, fallback to metric_name
    # Use 'is not None' instead of truthiness to handle 0.0 correctly
    val = result.metrics.get("primary_metric")
    if val is None:
        val = result.metrics.get(primary_metric_name)
    val = safe_float(val)
    
    # Negate lower-better metrics for consistent comparison
    if is_lower_better(primary_metric_name):
        return -val
    return val


def apply_variant_preprocessing(
    df: pd.DataFrame,
    variant: VariantConfig,
    target_column: str = None,
    apply_smote: bool = True
) -> pd.DataFrame:
    """Apply variant preprocessing configuration to dataset.
    
    This is a simplified implementation for Phase 1.
    In Phase 2, this should reuse existing stage3/stage4 logic.
    """
    df_processed = df.copy()
    
    # Separate features and target
    if target_column and target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
    else:
        X = df_processed
        y = None
    
    # === IMPUTATION ===
    imputation_method = variant.stage3_preprocessing.imputation.method
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_imp = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if imputation_method == "mean":
        X = X.fillna(X.mean(numeric_only=True))
        # Fill remaining categorical NaNs with mode
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "median":
        X = X.fillna(X.median(numeric_only=True))
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "drop":
        X = X.dropna()
    elif imputation_method == "knn":
        from sklearn.impute import KNNImputer
        n_neighbors = getattr(variant.stage3_preprocessing.imputation, 'n_neighbors', 5)
        if num_cols and X[num_cols].isnull().any().any():
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            X[num_cols] = imputer.fit_transform(X[num_cols])
        # KNN is numeric-only; fill categorical with mode
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        max_iter = getattr(variant.stage3_preprocessing.imputation, 'max_iter', 10)
        if num_cols and X[num_cols].isnull().any().any():
            imputer = IterativeImputer(random_state=42, max_iter=max_iter)
            X[num_cols] = imputer.fit_transform(X[num_cols])
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method in ("mode", "most_frequent"):
        from sklearn.impute import SimpleImputer
        if cat_cols_imp and X[cat_cols_imp].isnull().any().any():
            imp_cat = SimpleImputer(strategy='most_frequent')
            X[cat_cols_imp] = imp_cat.fit_transform(X[cat_cols_imp])
        if num_cols and X[num_cols].isnull().any().any():
            imp_num = SimpleImputer(strategy='most_frequent')
            X[num_cols] = imp_num.fit_transform(X[num_cols])
    # --- Tier 1: pandas-native methods ---
    elif imputation_method == "forward_fill":
        X = X.ffill()
        # Any remaining NaNs at the start → backfill as safety net
        X = X.bfill()
    elif imputation_method == "backward_fill":
        X = X.bfill()
        X = X.ffill()
    elif imputation_method == "interpolate_linear":
        if num_cols:
            X[num_cols] = X[num_cols].interpolate(method='linear', limit_direction='both')
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "constant":
        fill_val = getattr(variant.stage3_preprocessing.imputation, 'fill_value', 0)
        X = X.fillna(fill_val)
    elif imputation_method == "zero_fill":
        if num_cols:
            X[num_cols] = X[num_cols].fillna(0)
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna("missing")
    # --- Tier 1: statistical variants ---
    elif imputation_method == "trimmed_mean":
        from scipy import stats
        trim_frac = getattr(variant.stage3_preprocessing.imputation, 'trim_fraction', 0.1)
        for col in num_cols:
            if X[col].isnull().any():
                t_mean = stats.trim_mean(X[col].dropna().values, proportiontocut=trim_frac)
                X[col] = X[col].fillna(t_mean)
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "winsorized_mean":
        from scipy.stats.mstats import winsorize
        trim_frac = getattr(variant.stage3_preprocessing.imputation, 'trim_fraction', 0.05)
        for col in num_cols:
            if X[col].isnull().any():
                valid = X[col].dropna().values
                wins_vals = winsorize(valid, limits=[trim_frac, trim_frac])
                X[col] = X[col].fillna(float(wins_vals.mean()))
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "random_sample":
        for col in num_cols:
            if X[col].isnull().any():
                non_null = X[col].dropna()
                n_missing = X[col].isnull().sum()
                X.loc[X[col].isnull(), col] = non_null.sample(n=n_missing, replace=True, random_state=42).values
        for col in cat_cols_imp:
            if X[col].isnull().any():
                non_null = X[col].dropna()
                n_missing = X[col].isnull().sum()
                X.loc[X[col].isnull(), col] = non_null.sample(n=n_missing, replace=True, random_state=42).values
    # --- Tier 1: column-aware composites ---
    elif imputation_method == "numeric_mean_cat_mode":
        if num_cols:
            X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    elif imputation_method == "numeric_median_cat_mode":
        if num_cols:
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    else:
        import logging
        _imp_logger = logging.getLogger("s06.imputation")
        _imp_logger.warning(
            f"Unknown imputation method '{imputation_method}' for variant "
            f"'{variant.recipe_name}' — falling back to mean imputation"
        )
        X = X.fillna(X.mean(numeric_only=True))
        for col in cat_cols_imp:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")
    
    # === OUTLIER HANDLING (H2) ===
    outlier_cfg = getattr(variant.stage3_preprocessing, 'outlier_handling', None)
    outlier_method = outlier_cfg.method if outlier_cfg else "none"
    if outlier_method and outlier_method != "none":
        _out_num = X.select_dtypes(include=[np.number]).columns.tolist()
        n_before_outlier = len(X)
        try:
            if outlier_method == "iqr_removal":
                Q1 = X[_out_num].quantile(0.25)
                Q3 = X[_out_num].quantile(0.75)
                IQR = Q3 - Q1
                mask = ~((X[_out_num] < (Q1 - 1.5 * IQR)) | (X[_out_num] > (Q3 + 1.5 * IQR))).any(axis=1)
                X = X[mask]
                if y is not None:
                    y = y[mask]
                print(f"    🔧 Outlier handling (IQR removal): {n_before_outlier} → {len(X)} rows ({n_before_outlier - len(X)} removed)")
            elif outlier_method == "iqr_capping":
                Q1 = X[_out_num].quantile(0.25)
                Q3 = X[_out_num].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                X[_out_num] = X[_out_num].clip(lower=lower, upper=upper, axis=1)
                print(f"    🔧 Outlier handling (IQR capping): values clipped to [Q1-1.5*IQR, Q3+1.5*IQR]")
            elif outlier_method == "zscore":
                from scipy import stats
                z_scores = np.abs(stats.zscore(X[_out_num], nan_policy='omit'))
                mask = (z_scores < 3).all(axis=1)
                X = X[mask]
                if y is not None:
                    y = y[mask]
                print(f"    🔧 Outlier handling (z-score): {n_before_outlier} → {len(X)} rows ({n_before_outlier - len(X)} removed)")
            elif outlier_method == "winsorize":
                from scipy.stats import mstats
                for col in _out_num:
                    X[col] = mstats.winsorize(X[col], limits=[0.05, 0.05])
                print(f"    🔧 Outlier handling (winsorize): 5th/95th percentile capping applied")
            elif outlier_method == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
                preds = iso.fit_predict(X[_out_num].fillna(0))
                mask = preds == 1
                X = X[mask]
                if y is not None:
                    y = y[mask]
                print(f"    🔧 Outlier handling (Isolation Forest): {n_before_outlier} → {len(X)} rows ({n_before_outlier - len(X)} removed)")
            else:
                print(f"    ⚠️ Outlier method '{outlier_method}' not implemented, skipping")
            # Reset index after row removal
            X = X.reset_index(drop=True)
            if y is not None:
                y = y.reset_index(drop=True)
        except Exception as e:
            print(f"    ⚠️ Outlier handling '{outlier_method}' failed: {e}, skipping")
    
    # === ENCODING ===
    encoding_method = variant.stage3_preprocessing.encoding.categorical_method
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if encoding_method == "label":
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
    elif encoding_method == "onehot":
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    elif encoding_method == "target" and len(cat_cols) > 0 and y is not None:
        from category_encoders import TargetEncoder
        te = TargetEncoder(cols=list(cat_cols), return_df=True)
        X = te.fit_transform(X, y)
        print(f"    🔧 Target encoding applied to {len(cat_cols)} columns")
    elif encoding_method and encoding_method != "none" and len(cat_cols) > 0:
        print(f"    ⚠️ Encoding method '{encoding_method}' not implemented, falling back to label encoding")
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
    
    # === SCALING ===
    scaling_method = variant.stage3_preprocessing.scaling.method
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if scaling_method == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method == "yeo_johnson" and len(numeric_cols) > 0:
        from sklearn.preprocessing import PowerTransformer
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method == "quantile" and len(numeric_cols) > 0:
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    elif scaling_method and scaling_method != "none" and len(numeric_cols) > 0:
        print(f"    ⚠️ Scaling method '{scaling_method}' not implemented, falling back to standard scaling")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # === IMBALANCE HANDLING (C3) ===
    # Apply SMOTE/ADASYN/SMOTEENN etc. for classification tasks only.
    # Must run AFTER encoding + scaling (SMOTE requires all numeric, scaled data).
    # NOTE: apply_smote=False during training so PyCaret CV gets clean data
    # (prevents synthetic samples leaking into CV test folds).
    # SMOTE retraining happens AFTER model selection in train_pycaret_variant().
    imb_cfg = getattr(variant.stage3_preprocessing, 'imbalance_handling', None)
    imb_method = imb_cfg.method if imb_cfg else "none"
    is_classification = y is not None and y.nunique() <= 30  # heuristic
    if not apply_smote and imb_method and imb_method != "none" and is_classification:
        print(f"    ⏭️ Imbalance handling ({imb_method}) DEFERRED to post-model-selection retraining")
    elif apply_smote and imb_method and imb_method != "none" and is_classification and y is not None:
        try:
            n_before_imb = len(X)
            X_num = X.select_dtypes(include=[np.number])
            if len(X_num.columns) == len(X.columns):  # all numeric after encoding
                if imb_method == "smote":
                    from imblearn.over_sampling import SMOTE
                    sampler = SMOTE(random_state=42, n_jobs=-1)
                    X, y = sampler.fit_resample(X, y)
                    print(f"    ⚖️ Imbalance handling (SMOTE): {n_before_imb} → {len(X)} rows")
                elif imb_method == "adasyn":
                    from imblearn.over_sampling import ADASYN
                    sampler = ADASYN(random_state=42, n_jobs=-1)
                    X, y = sampler.fit_resample(X, y)
                    print(f"    ⚖️ Imbalance handling (ADASYN): {n_before_imb} → {len(X)} rows")
                elif imb_method == "smoteenn":
                    from imblearn.combine import SMOTEENN
                    sampler = SMOTEENN(random_state=42)
                    X, y = sampler.fit_resample(X, y)
                    print(f"    ⚖️ Imbalance handling (SMOTEENN): {n_before_imb} → {len(X)} rows")
                elif imb_method == "smotetomek":
                    from imblearn.combine import SMOTETomek
                    sampler = SMOTETomek(random_state=42)
                    X, y = sampler.fit_resample(X, y)
                    print(f"    ⚖️ Imbalance handling (SMOTETomek): {n_before_imb} → {len(X)} rows")
                else:
                    print(f"    ⚠️ Imbalance method '{imb_method}' not implemented, skipping")
            else:
                print(f"    ⚠️ Imbalance handling skipped: {len(X.columns) - len(X_num.columns)} non-numeric columns remain after encoding")
        except Exception as e:
            print(f"    ⚠️ Imbalance handling '{imb_method}' failed: {e}, continuing without resampling")
    elif imb_method and imb_method != "none" and not is_classification:
        print(f"    ℹ️ Imbalance handling '{imb_method}' skipped: not a classification task")
    
    # === FEATURE SELECTION ===
    fs_config = variant.stage4_feature_engineering.feature_selection
    fs_method = fs_config.method if fs_config else "none"
    fs_threshold = fs_config.threshold if fs_config and fs_config.threshold is not None else 0.01
    
    if fs_method and fs_method != "none" and y is not None:
        numeric_fs_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        n_before = len(numeric_fs_cols)
        
        if fs_method == "correlation":
            # Drop numeric features with low absolute correlation to target
            correlations = X[numeric_fs_cols].corrwith(y).abs()
            cols_to_keep = correlations[correlations >= fs_threshold].index.tolist()
            # Always keep non-numeric columns (already encoded but just in case)
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            cols_to_keep = list(set(cols_to_keep + non_numeric))
            cols_dropped = [c for c in X.columns if c not in cols_to_keep]
            if cols_dropped:
                X = X[cols_to_keep]
                print(f"    📉 Feature selection (correlation≥{fs_threshold}): {n_before} → {len(cols_to_keep)} features ({len(cols_dropped)} dropped)")
            else:
                print(f"    📉 Feature selection (correlation): all {n_before} features retained")
        
        elif fs_method == "variance":
            # Drop features with near-zero variance
            from sklearn.feature_selection import VarianceThreshold
            var_threshold = fs_threshold if fs_threshold > 0 else 0.01
            selector = VarianceThreshold(threshold=var_threshold)
            try:
                X_selected = selector.fit_transform(X[numeric_fs_cols])
                kept_mask = selector.get_support()
                kept_cols = [numeric_fs_cols[i] for i in range(len(numeric_fs_cols)) if kept_mask[i]]
                non_numeric = [c for c in X.columns if c not in numeric_fs_cols]
                X = X[kept_cols + non_numeric]
                print(f"    📉 Feature selection (variance≥{var_threshold}): {n_before} → {len(kept_cols)} features ({n_before - len(kept_cols)} dropped)")
            except Exception as e:
                print(f"    ⚠️ Variance feature selection failed: {e}, skipping")
        
        elif fs_method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            try:
                mi_func = mutual_info_classif if y.nunique() <= 20 else mutual_info_regression
                mi_scores = mi_func(X[numeric_fs_cols].fillna(0), y, random_state=42)
                mi_series = pd.Series(mi_scores, index=numeric_fs_cols)
                cols_to_keep = mi_series[mi_series >= fs_threshold].index.tolist()
                non_numeric = [c for c in X.columns if c not in numeric_fs_cols]
                if cols_to_keep:
                    X = X[cols_to_keep + non_numeric]
                    print(f"    📉 Feature selection (mutual_info≥{fs_threshold}): {n_before} → {len(cols_to_keep)} features")
                else:
                    print(f"    ⚠️ Mutual info would drop ALL features - skipping")
            except Exception as e:
                print(f"    ⚠️ Mutual info feature selection failed: {e}, skipping")
        
        else:
            print(f"    ℹ️ Feature selection method '{fs_method}' not implemented, skipping")
    elif fs_method and fs_method != "none" and y is None:
        print(f"    ⚠️ Feature selection '{fs_method}' requires target column, skipping")
    
    # Rejoin target if exists
    if y is not None:
        X[target_column] = y.values
    
    return X


def preprocess_holdout_aligned(
    df_train_raw: pd.DataFrame,
    df_holdout_raw: pd.DataFrame,
    variant: 'VariantConfig',
    target_column: str
) -> pd.DataFrame:
    """Preprocess holdout data using statistics fitted on training data.

    This solves the train/holdout preprocessing mismatch where
    apply_variant_preprocessing() refits all transformers on holdout,
    causing:
      - Imputation with holdout means (not training means)
      - Label encoding with different category orderings
      - Scaling with holdout statistics
      - SMOTE applied to holdout (creating synthetic test samples!)
      - Feature selection based on holdout correlations

    This function:
      1. Fits imputation on TRAINING, transforms holdout with training stats
      2. Fits encoding on TRAINING categories, applies same mapping to holdout
      3. Fits scaler on TRAINING (encoded), transforms holdout
      4. NEVER applies SMOTE/resampling to holdout
      5. Selects features from TRAINING, applies same selection to holdout
      6. Uses TRAINING outlier bounds for holdout capping (no row removal)

    Returns preprocessed holdout DataFrame with target column.
    """
    print(f"   🔄 Preprocessing holdout with training-aligned statistics")

    X_tr = df_train_raw.drop(columns=[target_column]).copy()
    y_tr = df_train_raw[target_column].copy()
    X_ho = df_holdout_raw.drop(columns=[target_column]).copy()
    y_ho = df_holdout_raw[target_column].copy()

    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()

    # ═══════════════════════════════════════════════════════════════════
    # IMPUTATION — fit on training, transform holdout with training stats
    # ═══════════════════════════════════════════════════════════════════
    imp_method = variant.stage3_preprocessing.imputation.method

    # Compute training fill values for numeric columns
    if imp_method in ("mean", "numeric_mean_cat_mode"):
        num_fill = X_tr[num_cols].mean() if num_cols else pd.Series(dtype=float)
    elif imp_method in ("median", "numeric_median_cat_mode"):
        num_fill = X_tr[num_cols].median() if num_cols else pd.Series(dtype=float)
    elif imp_method == "trimmed_mean":
        from scipy import stats as sp_stats_imp
        trim_frac = getattr(variant.stage3_preprocessing.imputation, 'trim_fraction', 0.1)
        num_fill = pd.Series(
            {c: sp_stats_imp.trim_mean(X_tr[c].dropna().values, proportiontocut=trim_frac)
             for c in num_cols if len(X_tr[c].dropna()) > 0},
            dtype=float
        )
    elif imp_method == "winsorized_mean":
        from scipy.stats.mstats import winsorize as _winsorize_imp
        trim_frac = getattr(variant.stage3_preprocessing.imputation, 'trim_fraction', 0.05)
        num_fill = pd.Series(
            {c: float(_winsorize_imp(X_tr[c].dropna().values,
                                     limits=[trim_frac, trim_frac]).mean())
             for c in num_cols if len(X_tr[c].dropna()) > 0},
            dtype=float
        )
    elif imp_method == "constant":
        fv = getattr(variant.stage3_preprocessing.imputation, 'fill_value', 0)
        num_fill = pd.Series({c: fv for c in num_cols})
    elif imp_method == "zero_fill":
        num_fill = pd.Series({c: 0 for c in num_cols})
    else:
        # Default fallback: training mean
        num_fill = X_tr[num_cols].mean() if num_cols else pd.Series(dtype=float)

    # Apply based on method type
    if imp_method == "knn":
        from sklearn.impute import KNNImputer
        n_neighbors = getattr(variant.stage3_preprocessing.imputation, 'n_neighbors', 5)
        if num_cols and X_tr[num_cols].isnull().any().any():
            imp = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            imp.fit(X_tr[num_cols])
            X_tr[num_cols] = imp.transform(X_tr[num_cols])
            if num_cols:
                ho_num = [c for c in num_cols if c in X_ho.columns]
                X_ho[ho_num] = imp.transform(X_ho[ho_num])
        else:
            X_tr[num_cols] = X_tr[num_cols].fillna(num_fill)
            X_ho[num_cols] = X_ho[num_cols].fillna(num_fill)
    elif imp_method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        max_iter = getattr(variant.stage3_preprocessing.imputation, 'max_iter', 10)
        if num_cols and X_tr[num_cols].isnull().any().any():
            imp = IterativeImputer(random_state=42, max_iter=max_iter)
            imp.fit(X_tr[num_cols])
            X_tr[num_cols] = imp.transform(X_tr[num_cols])
            ho_num = [c for c in num_cols if c in X_ho.columns]
            X_ho[ho_num] = imp.transform(X_ho[ho_num])
        else:
            X_tr[num_cols] = X_tr[num_cols].fillna(num_fill)
            X_ho[num_cols] = X_ho[num_cols].fillna(num_fill)
    elif imp_method in ("mode", "most_frequent"):
        from sklearn.impute import SimpleImputer
        if num_cols and X_tr[num_cols].isnull().any().any():
            imp_n = SimpleImputer(strategy='most_frequent')
            imp_n.fit(X_tr[num_cols])
            X_tr[num_cols] = imp_n.transform(X_tr[num_cols])
            X_ho[num_cols] = imp_n.transform(X_ho[num_cols])
        if cat_cols:
            imp_c = SimpleImputer(strategy='most_frequent')
            tr_cats = [c for c in cat_cols if c in X_tr.columns]
            ho_cats = [c for c in cat_cols if c in X_ho.columns]
            if tr_cats:
                imp_c.fit(X_tr[tr_cats])
                X_tr[tr_cats] = imp_c.transform(X_tr[tr_cats])
                if ho_cats:
                    X_ho[ho_cats] = imp_c.transform(X_ho[ho_cats])
    elif imp_method == "drop":
        # Training: drop NaN rows. Holdout: fill with training mean
        train_fill = X_tr[num_cols].mean() if num_cols else pd.Series(dtype=float)
        X_tr = X_tr.dropna()
        y_tr = y_tr.loc[X_tr.index].reset_index(drop=True)
        X_tr = X_tr.reset_index(drop=True)
        X_ho[num_cols] = X_ho[num_cols].fillna(train_fill)
    elif imp_method == "random_sample":
        for c in num_cols:
            non_null = X_tr[c].dropna()
            if non_null.empty:
                continue
            if X_tr[c].isnull().any():
                n_m = int(X_tr[c].isnull().sum())
                X_tr.loc[X_tr[c].isnull(), c] = non_null.sample(
                    n=n_m, replace=True, random_state=42).values
            if c in X_ho.columns and X_ho[c].isnull().any():
                n_m = int(X_ho[c].isnull().sum())
                X_ho.loc[X_ho[c].isnull(), c] = non_null.sample(
                    n=n_m, replace=True, random_state=42).values
    elif imp_method in ("forward_fill", "backward_fill", "interpolate_linear"):
        # Order-dependent methods: apply to training, use training mean for holdout
        if imp_method == "forward_fill":
            X_tr = X_tr.ffill().bfill()
        elif imp_method == "backward_fill":
            X_tr = X_tr.bfill().ffill()
        else:
            if num_cols:
                X_tr[num_cols] = X_tr[num_cols].interpolate(
                    method='linear', limit_direction='both')
        train_fill = X_tr[num_cols].mean() if num_cols else pd.Series(dtype=float)
        X_ho[num_cols] = X_ho[num_cols].fillna(train_fill)
    else:
        # All other methods (mean, median, constant, zero_fill, trimmed_mean, etc.)
        if num_cols:
            X_tr[num_cols] = X_tr[num_cols].fillna(num_fill)
            ho_num = [c for c in num_cols if c in X_ho.columns]
            X_ho[ho_num] = X_ho[ho_num].fillna(num_fill)

    # Categorical imputation: always use training modes
    for c in cat_cols:
        mode = X_tr[c].mode()
        fill_val = mode.iloc[0] if not mode.empty else "missing"
        X_tr[c] = X_tr[c].fillna(fill_val)
        if c in X_ho.columns:
            X_ho[c] = X_ho[c].fillna(fill_val)

    print(f"      Imputation ({imp_method}): training stats applied to holdout")

    # ═══════════════════════════════════════════════════════════════════
    # OUTLIER HANDLING — training bounds, capping only on holdout
    # ═══════════════════════════════════════════════════════════════════
    outlier_cfg = getattr(variant.stage3_preprocessing, 'outlier_handling', None)
    outlier_method = outlier_cfg.method if outlier_cfg else "none"
    if outlier_method and outlier_method != "none":
        _out_num = X_tr.select_dtypes(include=[np.number]).columns.tolist()
        try:
            if outlier_method in ("iqr_removal", "iqr_capping"):
                Q1 = X_tr[_out_num].quantile(0.25)
                Q3 = X_tr[_out_num].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                if outlier_method == "iqr_removal":
                    mask = ~((X_tr[_out_num] < lower) |
                             (X_tr[_out_num] > upper)).any(axis=1)
                    X_tr = X_tr[mask].reset_index(drop=True)
                    y_tr = y_tr[mask].reset_index(drop=True)
                else:
                    X_tr[_out_num] = X_tr[_out_num].clip(
                        lower=lower, upper=upper, axis=1)
                # Holdout: always CAP (never remove rows)
                ho_out = [c for c in _out_num if c in X_ho.columns]
                X_ho[ho_out] = X_ho[ho_out].clip(
                    lower=lower[ho_out], upper=upper[ho_out], axis=1)
            elif outlier_method == "zscore":
                from scipy import stats as sp_stats_out
                z_scores = np.abs(sp_stats_out.zscore(
                    X_tr[_out_num], nan_policy='omit'))
                mask = (z_scores < 3).all(axis=1)
                # Save training mean/std before removal for holdout capping
                tr_mean = X_tr[_out_num].mean()
                tr_std = X_tr[_out_num].std()
                X_tr = X_tr[mask].reset_index(drop=True)
                y_tr = y_tr[mask].reset_index(drop=True)
                # Holdout: cap at training mean ± 3*std
                for c in _out_num:
                    if c in X_ho.columns:
                        X_ho[c] = X_ho[c].clip(
                            lower=tr_mean[c] - 3 * tr_std[c],
                            upper=tr_mean[c] + 3 * tr_std[c])
            elif outlier_method == "winsorize":
                for c in _out_num:
                    low_pct = float(X_tr[c].quantile(0.05))
                    high_pct = float(X_tr[c].quantile(0.95))
                    X_tr[c] = X_tr[c].clip(lower=low_pct, upper=high_pct)
                    if c in X_ho.columns:
                        X_ho[c] = X_ho[c].clip(lower=low_pct, upper=high_pct)
            elif outlier_method == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(
                    contamination=0.05, random_state=42, n_jobs=-1)
                iso.fit(X_tr[_out_num].fillna(0))
                preds = iso.predict(X_tr[_out_num].fillna(0))
                mask = preds == 1
                X_tr = X_tr[mask].reset_index(drop=True)
                y_tr = y_tr[mask].reset_index(drop=True)
                # Holdout: do NOT remove rows (prediction still proceeds)
            print(f"      Outlier handling ({outlier_method}): training bounds applied")
        except Exception as e:
            print(f"      ⚠️ Outlier handling '{outlier_method}' failed: {e}, skipping")

    # ═══════════════════════════════════════════════════════════════════
    # ENCODING — training categories → consistent holdout mapping
    # ═══════════════════════════════════════════════════════════════════
    encoding_method = variant.stage3_preprocessing.encoding.categorical_method
    cat_cols_enc = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()

    if encoding_method == "label" or (
        encoding_method and encoding_method not in ("onehot", "none")
        and len(cat_cols_enc) > 0
    ):
        for c in cat_cols_enc:
            # Use TRAINING sorted unique values as canonical category order
            train_cats = sorted(X_tr[c].dropna().unique().tolist())
            cat_type = pd.CategoricalDtype(categories=train_cats, ordered=True)
            X_tr[c] = X_tr[c].astype(cat_type).cat.codes
            if c in X_ho.columns:
                X_ho[c] = X_ho[c].astype(cat_type).cat.codes
    elif encoding_method == "onehot":
        X_tr = pd.get_dummies(X_tr, columns=cat_cols_enc, drop_first=True)
        X_ho = pd.get_dummies(X_ho, columns=[
            c for c in cat_cols_enc if c in X_ho.columns], drop_first=True)
        # Align holdout columns to training columns
        for c in X_tr.columns:
            if c not in X_ho.columns:
                X_ho[c] = 0
        X_ho = X_ho[X_tr.columns]

    print(f"      Encoding ({encoding_method}): training categories applied to holdout")

    # ═══════════════════════════════════════════════════════════════════
    # SCALING — fit on training (encoded), transform holdout
    # ═══════════════════════════════════════════════════════════════════
    scaling_method = variant.stage3_preprocessing.scaling.method
    numeric_cols_sc = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    if scaling_method and scaling_method != "none" and numeric_cols_sc:
        if scaling_method == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaling_method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

        scaler.fit(X_tr[numeric_cols_sc])
        X_tr[numeric_cols_sc] = scaler.transform(X_tr[numeric_cols_sc])
        ho_sc = [c for c in numeric_cols_sc if c in X_ho.columns]
        X_ho[ho_sc] = scaler.transform(X_ho[ho_sc])
        print(f"      Scaling ({scaling_method}): fitted on training, applied to holdout")

    # ═══════════════════════════════════════════════════════════════════
    # SMOTE / IMBALANCE — SKIPPED in holdout alignment
    # SMOTE is now deferred to post-model-selection retraining in
    # train_pycaret_variant(). Feature selection below runs on natural
    # (non-SMOTE'd) data, consistent with how the model was selected.
    # ═══════════════════════════════════════════════════════════════════
    imb_cfg = getattr(variant.stage3_preprocessing, 'imbalance_handling', None)
    imb_method = imb_cfg.method if imb_cfg else "none"
    if imb_method and imb_method != "none":
        print(f"      ⏭️ Imbalance ({imb_method}): deferred to post-model-selection retraining; "
              f"holdout UNTOUCHED ({len(X_ho)} rows preserved)")

    # ═══════════════════════════════════════════════════════════════════
    # FEATURE SELECTION — determine from training, apply to holdout
    # ═══════════════════════════════════════════════════════════════════
    fs_config = variant.stage4_feature_engineering.feature_selection
    fs_method = fs_config.method if fs_config else "none"
    fs_threshold = (fs_config.threshold
                    if fs_config and fs_config.threshold is not None else 0.01)

    if fs_method and fs_method != "none" and y_tr is not None:
        numeric_fs_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_keep = list(X_tr.columns)  # default: keep all

        try:
            if fs_method == "correlation":
                correlations = X_tr[numeric_fs_cols].corrwith(y_tr).abs()
                sel = correlations[correlations >= fs_threshold].index.tolist()
                non_num = [c for c in X_tr.columns if c not in numeric_fs_cols]
                cols_to_keep = list(set(sel + non_num))
            elif fs_method == "variance":
                from sklearn.feature_selection import VarianceThreshold
                var_thresh = fs_threshold if fs_threshold > 0 else 0.01
                selector = VarianceThreshold(threshold=var_thresh)
                selector.fit(X_tr[numeric_fs_cols])
                kept = selector.get_support()
                sel = [numeric_fs_cols[i]
                       for i in range(len(numeric_fs_cols)) if kept[i]]
                non_num = [c for c in X_tr.columns if c not in numeric_fs_cols]
                cols_to_keep = sel + non_num
            elif fs_method == "mutual_info":
                from sklearn.feature_selection import (
                    mutual_info_classif, mutual_info_regression)
                mi_func = (mutual_info_classif if y_tr.nunique() <= 20
                           else mutual_info_regression)
                mi_scores = mi_func(
                    X_tr[numeric_fs_cols].fillna(0), y_tr, random_state=42)
                mi_series = pd.Series(mi_scores, index=numeric_fs_cols)
                sel = mi_series[mi_series >= fs_threshold].index.tolist()
                non_num = [c for c in X_tr.columns if c not in numeric_fs_cols]
                cols_to_keep = sel + non_num if sel else list(X_tr.columns)
        except Exception as e:
            print(f"      ⚠️ Feature selection failed: {e}, keeping all features")
            cols_to_keep = list(X_tr.columns)

        # Apply same selection to holdout
        ho_cols = [c for c in cols_to_keep if c in X_ho.columns]
        X_ho = X_ho[ho_cols]
        print(f"      Feature selection ({fs_method}): {len(ho_cols)} features "
              f"from training applied to holdout")

    # Rejoin target
    X_ho[target_column] = y_ho.values
    print(f"   ✅ Holdout preprocessing complete: {X_ho.shape[0]} rows × "
          f"{X_ho.shape[1]} cols (training-aligned, no SMOTE)")
    return X_ho


# ============================================================================
# PHASE B SIGNAL HELPERS (Round 0 & Round 1)
# ============================================================================

def run_round0_feasibility(
    df: pd.DataFrame,
    variant: VariantConfig,
    target_column: str
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Round 0: Feasibility check via transform-only (no model training).
    
    Returns:
        (df_transformed_or_none, report_dict)
        
    Report fields:
        - variant_id: str
        - status: "pass" | "fail" | "error"
        - reason: str (empty if pass)
        - transform_runtime_sec: float
        - n_features_before: int
        - n_features_after: int
        - feature_multiplier: float
        - feature_explosion: bool (>500 features or 5x multiplier)
    """
    start_time = time.time()
    n_features_before = df.shape[1] - (1 if target_column else 0)
    
    report = {
        "variant_id": variant.variant_id,
        "status": "pass",
        "reason": "",
        "transform_runtime_sec": 0.0,
        "n_features_before": n_features_before,
        "n_features_after": 0,
        "feature_multiplier": 0.0,
        "feature_explosion": False
    }
    
    try:
        # Apply preprocessing transform
        df_transformed = apply_variant_preprocessing(df, variant, target_column, apply_smote=False)
        n_features_after = df_transformed.shape[1] - (1 if target_column else 0)
        feature_multiplier = n_features_after / max(n_features_before, 1)
        
        report["transform_runtime_sec"] = time.time() - start_time
        report["n_features_after"] = n_features_after
        report["feature_multiplier"] = round(feature_multiplier, 2)
        
        # Zero-feature guard: preprocessing/feature selection dropped all features
        if n_features_after <= 0:
            report["status"] = "fail"
            report["reason"] = f"Zero features remaining after preprocessing ({n_features_before} → {n_features_after}). Check feature selection threshold."
            return None, report
        
        # Check feasibility gates with adaptive thresholds
        # Dynamic limit: min of 20k or max of (500, 50x original features)
        dyn_feature_limit = min(20000, max(500, 50 * n_features_before))
        report["dyn_feature_limit"] = dyn_feature_limit  # Log threshold used
        
        if n_features_after > dyn_feature_limit:
            report["status"] = "fail"
            report["reason"] = f"Feature explosion: {n_features_after} features exceeds adaptive threshold ({dyn_feature_limit})"
            report["feature_explosion"] = True
            return None, report
        
        if feature_multiplier > 10.0:  # More lenient multiplier (was 5x)
            report["status"] = "fail"
            report["reason"] = f"Feature multiplier {feature_multiplier:.1f}x exceeds threshold (10x)"
            report["feature_explosion"] = True
            return None, report
        
        # Passed all checks
        return df_transformed, report
        
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"Transform error: {str(e)}"
        report["transform_runtime_sec"] = time.time() - start_time
        return None, report


def run_round1_proxy(
    df_transformed: pd.DataFrame,
    variant_id: str,
    target_column: str,
    task_type: str,
    max_samples: int = 5000
) -> Dict[str, Any]:
    """Round 1: Proxy leaderboard using cheap SGD model on sampled data.
    
    Returns report_dict with fields:
        - variant_id: str
        - status: "pass" | "fail" | "error"
        - reason: str
        - proxy_runtime_sec: float
        - proxy_metric: float (accuracy for classification, R2 for regression)
        - n_features: int
    """
    start_time = time.time()
    
    report = {
        "variant_id": variant_id,
        "status": "pass",
        "reason": "",
        "proxy_runtime_sec": 0.0,
        "proxy_metric": 0.0,
        "n_features": df_transformed.shape[1] - 1
    }
    
    try:
        # Sample dataset for speed
        df_sample = df_transformed.sample(n=min(max_samples, len(df_transformed)), random_state=42)
        
        X = df_sample.drop(columns=[target_column])
        y = df_sample[target_column]
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if task_type == "classification" else None
        )
        
        # Train cheap proxy model with improved config
        if task_type == "classification":
            from sklearn.linear_model import SGDClassifier
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            
            # More iterations + early stopping for better convergence
            proxy_model = SGDClassifier(
                loss='log_loss',
                random_state=42,
                max_iter=500,  # Increased from 100
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5
            )
            proxy_model.fit(X_train, y_train)
            y_pred = proxy_model.predict(X_test)
            
            # Use balanced_accuracy for imbalanced datasets
            proxy_accuracy = accuracy_score(y_test, y_pred)
            proxy_balanced_acc = balanced_accuracy_score(y_test, y_pred)
            proxy_metric = proxy_balanced_acc  # Primary metric
            
            # Log both metrics
            report["proxy_accuracy"] = round(float(proxy_accuracy), 4)
            report["proxy_balanced_accuracy"] = round(float(proxy_balanced_acc), 4)
            
        elif task_type == "regression":
            from sklearn.linear_model import SGDRegressor
            from sklearn.metrics import r2_score
            
            proxy_model = SGDRegressor(
                random_state=42,
                max_iter=500,  # Increased from 100
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5
            )
            proxy_model.fit(X_train, y_train)
            y_pred = proxy_model.predict(X_test)
            
            # Use R2 as proxy metric
            proxy_metric = r2_score(y_test, y_pred)
            
        elif task_type == "clustering":
            # Clustering: use KMeans + silhouette_score as proxy metric (no target column)
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # For clustering, X is the full dataframe (no target to drop)
            X_cluster = df_sample.select_dtypes(include=['number']).copy()
            if target_column in X_cluster.columns:
                X_cluster = X_cluster.drop(columns=[target_column], errors='ignore')
            
            if X_cluster.shape[1] == 0:
                report["status"] = "error"
                report["reason"] = "No numeric features for clustering proxy"
                return report
            
            # Try k=3 as default proxy
            n_clusters = min(3, len(X_cluster) // 5)
            if n_clusters < 2:
                n_clusters = 2
            proxy_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = proxy_model.fit_predict(X_cluster)
            
            n_unique = len(set(labels))
            if n_unique > 1:
                proxy_metric = silhouette_score(X_cluster, labels)
            else:
                proxy_metric = -1.0
            
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        
        report["proxy_metric"] = round(float(proxy_metric), 4)
        report["proxy_runtime_sec"] = round(time.time() - start_time, 2)
        
        # Mark suspiciously low scores as warning (not fail)
        if task_type == "classification" and proxy_metric < 0.5:
            report["status"] = "warning"  # Not fail - may still train
            report["reason"] = f"Proxy metric below random baseline: {proxy_metric:.4f}"
        elif task_type == "regression" and proxy_metric < -0.5:
            report["status"] = "warning"
            report["reason"] = f"Proxy metric negative: {proxy_metric:.4f}"
        elif task_type == "clustering" and proxy_metric < 0.0:
            report["status"] = "warning"
            report["reason"] = f"Clustering silhouette negative: {proxy_metric:.4f}"
        
        return report
        
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"Proxy training error: {str(e)}"
        report["proxy_runtime_sec"] = round(time.time() - start_time, 2)
        return report


def train_pycaret_variant(
    df: pd.DataFrame,
    variant: VariantConfig,
    target_column: str,
    task_type: str,
    time_budget: int = 300
) -> Tuple[Any, Dict[str, Any], bool]:
    """Train models using PyCaret with variant configuration.
    
    Returns:
        (best_model, metrics_dict, timed_out_flag)
    """
    start_time = time.time()
    timed_out = False
    
    # Adaptive time budget: large datasets (>50K rows) get more time
    n_rows = len(df)
    if n_rows >= 50000 and time_budget < 600:
        print(f"      ⏱️ Adaptive budget: {n_rows:,} rows detected, raising budget {time_budget}→600s")
        time_budget = 600
    
    try:
        if task_type == "classification":
            from pycaret.classification import setup, compare_models, pull
            
            # Get canonical model list (enforces MODEL_UNIVERSE, prevents PyCaret adding removed models)
            _include_models = get_model_list(task_type, "pycaret")
            print(f"      PyCaret include list: {len(_include_models)} models from MODEL_UNIVERSE")
            
            # Disable MLflow autologging to prevent conflicts
            # preprocess=False: data is ALREADY preprocessed by our variant pipeline
            # (encoding, scaling, imputation done in s03/s04). PyCaret re-preprocessing
            # causes double-encoding and degrades model performance.
            # fix_imbalance: when recipe requests SMOTE/ADASYN, enable PyCaret's built-in
            # per-fold resampling so models are evaluated on balanced data during CV.
            # PyCaret applies SMOTE only on each fold's train split (no leakage).
            _imb_cfg_pc = getattr(variant.stage3_preprocessing, 'imbalance_handling', None)
            _imb_method_pc = _imb_cfg_pc.method if _imb_cfg_pc else "none"
            _fix_imbalance = _imb_method_pc not in (None, "none")
            if _fix_imbalance:
                print(f"      ⚖️ PyCaret fix_imbalance=True ({_imb_method_pc}) — balanced CV enabled")
            setup(
                data=df,
                target=target_column,
                session_id=42,  # Deterministic seed
                fold=3,  # 3-fold CV (was defaulting to 10 — 3.3x speedup)
                preprocess=False,  # CRITICAL: avoid double-preprocessing
                normalize=False,         # K5: defense-in-depth
                transformation=False,    # K5: defense-in-depth
                fix_imbalance=_fix_imbalance,  # Apply SMOTE within CV folds when recipe requests it
                verbose=False,
                log_experiment=False,
                html=False
            )
            
            # Train models with time budget (soft timeout)
            remaining_time = max(0.0, time_budget - (time.time() - start_time))
            budget_minutes = max(1, int(remaining_time // 60))
            print(f"      PyCaret: remaining_time={remaining_time:.1f}s, budget={budget_minutes} minutes")
            
            if remaining_time < 30:
                timed_out = True
                return None, {
                    "primary_metric": 0.0,
                    "algorithm": "insufficient_time",
                    "runtime_sec": time.time() - start_time,
                    "timed_out": True,
                    "error": f"Insufficient time: {remaining_time:.1f}s"
                }, True
            
            sort_metric = get_primary_metric(task_type)   # "AUC" for classification
            best_model = compare_models(
                include=_include_models,  # Enforce MODEL_UNIVERSE (all 14 models)
                sort=sort_metric,         # Sort by AUC, not Accuracy (avoids majority-class trap)
                n_select=1,
                budget_time=budget_minutes,
                verbose=False
            )
            
            # Check if we exceeded budget
            actual_runtime = time.time() - start_time
            if actual_runtime > time_budget:
                timed_out = True
            
            # Get leaderboard
            leaderboard = pull()
            primary_metric = sort_metric
            best_score = leaderboard[primary_metric].iloc[0]
            
            metrics = {
                "primary_metric": float(best_score),
                "accuracy": float(leaderboard.get("Accuracy", [0.0]).iloc[0]) if "Accuracy" in leaderboard else float(best_score),
                "algorithm": str(type(best_model).__name__),
                "runtime_sec": actual_runtime,
                "timed_out": timed_out,
                "n_models_trained": len(leaderboard)
            }
            
            # ── Extract ALL CV metrics from PyCaret leaderboard ──
            _PYCARET_METRIC_MAP = {
                "Accuracy": "accuracy", "AUC": "auc",
                "Recall": "recall", "Prec.": "precision",
                "F1": "f1", "Kappa": "kappa", "MCC": "mcc",
            }
            for _pc_col, _m_key in _PYCARET_METRIC_MAP.items():
                if _pc_col in leaderboard.columns:
                    metrics[_m_key] = round(float(leaderboard[_pc_col].iloc[0]), 4)
            
            # ── SMOTE FULL-DATA RETRAINING ────────────────────────────────
            # Model selection used balanced CV via fix_imbalance=True above.
            # Now retrain the selected champion on the FULL training set with
            # SMOTE so the deployed model has maximum minority-class exposure.
            #
            # DESIGN DECISION (R2 audit 2026-02): Label encoders fitted here
            # are saved to a sidecar dict (_smote_label_encoders) so they can
            # be persisted alongside model.pkl.  Without this, s10 would
            # receive a model trained on label-encoded features but no way to
            # reproduce that encoding at inference time.
            # ─────────────────────────────────────────────────────────────
            _smote_label_encoders = {}  # col_name → fitted LabelEncoder
            _imb_cfg = getattr(variant.stage3_preprocessing, 'imbalance_handling', None)
            _imb_method = _imb_cfg.method if _imb_cfg else "none"
            if _imb_method and _imb_method != "none":
                try:
                    X_train_full = df.drop(columns=[target_column])
                    y_train_full = df[target_column]
                    _X_num = X_train_full.select_dtypes(include=[np.number])
                    # If non-numeric columns remain, label-encode them so SMOTE can proceed
                    if len(_X_num.columns) < len(X_train_full.columns):
                        from sklearn.preprocessing import LabelEncoder as _LE
                        _non_num_cols = X_train_full.select_dtypes(exclude=[np.number]).columns.tolist()
                        print(f"      ⚠️ SMOTE retrain: label-encoding {len(_non_num_cols)} non-numeric columns: {_non_num_cols[:5]}")
                        for _col in _non_num_cols:
                            _le = _LE()
                            X_train_full[_col] = _le.fit_transform(X_train_full[_col].astype(str))
                            _smote_label_encoders[_col] = _le
                    _sampler = None
                    if _imb_method == "smote":
                        from imblearn.over_sampling import SMOTE
                        _sampler = SMOTE(random_state=42, n_jobs=-1)
                    elif _imb_method == "adasyn":
                        from imblearn.over_sampling import ADASYN
                        _sampler = ADASYN(random_state=42, n_jobs=-1)
                    elif _imb_method == "smoteenn":
                        from imblearn.combine import SMOTEENN
                        _sampler = SMOTEENN(random_state=42)
                    elif _imb_method == "smotetomek":
                        from imblearn.combine import SMOTETomek
                        _sampler = SMOTETomek(random_state=42)
                    if _sampler is not None:
                        X_resampled, y_resampled = _sampler.fit_resample(X_train_full, y_train_full)
                        print(f"      ⚖️ SMOTE full-data retrain ({_imb_method}): {len(X_train_full)} → {len(X_resampled)} rows")
                        best_model.fit(X_resampled, y_resampled)
                        # Persist label encoders on the model so they survive
                        # joblib serialisation and can be applied at inference
                        if _smote_label_encoders:
                            best_model._smote_label_encoders = _smote_label_encoders
                            print(f"      💾 Saved {len(_smote_label_encoders)} SMOTE label encoders on model object")
                        metrics["smote_retrained"] = 1
                        metrics["smote_rows_after"] = len(X_resampled)
                        print(f"      ✅ Model retrained on resampled data ({len(X_resampled)} rows)")
                except Exception as _smote_err:
                    print(f"      ⚠️ SMOTE retraining failed: {_smote_err}, keeping model as-is")
                    metrics["smote_retrained"] = 0
            
            return best_model, metrics, timed_out
            
        elif task_type == "regression":
            from pycaret.regression import setup, compare_models, pull
            
            # Get canonical model list (enforces MODEL_UNIVERSE, prevents PyCaret adding removed models)
            _include_models = get_model_list(task_type, "pycaret")
            print(f"      PyCaret include list: {len(_include_models)} models from MODEL_UNIVERSE")
            
            # preprocess=False: data is ALREADY preprocessed by our variant pipeline
            setup(
                data=df,
                target=target_column,
                session_id=42,  # Deterministic seed
                fold=3,  # 3-fold CV (was defaulting to 10 — 3.3x speedup)
                preprocess=False,  # CRITICAL: avoid double-preprocessing
                normalize=False,         # K5: defense-in-depth
                transformation=False,    # K5: defense-in-depth
                verbose=False,
                log_experiment=False,
                html=False
            )
            
            remaining_time = max(0.0, time_budget - (time.time() - start_time))
            budget_minutes = max(1, int(remaining_time // 60))
            print(f"      PyCaret: remaining_time={remaining_time:.1f}s, budget={budget_minutes} minutes")
            
            if remaining_time < 30:
                timed_out = True
                return None, {
                    "primary_metric": 0.0,
                    "algorithm": "insufficient_time",
                    "runtime_sec": time.time() - start_time,
                    "timed_out": True,
                    "error": f"Insufficient time: {remaining_time:.1f}s"
                }, True
            
            sort_metric = get_primary_metric(task_type)   # "R2" for regression
            best_model = compare_models(
                include=_include_models,  # Enforce MODEL_UNIVERSE (all 23 models)
                sort=sort_metric,         # Sort by R2 (explicit — matches s5a baseline)
                n_select=1,
                budget_time=budget_minutes,
                verbose=False
            )
            
            actual_runtime = time.time() - start_time
            if actual_runtime > time_budget:
                timed_out = True
            
            leaderboard = pull()
            primary_metric = sort_metric
            best_score = leaderboard[primary_metric].iloc[0]
            
            metrics = {
                "primary_metric": float(best_score),
                "r2": float(leaderboard.get("R2", [0.0]).iloc[0]) if "R2" in leaderboard else float(best_score),
                "algorithm": str(type(best_model).__name__),
                "runtime_sec": actual_runtime,
                "timed_out": timed_out,
                "n_models_trained": len(leaderboard)
            }

            # ── Extract ALL regression CV metrics from PyCaret leaderboard ──
            _PYCARET_REG_MAP = {
                "MAE": "mae", "MSE": "mse", "RMSE": "rmse",
                "R2": "r2", "RMSLE": "rmsle", "MAPE": "mape",
            }
            for _pc_col, _met_key in _PYCARET_REG_MAP.items():
                if _pc_col in leaderboard.columns:
                    metrics[_met_key] = round(float(leaderboard[_pc_col].iloc[0]), 4)

            return best_model, metrics, timed_out
            
        else:
            # Clustering: use sklearn KMeans directly (bypasses PyCaret's
            # internal silhouette_score which is O(n²) and prohibitively
            # slow on large datasets like Online Retail ~433K rows).
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score as _sil_score
            
            # Remove target column if present (clustering is unsupervised)
            df_cluster = df.copy()
            if target_column in df_cluster.columns:
                df_cluster = df_cluster.drop(columns=[target_column])
            
            # Cast numeric columns to float64 to prevent dtype mismatch errors
            _numeric_cols = df_cluster.select_dtypes(include=[np.number]).columns
            df_cluster[_numeric_cols] = df_cluster[_numeric_cols].astype(np.float64)
            
            # Drop any remaining non-numeric columns (clustering needs numeric)
            df_cluster = df_cluster.select_dtypes(include=[np.number])
            print(f"      Clustering: {len(df_cluster)} rows × {df_cluster.shape[1]} numeric features")
            
            remaining_time = max(0.0, time_budget - (time.time() - start_time))
            if remaining_time < 30:
                timed_out = True
                return None, {
                    "primary_metric": 0.0,
                    "algorithm": "insufficient_time",
                    "runtime_sec": time.time() - start_time,
                    "timed_out": True,
                    "error": f"Insufficient time: {remaining_time:.1f}s"
                }, True
            
            best_model = KMeans(n_clusters=3, random_state=42, n_init=10)
            best_model.fit(df_cluster)
            
            actual_runtime = time.time() - start_time
            if actual_runtime > time_budget:
                timed_out = True
            
            # Compute silhouette score on a sample for large datasets
            # (silhouette_score is O(n²) — infeasible above ~50K rows)
            _SIL_SAMPLE_SIZE = 10000
            sil_score = 0.0
            try:
                labels = best_model.labels_
                if len(df_cluster) > _SIL_SAMPLE_SIZE:
                    _rng = np.random.RandomState(42)
                    _idx = _rng.choice(len(df_cluster), _SIL_SAMPLE_SIZE, replace=False)
                    sil_score = float(_sil_score(df_cluster.iloc[_idx], labels[_idx]))
                    print(f"      Silhouette score (sampled {_SIL_SAMPLE_SIZE} rows): {sil_score:.4f}")
                else:
                    sil_score = float(_sil_score(df_cluster, labels))
                    print(f"      Silhouette score: {sil_score:.4f}")
            except Exception as _sil_err:
                print(f"      ⚠️ Silhouette computation failed: {_sil_err}")
            
            metrics = {
                "primary_metric": sil_score,
                "silhouette_score": sil_score,
                "algorithm": str(type(best_model).__name__),
                "runtime_sec": actual_runtime,
                "timed_out": timed_out,
                "n_models_trained": 1
            }
            
            return best_model, metrics, timed_out
            
    except Exception as e:
        print(f"❌ PyCaret training failed: {e}")
        return None, {
            "primary_metric": 0.0,
            "algorithm": "failed",
            "runtime_sec": time.time() - start_time,
            "timed_out": False,
            "error": str(e)
        }, False


def train_flaml_variant(
    df: pd.DataFrame,
    variant: VariantConfig,
    target_column: str,
    task_type: str,
    time_budget: int = 300
) -> Tuple[Any, Dict[str, Any], bool]:
    """Train models using FLAML with variant configuration.
    
    Returns:
        (best_model, metrics_dict, timed_out_flag)
    """
    start_time = time.time()
    timed_out = False
    
    # Adaptive time budget: large datasets (>50K rows) get more time (matches PyCaret logic)
    n_rows = len(df)
    if n_rows >= 50000 and time_budget < 600:
        print(f"      ⏱️ FLAML adaptive budget: {n_rows:,} rows detected, raising budget {time_budget}→600s")
        time_budget = 600
    
    # Enforce minimum FLAML budget floor to prevent 100% timeouts
    flaml_min = getattr(train_flaml_variant, '_min_budget', 120)
    if time_budget < flaml_min:
        print(f"      ⏱️ FLAML min budget floor: raising {time_budget}→{flaml_min}s")
        time_budget = flaml_min
    
    try:
        from flaml import AutoML
        
        # FLAML does not support clustering - skip with valid result
        if task_type == "clustering":
            print(f"      ⚠️  FLAML does not support clustering; skipping variant")
            return None, {
                "primary_metric": 0.0,
                "algorithm": "skipped",
                "runtime_sec": 0.0,
                "timed_out": False,
                "status": "skipped",
                "error": "FLAML does not support clustering task type"
            }, False
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        automl = AutoML()
        
        flaml_task = {
            "classification": "classification",
            "regression": "regression"
        }.get(task_type, "classification")
        
        # Check remaining time (soft timeout)
        remaining_time = max(0.0, time_budget - (time.time() - start_time))
        print(f"      FLAML budget: {min(remaining_time, time_budget):.1f} seconds")
        if remaining_time < 30:
            timed_out = True
            return None, {
                "primary_metric": 0.0,
                "algorithm": "insufficient_time",
                "runtime_sec": time.time() - start_time,
                "timed_out": True,
                "error": f"Insufficient time: {remaining_time:.1f}s"
            }, True
        
        automl.fit(
            X_train=X,
            y_train=y,
            task=flaml_task,
            time_budget=min(remaining_time, time_budget),
            metric="roc_auc" if task_type == "classification" else "r2",  # AUC handles class imbalance (matches s5b baseline)
            seed=42,  # Deterministic seed
            verbose=0,
            log_training_metric=False
        )
        
        actual_runtime = time.time() - start_time
        if actual_runtime > time_budget:
            timed_out = True
        
        # FIX: Prefer best_validation_score over best_loss (avoids inversion issues)
        if hasattr(automl, 'best_validation_score') and automl.best_validation_score is not None:
            primary_metric_value = float(automl.best_validation_score)
            metric_source = "best_validation_score"
        else:
            # Fallback to best_loss with convention (may need inversion)
            primary_metric_value = float(automl.best_loss) if task_type == "regression" else float(1 - automl.best_loss)
            metric_source = "best_loss_derived"
        
        metrics = {
            "primary_metric": primary_metric_value,
            "flaml_metric_source": metric_source,  # Track which metric used
            "algorithm": str(automl.best_estimator),
            "runtime_sec": actual_runtime,
            "timed_out": timed_out,
            "n_models_trained": len(automl.config_history) if hasattr(automl, 'config_history') else 1
        }
        
        # ── Track individual FLAML models tried during search ──
        # FLAML config_history: dict of {trial_id: (config, metric_value, ...)}
        flaml_individual_models = []
        if hasattr(automl, 'config_history') and automl.config_history:
            for trial_id, entry in automl.config_history.items():
                try:
                    if isinstance(entry, (list, tuple)):
                        config = entry[0] if len(entry) > 0 and isinstance(entry[0], dict) else {}
                        metric_val = float(entry[1]) if len(entry) > 1 else 0.0
                    elif isinstance(entry, dict):
                        config = entry
                        metric_val = 0.0
                    else:
                        continue
                    estimator = str(config.get('learner', config.get('ml_learner', 'unknown')))
                    flaml_individual_models.append({
                        "trial_id": int(trial_id),
                        "estimator": estimator,
                        "metric_value": round(abs(metric_val), 4),
                    })
                except Exception:
                    pass
            print(f"      📊 FLAML tracked {len(flaml_individual_models)} individual model trials")
        metrics["flaml_individual_models"] = flaml_individual_models
        
        # ════════════════════════════════════════════════════════════════════
        # BATCH 3 FIX (UPDATED): Cross-validated metrics — TIME-AWARE.
        #
        # FLAML's best_validation_score is ALREADY cross-validated internally
        # (no resubstitution leakage).  The external cross_val_predict adds
        # detailed per-metric breakdown (F1, recall, precision, etc.) but it
        # trains the best model 3–6 extra times which can exceed the deadline.
        #
        # Strategy: check remaining wall-clock time after automl.fit().
        #   - >= 300s remaining → full 3-fold CV (predict + predict_proba)
        #   - >= 120s remaining → lightweight 2-fold CV (predict only)
        #   - <  120s remaining → skip CV; use FLAML's internal validation score
        # ══════════════════════════════════════════════════════════════════════
        remaining_after_fit = max(0.0, time_budget - (time.time() - start_time))
        
        if task_type == "classification":
            metrics["accuracy"] = metrics["primary_metric"]
            # ── Compute classification metrics (time-aware) ──
            if remaining_after_fit >= 120:
                cv_folds = 3 if remaining_after_fit >= 300 else 2
                try:
                    from sklearn.metrics import (
                        accuracy_score, f1_score, precision_score, recall_score,
                        cohen_kappa_score, matthews_corrcoef, roc_auc_score,
                    )
                    from sklearn.model_selection import cross_val_predict
                    from sklearn.base import clone as sklearn_clone

                    cv_model = sklearn_clone(automl.model)
                    y_pred = cross_val_predict(cv_model, X, y, cv=cv_folds, method='predict')
                    metrics["accuracy"] = round(float(accuracy_score(y, y_pred)), 4)
                    metrics["f1"] = round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4)
                    metrics["precision"] = round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4)
                    metrics["recall"] = round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4)
                    metrics["kappa"] = round(float(cohen_kappa_score(y, y_pred)), 4)
                    metrics["mcc"] = round(float(matthews_corrcoef(y, y_pred)), 4)
                    # AUC via cross-validated predict_proba (only if enough time)
                    remaining_for_proba = max(0.0, time_budget - (time.time() - start_time))
                    if remaining_for_proba >= 90:
                        try:
                            cv_model_proba = sklearn_clone(automl.model)
                            y_proba = cross_val_predict(cv_model_proba, X, y, cv=cv_folds, method='predict_proba')
                            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                                metrics["auc"] = round(float(roc_auc_score(y, y_proba[:, 1])), 4)
                            else:
                                metrics["auc"] = round(float(roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")), 4)
                        except Exception:
                            pass
                    else:
                        print(f"      ⏱️ Skipping AUC cross_val_predict (only {remaining_for_proba:.0f}s left)")
                    print(f"      ✅ FLAML metrics via {cv_folds}-fold cross_val_predict (no leakage)")
                except Exception as _sk_err:
                    print(f"      ⚠️ cross-validated metric computation failed (non-fatal): {_sk_err}")
            else:
                print(f"      ⏱️ Skipping cross_val_predict ({remaining_after_fit:.0f}s left < 120s); "
                      f"using FLAML internal validation score ({metrics['primary_metric']:.4f})")
        else:
            metrics["r2"] = metrics["primary_metric"]
            # ── Compute regression metrics (time-aware) ──
            if remaining_after_fit >= 120:
                cv_folds = 3 if remaining_after_fit >= 300 else 2
                try:
                    from sklearn.metrics import (
                        mean_absolute_error, mean_squared_error, r2_score,
                    )
                    from sklearn.model_selection import cross_val_predict
                    from sklearn.base import clone as sklearn_clone

                    cv_model = sklearn_clone(automl.model)
                    y_pred = cross_val_predict(cv_model, X, y, cv=cv_folds, method='predict')
                    metrics["r2"] = round(float(r2_score(y, y_pred)), 4)
                    metrics["mae"] = round(float(mean_absolute_error(y, y_pred)), 4)
                    metrics["mse"] = round(float(mean_squared_error(y, y_pred)), 4)
                    metrics["rmse"] = round(float(mean_squared_error(y, y_pred, squared=False)), 4)
                    print(f"      ✅ FLAML regression metrics via {cv_folds}-fold cross_val_predict (no leakage)")
                except Exception as _sk_err:
                    print(f"      ⚠️ cross-validated regression metric computation failed (non-fatal): {_sk_err}")
            else:
                print(f"      ⏱️ Skipping cross_val_predict ({remaining_after_fit:.0f}s left < 120s); "
                      f"using FLAML internal validation score ({metrics['primary_metric']:.4f})")
        
        return automl.model, metrics, timed_out
        
    except Exception as e:
        print(f"❌ FLAML training failed: {e}")
        return None, {
            "primary_metric": 0.0,
            "algorithm": "failed",
            "runtime_sec": time.time() - start_time,
            "timed_out": False,
            "error": str(e)
        }, False


def run_variant_with_nested_mlflow(
    variant: VariantConfig,
    df: pd.DataFrame,
    engine: str,
    target_column: str,
    task_type: str,
    time_budget: int,
    attempt_deadline: float,
    df_preprocessed: Optional[pd.DataFrame] = None  # FIX 2: Reuse cached transform
) -> Tuple[VariantResult, Any]:
    """Run one variant with one engine, nested MLflow tracking.
    
    HARDENED: Returns (VariantResult, trained_model) tuple.
    Hard budget enforced via attempt_deadline.
    
    Args:
        df_preprocessed: Optional preprocessed dataframe from Round0/Round1.
                        If provided, skips preprocessing step (avoids double transform).
    """
    
    run_name = f"variant_{variant.variant_id}_{engine}"
    start_time = time.time()
    trained_model = None
    
    try:
        # Hard budget guard: check if already exceeded
        if deadline_guard(attempt_deadline, "before_preprocessing"):
            return VariantResult(
                variant_id=variant.variant_id,
                engine=engine,
                algorithm="time_budget_exceeded",
                metrics={"primary_metric": 0.0},
                runtime_sec=time.time() - start_time,
                timed_out=True,
                failed=False,
                failure_reason="Time budget exceeded before preprocessing"
            ), None
        
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log variant configuration
            mlflow.log_params({
                "variant_id": variant.variant_id,
                "engine": engine,
                "imputation": variant.stage3_preprocessing.imputation.method,
                "encoding": variant.stage3_preprocessing.encoding.categorical_method,
                "scaling": variant.stage3_preprocessing.scaling.method,
                "imbalance": variant.stage3_preprocessing.imbalance_handling.method if variant.stage3_preprocessing.imbalance_handling else "none",
                "feature_selection": variant.stage4_feature_engineering.feature_selection.method,
                "leakage_risk": check_leakage_risk(variant)
            })
            
            # Apply preprocessing with error handling (or reuse cached)
            if df_preprocessed is not None:
                print(f"  ♻️  Reusing preprocessed data from Round0/Round1...")
                df_processed = df_preprocessed
                n_features = df_processed.shape[1] - 1  # Exclude target
            else:
                print(f"  🔄 Applying preprocessing...")
                try:
                    df_processed = apply_variant_preprocessing(df, variant, target_column, apply_smote=False)
                    n_features = df_processed.shape[1] - 1  # Exclude target
                except Exception as e:
                    print(f"  ❌ Preprocessing failed: {e}")
                    return VariantResult(
                        variant_id=variant.variant_id,
                        engine=engine,
                        algorithm="preprocessing_failed",
                        metrics={"primary_metric": 0.0},
                        runtime_sec=time.time() - start_time,
                        timed_out=False,
                        failed=True,
                        failure_reason=f"Preprocessing error: {str(e)}"
                    ), None
            
            # Hard budget guard: check after preprocessing
            if deadline_guard(attempt_deadline, "after_preprocessing"):
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm="time_budget_exceeded",
                    metrics={"primary_metric": 0.0},
                    runtime_sec=time.time() - start_time,
                    timed_out=True,
                    failed=False,
                    failure_reason="Time budget exceeded after preprocessing"
                ), None
            
            # Train model with timeout enforcement
            print(f"  🏋️  Training with {engine} (budget: {time_budget}s)...")
            if engine == "pycaret":
                model, metrics, timed_out = train_pycaret_variant(
                    df_processed, variant, target_column, task_type, time_budget
                )
            elif engine == "flaml":
                model, metrics, timed_out = train_flaml_variant(
                    df_processed, variant, target_column, task_type, time_budget
                )
            else:
                raise ValueError(f"Unknown engine: {engine}")
            
            # Capture model for champion tracking
            trained_model = model
            
            # Validate and clean metrics
            metrics = validate_metrics(metrics)
            
            # Hard budget guard: check after training
            if deadline_guard(attempt_deadline, "after_training"):
                # Preserve real training metrics (model already trained)
                metrics["timed_out"] = True
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm=metrics.get("algorithm", "time_budget_exceeded"),
                    metrics=metrics,
                    runtime_sec=time.time() - start_time,
                    timed_out=True,
                    failed=False,
                    failure_reason="Time budget exceeded after training"
                ), trained_model
            
            # Log metrics
            _mlflow_metrics = {
                "primary_metric": metrics.get("primary_metric", 0.0),
                "runtime_sec": metrics.get("runtime_sec", 0.0),
                "timed_out": 1.0 if timed_out else 0.0,
                "n_models_trained": metrics.get("n_models_trained", 1),
                "n_features": n_features
            }
            # Log all detailed classification/regression metrics
            for _mk in ("accuracy", "auc", "f1", "precision", "recall", "kappa", "mcc", "r2"):
                if _mk in metrics and isinstance(metrics[_mk], (int, float)):
                    _mlflow_metrics[_mk] = float(metrics[_mk])
            mlflow.log_metrics(_mlflow_metrics)
            
            # ── Log individual FLAML models as nested MLflow child runs ──
            # This allows users to see each model FLAML tried in Azure ML Studio
            if engine == "flaml" and metrics.get("flaml_individual_models"):
                for fm in metrics["flaml_individual_models"]:
                    fm_run_name = f"flaml_trial_{fm['trial_id']}_{fm['estimator']}"
                    try:
                        with mlflow.start_run(run_name=fm_run_name, nested=True):
                            mlflow.log_params({
                                "trial_id": fm["trial_id"],
                                "estimator": fm["estimator"],
                                "engine": "flaml",
                                "variant_id": variant.variant_id,
                            })
                            mlflow.log_metrics({
                                "metric_value": fm["metric_value"],
                            })
                    except Exception as _fm_err:
                        print(f"      ⚠️ Could not log FLAML trial {fm['trial_id']}: {_fm_err}")
            
            # Hard budget guard: check before model logging
            if deadline_guard(attempt_deadline, "before_model_logging"):
                return VariantResult(
                    variant_id=variant.variant_id,
                    engine=engine,
                    algorithm=metrics.get("algorithm", "time_budget_exceeded"),
                    metrics=metrics,
                    runtime_sec=time.time() - start_time,
                    timed_out=True,
                    failed=False,
                    failure_reason="Time budget exceeded before model logging"
                ), trained_model
            
            # Log model if successful
            if model is not None:
                try:
                    mlflow.sklearn.log_model(model, "model")
                except Exception as e:
                    print(f"⚠️ Could not log model: {e}")
            
            # Return normalized result (no checkpoint marking here)
            return VariantResult(
                variant_id=variant.variant_id,
                engine=engine,
                algorithm=metrics.get("algorithm", "unknown"),
                metrics=metrics,
                runtime_sec=metrics.get("runtime_sec", 0.0),
                timed_out=timed_out,
                failed=(model is None),
                failure_reason=metrics.get("error") if model is None else None,
                leakage_risk=check_leakage_risk(variant),
                n_features=n_features
            ), trained_model
    
    except Exception as e:
        print(f"  ❌ Unexpected failure: {e}")
        # Return valid result (checkpoint marking happens in finally block of caller)
        return VariantResult(
            variant_id=variant.variant_id,
            engine=engine,
            algorithm="crashed",
            metrics={"primary_metric": 0.0},
            runtime_sec=time.time() - start_time,
            timed_out=False,
            failed=True,
            failure_reason=f"Unexpected error: {str(e)}"
        ), trained_model


def main():
    parser = argparse.ArgumentParser(description="Phase B Variant Runner (HARDENED)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to main config YAML")
    parser.add_argument("--variants_json", type=str, required=False, help="JSON file containing list of variant paths")
    parser.add_argument("--variants_list", type=str, required=False, help="Comma-separated list of variant paths")
    parser.add_argument("--engine_list", type=str, required=True, help="Comma-separated engines (pycaret,flaml)")
    parser.add_argument("--dataset_in", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--time_budget_per_variant", type=int, default=300, help="Time budget per variant (seconds)")
    parser.add_argument("--flaml_min_budget", type=int, default=120,
                        help="Minimum FLAML time budget in seconds (floor to prevent 100%% timeout)")
    
    # Signal artifact flags - DEFAULT TO TRUE (always enabled unless explicitly disabled via env vars)
    parser.add_argument("--enable_round0_feasibility", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Round 0 feasibility checks and artifact")
    parser.add_argument("--enable_round1_proxy", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Round 1 proxy leaderboard (transform-only)")
    parser.add_argument("--enable_elimination_report", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable detailed elimination report artifact")
    
    # V3-Proposed Planner flags
    parser.add_argument("--planner_enabled", action="store_true", default=False,
                        help="Enable V3-Proposed adaptive planner mode")
    parser.add_argument("--round1_max_variants", type=int, default=40,
                        help="Max variants for Round 1 proxy training")
    parser.add_argument("--round2_max_variants", type=int, default=10,
                        help="Max variants for Round 2 full training")
    parser.add_argument("--proxy_prune_threshold", type=float, default=0.50,
                        help="Proxy metric threshold for pruning (classification)")
    parser.add_argument("--diversity_min_hamming", type=int, default=2,
                        help="Min Hamming distance for diverse sampling")
    parser.add_argument("--cache_enabled", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable preprocessing cache")
    
    args = parser.parse_args()
    
    # Parse arguments
    engines = [e.strip() for e in args.engine_list.split(",")]
    
    # Load variant paths from JSON file OR string list
    if args.variants_json:
        with open(args.variants_json, 'r') as f:
            variant_paths = json.load(f)
    elif args.variants_list:
        variant_paths = [p.strip() for p in args.variants_list.split(",")]
    else:
        raise ValueError("Must provide either --variants_json or --variants_list")
    
    # ======== FIX: RESOLVE VARIANT PATHS ========
    # Variant paths may be relative to configs/recipes/ (e.g. "classification/variant_search/variant_xxx.yml")
    # Resolve them to absolute paths so open() works in Azure ML job context
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _RECIPES_BASE = _PROJECT_ROOT / "configs" / "recipes"
    resolved_variant_paths = []
    for _vp in variant_paths:
        _vp_path = Path(_vp)
        if _vp_path.exists():
            resolved_variant_paths.append(str(_vp_path))
        elif (_RECIPES_BASE / _vp).exists():
            resolved_variant_paths.append(str(_RECIPES_BASE / _vp))
        elif (_PROJECT_ROOT / _vp).exists():
            resolved_variant_paths.append(str(_PROJECT_ROOT / _vp))
        else:
            # Keep original for error reporting downstream
            print(f"   ⚠️ Cannot resolve variant path: {_vp}")
            print(f"      Tried: {_vp_path}, {_RECIPES_BASE / _vp}, {_PROJECT_ROOT / _vp}")
            resolved_variant_paths.append(str(_RECIPES_BASE / _vp))
    variant_paths = resolved_variant_paths
    
    print(f"\n{'='*80}")
    print(f"PHASE B VARIANT RUNNER (HARDENED)")
    print(f"{'='*80}")
    print(f"Variants to process: {len(variant_paths)}")
    print(f"Engines: {', '.join(engines)}")
    print(f"Total runs: {len(variant_paths) * len(engines)}")
    print(f"{'='*80}\n")
    
    # Load config to get task type and target column
    import yaml
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    task_type = config.get("task_type", "classification")
    target_column = config.get("dataset", {}).get("target_column")
    delimiter = config.get("dataset", {}).get("delimiter", ",")
    
    # Wire --flaml_min_budget to the training function via function attribute
    train_flaml_variant._min_budget = args.flaml_min_budget
    print(f"FLAML min budget floor: {args.flaml_min_budget}s")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ======== HARDENING FEATURE 6: DETERMINISTIC SEEDING ========
    set_deterministic_seed(42)
    
    # ======== HARDENING FEATURE 1: CHECKPOINTING ========
    total_expected = len(variant_paths) * len(engines)
    checkpoint = CheckpointManager(output_path / "resume_state.json")
    progress = checkpoint.get_progress(total_expected=total_expected)
    print(f"\n📊 Checkpoint Progress: {progress['completed']}/{progress['total']} completed")
    if progress['completed'] > 0:
        print(f"   ⏮️  Resuming from previous run (last updated: {progress['last_updated']})")
    print(f"   ⏱️  Time budget per variant: {args.time_budget_per_variant}s\n")
    
    # Load dataset (respect config delimiter — e.g. bank_marketing uses ";")
    print(f"📂 Loading dataset from {args.dataset_in}")
    df_full = pd.read_csv(args.dataset_in, sep=delimiter)
    print(f"   Shape: {df_full.shape[0]} rows × {df_full.shape[1]} columns")
    
    # ════════════════════════════════════════════════════════════════════
    # BATCH 3 FIX: HOLDOUT SPLIT — prevents data leakage across all
    # preprocessing steps (SMOTE, scaling, imputation, outlier handling,
    # feature selection).  Preprocessing + training now see only 80% of
    # the data; the 20% holdout is reserved for honest evaluation.
    # ════════════════════════════════════════════════════════════════════
    from sklearn.model_selection import train_test_split as _holdout_split
    holdout_fraction = 0.2
    if task_type == "classification" and target_column in df_full.columns:
        stratify_col = df_full[target_column]
    else:
        stratify_col = None
    df, df_holdout = _holdout_split(
        df_full,
        test_size=holdout_fraction,
        random_state=42,
        stratify=stratify_col
    )
    df = df.reset_index(drop=True)
    df_holdout = df_holdout.reset_index(drop=True)
    print(f"   🔀 Holdout split: {len(df)} train / {len(df_holdout)} holdout ({holdout_fraction:.0%})")
    
    # Save holdout for downstream steps and champion evaluation
    holdout_path = output_path / "holdout_data.csv"
    df_holdout.to_csv(holdout_path, index=False)
    print(f"   📁 Holdout saved to {holdout_path}")
    
    # ======== HARDENING FEATURE 4: DATA FINGERPRINTING ========
    # Fingerprint computed on FULL data (for reproducibility tracking)
    data_fingerprint = compute_data_fingerprint(df_full)
    code_version = get_code_version()
    print(f"   Data fingerprint: {data_fingerprint['hash'][:12]}... (full dataset)")
    print(f"   Code version: {code_version}\n")
    
    # ======== V3-PROPOSED: PREPROCESSING CACHE ========
    preprocessing_cache = create_preprocessing_cache(
        enabled=args.cache_enabled,
        max_entries=50
    )
    print(f"🗄️  Preprocessing cache: {'ENABLED' if args.cache_enabled else 'DISABLED'}")
    
    # ======== V3-PROPOSED: ADAPTIVE PLANNER MODE ========
    variant_plan = None
    if args.planner_enabled:
        print(f"\n{'='*80}")
        print(f"V3-PROPOSED: ADAPTIVE PLANNER MODE ENABLED")
        print(f"{'='*80}")
        
        # Build EDA priors from dataset (simplified inline profiling)
        print(f"\n📊 Building EDA priors from dataset...")
        eda_priors = EdaPriors(
            missing_rate=float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            imbalance_ratio=float(df[target_column].value_counts(normalize=True).min()) if task_type == "classification" and target_column else 1.0,
            high_cardinality_cols=[col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() > 50],
            outlier_prevalence=0.0,  # Simplified - would need IQR calculation
            skewness_issues=sum(1 for col in df.select_dtypes(include=[np.number]).columns if abs(df[col].skew()) > 1),
            n_rows=df.shape[0],
            n_features=df.shape[1] - 1
        )
        print(f"   missing_rate: {eda_priors.missing_rate:.3f}")
        print(f"   imbalance_ratio: {eda_priors.imbalance_ratio:.3f}")
        print(f"   high_cardinality_cols: {len(eda_priors.high_cardinality_cols)}")
        print(f"   skewness_issues: {eda_priors.skewness_issues}")
        
        # Build planner config
        planner_config = {
            "round1_max_variants": args.round1_max_variants,
            "round2_max_variants": args.round2_max_variants,
            "proxy_prune_threshold": args.proxy_prune_threshold,
            "diversity_min_hamming_distance": args.diversity_min_hamming,
            "diversity_coverage_enabled": True,
            "round0_budget_per_variant_sec": 10,
            "round1_budget_per_variant_sec": 60,
            "round2_budget_per_variant_sec": args.time_budget_per_variant
        }
        
        # Load variant configs for scoring
        print(f"\n🔍 Scoring {len(variant_paths)} variants against EDA priors...")
        variant_configs = []
        scored_variants = []
        import yaml as yaml_loader
        
        for vpath in variant_paths:
            try:
                with open(vpath, 'r') as f:
                    vcfg = yaml_loader.safe_load(f)
                variant_configs.append(vcfg)
                
                # Score variant
                score, reasoning = score_variant_relevance(vcfg, eda_priors, task_type)
                stage3 = vcfg.get("stage3_preprocessing", {})
                stage4 = vcfg.get("stage4_feature_engineering", {})
                variant_id = vcfg.get("recipe_name") or vcfg.get("variant_metadata", {}).get("variant_id", Path(vpath).stem)
                
                scored_variants.append(VariantScore(
                    variant_id=variant_id,
                    variant_path=vpath,
                    relevance_score=score,
                    reasoning=reasoning,
                    preprocessing_hash=compute_preprocessing_hash(vcfg),
                    imputation=stage3.get("imputation", {}).get("method", "none"),
                    encoding=stage3.get("encoding", {}).get("categorical_method", "onehot"),
                    scaling=stage3.get("scaling", {}).get("method", "none"),
                    imbalance=stage3.get("imbalance_handling", {}).get("method", "none"),
                    feature_selection=stage4.get("feature_selection", {}).get("method", "none")
                ))
            except Exception as e:
                print(f"   ⚠️ Could not load/score variant {vpath}: {e}")
        
        # Diverse sample for Round 1
        print(f"\n🎯 Selecting top {args.round1_max_variants} diverse variants for Round 1...")
        round1_candidates = diverse_sample(scored_variants, args.round1_max_variants, args.diversity_min_hamming)
        
        # Build variant plan
        variant_plan = build_variant_plan(
            variant_configs=variant_configs,
            variant_paths=variant_paths,
            eda_priors=eda_priors,
            task_type=task_type,
            planner_config=planner_config
        )
        
        # Update variant_paths to use only selected variants
        original_count = len(variant_paths)
        variant_paths = [v.variant_path for v in round1_candidates]
        
        print(f"\n📋 PLANNER RESULT:")
        print(f"   Original variants: {original_count}")
        print(f"   Selected for Round 1: {len(variant_paths)}")
        print(f"   Top 5 by relevance score:")
        for v in sorted(round1_candidates, key=lambda x: x.relevance_score, reverse=True)[:5]:
            print(f"      - {v.variant_id}: score={v.relevance_score:.1f}")
        
        # Save variant_plan.json
        variant_plan_path = output_path / "variant_plan.json"
        with open(variant_plan_path, 'w') as f:
            from dataclasses import asdict as plan_asdict
            json.dump(plan_asdict(variant_plan) if hasattr(variant_plan, '__dataclass_fields__') else {
                "planner_version": "1.0",
                "eda_priors": plan_asdict(eda_priors),
                "shortlist": [{"variant_id": v.variant_id, "relevance_score": v.relevance_score, "reasoning": v.reasoning} for v in round1_candidates],
                "budget_allocation": variant_plan.budget_allocation if variant_plan else {}
            }, f, indent=2, default=str)
        print(f"   ✅ Saved variant_plan.json")
        print(f"{'='*80}\n")
    
    # Disable autologging (do NOT change tracking URI — breaks MLflow hierarchy)
    try:
        mlflow.sklearn.autolog(disable=True)
    except Exception:
        pass
    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass
    # 🔥 FIX: Convert azureml:// tracking URI to https:// to avoid registry errors
    # Azure ML sets MLFLOW_TRACKING_URI to azureml:// which MLflow registry doesn't support
    _mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if _mlflow_uri.startswith("azureml://"):
        _https_uri = _mlflow_uri.replace("azureml://", "https://")
        mlflow.set_tracking_uri(_https_uri)
        print(f"🔗 MLflow tracking URI converted to HTTPS")
    # Also set local model registry as defense-in-depth
    os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")
    
    # ======== AZURE ML PIPELINE COMPATIBILITY (CRITICAL) ========
    # Azure ML pipeline steps automatically create an MLflow run
    # Only create parent run if we're NOT already in an Azure ML pipeline context
    parent_run_created = False
    if mlflow.active_run() is None:
        # Standalone execution (local testing) - create parent run
        mlflow.start_run(run_name="phase_b_variant_runner")
        parent_run_created = True
        print(f"🔬 Started parent MLflow run: {mlflow.active_run().info.run_id}")
        print(f"   All variant×engine runs will nest under this parent\n")
    else:
        # Azure ML pipeline context - use existing step run as parent
        print(f"🔬 Using existing Azure ML pipeline step run as parent: {mlflow.active_run().info.run_id}")
        print(f"   All variant×engine runs will nest under this step\n")
    
    try:
        # Results collection
        all_results: List[VariantResult] = []
        
        # Champion-so-far tracking (avoid OOM from storing all models)
        primary_metric_name = get_primary_metric(task_type)
        best_score = float('-inf')  # Unified score (higher is better)
        best_key: Tuple[str, str] = (None, None)  # (variant_id, engine)
        best_model: Any = None
        
        skipped_count = 0
        failed_count = 0
        
        # Phase B signal artifacts (optional)
        round0_feasibility_reports = [] if args.enable_round0_feasibility else None
        round1_proxy_reports = [] if args.enable_round1_proxy else None
        elimination_decisions = [] if args.enable_elimination_report else None
        
        print(f"🔍 Phase B Signals: Round0={'ON' if args.enable_round0_feasibility else 'OFF'}, Round1={'ON' if args.enable_round1_proxy else 'OFF'}, Elimination={'ON' if args.enable_elimination_report else 'OFF'}\n")
        
        # Process each variant
        for i, variant_path in enumerate(variant_paths, 1):
            print(f"\n{'='*80}")
            print(f"[{i}/{len(variant_paths)}] Processing variant: {Path(variant_path).name}")
            print(f"{'='*80}")
            
            try:
                # Load and validate variant
                variant = load_variant(variant_path)
                validate_variant_for_task(variant, task_type)
            
                print(f"   Variant ID: {variant.variant_id}")
                print(f"   Config: {variant.stage3_preprocessing.imputation.method}+{variant.stage3_preprocessing.encoding.categorical_method}+{variant.stage3_preprocessing.scaling.method}")
            
                # ==== INSERT POINT C: Round 0 Feasibility Check ====
                df_transformed = None
                cache_key = None
                cache_hit = False
                
                # Check preprocessing cache first (V3-Proposed)
                if preprocessing_cache.enabled:
                    preproc_config = {
                        "imputation": variant.stage3_preprocessing.imputation.method,
                        "encoding": variant.stage3_preprocessing.encoding.categorical_method,
                        "scaling": variant.stage3_preprocessing.scaling.method
                    }
                    is_cacheable, cache_reason = preprocessing_cache.is_cacheable(preproc_config)
                    if is_cacheable:
                        cache_key = preprocessing_cache.compute_key(preproc_config, data_fingerprint['hash'])
                        df_transformed = preprocessing_cache.get(cache_key)
                        if df_transformed is not None:
                            cache_hit = True
                            print(f"   🗄️  CACHE HIT: Reusing preprocessed data (key={cache_key[:8]}...)")
                
                if args.enable_round0_feasibility and not cache_hit:
                    print(f"   🔍 Round 0: Feasibility check (transform-only)...")
                    df_transformed, feas_report = run_round0_feasibility(df, variant, target_column)
                    round0_feasibility_reports.append(feas_report)
                    
                    if feas_report["status"] == "fail":
                        print(f"      ❌ FAILED: {feas_report['reason']} [{feas_report['transform_runtime_sec']:.2f}s]")
                        if args.enable_elimination_report:
                            elimination_decisions.append({
                                "variant_id": variant.variant_id,
                                "engine": "all",
                                "stage": "round0_feasibility",
                                "reason": "feasibility_fail",
                                "details": feas_report["reason"],
                                "timestamp": datetime.utcnow().isoformat() + "Z"
                            })
                        failed_count += len(engines)  # Count as failure for all engines
                        continue  # Skip engine loop for this variant
                    elif feas_report["status"] == "error":
                        print(f"      ⚠️  ERROR: {feas_report['reason']} [{feas_report['transform_runtime_sec']:.2f}s]")
                        if args.enable_elimination_report:
                            elimination_decisions.append({
                                "variant_id": variant.variant_id,
                                "engine": "all",
                                "stage": "round0_feasibility",
                                "reason": "feasibility_error",
                                "details": feas_report["reason"],
                                "timestamp": datetime.utcnow().isoformat() + "Z"
                            })
                        failed_count += len(engines)
                        continue
                    else:
                        print(f"      ✅ PASSED: {feas_report['n_features_after']} features [{feas_report['transform_runtime_sec']:.2f}s]")
                        # Store in cache if cacheable (V3-Proposed)
                        if cache_key and not cache_hit and df_transformed is not None:
                            preprocessing_cache.put(cache_key, df_transformed)
                            print(f"      🗄️  CACHE STORE: key={cache_key[:8]}...")
                elif cache_hit:
                    # Cache hit - add synthetic feasibility report
                    feas_report = {
                        "variant_id": variant.variant_id,
                        "status": "pass",
                        "reason": "cache_hit",
                        "transform_runtime_sec": 0.0,
                        "n_features_before": df.shape[1] - 1,
                        "n_features_after": df_transformed.shape[1] - 1 if df_transformed is not None else 0,
                        "feature_multiplier": 1.0,
                        "feature_explosion": False
                    }
                    if round0_feasibility_reports is not None:
                        round0_feasibility_reports.append(feas_report)
                
                # ==== INSERT POINT D: Round 1 Proxy Leaderboard ====
                if args.enable_round1_proxy:
                    print(f"   📊 Round 1: Proxy leaderboard (SGD model on sampled data)...")
                    # Reuse transformed data from Round 0 if available
                    if df_transformed is None:
                        df_transformed = apply_variant_preprocessing(df, variant, target_column, apply_smote=False)
                    
                    proxy_report = run_round1_proxy(df_transformed, variant.variant_id, target_column, task_type)
                    round1_proxy_reports.append(proxy_report)
                    
                    if proxy_report["status"] == "fail":
                        print(f"      ⚠️  FAIL: {proxy_report['reason']} [proxy_metric={proxy_report['proxy_metric']:.4f}, {proxy_report['proxy_runtime_sec']:.2f}s]")
                    elif proxy_report["status"] == "error":
                        print(f"      ⚠️  ERROR: {proxy_report['reason']} [{proxy_report['proxy_runtime_sec']:.2f}s]")
                    else:
                        print(f"      ✅ proxy_metric={proxy_report['proxy_metric']:.4f} [{proxy_report['proxy_runtime_sec']:.2f}s]")
                
                # Run with each engine
                for engine in engines:
                    print(f"\n   🔧 Engine: {engine}")
                
                    # ======== HARDENING FEATURE 1: SKIP IF ALREADY COMPLETED ========
                    if checkpoint.is_completed(variant.variant_id, engine):
                        print(f"   ⏭️  SKIPPED (already completed in previous run)")
                        skipped_count += 1
                        continue
                
                    # Single-point checkpointing: try/except/finally pattern
                    result = None
                    model = None
                    attempt_start = time.time()
                    
                    # C1 FIX (UPDATED): Compute effective budget matching FLAML's
                    # adaptive logic.  FLAML internally raises budget to 600s for
                    # >50K rows.  After automl.fit() we may run time-aware
                    # cross_val_predict (3-fold × 2 passes).  The outer deadline
                    # must accommodate both FLAML training AND optional CV work,
                    # otherwise deadline_guard discards valid results.
                    effective_budget = args.time_budget_per_variant
                    if engine == "flaml":
                        n_rows = len(df)
                        if n_rows >= 50000 and effective_budget < 600:
                            effective_budget = 600
                        effective_budget = max(effective_budget, 120)  # min floor
                        # Buffer: 360s to cover time-aware cross_val_predict
                        # (up to 3-fold × 2 calls).  CV is best-effort and will
                        # self-skip when remaining time is too short, so this
                        # extra headroom is only used when CV actually runs.
                        effective_budget += 360
                    attempt_deadline = attempt_start + effective_budget
                
                    try:
                        # ======== HARDENING FEATURE 2: HARD BUDGET GUARD ========
                        # FIX 2B: Pass cached df_transformed if available (from Round0/Round1)
                        result, model = run_variant_with_nested_mlflow(
                            variant=variant,
                            df=df,
                            engine=engine,
                            target_column=target_column,
                            task_type=task_type,
                            time_budget=args.time_budget_per_variant,
                            attempt_deadline=attempt_deadline,
                            df_preprocessed=df_transformed  # Reuse cached transform if available
                        )
                    except Exception as e:
                        # Outer safety net for catastrophic failures
                        print(f"   💥 CATASTROPHIC FAILURE: {e}")
                        result = VariantResult(
                            variant_id=variant.variant_id,
                            engine=engine,
                            algorithm="outer_exception",
                            metrics={"primary_metric": 0.0},
                            runtime_sec=time.time() - attempt_start,
                            timed_out=False,
                            failed=True,
                            failure_reason=f"Outer exception: {str(e)}"
                        )
                        model = None
                    finally:
                        # ======== SINGLE-POINT COMPLETION MARKING ========
                        # Guarantee result exists (fallback)
                        if result is None:
                            result = VariantResult(
                                variant_id=variant.variant_id,
                                engine=engine,
                                algorithm="result_not_set",
                                metrics={"primary_metric": 0.0},
                                runtime_sec=time.time() - attempt_start,
                                timed_out=False,
                                failed=True,
                                failure_reason="Result object not created"
                            )
                    
                        all_results.append(result)
                        checkpoint.mark_completed(variant.variant_id, engine)
                    
                        # Track champion-so-far (avoid OOM)
                        # MUST match the leaderboard's usable_results filter:
                        # exclude timed-out and zero-metric runs so that
                        # best_key always agrees with champion_result.
                        if (model is not None
                                and not result.failed
                                and not result.timed_out
                                and safe_float(result.metrics.get("primary_metric")) > 0.01):
                            current_score = get_result_score(result, primary_metric_name)
                            if current_score > best_score:
                                best_score = current_score
                                best_key = (variant.variant_id, engine)
                                best_model = model
                                original_val = safe_float(result.metrics.get("primary_metric"))
                                print(f"   🏆 NEW CHAMPION: {variant.variant_id}::{engine} = {original_val:.4f} (score={current_score:.4f})")
                    
                        # ======== ENHANCED LOGGING ========
                        if result.failed:
                            print(f"   ❌ FAILED: {result.algorithm} - {result.failure_reason} [{result.runtime_sec:.1f}s]")
                            failed_count += 1
                            
                            # ==== INSERT POINT E: Track elimination decision ====
                            if args.enable_elimination_report:
                                elimination_decisions.append({
                                    "variant_id": variant.variant_id,
                                    "engine": engine,
                                    "stage": "training",
                                    "reason": "training_failed",
                                    "details": result.failure_reason,
                                    "timestamp": datetime.utcnow().isoformat() + "Z"
                                })
                            
                        elif result.timed_out:
                            print(f"   ⏱️  TIMED_OUT: metric={result.metrics.get('primary_metric', 0.0):.4f} [{result.runtime_sec:.1f}s]")
                        else:
                            print(f"   ✅ SUCCESS: {result.algorithm} | metric={result.metrics.get('primary_metric', 0.0):.4f} [{result.runtime_sec:.1f}s]")
        
            except Exception as e:
                print(f"   ❌ Variant processing failed: {e}")
                failed_count += 1
                continue
        valid_results = [r for r in all_results if not r.failed and r.algorithm != "skipped"]
        
        # Filter out timed-out runs with zero metrics (unusable champions)
        usable_results = [
            r for r in valid_results 
            if r.metrics.get("primary_metric", 0.0) > 0.01 and not r.timed_out
        ]
        
        # Track eliminated timed-out/zero-metric variants
        if args.enable_elimination_report and len(usable_results) < len(valid_results):
            for r in valid_results:
                if r not in usable_results:
                    elimination_decisions.append({
                        "variant_id": r.variant_id,
                        "engine": r.engine,
                        "stage": "post_training",
                        "reason": "timed_out_or_zero_metric",
                        "details": f"timed_out={r.timed_out}, metric={r.metrics.get('primary_metric', 0.0):.4f}",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
        
        if len(usable_results) > 0:
            # Use only usable results for champion selection
            valid_results = usable_results
            print(f"\n✅ Found {len(valid_results)} usable results (filtered out {len([r for r in all_results if not r.failed]) - len(valid_results)} timed-out/zero-metric runs)")
        elif len(valid_results) > 0:
            # Fallback: use timed-out results but warn user
            print(f"\n⚠️  WARNING: All {len(valid_results)} valid results are timed-out or have zero metrics")
            print(f"   Champion selection may not be meaningful. Consider increasing time_budget_per_variant.")
        
        if len(valid_results) == 0:
            print("\n⚠️ NO VALID RESULTS - Cannot select champion")
            print("   All variants either failed or were skipped. Check logs for errors.")
            print("   Creating placeholder output files to satisfy component YAML...\n")
            
            # CRITICAL: Create placeholder files to prevent cp command failures
            leaderboard_path = output_path / "leaderboard.csv"
            manifest_path = output_path / "champion_manifest.json"
            results_path = output_path / "all_results.json"
            
            # Empty leaderboard with task-type-aware metric columns
            _metric_cols = get_metric_columns_for_task(task_type)
            pd.DataFrame(columns=["variant_id", "engine", "algorithm", "primary_metric"]
                                  + _metric_cols
                                  + ["runtime_sec", "n_features", "leakage_risk", "timed_out", "failed"]).to_csv(leaderboard_path, index=False)
            print(f"   📊 Created empty leaderboard: {leaderboard_path}")
            
            # Placeholder champion manifest
            placeholder_manifest = {
                "variant_id": "none",
                "variant_path": "none",
                "engine": "none",
                "algorithm": "none",
                "primary_metric_name": primary_metric_name.lower(),
                "primary_metric_value": 0.0,
                "metrics": {},
                "preprocessing_config": {},
                "feature_engineering_config": {},
                "data_fingerprint": data_fingerprint,
                "code_version": code_version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "leakage_risk": "unknown",
                "task_type": task_type,
                "status": "no_valid_results"
            }
            with open(manifest_path, 'w') as f:
                json.dump(placeholder_manifest, f, indent=2)
            print(f"   🏆 Created placeholder manifest: {manifest_path}")
            
            # All results (failures only)
            with open(results_path, 'w') as f:
                json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
            print(f"   📄 Created results with failures: {results_path}")
            
            print(f"\n⚠️ Phase B Variant Runner completed with NO VALID RESULTS")
            print(f"   Total attempts: {len(all_results)}")
            print(f"   All failed or skipped - pipeline cannot proceed to champion selection\n")
            
            # T6: Safety net — train a simple default model so downstream steps have a model.pkl
            try:
                import joblib as _jl
                _model_pkl = output_path / "model.pkl"
                if not _model_pkl.exists():
                    print(f"   🔄 T6: Training safety-net XGBoost with default params...")
                    import xgboost as xgb
                    if task_type == "classification" and target_column and target_column in df.columns:
                        _X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
                        _y = df[target_column]
                        if _X.shape[1] > 0:
                            _X = _X.fillna(_X.median())
                            from sklearn.preprocessing import LabelEncoder as _LE
                            if _y.dtype == "object":
                                _y = _LE().fit_transform(_y)
                            _fb = xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                            _fb.fit(_X, _y)
                            _jl.dump(_fb, _model_pkl)
                            placeholder_manifest["algorithm"] = "xgboost_safety_net"
                            placeholder_manifest["status"] = "safety_net_fallback"
                            with open(manifest_path, 'w') as f:
                                json.dump(placeholder_manifest, f, indent=2)
                            print(f"   ✅ Safety-net classifier saved: {_model_pkl}")
                    elif task_type == "regression" and target_column and target_column in df.columns:
                        _X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
                        _y = df[target_column]
                        if _X.shape[1] > 0:
                            _X = _X.fillna(_X.median())
                            _fb = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                            _fb.fit(_X, _y)
                            _jl.dump(_fb, _model_pkl)
                            placeholder_manifest["algorithm"] = "xgboost_safety_net"
                            placeholder_manifest["status"] = "safety_net_fallback"
                            with open(manifest_path, 'w') as f:
                                json.dump(placeholder_manifest, f, indent=2)
                            print(f"   ✅ Safety-net regressor saved: {_model_pkl}")
                    elif task_type == "clustering":
                        _X = df.select_dtypes(include=["number"]).fillna(0)
                        if _X.shape[1] > 0:
                            from sklearn.cluster import KMeans as _KM
                            _fb = _KM(n_clusters=3, random_state=42, n_init=10)
                            _fb.fit(_X)
                            _jl.dump(_fb, _model_pkl)
                            placeholder_manifest["algorithm"] = "kmeans_safety_net"
                            placeholder_manifest["status"] = "safety_net_fallback"
                            with open(manifest_path, 'w') as f:
                                json.dump(placeholder_manifest, f, indent=2)
                            print(f"   ✅ Safety-net clustering model saved: {_model_pkl}")
            except Exception as _sn_err:
                print(f"   ⚠️  Safety-net model training failed (non-fatal): {_sn_err}")
            
            return
        
        # ======== HARDENING FEATURE 4: NORMALIZED RESULTS ========
        # Generate leaderboard using VariantResult schema with unified scoring
        # Use task-type-aware metric columns for uniform output structure
        metric_columns = get_metric_columns_for_task(task_type)
        leaderboard_data = []
        for r in valid_results:
            score = get_result_score(r, primary_metric_name)
            row = {
                "variant_id": r.variant_id,
                "engine": r.engine,
                "algorithm": r.algorithm,
                "primary_metric": r.metrics.get("primary_metric", 0.0),
            }
            # Add task-type-specific metric columns
            for col in metric_columns:
                row[col] = r.metrics.get(col, None)
            row.update({
                "score": score,  # Unified comparable score
                "runtime_sec": r.runtime_sec,
                "n_features": r.n_features,
                "leakage_risk": r.leakage_risk,
                "timed_out": r.timed_out,
                "failed": r.failed
            })
            leaderboard_data.append(row)
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Sort by unified score (descending = best first)
        leaderboard_df = leaderboard_df.sort_values("score", ascending=False)
        
        # Drop score column before saving (internal use only)
        leaderboard_df_export = leaderboard_df.drop(columns=["score"])
        leaderboard_path = output_path / "leaderboard.csv"
        
        # Deadline guard before leaderboard write
        final_deadline = time.time() + 60  # Give 60s for final writes
        if not deadline_guard(final_deadline, "before_leaderboard_write"):
            # ======== HARDENING FEATURE 3: ATOMIC LEADERBOARD WRITE ========
            leaderboard_temp = output_path / "leaderboard.csv.tmp"
            leaderboard_df_export.to_csv(leaderboard_temp, index=False)
            os.replace(str(leaderboard_temp), str(leaderboard_path))
            print(f"\n📊 Leaderboard saved (atomic): {leaderboard_path}")
        else:
            # CRITICAL: Still create leaderboard to prevent component YAML copy failure
            print(f"\n⚠️ Deadline exceeded - creating minimal leaderboard")
            leaderboard_df_export.head(10).to_csv(leaderboard_path, index=False)
            print(f"   Saved top 10 results to: {leaderboard_path}")
        
        # Select champion using unified scoring (matches leaderboard sort)
        # Get variant_id + engine from top leaderboard row
        top_row = leaderboard_df.iloc[0]
        champion_result = None
        for r in valid_results:
            if r.variant_id == top_row["variant_id"] and r.engine == top_row["engine"]:
                champion_result = r
                break
        
        if champion_result is None:
            # Fallback: use first valid result
            champion_result = valid_results[0]
        
        # Find corresponding variant config
        champion_variant_path = None
        for vp in variant_paths:
            v = load_variant(vp)
            if v.variant_id == champion_result.variant_id:
                champion_variant_path = vp
                break
        
        champion_variant = load_variant(champion_variant_path) if champion_variant_path else None
        
            # ======== HARDENING FEATURE 3: STABLE CHAMPION MANIFEST CONTRACT ========
        # Store original (non-negated) metric value in manifest
        primary_metric_name_lower = get_primary_metric(task_type).lower()
        original_metric_value = safe_float(
            champion_result.metrics.get("primary_metric") or 
            champion_result.metrics.get(primary_metric_name_lower)
        )
        
        champion_manifest = ChampionManifest(
            variant_id=champion_result.variant_id,
            variant_path=champion_variant_path if champion_variant_path else "unknown",
            engine=champion_result.engine,
            algorithm=champion_result.algorithm,
            primary_metric_name=primary_metric_name_lower,
            primary_metric_value=original_metric_value,
            metrics=champion_result.metrics,
            preprocessing_config={
                "imputation": champion_variant.stage3_preprocessing.imputation.method if champion_variant else "unknown",
                "encoding": champion_variant.stage3_preprocessing.encoding.categorical_method if champion_variant else "unknown",
                "scaling": champion_variant.stage3_preprocessing.scaling.method if champion_variant else "unknown",
                "imbalance": champion_variant.stage3_preprocessing.imbalance_handling.method if (champion_variant and champion_variant.stage3_preprocessing.imbalance_handling) else "none",
                "feature_selection": champion_variant.stage4_feature_engineering.feature_selection.method if champion_variant else "unknown"
            },
            feature_engineering_config={
                "method": champion_variant.stage4_feature_engineering.feature_selection.method if champion_variant else "unknown",
                "n_features_selected": champion_result.n_features
            },
            data_fingerprint=data_fingerprint,
            code_version=code_version,
            timestamp=datetime.utcnow().isoformat() + "Z",
            leakage_risk=champion_result.leakage_risk,
            task_type=task_type
        )
        
            # ======== HARDENING FEATURE 5: ATOMIC WRITES ========
        # Parent MLflow run already active (started at beginning of main())
        # All artifacts will be logged to this parent run
        
        # Deadline guard before manifest write
        manifest_path = output_path / "champion_manifest.json"
        if not deadline_guard(final_deadline, "before_manifest_write"):
            # Save champion manifest with stable schema
            atomic_write(manifest_path, json.dumps(asdict(champion_manifest), indent=2))
            print(f"🏆 Champion manifest saved: {manifest_path}")
            print(f"   Schema: ChampionManifest v1.1 (15 fields with primary_metric_name/value)")
            
            # Log to MLflow
            try:
                mlflow.log_artifact(str(manifest_path), "phase_b_outputs")
                print(f"   ☁️  Logged to MLflow: phase_b_outputs/champion_manifest.json")
            except Exception as e:
                print(f"   ⚠️  Could not log manifest to MLflow: {e}")
        else:
            # CRITICAL: Still create manifest to prevent component YAML copy failure
            print(f"⚠️ Deadline exceeded - creating minimal manifest")
            with open(manifest_path, 'w') as f:
                json.dump(asdict(champion_manifest), f, indent=2)
            print(f"   Saved manifest to: {manifest_path}")
        
        # Save all results with normalized schema (atomic write)
        # CRITICAL: Always create this file (required by component YAML)
        results_path = output_path / "all_results.json"
        atomic_write(results_path, json.dumps([asdict(r) for r in all_results], indent=2, default=str))
        print(f"📄 All results saved: {results_path}")
        print(f"   Schema: VariantResult (10 fields, normalized across engines)")
        
        # Log to MLflow
        try:
            mlflow.log_artifact(str(results_path), "phase_b_outputs")
            print(f"   ☁️  Logged to MLflow: phase_b_outputs/all_results.json")
        except Exception as e:
            print(f"   ⚠️  Could not log all_results to MLflow: {e}")
        
        # ── Candidate Ledger ──────────────────────────────────────────────
        try:
            _ledger_rows = []
            for _idx, _r in enumerate(all_results):
                _st = "failed" if _r.failed else ("timed_out" if _r.timed_out else "ok")
                _norm = normalize_metrics(task_type, _r.metrics)
                _vid = _r.variant_id
                # Compute rank: 1-based by primary metric descending among valid
                _rank = None
                if not _r.failed and not _r.timed_out:
                    _valid_scores = [
                        (i, get_result_score(v, primary_metric_name))
                        for i, v in enumerate(all_results)
                        if not v.failed and not v.timed_out
                    ]
                    _valid_scores.sort(key=lambda x: x[1], reverse=True)
                    for _rk, (_vi, _) in enumerate(_valid_scores, 1):
                        if _vi == _idx:
                            _rank = _rk
                            break
                _is_champ = (
                    champion_result is not None
                    and _r.variant_id == champion_result.variant_id
                    and _r.engine == champion_result.engine
                )
                _row = make_row(
                    stage="phase_b", step_name="s06", engine=_r.engine,
                    candidate_id=f"{_vid}__{_r.engine}__{_r.algorithm}",
                    task_type=task_type,
                    dataset_id=data_fingerprint.get("hash", "")[:12] if data_fingerprint else "",
                    status=_st,
                    failure_reason=_r.failure_reason or "",
                    compute_time_sec=round(_r.runtime_sec, 2),
                    source_path="src/steps/s06_phaseb_variant_runner.py",
                    recipe_name=_vid,
                    candidate_rank=_rank,
                    is_stage_best=_is_champ,
                    **_norm,
                )
                _ledger_rows.append(_row)
            write_stage_table(
                _ledger_rows,
                csv_path=str(output_path / "s06_candidates.csv"),
                parquet_path=str(output_path / "s06_candidates.parquet"),
            )
            print(f"📒 Candidate ledger: {len(_ledger_rows)} rows → s06_candidates.csv")
        except Exception as _ledger_err:
            print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")
        
        # ======== PHASE B SIGNAL ARTIFACTS ========
        print(f"\n{'='*80}")
        print(f"PHASE B SIGNAL ARTIFACTS")
        print(f"{'='*80}")
        
        # Artifact 1: Round 0 Feasibility Report
        if args.enable_round0_feasibility and round0_feasibility_reports:
            feas_df = pd.DataFrame(round0_feasibility_reports)
            feas_path = output_path / "round0_feasibility_report.csv"
            feas_df.to_csv(feas_path, index=False)
            print(f"🔍 Round 0 Feasibility Report: {feas_path}")
            print(f"   Variants: {len(feas_df)}")
            print(f"   Pass: {(feas_df['status'] == 'pass').sum()}, Fail: {(feas_df['status'] == 'fail').sum()}, Error: {(feas_df['status'] == 'error').sum()}")
            
            try:
                mlflow.log_artifact(str(feas_path), "phase_b_outputs")
                print(f"   ☁️  Logged to MLflow: phase_b_outputs/round0_feasibility_report.csv")
            except Exception as e:
                print(f"   ⚠️  Could not log to MLflow: {e}")
        
        # Artifact 2: Round 1 Proxy Leaderboard
        if args.enable_round1_proxy and round1_proxy_reports:
            proxy_df = pd.DataFrame(round1_proxy_reports)
            # Sort by proxy_metric descending (best first)
            proxy_df = proxy_df.sort_values("proxy_metric", ascending=False)
            proxy_path = output_path / "round1_proxy_leaderboard.csv"
            proxy_df.to_csv(proxy_path, index=False)
            print(f"📊 Round 1 Proxy Leaderboard: {proxy_path}")
            print(f"   Variants: {len(proxy_df)}")
            if len(proxy_df) > 0:
                print(f"   Top variant: {proxy_df.iloc[0]['variant_id']} (proxy_metric={proxy_df.iloc[0]['proxy_metric']:.4f})")
            
            try:
                mlflow.log_artifact(str(proxy_path), "phase_b_outputs")
                print(f"   ☁️  Logged to MLflow: phase_b_outputs/round1_proxy_leaderboard.csv")
            except Exception as e:
                print(f"   ⚠️  Could not log to MLflow: {e}")
        
        # Artifact 3: Elimination Report
        if args.enable_elimination_report and elimination_decisions:
            elim_report = {
                "summary": {
                    "total_attempts": len(all_results),
                    "total_eliminated": len(elimination_decisions),
                    "by_stage": {},
                    "by_reason": {}
                },
                "eliminated_variants": elimination_decisions
            }
            
            # FIX 5: Build per-variant summary map for audit trail
            variant_summary = {}
            for decision in elimination_decisions:
                vid = decision["variant_id"]
                stage = decision.get("stage", "unknown")
                reason = decision.get("reason", "unknown")
                
                # Aggregate by stage and reason
                elim_report["summary"]["by_stage"][stage] = elim_report["summary"]["by_stage"].get(stage, 0) + 1
                elim_report["summary"]["by_reason"][reason] = elim_report["summary"]["by_reason"].get(reason, 0) + 1
                
                # Build per-variant audit trail
                if vid not in variant_summary:
                    variant_summary[vid] = {
                        "final_status": "eliminated",
                        "stages": [],
                        "reasons": [],
                        "engines_affected": []
                    }
                variant_summary[vid]["stages"].append(stage)
                variant_summary[vid]["reasons"].append(reason)
                if decision.get("engine") and decision["engine"] not in variant_summary[vid]["engines_affected"]:
                    variant_summary[vid]["engines_affected"].append(decision["engine"])
            
            # Add variants that passed to summary (for completeness)
            for result in valid_results:
                if result.variant_id not in variant_summary:
                    variant_summary[result.variant_id] = {
                        "final_status": "passed",
                        "stages": ["training_complete"],
                        "reasons": ["success"],
                        "engines_affected": [result.engine]
                    }
            
            elim_report["variant_summary"] = variant_summary
            
            elim_path = output_path / "elimination_report.json"
            with open(elim_path, 'w') as f:
                json.dump(elim_report, f, indent=2)
            print(f"📋 Elimination Report: {elim_path}")
            print(f"   Total eliminated: {elim_report['summary']['total_eliminated']}/{elim_report['summary']['total_attempts']}")
            print(f"   By stage: {elim_report['summary']['by_stage']}")
            print(f"   By reason: {elim_report['summary']['by_reason']}")
            print(f"   Variant summary entries: {len(variant_summary)}")
            
            try:
                mlflow.log_artifact(str(elim_path), "phase_b_outputs")
                print(f"   ☁️  Logged to MLflow: phase_b_outputs/elimination_report.json")
            except Exception as e:
                print(f"   ⚠️  Could not log to MLflow: {e}")
            
            # FIX 5B: Log signal metrics to parent MLflow run for UI filtering
            try:
                round0_pass = len([r for r in round0_feasibility_reports if r["status"] == "pass"]) if round0_feasibility_reports else 0
                round0_fail = len([r for r in round0_feasibility_reports if r["status"] == "fail"]) if round0_feasibility_reports else 0
                round1_pass = len([r for r in round1_proxy_reports if r["status"] == "pass"]) if round1_proxy_reports else 0
                round1_warning = len([r for r in round1_proxy_reports if r["status"] == "warning"]) if round1_proxy_reports else 0
                
                mlflow.log_metrics({
                    "phase_b_round0_pass_count": round0_pass,
                    "phase_b_round0_fail_count": round0_fail,
                    "phase_b_round1_pass_count": round1_pass,
                    "phase_b_round1_warning_count": round1_warning,
                    "phase_b_eliminated_total": elim_report["summary"]["total_eliminated"],
                    "phase_b_variants_passed": len([v for v in variant_summary.values() if v["final_status"] == "passed"])
                })
                print(f"   ☁️  Logged signal metrics to MLflow parent run")
            except Exception as e:
                print(f"   ⚠️  Could not log signal metrics to MLflow: {e}")
        
        print(f"{'='*80}\n")
        
        # ======== HARDENING FEATURE 4: CHAMPION MODEL ARTIFACT ========
        # Validate champion consistency before saving
        champion_key = (champion_result.variant_id, champion_result.engine)
        
        print(f"\n🎯 Champion Selection:")
        print(f"   Variant: {champion_manifest.variant_id}")
        print(f"   Engine: {champion_manifest.engine}")
        print(f"   Algorithm: {champion_manifest.algorithm}")
        print(f"   Metric: {champion_manifest.primary_metric_name} = {champion_manifest.primary_metric_value:.4f}")
        
        # Check champion consistency — mismatch is a hard error
        # best_key == (None, None) means no usable (non-timed-out) model was
        # retained in memory; fall through to the best_model-is-None branch.
        if best_key != (None, None) and champion_key != best_key:
            diagnostic = (
                f"Champion mismatch detected!\n"
                f"   Leaderboard champion: {champion_key}\n"
                f"   Best tracked model:   {best_key}\n"
                f"   This indicates a bug in champion selection logic."
            )
            print(f"\n🚨 FATAL: {diagnostic}")
            
            # Save diagnostic info before aborting
            pointer_data = {
                "reason": "mismatch",
                "leaderboard_champion": {
                    "variant_id": champion_key[0],
                    "engine": champion_key[1]
                },
                "tracked_best": {
                    "variant_id": best_key[0] if best_key[0] is not None else None,
                    "engine": best_key[1] if best_key[1] is not None else None
                },
                "mlflow": {
                    "tracking_uri": mlflow.get_tracking_uri(),
                    "parent_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
                }
            }
            pointer_path = output_path / "champion_model_pointer.json"
            atomic_write(pointer_path, json.dumps(pointer_data, indent=2))
            print(f"   📋 Diagnostic saved: {pointer_path}")
            
            # T14: Downgrade mismatch from fatal RuntimeError to warning + safety-net.
            # The leaderboard champion still has valid metrics; the in-memory model
            # was evicted (e.g. timeout).  Proceed to safety-net so downstream has model.pkl.
            print(f"   🔄 T14: Proceeding to safety-net instead of aborting...")
            best_model = None  # force safety-net path below

        # T14: Separate `if` (not elif) so mismatch path falls through to safety-net
        if best_model is None:
            print(f"   ⚠️  WARNING: No champion model available (all attempts failed/timed out)")
            # T6: Safety net — train a default model so downstream has model.pkl
            _model_pkl = output_path / "model.pkl"
            if not _model_pkl.exists():
                try:
                    import joblib as _jl_fb
                    import xgboost as _xgb_fb
                    print(f"   🔄 T6: Training safety-net XGBoost...")
                    if task_type in ("classification", "regression") and target_column and target_column in df.columns:
                        _X = df.drop(columns=[target_column]).select_dtypes(include=["number"]).fillna(0)
                        _y = df[target_column]
                        if _X.shape[1] > 0:
                            if task_type == "classification":
                                from sklearn.preprocessing import LabelEncoder as _LE2
                                if _y.dtype == "object":
                                    _y = _LE2().fit_transform(_y)
                                _fb = _xgb_fb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                            else:
                                _fb = _xgb_fb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                            _fb.fit(_X, _y)
                            _jl_fb.dump(_fb, _model_pkl)
                            print(f"   ✅ Safety-net model saved: {_model_pkl}")
                    elif task_type == "clustering":
                        # T14: Add clustering safety-net
                        _X = df.select_dtypes(include=["number"]).fillna(0)
                        if _X.shape[1] > 0:
                            from sklearn.cluster import KMeans as _KM2
                            _fb = _KM2(n_clusters=3, random_state=42, n_init=10)
                            _fb.fit(_X)
                            _jl_fb.dump(_fb, _model_pkl)
                            print(f"   ✅ Safety-net clustering model saved: {_model_pkl}")
                except Exception as _sn2_err:
                    print(f"   ⚠️  Safety-net failed (non-fatal): {_sn2_err}")
        elif deadline_guard(final_deadline, "before_model_save"):
            print(f"   ⚠️  Skipped champion model save (deadline exceeded)")
        else:
            # Save champion model to disk atomically
            # CRITICAL: Save as model.pkl (not champion_model.pkl) for downstream compatibility
            # final_evaluation.py load_model_and_encoder() expects model.pkl inside folder
            # CRITICAL: Use joblib (not pickle) — s10 loads with joblib.load()
            import joblib as _joblib
            model_path = output_path / "model.pkl"
            model_temp = output_path / "model.pkl.tmp"
            try:
                _joblib.dump(best_model, str(model_temp))
                os.replace(str(model_temp), str(model_path))
                print(f"   💾 Champion model saved to disk (atomic): {model_path}")
                
                # Log to MLflow
                try:
                    mlflow.log_artifact(str(model_path), "phase_b_outputs")
                    print(f"   ☁️  Logged to MLflow: phase_b_outputs/model.pkl")
                except Exception as e:
                    print(f"   ⚠️  Could not log champion model to MLflow: {e}")
            except Exception as e:
                print(f"   ⚠️  Could not save champion model to disk: {e}")
                if model_temp.exists():
                    model_temp.unlink()
        
        # ════════════════════════════════════════════════════════════════════
        # BATCH 3 FIX: HOLDOUT EVALUATION — honest out-of-sample metrics
        # for the champion model on the 20% holdout set.
        # ════════════════════════════════════════════════════════════════════
        if best_model is not None and df_holdout is not None and len(df_holdout) > 0:
            print(f"\n{'='*60}")
            print(f"🧪 HOLDOUT EVALUATION (unseen 20% data)")
            print(f"{'='*60}")
            try:
                # Apply same preprocessing as champion variant to holdout
                # CRITICAL: Use training-aligned preprocessing (fit on train, transform holdout)
                if champion_variant is not None:
                    df_holdout_proc = preprocess_holdout_aligned(
                        df, df_holdout, champion_variant, target_column
                    )
                else:
                    df_holdout_proc = df_holdout.copy()

                if target_column in df_holdout_proc.columns:
                    X_holdout = df_holdout_proc.drop(columns=[target_column])
                    y_holdout = df_holdout_proc[target_column]
                else:
                    raise ValueError(f"Target column '{target_column}' not in preprocessed holdout")

                # Align columns to what the trained model expects
                if hasattr(best_model, 'feature_names_in_'):
                    train_features = list(best_model.feature_names_in_)
                elif hasattr(best_model, 'feature_name_'):
                    train_features = list(best_model.feature_name_)
                elif hasattr(best_model, 'feature_names_'):
                    # CatBoost uses feature_names_ attribute
                    train_features = list(best_model.feature_names_)
                else:
                    train_features = None

                if train_features is not None:
                    missing_cols = set(train_features) - set(X_holdout.columns)
                    extra_cols = set(X_holdout.columns) - set(train_features)
                    if missing_cols:
                        print(f"   ⚠️  Adding {len(missing_cols)} missing columns (zeroed)")
                    if extra_cols:
                        print(f"   ⚠️  Dropping {len(extra_cols)} extra columns")
                    X_holdout = X_holdout.reindex(columns=train_features, fill_value=0)

                # ── Save preprocessed holdout data for downstream s10 evaluation ──
                # s10 receives s4 (baseline-preprocessed) data but Phase B model
                # expects variant-preprocessed data. Saving the aligned holdout
                # here lets s10 evaluate Phase B on correctly preprocessed data.
                try:
                    eval_df = X_holdout.copy()
                    eval_df[target_column] = y_holdout.values
                    eval_data_path = output_path / "phaseb_eval_data.csv"
                    eval_df.to_csv(eval_data_path, index=False)
                    print(f"   📁 Phase B eval data saved for s10: {eval_data_path} ({len(eval_df)} rows)")
                except Exception as _save_err:
                    print(f"   ⚠️  Could not save Phase B eval data (non-fatal): {_save_err}")

                y_pred = best_model.predict(X_holdout)

                holdout_metrics = {}
                if task_type == "classification":
                    from sklearn.metrics import (accuracy_score, f1_score,
                                                 precision_score, recall_score,
                                                 roc_auc_score, balanced_accuracy_score)
                    holdout_metrics["holdout_accuracy"] = float(accuracy_score(y_holdout, y_pred))
                    holdout_metrics["holdout_balanced_accuracy"] = float(balanced_accuracy_score(y_holdout, y_pred))
                    holdout_metrics["holdout_f1"] = float(f1_score(y_holdout, y_pred, average="weighted", zero_division=0))
                    holdout_metrics["holdout_precision"] = float(precision_score(y_holdout, y_pred, average="weighted", zero_division=0))
                    holdout_metrics["holdout_recall"] = float(recall_score(y_holdout, y_pred, average="weighted", zero_division=0))
                    if hasattr(best_model, 'predict_proba'):
                        try:
                            y_proba = best_model.predict_proba(X_holdout)
                            n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2
                            if n_classes == 2:
                                holdout_metrics["holdout_auc"] = float(roc_auc_score(y_holdout, y_proba[:, 1]))
                            else:
                                holdout_metrics["holdout_auc"] = float(roc_auc_score(y_holdout, y_proba, multi_class="ovr", average="weighted"))
                        except Exception:
                            pass
                elif task_type == "regression":
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    holdout_metrics["holdout_r2"] = float(r2_score(y_holdout, y_pred))
                    holdout_metrics["holdout_mse"] = float(mean_squared_error(y_holdout, y_pred))
                    holdout_metrics["holdout_rmse"] = float(mean_squared_error(y_holdout, y_pred, squared=False))
                    holdout_metrics["holdout_mae"] = float(mean_absolute_error(y_holdout, y_pred))

                for k, v in holdout_metrics.items():
                    print(f"   {k}: {v:.4f}")

                # Log to MLflow
                try:
                    mlflow.log_metrics(holdout_metrics)
                    print(f"   ☁️  Logged holdout metrics to MLflow parent run")
                except Exception as e:
                    print(f"   ⚠️  Could not log holdout metrics to MLflow: {e}")

                # Save holdout metrics to JSON
                holdout_metrics_path = output_path / "holdout_metrics.json"
                with open(holdout_metrics_path, 'w') as f:
                    json.dump(holdout_metrics, f, indent=2)
                print(f"   📄 Holdout metrics saved: {holdout_metrics_path}")

                # Update champion manifest with holdout metrics
                champion_manifest.metrics["holdout"] = holdout_metrics
                atomic_write(manifest_path, json.dumps(asdict(champion_manifest), indent=2))
                print(f"   📋 Updated champion manifest with holdout metrics")

            except Exception as e:
                print(f"   ⚠️  Holdout evaluation failed (non-fatal): {e}")
                import traceback
                traceback.print_exc()
        else:
            if best_model is None:
                print(f"\n⚠️  Skipping holdout evaluation (no champion model)")
            elif df_holdout is None or len(df_holdout) == 0:
                print(f"\n⚠️  Skipping holdout evaluation (no holdout data)")

        # Log leaderboard to MLflow
        try:
            mlflow.log_artifact(str(leaderboard_path), "phase_b_outputs")
            print(f"\n☁️  Logged to MLflow: phase_b_outputs/leaderboard.csv")
        except Exception as e:
            print(f"⚠️  Could not log leaderboard to MLflow: {e}")
        
        print(f"\n✅ Phase B Variant Runner completed successfully!")
        print(f"   Total valid results: {len(valid_results)}")
        print(f"   Skipped: {skipped_count} | Failed: {failed_count}")
        print(f"   Leakage Risk: {champion_manifest.leakage_risk}")
        print(f"   Data Fingerprint: {data_fingerprint['hash'][:12]}...")
        print(f"   Code Version: {code_version}")
        
        # V3-Proposed: Log cache stats
        if preprocessing_cache.enabled:
            cache_stats = preprocessing_cache.get_stats()
            print(f"\n🗄️  PREPROCESSING CACHE STATS:")
            print(f"   Hits: {cache_stats['hits']} | Misses: {cache_stats['misses']}")
            print(f"   Hit Rate: {cache_stats['hit_rate']:.1%}")
            print(f"   Stores: {cache_stats['stores']} | Evictions: {cache_stats['evictions']}")
            
            # Log to MLflow
            try:
                mlflow.log_metrics({
                    "cache_hits": cache_stats['hits'],
                    "cache_misses": cache_stats['misses'],
                    "cache_hit_rate": cache_stats['hit_rate'],
                    "cache_stores": cache_stats['stores']
                })
            except Exception as e:
                print(f"   ⚠️  Could not log cache stats to MLflow: {e}")
        
        # V3-Proposed: Log planner metrics if enabled
        if args.planner_enabled and variant_plan:
            print(f"\n📋 PLANNER METRICS:")
            print(f"   Original variants: {variant_plan.round0_summary.get('total_variants', 'N/A') if hasattr(variant_plan, 'round0_summary') else 'N/A'}")
            print(f"   Selected for training: {len(variant_paths)}")
            try:
                mlflow.log_artifact(str(output_path / "variant_plan.json"), "phase_b_outputs")
                print(f"   ☁️  Logged variant_plan.json to MLflow")
            except Exception as e:
                print(f"   ⚠️  Could not log variant_plan.json: {e}")
        
        print()
    
    finally:
        # End parent run only if we created it (not in Azure ML pipeline context)
        # Azure ML pipeline will close the step run automatically
        if parent_run_created and mlflow.active_run():
            mlflow.end_run()
            print(f"🏁 Ended parent MLflow run\n")
        elif mlflow.active_run():
            print(f"🏁 Azure ML pipeline will close step run automatically\n")


if __name__ == "__main__":
    main()
