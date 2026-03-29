"""
Dataset Profiler - Intelligent Recipe Recommendation System

Analyzes dataset characteristics to recommend optimal preprocessing strategies.
This enables data-driven variant selection instead of blind grid search.

Author: MLOps Solution Accelerator V3
Date: 2026-01-26
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class DatasetProfile:
    """Rich dataset characteristics for intelligent recipe recommendation.
    
    This profile captures statistical properties, quality issues, and domain signals
    that drive preprocessing strategy selection.
    """
    
    # Basic dimensions
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    target_column: Optional[str] = None
    target_type: str = "unknown"  # binary, multiclass, continuous, none (clustering)
    
    # Quality issues
    missing_rate: float = 0.0
    missing_patterns: Dict[str, float] = field(default_factory=dict)
    imbalance_ratio: float = 1.0  # minority/majority class (classification only)
    outlier_prevalence: float = 0.0
    high_cardinality_cats: List[str] = field(default_factory=list)
    
    # Correlation structure
    feature_correlation_mean: float = 0.0
    feature_correlation_max: float = 0.0
    multicollinearity_detected: bool = False
    
    # Domain signals
    domain_hints: List[str] = field(default_factory=list)
    data_sensitivity: str = "low"  # low, medium, high (PII detection)
    
    # Recommendations cache
    _recommendations: Optional[Dict[str, Any]] = None
    
    def recommend_preprocessing_strategies(self) -> Dict[str, Any]:
        """Generate preprocessing recommendations based on dataset profile.
        
        Returns:
            Dict with recommended strategies for each preprocessing dimension:
            - imputation: List of methods to try
            - encoding: List of encoding strategies
            - scaling: List of scaling methods
            - imbalance_handling: List of resampling techniques
            - feature_selection: List of feature selection methods
            - priority_scores: Which dimensions matter most for this dataset
        """
        if self._recommendations is not None:
            return self._recommendations
            
        recommendations = {
            "imputation": [],
            "encoding": [],
            "scaling": [],
            "imbalance_handling": [],
            "feature_selection": [],
            "priority_scores": {},
            "reasoning": []
        }
        
        # === IMPUTATION STRATEGY ===
        if self.missing_rate > 0.2:
            recommendations["imputation"] = [
                "knn", "iterative", "median",
                "forward_fill", "interpolate_linear",  # Tier 1 additions
            ]
            recommendations["priority_scores"]["imputation"] = 0.9
            recommendations["reasoning"].append(
                f"High missing rate ({self.missing_rate:.1%}) → Advanced imputation needed (ML-based + interpolation)"
            )
        elif self.missing_rate > 0.05:
            recommendations["imputation"] = [
                "mean", "median", "knn",
                "mode", "constant",  # Tier 1 additions
            ]
            recommendations["priority_scores"]["imputation"] = 0.6
            recommendations["reasoning"].append(
                f"Moderate missing rate ({self.missing_rate:.1%}) → Standard + mode/constant imputation"
            )
        else:
            recommendations["imputation"] = ["mean", "drop"]
            recommendations["priority_scores"]["imputation"] = 0.2
            recommendations["reasoning"].append(
                f"Low missing rate ({self.missing_rate:.1%}) → Simple imputation sufficient"
            )
        
        # === ENCODING STRATEGY ===
        if len(self.high_cardinality_cats) > 0:
            recommendations["encoding"] = ["target", "catboost", "onehot"]
            recommendations["priority_scores"]["encoding"] = 0.8
            recommendations["reasoning"].append(
                f"High-cardinality categoricals detected ({len(self.high_cardinality_cats)}) → Target encoding recommended"
            )
        elif self.n_categorical > self.n_numeric:
            recommendations["encoding"] = ["onehot", "label"]
            recommendations["priority_scores"]["encoding"] = 0.7
        else:
            recommendations["encoding"] = ["onehot", "label"]
            recommendations["priority_scores"]["encoding"] = 0.5
        
        # === SCALING STRATEGY ===
        if "finance" in self.domain_hints or self.outlier_prevalence > 0.1:
            recommendations["scaling"] = ["robust", "minmax"]
            recommendations["priority_scores"]["scaling"] = 0.8
            recommendations["reasoning"].append(
                f"Finance domain or outliers ({self.outlier_prevalence:.1%}) → Robust scaling"
            )
        elif "academic" in self.domain_hints:
            recommendations["scaling"] = ["standard", "none"]
            recommendations["priority_scores"]["scaling"] = 0.6
        else:
            recommendations["scaling"] = ["standard", "robust", "minmax", "none"]
            recommendations["priority_scores"]["scaling"] = 0.5
        
        # === IMBALANCE HANDLING (Classification only) ===
        if self.target_type in ["binary", "multiclass"]:
            if self.imbalance_ratio < 0.3:  # Severe imbalance
                recommendations["imbalance_handling"] = ["smote", "smoteenn", "smotetomek"]
                recommendations["priority_scores"]["imbalance_handling"] = 0.95
                recommendations["reasoning"].append(
                    f"Severe class imbalance ({self.imbalance_ratio:.2f}) → SMOTE variants critical"
                )
            elif self.imbalance_ratio < 0.5:
                recommendations["imbalance_handling"] = ["smote", "none"]
                recommendations["priority_scores"]["imbalance_handling"] = 0.7
                recommendations["reasoning"].append(
                    f"Moderate class imbalance ({self.imbalance_ratio:.2f}) → SMOTE recommended"
                )
            else:
                recommendations["imbalance_handling"] = ["none"]
                recommendations["priority_scores"]["imbalance_handling"] = 0.2
                recommendations["reasoning"].append(
                    f"Balanced classes ({self.imbalance_ratio:.2f}) → No resampling needed"
                )
        else:
            recommendations["imbalance_handling"] = ["none"]
            recommendations["priority_scores"]["imbalance_handling"] = 0.0
        
        # === FEATURE SELECTION ===
        if self.multicollinearity_detected or self.feature_correlation_max > 0.9:
            recommendations["feature_selection"] = ["correlation", "variance", "mutual_info"]
            recommendations["priority_scores"]["feature_selection"] = 0.85
            recommendations["reasoning"].append(
                f"High multicollinearity detected (max={self.feature_correlation_max:.2f}) → Feature selection critical"
            )
        elif self.n_features > 50:
            recommendations["feature_selection"] = ["variance", "mutual_info", "correlation"]
            recommendations["priority_scores"]["feature_selection"] = 0.7
            recommendations["reasoning"].append(
                f"High dimensionality ({self.n_features} features) → Feature selection recommended"
            )
        else:
            recommendations["feature_selection"] = ["none", "correlation"]
            recommendations["priority_scores"]["feature_selection"] = 0.3
        
        self._recommendations = recommendations
        return recommendations
    
    def generate_profile_summary(self) -> str:
        """Generate human-readable profile summary."""
        recommendations = self.recommend_preprocessing_strategies()
        
        summary = f"""
=== DATASET PROFILE SUMMARY ===
Dimensions: {self.n_rows} rows × {self.n_features} features
  - Numeric: {self.n_numeric}
  - Categorical: {self.n_categorical}
Target: {self.target_column} ({self.target_type})

Quality Issues:
  - Missing rate: {self.missing_rate:.1%}
  - Imbalance ratio: {self.imbalance_ratio:.2f}
  - Outlier prevalence: {self.outlier_prevalence:.1%}
  - High-cardinality categoricals: {len(self.high_cardinality_cats)}
  - Multicollinearity: {"YES" if self.multicollinearity_detected else "NO"} (max={self.feature_correlation_max:.2f})

Domain Signals: {', '.join(self.domain_hints) if self.domain_hints else 'Generic'}

PREPROCESSING RECOMMENDATIONS:
"""
        for reason in recommendations["reasoning"]:
            summary += f"  • {reason}\n"
        
        summary += f"\nPRIORITY DIMENSIONS (sorted by impact):\n"
        sorted_priorities = sorted(
            recommendations["priority_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for dim, score in sorted_priorities:
            if score > 0.5:
                summary += f"  • {dim}: {score:.1f}/1.0\n"
        
        return summary


class DatasetProfiler:
    """Analyzes datasets and generates preprocessing recommendations."""
    
    def __init__(self, task_type: str):
        """
        Args:
            task_type: "classification", "regression", or "clustering"
        """
        self.task_type = task_type
    
    def profile_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> DatasetProfile:
        """Profile a dataset and generate preprocessing recommendations.
        
        Args:
            df: Input dataframe
            target_column: Target column name (None for clustering)
            
        Returns:
            DatasetProfile with rich characteristics and recommendations
        """
        profile = DatasetProfile(
            n_rows=len(df),
            n_features=len(df.columns) - (1 if target_column else 0),
            n_numeric=len(df.select_dtypes(include=[np.number]).columns),
            n_categorical=len(df.select_dtypes(include=['object', 'category']).columns),
            target_column=target_column
        )
        
        # Detect target type
        if target_column and target_column in df.columns:
            if self.task_type == "classification":
                n_classes = df[target_column].nunique()
                profile.target_type = "binary" if n_classes == 2 else "multiclass"
            elif self.task_type == "regression":
                profile.target_type = "continuous"
        elif self.task_type == "clustering":
            profile.target_type = "none"
        
        # Calculate missing rate
        profile.missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        profile.missing_patterns = {
            col: df[col].isnull().mean()
            for col in df.columns
            if df[col].isnull().any()
        }
        
        # Class imbalance (classification only)
        if target_column and profile.target_type in ["binary", "multiclass"]:
            value_counts = df[target_column].value_counts()
            if len(value_counts) > 0:
                profile.imbalance_ratio = value_counts.min() / value_counts.max()
        
        # Outlier prevalence (numeric features only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_counts = []
            for col in numeric_cols:
                if col != target_column:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_counts.append(outliers / len(df))
            profile.outlier_prevalence = np.mean(outlier_counts) if outlier_counts else 0.0
        
        # High-cardinality categoricals (>100 unique values)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        profile.high_cardinality_cats = [
            col for col in cat_cols
            if col != target_column and df[col].nunique() > 100
        ]
        
        # Correlation structure
        if len(numeric_cols) > 1:
            feature_cols = [c for c in numeric_cols if c != target_column]
            if len(feature_cols) > 1:
                corr_matrix = df[feature_cols].corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                profile.feature_correlation_mean = corr_matrix.mean().mean()
                profile.feature_correlation_max = corr_matrix.max().max()
                profile.multicollinearity_detected = profile.feature_correlation_max > 0.85
        
        # Domain detection heuristics
        profile.domain_hints = self._detect_domain_hints(df)
        
        return profile
    
    def _detect_domain_hints(self, df: pd.DataFrame) -> List[str]:
        """Detect domain signals from column names and data patterns."""
        hints = []
        
        # Finance signals
        finance_keywords = ['price', 'amount', 'balance', 'transaction', 'credit', 'debit', 'revenue']
        if any(keyword in ' '.join(df.columns).lower() for keyword in finance_keywords):
            hints.append("finance")
        
        # Time series signals
        time_keywords = ['date', 'time', 'timestamp', 'year', 'month', 'day']
        if any(keyword in ' '.join(df.columns).lower() for keyword in time_keywords):
            hints.append("time_series")
        
        # Academic/research signals
        academic_keywords = ['score', 'grade', 'test', 'exam', 'student']
        if any(keyword in ' '.join(df.columns).lower() for keyword in academic_keywords):
            hints.append("academic")
        
        # Healthcare signals
        healthcare_keywords = ['patient', 'diagnosis', 'treatment', 'symptom', 'medical']
        if any(keyword in ' '.join(df.columns).lower() for keyword in healthcare_keywords):
            hints.append("healthcare")
        
        return hints
