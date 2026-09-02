import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import mlflow
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Azure ML
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_style('whitegrid')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger
from utils.data_identity import canonical_dataframe_sha256
from utils.holdout_partition import (
    ROW_ID_COLUMN,
    SPLIT_COLUMN,
    TRAIN_PARTITION,
    ensure_holdout_partition,
)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config from local file path."""
    print(f"📖 Loading config from: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset_uri(cfg: Dict[str, Any]) -> str:
    """Build azureml:// datastore URI from config."""
    azure_cfg = cfg.get("azure_ml") or cfg.get("azureml") or {}
    ds_cfg = cfg.get("dataset", {}) or {}
    
    datastore_name = ds_cfg.get("datastore_name")
    blob_path = ds_cfg.get("blob_path")
    
    if not (azure_cfg and datastore_name and blob_path):
        raise ValueError(
            "Missing required config fields. Need: "
            "azure_ml.{subscription_id,resource_group,workspace_name}, "
            "dataset.{datastore_name,blob_path}"
        )
    
    return (
        f"azureml://subscriptions/{azure_cfg['subscription_id']}"
        f"/resourcegroups/{azure_cfg['resource_group']}"
        f"/workspaces/{azure_cfg['workspace_name']}"
        f"/datastores/{datastore_name}"
        f"/paths/{blob_path}"
    )


def validate_data_quality(df: pd.DataFrame, config: Dict[str, Any], task_type: str, target_col: str) -> List[str]:
    """
    Validate data quality with RED/YELLOW/GREEN decision gates.
    Returns list of issues with severity indicators.
    """
    issues = []
    stage1_cfg = config.get("stage1", {})
    
    # RED: Insufficient rows
    min_rows = stage1_cfg.get("min_rows", 1000)
    if len(df) < min_rows:
        issues.append(f"🔴 RED: Insufficient rows ({len(df)} < {min_rows}). Cannot proceed with reliable modeling.")
    
    # RED: Excessive missing data
    overall_missing = df.isna().mean().mean() * 100
    max_missing_pct = stage1_cfg.get("max_missing_pct", 50)
    if overall_missing > max_missing_pct:
        issues.append(f"🔴 RED: Excessive missing data ({overall_missing:.1f}% > {max_missing_pct}%). Data quality too low.")
    
    # YELLOW: High missing data (warning)
    elif overall_missing > max_missing_pct * 0.5:
        issues.append(f"🟡 YELLOW: High missing data ({overall_missing:.1f}%). Consider data quality improvements.")
    
    # RED/YELLOW: Classification-specific checks
    if task_type == "classification" and target_col:
        value_counts = df[target_col].value_counts()
        min_samples = stage1_cfg.get("classification_min_samples_per_class", 30)
        
        # Check minimum samples per class
        if value_counts.min() < min_samples:
            minority_class = value_counts.idxmin()
            minority_count = value_counts.min()
            if minority_count < min_samples * 0.5:
                issues.append(f"🔴 RED: Insufficient samples in minority class '{minority_class}' ({minority_count} < {min_samples}). Risk of overfitting.")
            else:
                issues.append(f"🟡 YELLOW: Low samples in minority class '{minority_class}' ({minority_count}). Consider oversampling.")
        
        # Check class imbalance
        imbalance_ratio = value_counts.min() / value_counts.max()
        if imbalance_ratio < 0.1:
            issues.append(f"🟡 YELLOW: Severe class imbalance (ratio {imbalance_ratio:.3f}). Consider SMOTE or class weights.")
    
    # RED/YELLOW: Regression-specific checks
    if task_type == "regression" and target_col:
        target_variance = df[target_col].var()
        min_variance = stage1_cfg.get("regression_target_min_variance", 0.01)
        
        if target_variance < min_variance:
            issues.append(f"🔴 RED: Target variance too low ({target_variance:.6f}). Target may be constant or near-constant.")
        
        # Check for outliers using IQR method
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[target_col] < Q1 - 3*IQR) | (df[target_col] > Q3 + 3*IQR)]
        outlier_pct = len(outliers) / len(df) * 100
        
        if outlier_pct > 10:
            issues.append(f"🟡 YELLOW: High outlier percentage ({outlier_pct:.1f}%). Consider outlier treatment.")
    
    # YELLOW: High duplicate rows
    duplicate_pct = df.duplicated().sum() / len(df) * 100
    if duplicate_pct > 5:
        issues.append(f"🟡 YELLOW: High duplicate rows ({duplicate_pct:.1f}%). Consider deduplication.")
    
    # GREEN: All checks passed
    if not issues:
        issues.append("🟢 GREEN: Data quality checks passed. Dataset ready for modeling.")
    
    return issues


def calculate_decision_gate_status(issues: List[str]) -> str:
    """Calculate overall decision gate status from issues list."""
    if any("🔴 RED" in issue for issue in issues):
        return "RED"
    elif any("🟡 YELLOW" in issue for issue in issues):
        return "YELLOW"
    else:
        return "GREEN"


def generate_advanced_eda_visualizations(df: pd.DataFrame, target_col: str, task_type: str, output_dir: Path, col_types: dict) -> dict:
    """
    Generate advanced EDA visualizations: missing value heatmap, outlier boxplots, target distribution.
    Returns: dict of generated file paths
    """
    print("📊 Generating advanced EDA visualizations...")
    generated_files = {}
    
    # 1. MISSING VALUE HEATMAP
    try:
        plt.figure(figsize=(12, 8))
        missing_data = df.isna()
        if missing_data.any().any():
            # Show only columns with missing values
            cols_with_missing = [col for col in df.columns if df[col].isna().any()]
            if cols_with_missing:
                sample_size = min(500, len(df))  # Heatmap for first 500 rows
                sns.heatmap(missing_data[cols_with_missing].head(sample_size), cbar=True, yticklabels=False, cmap='viridis')
                plt.title(f'Missing Value Heatmap (First {sample_size} rows)', fontsize=14, fontweight='bold')
                plt.xlabel('Columns', fontsize=12)
                plt.ylabel('Row Index', fontsize=12)
                plt.tight_layout()
                heatmap_path = output_dir / 'stage1_missing_value_heatmap.png'
                plt.savefig(heatmap_path, dpi=100, bbox_inches='tight')
                plt.close()
                generated_files['missing_heatmap'] = str(heatmap_path)
                print(f"  ✅ Missing value heatmap: {heatmap_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to generate missing value heatmap: {e}")
    
    # 2. OUTLIER DETECTION BOXPLOTS (Top 6 numeric columns)
    try:
        numeric_cols = col_types.get('numeric', [])
        if numeric_cols:
            top_numeric = numeric_cols[:6]  # Limit to 6 for readability
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for idx, col in enumerate(top_numeric):
                if idx < 6:
                    sns.boxplot(data=df, y=col, ax=axes[idx], color='skyblue')
                    axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
                    axes[idx].set_ylabel('Value', fontsize=10)
            # Hide unused subplots
            for idx in range(len(top_numeric), 6):
                axes[idx].axis('off')
            plt.suptitle('Outlier Detection (Top 6 Numeric Features)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            boxplot_path = output_dir / 'stage1_outlier_boxplots.png'
            plt.savefig(boxplot_path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_files['outlier_boxplots'] = str(boxplot_path)
            print(f"  ✅ Outlier boxplots: {boxplot_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to generate outlier boxplots: {e}")
    
    # 3. TARGET DISTRIBUTION
    try:
        if target_col and target_col in df.columns and task_type != 'clustering':
            plt.figure(figsize=(10, 6))
            if task_type == 'classification':
                target_counts = df[target_col].value_counts()
                target_counts.plot(kind='bar', color='coral', edgecolor='black')
                plt.title(f'Target Distribution: {target_col}', fontsize=14, fontweight='bold')
                plt.xlabel('Class', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                # Add percentage labels
                total = len(df)
                for i, v in enumerate(target_counts):
                    plt.text(i, v + total*0.01, f'{v/total*100:.1f}%', ha='center', fontsize=10)
            elif task_type == 'regression':
                sns.histplot(df[target_col].dropna(), bins=50, kde=True, color='coral', edgecolor='black')
                plt.title(f'Target Distribution: {target_col}', fontsize=14, fontweight='bold')
                plt.xlabel(target_col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            target_dist_path = output_dir / 'stage1_target_distribution.png'
            plt.savefig(target_dist_path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_files['target_distribution'] = str(target_dist_path)
            print(f"  ✅ Target distribution: {target_dist_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to generate target distribution: {e}")
    
    return generated_files


def generate_html_profile_report(df: pd.DataFrame, target_col: str, task_type: str, output_dir: Path, config: Dict[str, Any]) -> bool:
    """
    Generate HTML EDA report using sweetviz (if available) or basic HTML fallback.
    Returns True if sweetviz report was generated, False if fallback was used.
    """
    stage1_cfg = config.get("stage1", {})
    generate_sweetviz = stage1_cfg.get("generate_sweetviz", False)
    eda_sample_size = stage1_cfg.get("eda_sample_size", 10000)
    
    if not generate_sweetviz:
        print("ℹ️  Sweetviz report generation disabled (stage1.generate_sweetviz=false)")
        return False
    
    try:
        import sweetviz as sv
        print(f"📊 Generating sweetviz HTML report...")
        
        # Sample if dataset is large
        df_sample = df.sample(n=min(len(df), eda_sample_size), random_state=42) if len(df) > eda_sample_size else df
        
        # Generate sweetviz report
        target_feat = target_col if task_type != "clustering" else None
        report = sv.analyze(df_sample, target_feat=target_feat)
        
        # Save report
        report_path = output_dir / "sweetviz_eda_report.html"
        report.show_html(str(report_path), open_browser=False, layout="vertical")
        
        print(f"✅ Sweetviz report saved: {report_path}")
        return True
        
    except ImportError:
        print("⚠️  Sweetviz not available, falling back to basic HTML report")
        return _generate_basic_html_report(df, target_col, task_type, output_dir)
    except Exception as e:
        print(f"⚠️  Sweetviz report generation failed: {e}")
        print("   Falling back to basic HTML report...")
        return _generate_basic_html_report(df, target_col, task_type, output_dir)


def _generate_basic_html_report(df: pd.DataFrame, target_col: str, task_type: str, output_dir: Path) -> bool:
    """
    Generate basic HTML report as fallback when sweetviz is unavailable.
    """
    try:
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EDA Report - Basic</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background-color: white; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #007bff; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>EDA Report - Dataset Overview</h1>
    <div class="metric">
        <h2>Dataset Shape</h2>
        <p><strong>Rows:</strong> {len(df):,}</p>
        <p><strong>Columns:</strong> {len(df.columns)}</p>
        <p><strong>Task Type:</strong> {task_type}</p>
        {f'<p><strong>Target Column:</strong> {target_col}</p>' if target_col else ''}
    </div>
    
    <div class="metric">
        <h2>Column Information</h2>
        <table>
            <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Missing (%)</th>
                <th>Unique Values</th>
            </tr>
        """
        
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            html_content += f"""
            <tr>
                <td>{col}</td>
                <td>{df[col].dtype}</td>
                <td>{missing_pct:.2f}%</td>
                <td>{df[col].nunique():,}</td>
            </tr>
            """
        
        html_content += """
        </table>
    </div>
    
    <div class="metric">
        <h2>Data Quality Summary</h2>
        <p><strong>Overall Missing:</strong> {:.2f}%</p>
        <p><strong>Duplicate Rows:</strong> {:,} ({:.2f}%)</p>
    </div>
</body>
</html>
        """.format(
            df.isna().mean().mean() * 100,
            df.duplicated().sum(),
            df.duplicated().sum() / len(df) * 100
        )
        
        report_path = output_dir / "basic_eda_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)
        
        print(f"✅ Basic HTML report saved: {report_path}")
        return False
        
    except Exception as e:
        print(f"⚠️  Basic HTML report generation failed: {e}")
        return False


def analyze_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze column types and return structured counts."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "boolean": boolean_cols,
        "counts": {
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
            "datetime": len(datetime_cols),
            "boolean": len(boolean_cols),
        }
    }


def analyze_column_statistics(df: pd.DataFrame, col_types: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate per-column statistics."""
    stats = []
    
    for col in df.columns:
        col_stat = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_count": int(df[col].nunique()),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().sum() / len(df) * 100),
        }
        
        # Numeric statistics
        if col in col_types["numeric"]:
            col_stat.update({
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "median": float(df[col].median()) if not df[col].isna().all() else None,
                "q25": float(df[col].quantile(0.25)) if not df[col].isna().all() else None,
                "q75": float(df[col].quantile(0.75)) if not df[col].isna().all() else None,
            })
        
        # Categorical statistics
        if col in col_types["categorical"]:
            mode_val = df[col].mode()
            col_stat.update({
                "mode": str(mode_val[0]) if len(mode_val) > 0 else None,
                "mode_count": int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0,
                "mode_pct": float(df[col].value_counts().iloc[0] / len(df) * 100) if len(df[col].value_counts()) > 0 else 0.0,
            })
        
        stats.append(col_stat)
    
    return stats


def analyze_data_quality(df: pd.DataFrame, col_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze data quality issues."""
    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = float(duplicate_count / len(df) * 100)
    
    # Constant columns (only 1 unique value)
    constant_cols = [s["name"] for s in col_stats if s["unique_count"] <= 1]
    
    # High missing columns (>50%)
    high_missing_cols = [s["name"] for s in col_stats if s["missing_pct"] > 50.0]
    
    # High cardinality columns (>100 unique for categorical)
    high_cardinality_cols = [
        s["name"] for s in col_stats 
        if s["dtype"] in ['object', 'category'] and s["unique_count"] > 100
    ]
    
    return {
        "duplicate_rows": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "constant_columns": constant_cols,
        "constant_count": len(constant_cols),
        "high_missing_columns": high_missing_cols,
        "high_missing_count": len(high_missing_cols),
        "high_cardinality_columns": high_cardinality_cols,
        "high_cardinality_count": len(high_cardinality_cols),
    }


def analyze_target(df: pd.DataFrame, target_col: str, task_type: str) -> Dict[str, Any]:
    """Analyze target column based on task type."""
    if not target_col or target_col not in df.columns:
        return {}
    
    target_analysis = {
        "name": target_col,
        "dtype": str(df[target_col].dtype),
        "missing_count": int(df[target_col].isna().sum()),
        "missing_pct": float(df[target_col].isna().sum() / len(df) * 100),
    }
    
    if task_type == "classification":
        # Class distribution
        value_counts = df[target_col].value_counts()
        target_analysis.update({
            "class_count": int(len(value_counts)),
            "class_distribution": {str(k): int(v) for k, v in value_counts.items()},
            "class_distribution_pct": {str(k): float(v / len(df) * 100) for k, v in value_counts.items()},
            "majority_class": str(value_counts.idxmax()),
            "majority_class_pct": float(value_counts.max() / len(df) * 100),
            "minority_class": str(value_counts.idxmin()),
            "minority_class_pct": float(value_counts.min() / len(df) * 100),
            "is_imbalanced": bool(value_counts.min() / value_counts.max() < 0.3),
        })
    
    elif task_type == "regression":
        # Distribution statistics
        target_analysis.update({
            "mean": float(df[target_col].mean()) if not df[target_col].isna().all() else None,
            "std": float(df[target_col].std()) if not df[target_col].isna().all() else None,
            "min": float(df[target_col].min()) if not df[target_col].isna().all() else None,
            "max": float(df[target_col].max()) if not df[target_col].isna().all() else None,
            "median": float(df[target_col].median()) if not df[target_col].isna().all() else None,
            "q25": float(df[target_col].quantile(0.25)) if not df[target_col].isna().all() else None,
            "q75": float(df[target_col].quantile(0.75)) if not df[target_col].isna().all() else None,
            "skewness": float(df[target_col].skew()) if not df[target_col].isna().all() else None,
            "kurtosis": float(df[target_col].kurtosis()) if not df[target_col].isna().all() else None,
        })
    
    return target_analysis


def generate_comprehensive_eda(df: pd.DataFrame, target_col: str, task_type: str) -> Dict[str, Any]:
    """Generate comprehensive EDA with 20-30 metrics."""
    print("📊 Generating comprehensive EDA...")
    
    # Column type analysis
    col_types = analyze_column_types(df)
    
    # Per-column statistics
    col_stats = analyze_column_statistics(df, col_types)
    
    # Data quality analysis
    quality = analyze_data_quality(df, col_stats)
    
    # Target analysis
    target_analysis = analyze_target(df, target_col, task_type)
    
    # Correlation matrix for numeric columns (top 20 correlations)
    correlations = {}
    if len(col_types["numeric"]) > 1:
        try:
            corr_matrix = df[col_types["numeric"]].corr()
            # Get upper triangle (avoid duplicates)
            upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            corr_pairs = [
                {
                    "col1": corr_matrix.index[i],
                    "col2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j])
                }
                for i, j in zip(*np.where(upper_tri))
                if not np.isnan(corr_matrix.iloc[i, j])
            ]
            # Sort by absolute correlation
            corr_pairs = sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:20]
            correlations = {"top_20_pairs": corr_pairs}
        except Exception as e:
            print(f"⚠️  Correlation calculation failed: {e}")
            correlations = {"error": str(e)}
    
    eda = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "column_types": col_types,
        "column_statistics": col_stats,
        "data_quality": quality,
        "target_analysis": target_analysis,
        "correlations": correlations,
    }
    
    return eda


def log_eda_to_mlflow(logger, eda: Dict[str, Any], task_type: str, target_col: str, dataset_uri: str):
    """Log comprehensive EDA metrics and params to MLflow and Azure ML."""
    try:
        # Log high-level params
        logger.log_param("task_type", str(task_type))
        if target_col:
            logger.log_param("target_column", str(target_col))
        logger.log_param("dataset_uri", dataset_uri)
        
        # Log dataset shape metrics
        logger.log_metric("dataset_rows", eda["dataset_shape"]["rows"])
        logger.log_metric("dataset_cols", eda["dataset_shape"]["columns"])
        
        # Log column type counts
        logger.log_metric("numeric_cols", eda["column_types"]["counts"]["numeric"])
        logger.log_metric("categorical_cols", eda["column_types"]["counts"]["categorical"])
        logger.log_metric("datetime_cols", eda["column_types"]["counts"]["datetime"])
        logger.log_metric("boolean_cols", eda["column_types"]["counts"]["boolean"])
        
        # Log data quality metrics
        logger.log_metric("duplicate_rows", eda["data_quality"]["duplicate_rows"])
        logger.log_metric("duplicate_pct", eda["data_quality"]["duplicate_pct"])
        logger.log_metric("constant_columns_count", eda["data_quality"]["constant_count"])
        logger.log_metric("high_missing_cols_count", eda["data_quality"]["high_missing_count"])
        logger.log_metric("high_cardinality_cols_count", eda["data_quality"]["high_cardinality_count"])
        
        # Log total missing
        total_missing = sum(s["missing_count"] for s in eda["column_statistics"])
        logger.log_metric("missing_total", total_missing)
        
        # Log average missing percentage
        avg_missing_pct = sum(s["missing_pct"] for s in eda["column_statistics"]) / len(eda["column_statistics"])
        logger.log_metric("avg_missing_pct", avg_missing_pct)
        
        # Log target-specific metrics
        if eda.get("target_analysis"):
            target = eda["target_analysis"]
            logger.log_metric("target_missing_count", target.get("missing_count", 0))
            logger.log_metric("target_missing_pct", target.get("missing_pct", 0.0))
            
            if task_type == "classification":
                logger.log_metric("target_class_count", target.get("class_count", 0))
                logger.log_metric("target_majority_pct", target.get("majority_class_pct", 0.0))
                logger.log_metric("target_minority_pct", target.get("minority_class_pct", 0.0))
                logger.log_param("target_is_imbalanced", str(target.get("is_imbalanced", False)))
            
            elif task_type == "regression":
                if target.get("mean") is not None:
                    logger.log_metric("target_mean", target["mean"])
                if target.get("std") is not None:
                    logger.log_metric("target_std", target["std"])
                if target.get("median") is not None:
                    logger.log_metric("target_median", target["median"])
                if target.get("skewness") is not None:
                    logger.log_metric("target_skewness", target["skewness"])
        
        # Log column type lists as params (comma-separated strings)
        if eda["column_types"]["numeric"]:
            logger.log_param("numeric_columns", ",".join(eda["column_types"]["numeric"][:10]))  # First 10
        if eda["column_types"]["categorical"]:
            logger.log_param("categorical_columns", ",".join(eda["column_types"]["categorical"][:10]))
        
        print("✅ Logged 20-30 EDA metrics to MLflow")
        
    except Exception as e:
        print(f"⚠️  MLflow EDA logging failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Time-Series Auto-Detection
# ──────────────────────────────────────────────────────────────────────────────

_TEMPORAL_KEYWORDS = {
    "date", "datetime", "timestamp", "time", "year", "month", "day",
    "week", "hour", "minute", "second", "period", "epoch", "ts",
    "created_at", "updated_at", "event_time", "record_date",
}


def detect_time_series(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> Dict[str, Any]:
    """Auto-detect whether a dataset is time-series data.

    Uses multiple heuristics (column names, dtypes, monotonicity,
    autocorrelation) and returns a confidence score.

    Returns a dict with:
      is_time_series   – bool (True if confidence >= 0.60)
      confidence       – float 0-1
      time_column      – str or None (best candidate column)
      frequency        – str or None ('D', 'H', 'M', etc.)
      signals_triggered – list[str] explaining why
    """
    result: Dict[str, Any] = {
        "is_time_series": False,
        "confidence": 0.0,
        "time_column": None,
        "frequency": None,
        "signals_triggered": [],
        "n_temporal_cols": 0,
    }
    signals: List[str] = []
    score = 0.0  # accumulates evidence; normalised to 0-1 at the end

    # ── 1. Datetime-typed columns ─────────────────────────────────────
    dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if dt_cols:
        signals.append(f"datetime_dtype_cols={dt_cols}")
        score += 0.35

    # ── 2. Column name heuristic ──────────────────────────────────────
    name_candidates: List[str] = []
    for col in df.columns:
        col_lower = col.lower().strip().replace(" ", "_")
        if any(kw == col_lower or col_lower.startswith(kw + "_") or col_lower.endswith("_" + kw)
               for kw in _TEMPORAL_KEYWORDS):
            name_candidates.append(col)
    if name_candidates:
        signals.append(f"temporal_column_names={name_candidates}")
        score += 0.20

    # ── 3. Parseable-as-datetime columns ──────────────────────────────
    parse_candidates: List[str] = []
    for col in (name_candidates or df.select_dtypes(include=["object"]).columns[:5]):
        if col in dt_cols:
            continue
        try:
            sample = df[col].dropna().head(20)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() >= 0.80:
                parse_candidates.append(col)
        except Exception as e:
            logger.debug("datetime parse-candidate check failed for %s: %s", col, e)
    if parse_candidates:
        signals.append(f"parseable_datetime_cols={parse_candidates}")
        score += 0.20

    # ── 4. Pick best time-column candidate ────────────────────────────
    all_time_candidates = dt_cols + parse_candidates + [c for c in name_candidates if c not in dt_cols and c not in parse_candidates]
    result["n_temporal_cols"] = len(set(all_time_candidates))

    best_time_col = None
    if all_time_candidates:
        best_time_col = all_time_candidates[0]
        result["time_column"] = best_time_col

    # ── 5. Monotonicity check (sorted dates = strong signal) ─────────
    if best_time_col is not None:
        try:
            ts_series = pd.to_datetime(df[best_time_col], errors="coerce").dropna()
            if len(ts_series) > 10:
                is_mono = ts_series.is_monotonic_increasing or ts_series.is_monotonic_decreasing
                if is_mono:
                    signals.append("monotonic_time_column")
                    score += 0.15
                # Detect frequency
                try:
                    freq = pd.infer_freq(ts_series.head(100))
                    if freq:
                        result["frequency"] = freq
                        signals.append(f"inferred_freq={freq}")
                        score += 0.10
                except Exception as e:
                    logger.debug("infer_freq failed: %s", e)
        except Exception as e:
            logger.debug("monotonicity check failed for %s: %s", best_time_col, e)

    # ── 6. Target autocorrelation (time-series targets are correlated) ─
    if target_col and target_col in df.columns:
        try:
            y = pd.to_numeric(df[target_col], errors="coerce").dropna()
            if len(y) > 30:
                # Lag-1 autocorrelation
                autocorr = y.autocorr(lag=1)
                if autocorr is not None and abs(autocorr) > 0.5:
                    signals.append(f"target_autocorr_lag1={autocorr:.3f}")
                    score += 0.15
        except Exception as e:
            logger.debug("target autocorrelation check failed: %s", e)

    # ── 7. Index is datetime ──────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        signals.append("datetime_index")
        score += 0.30

    # ── Normalise and threshold ───────────────────────────────────────
    confidence = min(score, 1.0)
    result["confidence"] = round(confidence, 3)
    result["signals_triggered"] = signals
    result["is_time_series"] = confidence >= 0.60

    return result


def generate_intelligent_recipe_recommendations(
    df: pd.DataFrame, 
    eda: Dict[str, Any], 
    config: Dict[str, Any], 
    task_type: str, 
    target_col: str
) -> Dict[str, Any]:
    """
    🤖 Intelligent Recipe Recommendation Engine (Data-Adaptive, Not Hardcoded)
    
    Analyzes dataset characteristics and recommends optimal preprocessing strategies
    based on ACTUAL data properties, not arbitrary thresholds.
    """
    recommendations = {
        "task_type": task_type,
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "dataset_name": config.get("dataset", {}).get("name", "unknown")
    }
    
    n_samples = eda.get("dataset_shape", {}).get("rows", 0)
    n_features = eda.get("dataset_shape", {}).get("columns", 0)
    col_stats = eda.get("column_statistics", [])
    quality = eda.get("data_quality", {})
    target_analysis = eda.get("target_analysis", {})
    
    print("\n🤖 Analyzing dataset characteristics for intelligent recipe recommendations...")
    
    # === 1. IMBALANCE HANDLING (Classification Only) ===
    if task_type == "classification" and target_col:
        is_imbalanced = target_analysis.get("is_imbalanced", False)
        minority_pct = target_analysis.get("minority_class_pct", 50)
        minority_count = int(minority_pct * n_samples / 100) if minority_pct else 0
        majority_pct = target_analysis.get("majority_class_pct", 50)
        imbalance_ratio = minority_pct / majority_pct if majority_pct > 0 else 1.0
        
        # Data-driven decision: consider BOTH ratio AND absolute counts
        if minority_count < 100:
            recommendations["imbalance_handling"] = "class_weights"
            recommendations["imbalance_reason"] = f"Few minority samples ({minority_count}), class weights safer than SMOTE"
        elif imbalance_ratio < 0.05:
            recommendations["imbalance_handling"] = "smote_tomek"
            recommendations["imbalance_reason"] = f"Severe imbalance (ratio={imbalance_ratio:.3f}), SMOTE+Tomek cleaning"
        elif imbalance_ratio < 0.3:
            recommendations["imbalance_handling"] = "smote"
            recommendations["imbalance_reason"] = f"Moderate imbalance (ratio={imbalance_ratio:.3f}), standard SMOTE"
        else:
            recommendations["imbalance_handling"] = "none"
            recommendations["imbalance_reason"] = f"Balanced classes (ratio={imbalance_ratio:.3f})"
    
    # === 2. OUTLIER TREATMENT (Adaptive to Distribution) ===
    numeric_cols = [c for c in col_stats if c.get("type") == "numeric"]
    if numeric_cols:
        # Calculate average skewness to understand distribution shape
        skewness_values = [abs(c.get("skewness", 0)) for c in numeric_cols if c.get("skewness") is not None]
        avg_skewness = np.mean(skewness_values) if skewness_values else 0
        
        # Outlier percentage from quality analysis
        outlier_pct = quality.get("outlier_percentage", 0)
        
        # Adaptive decision based on distribution + outlier density
        if avg_skewness > 2 and outlier_pct > 10:
            recommendations["outlier_treatment"] = "winsorize_95"
            recommendations["outlier_reason"] = f"Heavy-tailed distribution (skew={avg_skewness:.2f}), {outlier_pct:.1f}% outliers"
        elif outlier_pct > 15:
            recommendations["outlier_treatment"] = "iqr_removal"
            recommendations["outlier_reason"] = f"High outlier density ({outlier_pct:.1f}%), remove extreme values"
        elif outlier_pct > 5:
            recommendations["outlier_treatment"] = "winsorize_99"
            recommendations["outlier_reason"] = f"Moderate outliers ({outlier_pct:.1f}%), conservative capping"
        else:
            recommendations["outlier_treatment"] = "none"
            recommendations["outlier_reason"] = f"Low outliers ({outlier_pct:.1f}%), keep as-is"
    
    # === 3. SCALING METHOD (Distribution-Aware) ===
    if numeric_cols:
        avg_skewness = np.mean([abs(c.get("skewness", 0)) for c in numeric_cols if c.get("skewness") is not None])
        avg_kurtosis = np.mean([abs(c.get("kurtosis", 0)) for c in numeric_cols if c.get("kurtosis") is not None])
        has_heavy_tails = avg_kurtosis > 3
        
        if avg_skewness > 2 and has_heavy_tails:
            recommendations["scaling_method"] = "robust"
            recommendations["scaling_reason"] = f"Heavy-tailed (skew={avg_skewness:.2f}, kurt={avg_kurtosis:.2f}), robust scaler"
        elif avg_skewness > 1.5:
            recommendations["scaling_method"] = "yeo_johnson"
            recommendations["scaling_reason"] = f"Moderate skewness ({avg_skewness:.2f}), power transform recommended"
        else:
            recommendations["scaling_method"] = "standard"
            recommendations["scaling_reason"] = f"Near-normal distribution (skew={avg_skewness:.2f}), standard scaling"
        
        recommendations["scaling_note"] = "Tree-based models (XGBoost, LightGBM) don't require scaling"
    
    # === 4. IMPUTATION METHOD (Computational Feasibility) ===
    missing_pct = quality.get("missing_percentage", 0)
    
    # Estimate KNN computational cost: O(n²)
    knn_time_estimate = (n_samples ** 2) * n_features * 1e-7  # seconds
    knn_feasible = knn_time_estimate < 300  # 5 minute threshold
    
    if missing_pct > 20 and knn_feasible:
        recommendations["imputation_numeric"] = "knn"
        recommendations["imputation_categorical"] = "most_frequent"
        recommendations["imputation_reason"] = f"High missing ({missing_pct:.1f}%), KNN feasible ({knn_time_estimate:.0f}s)"
    elif missing_pct > 20:
        recommendations["imputation_numeric"] = "iterative"
        recommendations["imputation_categorical"] = "most_frequent"
        recommendations["imputation_reason"] = f"High missing ({missing_pct:.1f}%), iterative imputer (KNN too slow)"
    elif missing_pct > 5:
        recommendations["imputation_numeric"] = "median"
        recommendations["imputation_categorical"] = "most_frequent"
        recommendations["imputation_reason"] = f"Moderate missing ({missing_pct:.1f}%), median robust to outliers"
    else:
        recommendations["imputation_numeric"] = "mean"
        recommendations["imputation_categorical"] = "most_frequent"
        recommendations["imputation_reason"] = f"Low missing ({missing_pct:.1f}%), simple mean imputation"
    
    # === 5. ENCODING METHOD (Cardinality-Adaptive) ===
    cat_cols = [c for c in col_stats if c.get("type") == "categorical"]
    if cat_cols:
        cardinalities = [c.get("unique_count", 0) for c in cat_cols]
        max_cardinality = max(cardinalities)
        avg_cardinality = np.mean(cardinalities)
        
        if max_cardinality > 100:
            recommendations["encoding_method"] = "target_cv5"
            recommendations["encoding_reason"] = f"Very high cardinality (max={max_cardinality}), target encoding with CV"
            recommendations["encoding_warning"] = "Use 5-fold CV to prevent target leakage"
        elif avg_cardinality > 20:
            recommendations["encoding_method"] = "hashing"
            recommendations["encoding_reason"] = f"High cardinality (avg={avg_cardinality:.0f}), hash to fixed dimensions"
        else:
            recommendations["encoding_method"] = "onehot"
            recommendations["encoding_reason"] = f"Low cardinality (avg={avg_cardinality:.0f}), standard one-hot"
    
    # === 6. FEATURE SELECTION (Dimensionality-Aware) ===
    feature_to_sample_ratio = n_features / n_samples if n_samples > 0 else 0
    
    if feature_to_sample_ratio > 0.1:  # More than 10% (high-dimensional)
        if task_type == "classification":
            recommendations["feature_selection"] = "chi2_selectkbest"
            recommendations["feature_selection_k"] = min(int(n_samples * 0.05), 50)
        else:
            recommendations["feature_selection"] = "f_regression_selectkbest"
            recommendations["feature_selection_k"] = min(int(n_samples * 0.05), 50)
        recommendations["feature_selection_reason"] = f"High dimensionality (ratio={feature_to_sample_ratio:.3f}), aggressive selection"
    elif n_features > 100:
        recommendations["feature_selection"] = "lasso_l1"
        recommendations["feature_selection_reason"] = f"Many features ({n_features}), L1 regularization"
    elif n_features > 50:
        recommendations["feature_selection"] = "variance_threshold"
        recommendations["feature_selection_reason"] = f"Moderate features ({n_features}), remove low-variance"
    else:
        recommendations["feature_selection"] = "none"
        recommendations["feature_selection_reason"] = f"Few features ({n_features}), keep all"
    
    # === 7. RECIPE TIER (Performance vs Speed) ===
    if n_samples < 1000:
        recommendations["suggested_tier"] = "lightning_fast"
        recommendations["tier_reason"] = f"Small dataset ({n_samples:,} samples), fast iteration"
    elif n_samples < 10000:
        recommendations["suggested_tier"] = "quick_exploration"
        recommendations["tier_reason"] = f"Medium dataset ({n_samples:,} samples), balanced speed"
    elif n_samples < 100000:
        recommendations["suggested_tier"] = "balanced_performance"
        recommendations["tier_reason"] = f"Large dataset ({n_samples:,} samples), quality focus"
    else:
        recommendations["suggested_tier"] = "high_performance"
        recommendations["tier_reason"] = f"Very large dataset ({n_samples:,} samples), maximize accuracy"
    
    # Print summary
    print(f"\n📊 Intelligent Recommendations (Data-Driven):")
    if task_type == "classification":
        print(f"   🎯 Imbalance: {recommendations.get('imbalance_handling', 'N/A')}")
    print(f"   📈 Outliers: {recommendations.get('outlier_treatment', 'N/A')}")
    print(f"   📏 Scaling: {recommendations.get('scaling_method', 'N/A')}")
    print(f"   🔧 Imputation: {recommendations.get('imputation_numeric', 'N/A')}")
    print(f"   🏷️  Encoding: {recommendations.get('encoding_method', 'N/A')}")
    print(f"   🎯 Feature Selection: {recommendations.get('feature_selection', 'N/A')}")
    print(f"   ⚡ Suggested Tier: {recommendations.get('suggested_tier', 'N/A')}")
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Data Ingestion (Read-only via SDK v2)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dataset_in", type=str, required=False, help="Ignored (legacy mount param)")
    parser.add_argument("--dataset_out", type=str, required=True, help="Output path for dataset CSV")
    parser.add_argument("--eda_dir", type=str, required=True, help="Output directory for EDA report")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STAGE 1: DATA INGESTION (Read-only via azureml:// URI)")
    print("=" * 80)
    
    # Load config
    cfg = load_config(args.config)
    
    # Build dataset URI and read (no mounting, no datastore creation)
    dataset_uri = build_dataset_uri(cfg)
    # Get delimiter from config (V4 enhancement: support multiple delimiters)
    delimiter = cfg.get("dataset", {}).get("delimiter") or cfg.get("dataset", {}).get("csv_delimiter") or ","
    read_kwargs = {"sep": delimiter}
    # Get encoding from config (handles non-UTF-8 datasets like Online Retail with £ chars)
    csv_encoding = cfg.get("dataset", {}).get("encoding")
    if csv_encoding:
        read_kwargs["encoding"] = csv_encoding
        print(f"📝 Using encoding: {csv_encoding}")
    read_attempts: List[Tuple[str, str]] = [(dataset_uri, "azureml:// datastore URI")]

    # Fallback: use mounted dataset_in if available (joined with blob_path)
    blob_path = cfg.get("dataset", {}).get("blob_path")
    if args.dataset_in and blob_path:
        mounted_path = str(Path(args.dataset_in) / blob_path)
        read_attempts.append((mounted_path, "mounted datastore path"))

    df = None
    for path, label in read_attempts:
        try:
            print(f"🔗 Reading dataset via {label}: {path}")
            df = pd.read_csv(path, **read_kwargs)
            print(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} cols from {label}")
            break
        except Exception as e:
            print(f"⚠️  Read attempt failed for {label}: {e}")
    
    if df is None:
        attempted = "; ".join([f"{lbl} ({p})" for p, lbl in read_attempts])
        raise RuntimeError(f"Could not load dataset from any source. Tried: {attempted}")

    expected_content_sha256 = str(
        (cfg.get("dataset") or {}).get("content_sha256") or ""
    ).strip().lower()
    actual_content_sha256 = canonical_dataframe_sha256(df)
    if expected_content_sha256 and actual_content_sha256 != expected_content_sha256:
        raise RuntimeError(
            "Dataset content identity mismatch: "
            f"expected={expected_content_sha256}, actual={actual_content_sha256}"
        )
    print(f"Dataset content SHA-256: {actual_content_sha256}")

    # Optional: enable async logging if environment variable is set
    try:
        if os.environ.get("MLFLOW_ENABLE_ASYNC_LOGGING", "").lower() in ("1", "true", "yes"):
            mlflow.enable_async_logging()
    except Exception as e:
        print(f"⚠️  Could not enable MLflow async logging: {e}")
    
    # Validate target column (required for classification/regression, optional for clustering)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target = cfg.get("dataset", {}).get("target_column")
    
    if task_type == "clustering":
        # Clustering doesn't require a target column
        if target:
            print(f"⚠️  Target column '{target}' ignored for clustering task")
    else:
        # Classification/regression require a target column
        if not target:
            raise ValueError(f"Target column required for {task_type} task but not specified in config")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Available: {df.columns.tolist()}")
        print(f"✅ Target column '{target}' found")

    df = ensure_holdout_partition(
        df,
        target_col=target,
        task_type=task_type,
        holdout_fraction=float(cfg.get("holdout_fraction", 0.2)),
        random_seed=int(cfg.get("random_seed", 42)),
        split_strategy=str(cfg.get("holdout_split_strategy", "random")),
        time_column=cfg.get("holdout_time_column"),
    )
    training_df = df.loc[df[SPLIT_COLUMN].eq(TRAIN_PARTITION)].drop(
        columns=[SPLIT_COLUMN, ROW_ID_COLUMN]
    )
    print(
        "🔒 Canonical split assigned before data-driven recommendations: "
        f"train={len(training_df):,}, holdout={len(df) - len(training_df):,}"
    )

    # Create dual logger for MLflow and Azure ML
    logger = create_metrics_logger(
        run_name="s01_ingestion",
        tags={"pipeline": "v3_mlops", "phase": "ingestion", "step": "s01"}
    )

    # Generate comprehensive EDA
    eda = generate_comprehensive_eda(training_df, target, task_type)
    eda["partition_scope"] = {
        "recommendations_and_eda": "training_only",
        "n_train": int(len(training_df)),
        "n_holdout": int(len(df) - len(training_df)),
    }
    
    # V4 Enhancement: Decision gate validation
    print("\n🔍 Running data quality validation (decision gates)...")
    validation_issues = validate_data_quality(training_df, cfg, task_type, target)
    decision_gate_status = calculate_decision_gate_status(validation_issues)
    
    print(f"\n📋 Data Quality Decision Gate: {decision_gate_status}")
    for issue in validation_issues:
        print(f"   {issue}")
    
    # Add decision gate results to EDA
    eda["decision_gate"] = {
        "status": decision_gate_status,
        "issues": validation_issues,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Log comprehensive EDA to MLflow and Azure ML for Studio visibility
    log_eda_to_mlflow(logger, eda, task_type, target, dataset_uri)
    
    # Log decision gate status
    logger.log_param("decision_gate_status", decision_gate_status)
    logger.log_metric("decision_gate_issue_count", len(validation_issues))
    
    # 🤖 Intelligent Recipe Recommendations (Data-Adaptive)
    recipe_recommendations = generate_intelligent_recipe_recommendations(
        training_df,
        eda,
        cfg,
        task_type,
        target,
    )
    
    # 🕐 AUTO-DETECT TIME-SERIES DATA
    ts_detection = detect_time_series(training_df, target, task_type)
    recipe_recommendations["is_time_series"] = ts_detection["is_time_series"]
    recipe_recommendations["time_series_detection"] = ts_detection
    if ts_detection["is_time_series"]:
        print(f"\n🕐 TIME-SERIES DETECTED (confidence={ts_detection['confidence']:.0%})")
        print(f"   Time column : {ts_detection['time_column']}")
        print(f"   Frequency   : {ts_detection['frequency']}")
        print(f"   Signals     : {', '.join(ts_detection['signals_triggered'])}")
    else:
        print(f"\n🕐 Time-series check: NOT detected (confidence={ts_detection['confidence']:.0%})")
    
    # Log recommendations to MLflow
    for param_name, param_value in recipe_recommendations.items():
        if not param_name.endswith("_reason") and not param_name.endswith("_timestamp") and not param_name.endswith("_note") and not param_name.endswith("_warning"):
            logger.log_param(f"recommend_{param_name}", str(param_value))
    logger.log_param("is_time_series", str(ts_detection["is_time_series"]))
    
    # V4 Pattern: Use hardcoded outputs/ folder for Azure ML Studio visibility
    # Component output parameters (args.eda_dir) are NOT visible in Studio UI
    # Files written to outputs/ folder ARE automatically captured and visible
    job_outputs_dir = Path("outputs")
    job_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 🎯 NEW: Generate advanced EDA visualizations (missing heatmap, outliers, target distribution)
    col_types = analyze_column_types(training_df)
    advanced_viz = generate_advanced_eda_visualizations(
        training_df,
        target,
        task_type,
        job_outputs_dir,
        col_types,
    )
    print(f"✅ Advanced EDA visualizations generated: {len(advanced_viz)} plots")
    
    # V4 Enhancement: Generate HTML profile report with sweetviz (if enabled)
    sweetviz_generated = generate_html_profile_report(
        training_df,
        target,
        task_type,
        job_outputs_dir,
        cfg,
    )
    
    # Save dataset (write to component output parameter for inter-step passing)
    # V4 Enhancement: Preserve original delimiter (critical fix for semicolon-delimited datasets)
    os.makedirs(Path(args.dataset_out).parent, exist_ok=True)
    df.to_csv(args.dataset_out, sep=delimiter, index=False)
    print(f"💾 Saved dataset to: {args.dataset_out} (delimiter: '{delimiter}')")
    
    # Save comprehensive EDA artifacts to outputs/ folder (visible in Studio)
    
    # Save full EDA JSON
    eda_report_path = job_outputs_dir / "eda_report.json"
    with open(eda_report_path, "w") as f:
        json.dump(eda, f, indent=2)
    
    # Save column statistics as CSV for easy viewing
    col_stats_path = job_outputs_dir / "column_statistics.csv"
    col_stats_df = pd.DataFrame(eda["column_statistics"])
    col_stats_df.to_csv(col_stats_path, index=False)
    
    # Save correlation matrix as CSV (if available)
    if eda.get("correlations", {}).get("top_20_pairs"):
        corr_path = job_outputs_dir / "top_correlations.csv"
        corr_df = pd.DataFrame(eda["correlations"]["top_20_pairs"])
        corr_df.to_csv(corr_path, index=False)
    
    # Save intelligent recipe recommendations (NEW)
    recommendations_path = job_outputs_dir / "recipe_recommendations.json"
    with open(recommendations_path, "w") as f:
        json.dump(recipe_recommendations, f, indent=2)

    # Save time-series detection result (NEW)
    ts_path = job_outputs_dir / "time_series_detection.json"
    with open(ts_path, "w") as f:
        json.dump(ts_detection, f, indent=2)
    
    # Log key EDA metrics to MLflow (not artifacts - those fail with azureml:// scheme)
    logger.log_dict(eda, "eda_comprehensive.json")
    
    # Print output summary with Studio visibility guidance
    print(f"📊 Comprehensive EDA saved to: {job_outputs_dir}")
    print(f"   - eda_report.json ({len(eda)} sections)")
    print(f"   - column_statistics.csv ({len(eda['column_statistics'])} columns)")
    if eda.get("correlations", {}).get("top_20_pairs"):
        print(f"   - top_correlations.csv ({len(eda['correlations']['top_20_pairs'])} pairs)")
    if sweetviz_generated:
        print(f"   - sweetviz_eda_report.html (interactive visualization)")
    print(f"   - recipe_recommendations.json (🤖 intelligent preprocessing strategy)")
    
    print()
    print("💡 Azure ML Studio Visibility:")
    print(f"   → Navigate to: Outputs + logs → outputs/")
    print(f"   → Download HTML report for interactive exploration")
    
    # End logging
    logger.end_run()
    
    print("=" * 80)
    print("✅ Stage 1 completed successfully (read-only, no writes to datastore)")
    print("=" * 80)


if __name__ == "__main__":
    main()
