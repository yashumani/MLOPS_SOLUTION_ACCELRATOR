"""
Multi-Stage EDA Generator for MLOps Pipeline
Generates correlation heatmaps and Sweetviz reports for preprocessing audit trail
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_correlation_heatmap(df: pd.DataFrame, output_path: Path, stage_name: str):
    """
    Generate correlation heatmap for numeric features.
    Critical for tracking feature relationships across preprocessing stages.
    
    Args:
        df: DataFrame to analyze
        output_path: Path to save heatmap PNG
        stage_name: Name of the stage (for title)
    
    Returns:
        Path to saved heatmap, or None if failed
    """
    try:
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            print(f"   ⚠️  Skipping heatmap: Need at least 2 numeric columns (found {numeric_df.shape[1]})")
            return None
        
        print(f"   📊 Generating correlation heatmap ({numeric_df.shape[1]} features)...")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Determine figure size based on feature count
        fig_size = min(max(numeric_df.shape[1] * 0.4, 10), 20)
        
        # Create heatmap
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(
            corr_matrix, 
            annot=False,  # No annotations for large matrices
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f'Feature Correlation Heatmap - {stage_name}', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ Saved to: {output_path.name}")
        
        # Also save top correlations as CSV
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        corr_pairs = []
        for i, j in zip(*np.where(upper_tri)):
            if not np.isnan(corr_matrix.iloc[i, j]):
                corr_pairs.append({
                    "feature1": corr_matrix.index[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j])
                })
        
        # Sort by absolute correlation
        corr_pairs = sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:50]
        
        corr_csv_path = output_path.parent / f"{stage_name.lower().replace(' ', '_').replace('-', '_')}_top_correlations.csv"
        pd.DataFrame(corr_pairs).to_csv(corr_csv_path, index=False)
        print(f"      ✓ Top 50 correlations: {corr_csv_path.name}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"   ⚠️  Heatmap generation failed: {e}")
        return None


def generate_sweetviz_report(df: pd.DataFrame, output_path: Path, stage_name: str, target_col: str = None, config: dict = None):
    """
    Generate Sweetviz HTML report for interactive EDA.
    Shows distribution changes across preprocessing stages.
    
    Args:
        df: DataFrame to analyze
        output_path: Path to save HTML report
        stage_name: Name of the stage (for report title)
        target_col: Target column name (optional)
        config: Config dictionary (optional, for sampling settings)
    
    Returns:
        Path to saved report, or None if failed
    """
    try:
        # Check if sweetviz generation is enabled
        stage_config = config.get("stage1", {}) if config else {}
        generate_sweetviz = stage_config.get("generate_sweetviz", True)
        
        if not generate_sweetviz:
            print(f"   ⏭️  Sweetviz disabled in config")
            return None
        
        import sweetviz as sv
        
        print(f"   🍬 Generating Sweetviz report for {stage_name}...")
        
        # Sample large datasets for performance
        eda_sample_size = stage_config.get("eda_sample_size", 10000)
        if len(df) > eda_sample_size:
            df_sample = df.sample(n=eda_sample_size, random_state=42)
            print(f"      ℹ️  Sampling {eda_sample_size:,} rows for performance")
        else:
            df_sample = df
        
        # Generate report
        if target_col and target_col in df_sample.columns:
            report = sv.analyze(df_sample, target_feat=target_col)
        else:
            report = sv.analyze(df_sample)
        
        # Save HTML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.show_html(str(output_path), open_browser=False)
        
        print(f"      ✓ Saved to: {output_path.name}")
        return str(output_path)
        
    except Exception as e:
        print(f"   ⚠️  Sweetviz report generation failed: {e}")
        return None
