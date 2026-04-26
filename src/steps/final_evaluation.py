import argparse
import json
import logging
import time as _time_mod
from pathlib import Path
import sys
import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Add src to path for imports (single canonical insertion at front of sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger
from utils.stage_signals import StageSignal, write_stage_signal
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_stage_table,
    merge_ledgers, build_summary, build_readme_md,
)
from utils.aim_tournament import run_aim_tournament
from utils.model_universe import build_coverage_report, write_model_coverage

# Module-level logger for diagnostic/debug messages (does not shadow the
# per-run MetricsLogger created inside main()).
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """T15: Handle numpy types for JSON serialization — prevents bool_/float64 crashes."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return super().default(obj)


def _safe_disable_autolog():
    """Disable MLflow autologging and fix tracking URI for Azure ML compatibility."""
    # Suppression is expected: Azure ML's azureml:// tracking URI is not fully
    # compatible with mlflow.sklearn.autolog / mlflow.autolog model registry
    # calls. We log at debug to retain forensic visibility without noise.
    try:
        mlflow.sklearn.autolog(disable=True)
    except Exception as e:
        logger.debug(f"MLflow sklearn autolog disable suppression (Azure ML tracking URI incompatibility): {e}")
    try:
        mlflow.autolog(disable=True)
    except Exception as e:
        logger.debug(f"MLflow autolog disable suppression (Azure ML tracking URI incompatibility): {e}")
    # Fix: Convert azureml:// to https:// to avoid model registry errors
    import os as _os
    _mlflow_uri = _os.getenv("MLFLOW_TRACKING_URI", "")
    if _mlflow_uri.startswith("azureml://"):
        mlflow.set_tracking_uri(_mlflow_uri.replace("azureml://", "https://"))
    # Set local model registry as fallback
    _os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")


def collect_all_stage_metrics(experiment_name: str) -> dict:
    """
    Collect metrics from all 15 pipeline stages by scanning MLflow runs.
    Returns: {
        'preprocessing_stages': {s01_ingestion: {...}, s02_preparation: {...}, ...},
        'baseline_models': {pycaret: {...}, flaml: {...}},
        'phaseb_recipes': {recipe1_pycaret: {...}, recipe1_flaml: {...}, ...},
        'phasec_hpo': {trial1: {...}, trial2: {...}, ...},
        'aggregates': {baseline: {...}, phaseb: {...}, phasec: {...}}
    }
    """
    print("📊 Collecting metrics from all pipeline stages...")
    all_metrics = {
        'preprocessing_stages': {},
        'baseline_models': {},
        'phaseb_recipes': {},
        'phasec_hpo': {},
        'aggregates': {}
    }
    
    try:
        # Search for runs in the experiment
        client = mlflow.tracking.MlflowClient()
        
        # Get experiment
        try:
            exp = client.get_experiment_by_name(experiment_name)
            if not exp:
                print(f"  ⚠️ Experiment not found: {experiment_name}")
                return all_metrics
            exp_id = exp.experiment_id
        except Exception as e:
            print(f"  ⚠️ Could not get experiment: {e}")
            return all_metrics
        
        # Search for runs
        runs = client.search_runs(
            experiment_ids=[exp_id],
            max_results=500,
            order_by=["start_time DESC"]
        )
        
        print(f"  Found {len(runs)} runs in experiment '{experiment_name}'")
        
        for run in runs:
            run_name = run.data.tags.get('mlflow.runName', 'unknown')
            
            # Extract metrics and params
            metrics = {k: v for k, v in run.data.metrics.items()}
            params = {k: v for k, v in run.data.params.items()}
            
            # Categorize by run name pattern
            if 's01_ingestion' in run_name or 's02_preparation' in run_name or 's03_preprocessing' in run_name or 's04_feature_engineering' in run_name:
                all_metrics['preprocessing_stages'][run_name] = {'metrics': metrics, 'params': params}
            elif 's05a_baseline_pycaret' in run_name or 's05b_baseline_flaml' in run_name:
                engine = 'pycaret' if 'pycaret' in run_name else 'flaml'
                all_metrics['baseline_models'][engine] = {'metrics': metrics, 'params': params}
            elif 'phaseb' in run_name.lower() and 'aggregate' not in run_name.lower():
                all_metrics['phaseb_recipes'][run_name] = {'metrics': metrics, 'params': params}
            elif 'phasec' in run_name.lower() or 'optuna' in run_name.lower() and 'aggregate' not in run_name.lower():
                all_metrics['phasec_hpo'][run_name] = {'metrics': metrics, 'params': params}
            elif 'aggregate' in run_name.lower():
                phase = 'baseline' if 'baseline' in run_name.lower() else ('phaseb' if 'phaseb' in run_name.lower() else 'phasec')
                all_metrics['aggregates'][phase] = {'metrics': metrics, 'params': params}
        
        print(f"  ✅ Collected: {len(all_metrics['preprocessing_stages'])} preprocessing, {len(all_metrics['baseline_models'])} baseline models")
        print(f"               {len(all_metrics['phaseb_recipes'])} Phase B recipes, {len(all_metrics['phasec_hpo'])} HPO trials")
        print(f"               {len(all_metrics['aggregates'])} aggregate summaries")
        
    except Exception as e:
        print(f"  ⚠️ Error collecting MLflow metrics: {e}")
    
    return all_metrics


def generate_performance_visualizations(all_metrics: dict, outputs_dir: Path, task_type: str):
    """
    Generate comprehensive performance visualizations:
    - baseline_models_comparison.png (PyCaret vs FLAML)
    - phaseb_recipes_comparison.png (top 10 recipes)
    - phase_comparison.png (Baseline vs Phase B vs Phase C)
    """
    print("📊 Generating performance visualizations...")
    
    # Determine primary metric
    primary_metric = 'accuracy' if task_type == 'classification' else ('r2' if task_type == 'regression' else 'silhouette_score')
    
    # 1. BASELINE MODELS COMPARISON
    try:
        baseline_data = []
        for engine, data in all_metrics.get('baseline_models', {}).items():
            metrics = data.get('metrics', {})
            if primary_metric in metrics:
                baseline_data.append({'Engine': engine.upper(), 'Score': metrics[primary_metric]})
        
        if baseline_data:
            df_baseline = pd.DataFrame(baseline_data)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_baseline, x='Engine', y='Score', palette='viridis', edgecolor='black')
            plt.title(f'Baseline Models Comparison ({primary_metric.upper()})', fontsize=14, fontweight='bold')
            plt.xlabel('Engine', fontsize=12)
            plt.ylabel(f'{primary_metric.upper()} Score', fontsize=12)
            plt.ylim(0, 1.0 if task_type != 'clustering' else None)
            for i, row in enumerate(df_baseline.itertuples()):
                plt.text(i, row.Score + 0.02, f'{row.Score:.4f}', ha='center', fontsize=11, fontweight='bold')
            plt.tight_layout()
            baseline_path = outputs_dir / 'baseline_models_comparison.png'
            plt.savefig(baseline_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Baseline comparison: {baseline_path}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate baseline comparison: {e}")
    
    # 2. PHASE B RECIPES COMPARISON (Top 10)
    try:
        phaseb_data = []
        for recipe_name, data in all_metrics.get('phaseb_recipes', {}).items():
            metrics = data.get('metrics', {})
            if primary_metric in metrics:
                phaseb_data.append({'Recipe': recipe_name[:30], 'Score': metrics[primary_metric]})  # Truncate name
        
        if phaseb_data:
            df_phaseb = pd.DataFrame(phaseb_data).sort_values('Score', ascending=False).head(10)
            plt.figure(figsize=(12, 8))
            sns.barplot(data=df_phaseb, x='Score', y='Recipe', palette='coolwarm', edgecolor='black')
            plt.title(f'Top 10 Phase B Recipes ({primary_metric.upper()})', fontsize=14, fontweight='bold')
            plt.xlabel(f'{primary_metric.upper()} Score', fontsize=12)
            plt.ylabel('Recipe', fontsize=12)
            plt.xlim(0, 1.0 if task_type != 'clustering' else None)
            for i, row in enumerate(df_phaseb.itertuples()):
                plt.text(row.Score + 0.01, i, f'{row.Score:.4f}', va='center', fontsize=10)
            plt.tight_layout()
            phaseb_path = outputs_dir / 'phaseb_recipes_comparison.png'
            plt.savefig(phaseb_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Phase B recipes comparison: {phaseb_path}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate Phase B comparison: {e}")
    
    # 3. PHASE COMPARISON (Baseline vs Phase B vs Phase C)
    try:
        phase_data = []
        aggregates = all_metrics.get('aggregates', {})
        for phase_name, data in aggregates.items():
            metrics = data.get('metrics', {})
            if primary_metric in metrics:
                phase_data.append({'Phase': phase_name.upper(), 'Score': metrics[primary_metric]})
        
        if phase_data:
            df_phases = pd.DataFrame(phase_data)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_phases, x='Phase', y='Score', palette='plasma', edgecolor='black')
            plt.title(f'Phase-Level Comparison ({primary_metric.upper()})', fontsize=14, fontweight='bold')
            plt.xlabel('Phase', fontsize=12)
            plt.ylabel(f'{primary_metric.upper()} Score', fontsize=12)
            plt.ylim(0, 1.0 if task_type != 'clustering' else None)
            for i, row in enumerate(df_phases.itertuples()):
                plt.text(i, row.Score + 0.02, f'{row.Score:.4f}', ha='center', fontsize=11, fontweight='bold')
            plt.tight_layout()
            phase_path = outputs_dir / 'phase_comparison.png'
            plt.savefig(phase_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Phase comparison: {phase_path}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate phase comparison: {e}")


def generate_comprehensive_sweetviz_report(df: pd.DataFrame, outputs_dir: Path, target_col: str, task_type: str):
    """
    Generate final comprehensive Sweetviz HTML report for processed dataset.
    """
    print("📊 Generating final Sweetviz report for processed dataset...")
    
    try:
        import sweetviz as sv
        
        # Sample if too large (max 10,000 rows)
        df_sample = df.sample(n=min(10000, len(df)), random_state=42) if len(df) > 10000 else df
        
        # Generate report with target feature
        target_feat = target_col if task_type != 'clustering' and target_col in df.columns else None
        report = sv.analyze(df_sample, target_feat=target_feat)
        
        # Save report
        report_path = outputs_dir / 'final_dataset_sweetviz_report.html'
        report.show_html(str(report_path), open_browser=False, layout='vertical')
        
        print(f"  ✅ Final Sweetviz report: {report_path}")
        return True
        
    except ImportError:
        print("  ⚠️ Sweetviz not available, skipping final HTML report")
        return False
    except Exception as e:
        print(f"  ⚠️ Failed to generate final Sweetviz report: {e}")
        return False


def validate_and_log_outputs(output_path: Path, output_type: str = "model") -> dict:
    """
    Validate that output files exist and log their details.
    Returns: {"valid": bool, "files": [dict], "total_size": int, "errors": [str]}
    """
    validation = {"valid": False, "files": [], "total_size": 0, "errors": []}
    
    if not output_path.exists():
        validation["errors"].append(f"Output path does not exist: {output_path}")
        return validation
    
    # Collect all files
    files = list(output_path.rglob("*")) if output_path.is_dir() else [output_path]
    files = [f for f in files if f.is_file()]
    
    if not files:
        validation["errors"].append(f"Output folder is empty: {output_path}")
        return validation
    
    # Validate each file
    for file in files:
        size = file.stat().st_size
        validation["files"].append({"name": file.name, "size": size})
        validation["total_size"] += size
        print(f"  📦 {output_type}: {file.name} ({size:,} bytes)")
    
    validation["valid"] = validation["total_size"] > 0
    return validation


def validate_input_paths(args) -> dict:
    """
    Validate all required input paths exist and are accessible.
    Returns: {"valid": bool, "errors": [str], "warnings": [str]}
    """
    validation = {"valid": True, "errors": [], "warnings": []}
    
    # Check config file
    if not Path(args.config).exists():
        validation["errors"].append(f"Config file missing: {args.config}")
        validation["valid"] = False
    
    # Check dataset
    if not Path(args.dataset_in).exists():
        validation["errors"].append(f"Dataset missing: {args.dataset_in}")
        validation["valid"] = False
    else:
        try:
            df = pd.read_csv(args.dataset_in)
            if df.empty:
                validation["warnings"].append(f"Dataset is empty: {args.dataset_in}")
        except Exception as e:
            validation["errors"].append(f"Cannot read dataset: {e}")
            validation["valid"] = False
    
    # Check model paths (can be files or folders)
    model_paths = {
        "baseline": args.baseline_model,
        "phaseb": args.phaseb_model,
        "phasec": args.phasec_model
    }
    
    for phase_name, model_path in model_paths.items():
        path = Path(model_path)
        if not path.exists():
            validation["warnings"].append(f"{phase_name} model missing: {model_path}")
        elif path.is_dir():
            # Check if folder contains model.pkl
            model_pkl = path / "model.pkl"
            if not model_pkl.exists():
                validation["warnings"].append(f"{phase_name} model folder missing model.pkl: {model_path}")
        elif path.suffix != ".pkl":
            validation["warnings"].append(f"{phase_name} model not a .pkl file: {model_path}")
    
    return validation


def ensure_output_dirs(args):
    """Create output directories if they don't exist"""
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.champion_out).mkdir(parents=True, exist_ok=True)


def load_model_and_encoder(path: str):
    """Load model from file or folder path, label encoder, and optimal threshold if exists.
    
    Returns:
        (model, label_encoder, threshold) — threshold is None if not found.
    """
    try:
        import joblib
        from pathlib import Path
        path_obj = Path(path)
        
        model = None
        label_encoder = None
        threshold = None
        
        # If path is a folder, look for model.pkl and label_encoder.pkl inside
        if path_obj.is_dir():
            model_file = path_obj / "model.pkl"
            encoder_file = path_obj / "label_encoder.pkl"
            threshold_file = path_obj / "threshold_info.json"
            
            if model_file.exists():
                model = joblib.load(str(model_file))
            else:
                print(f"  ⚠️  Model folder exists but model.pkl missing: {path}")
                return None, None, None
            
            # Try to load label encoder (optional, only exists for classification with string labels)
            if encoder_file.exists():
                label_encoder = joblib.load(str(encoder_file))
                print(f"  ✅ Loaded label encoder from {path}")
            
            # Try to load optimal threshold (saved by s5a baseline)
            if threshold_file.exists():
                import json
                with open(threshold_file) as f:
                    tinfo = json.load(f)
                threshold = tinfo.get("optimal_threshold")
                if threshold is not None:
                    print(f"  🎯 Loaded threshold={threshold:.4f} from {threshold_file.name}")
        
        # If path is a file, load it directly (no encoder/threshold in this case)
        elif path_obj.exists() and path_obj.suffix == ".pkl":
            model = joblib.load(path)
        else:
            print(f"  ⚠️  Model path not found or invalid: {path}")
            return None, None, None
        
        return model, label_encoder, threshold
    except Exception as e:
        print(f"  ❌ Error loading model from {path}: {e}")
        return None, None, None


def get_primary_metric(task_type: str) -> str:
    """Return the primary metric name for a given task type.
    
    For classification, use balanced_accuracy to properly handle
    imbalanced datasets (e.g. 80/20 churn). Raw accuracy masks
    catastrophic minority-class failures.
    """
    if task_type == "classification":
        return "balanced_accuracy"
    elif task_type == "regression":
        return "r2"
    elif task_type == "clustering":
        return "silhouette_score"
    else:
        return "balanced_accuracy"  # default fallback


def eval_model(model, X_test, y_test, task: str, label_encoder=None, threshold=None):
    """Evaluate model and return comprehensive metrics.
    
    Args:
        threshold: For classification, override default 0.5 decision boundary.
                   When provided, uses predict_proba >= threshold instead of
                   model.predict(). Critical for imbalanced datasets.
    """
    if model is None:
        return None
    try:
        if task == "classification":
            # Use optimal threshold if provided AND model supports predict_proba
            if threshold is not None and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                # Binary: use column 1 (positive class probability)
                if proba.shape[1] == 2:
                    preds = (proba[:, 1] >= threshold).astype(int)
                    print(f"  🎯 Using optimal threshold={threshold:.4f} (predict_proba)")
                else:
                    # Multiclass: threshold not applicable, fall back to argmax
                    preds = model.predict(X_test)
                    print(f"  ⚠️  Multiclass: ignoring threshold, using model.predict()")
            else:
                preds = model.predict(X_test)
            
            # 🔥 FIX: Handle label encoding mismatch
            # If model predicts numeric (0, 1) but y_test has strings ('no', 'yes'), encode y_test
            y_test_encoded = y_test
            if label_encoder is not None:
                # Model was trained with encoded labels, encode y_test for comparison
                y_test_encoded = label_encoder.transform(y_test)
                print(f"  ✅ Encoded y_test using saved label encoder")
            elif pd.api.types.is_string_dtype(y_test) or y_test.dtype == 'object':
                # y_test is string but no encoder provided - create one from model's classes
                from sklearn.preprocessing import LabelEncoder
                temp_encoder = LabelEncoder()
                if hasattr(model, 'classes_'):
                    # Use model's training classes to ensure consistent label mapping
                    temp_encoder.classes_ = np.array(model.classes_)
                    y_test_encoded = temp_encoder.transform(y_test)
                    print(f"  ⚠️  Created label encoder from model.classes_: {list(model.classes_)}")
                else:
                    y_test_encoded = temp_encoder.fit_transform(y_test)
                    print(f"  ⚠️  Created temporary label encoder (no model.classes_ available)")
            
            metrics = {
                "accuracy": float(accuracy_score(y_test_encoded, preds)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_encoded, preds)),
            }
            
            # Detect binary vs multiclass for correct averaging
            n_classes = len(np.unique(y_test_encoded))
            avg_strategy = 'binary' if n_classes == 2 else 'weighted'
            
            metrics["precision"] = float(precision_score(y_test_encoded, preds, zero_division=0, average=avg_strategy))
            metrics["recall"] = float(recall_score(y_test_encoded, preds, zero_division=0, average=avg_strategy))
            metrics["f1"] = float(f1_score(y_test_encoded, preds, zero_division=0, average=avg_strategy))
            
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_test)
                    if n_classes == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test_encoded, prob[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_test_encoded, prob, multi_class="ovr", average="weighted"))
            except Exception as _auc_err:
                logging.getLogger(__name__).debug("ROC-AUC computation failed: %s", _auc_err)
            return metrics
        elif task == "regression":
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            metrics = {
                "r2": float(r2_score(y_test, preds)),
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(np.sqrt(mse)),
                "mse": float(mse)
            }
            return metrics
        elif task == "clustering":
            # For clustering, y_test contains cluster assignments (or original if no y_test available)
            preds = model.predict(X_test)
            
            # Only compute silhouette if we have more than 1 cluster
            n_clusters = len(np.unique(preds))
            metrics = {}
            
            if n_clusters > 1:
                # 🔥 FIX: Sample data for clustering metrics to prevent OOM on large datasets
                # silhouette_score is O(n²) — 541K rows will exhaust 16 GB RAM
                _CLUSTER_EVAL_CAP = 10_000
                n_total = len(X_test)
                if n_total > _CLUSTER_EVAL_CAP:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(n_total, size=_CLUSTER_EVAL_CAP, replace=False)
                    X_sample = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
                    preds_sample = preds[idx]
                    print(f"  📊 Sampling {_CLUSTER_EVAL_CAP:,}/{n_total:,} rows for clustering metrics")
                else:
                    X_sample = X_test
                    preds_sample = preds

                try:
                    metrics["silhouette_score"] = float(silhouette_score(X_sample, preds_sample))
                except Exception as e:
                    print(f"  ⚠️  Could not compute silhouette_score: {e}")
                    metrics["silhouette_score"] = -1.0
                
                try:
                    metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_sample, preds_sample))
                except Exception as e:
                    print(f"  ⚠️  Could not compute davies_bouldin_score: {e}")
                    metrics["davies_bouldin_score"] = float('inf')
                
                try:
                    metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_sample, preds_sample))
                except Exception as e:
                    print(f"  ⚠️  Could not compute calinski_harabasz_score: {e}")
                    metrics["calinski_harabasz_score"] = 0.0
            else:
                # Single cluster - all silhouette scores are 0, other metrics undefined
                metrics["silhouette_score"] = 0.0
                metrics["davies_bouldin_score"] = 0.0
                metrics["calinski_harabasz_score"] = 0.0
            
            metrics["n_clusters"] = int(n_clusters)
            return metrics
        else:
            # Default to regression behavior
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            metrics = {
                "r2": float(r2_score(y_test, preds)),
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(np.sqrt(mse)),
                "mse": float(mse)
            }
            return metrics
    except Exception as e:
        print(f"  ❌ Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return None


def _safe_metric(val, sentinel=-999.0):
    """T10: Clamp -inf/nan to sentinel so MLflow never rejects non-finite floats."""
    try:
        v = float(val)
        if np.isfinite(v):
            return v
        return sentinel
    except (ValueError, TypeError):
        return None          # caller should skip non-numeric


def log_metrics_to_mlflow(report: dict, task_type: str, logger):
    """Log all metrics to MLflow for Azure ML Studio visibility.
    
    Args:
        report: Final evaluation report dict
        task_type: classification/regression/clustering
        logger: MetricsLogger instance from create_metrics_logger()
    """
    try:
        # Log task metadata
        logger.log_param("task_type", task_type)
        logger.log_param("final_evaluation", "true")
        
        primary_metric = get_primary_metric(task_type)
        
        # Log baseline metrics (T10: clamp non-finite values)
        if report["baseline_metrics"]:
            for key, val in report["baseline_metrics"].items():
                safe = _safe_metric(val)
                if safe is not None:
                    logger.log_metric(f"baseline_{key}", safe)
        
        # Log Phase B metrics (T10: clamp non-finite values)
        if report["phaseb_metrics"]:
            for key, val in report["phaseb_metrics"].items():
                safe = _safe_metric(val)
                if safe is not None:
                    logger.log_metric(f"phaseb_{key}", safe)
        
        # Log Phase C metrics (T10: clamp non-finite values)
        if report["phasec_metrics"]:
            for key, val in report["phasec_metrics"].items():
                safe = _safe_metric(val)
                if safe is not None:
                    logger.log_metric(f"phasec_{key}", safe)
        
        # Log champion selection (T10: clamp champion_score)
        selection = report.get("selection", {})
        logger.log_param("champion_phase", selection.get("key", "none"))
        if selection.get("score") is not None:
            safe_score = _safe_metric(selection["score"])
            if safe_score is not None:
                logger.log_metric("champion_score", safe_score)
        
        # Log primary metric for comparison (task-aware, T10: clamp)
        if report.get("baseline_metrics") and primary_metric in report["baseline_metrics"]:
            safe = _safe_metric(report["baseline_metrics"][primary_metric])
            if safe is not None:
                logger.log_metric("final_baseline_metric", safe)
        if report.get("phaseb_metrics") and primary_metric in report["phaseb_metrics"]:
            safe = _safe_metric(report["phaseb_metrics"][primary_metric])
            if safe is not None:
                logger.log_metric("final_phaseb_metric", safe)
        if report.get("phasec_metrics") and primary_metric in report["phasec_metrics"]:
            safe = _safe_metric(report["phasec_metrics"][primary_metric])
            if safe is not None:
                logger.log_metric("final_phasec_metric", safe)
        
        print("  ✅ All metrics logged to MLflow")
        
    except Exception as e:
        print(f"  ⚠️  Failed to log some metrics to MLflow: {e}")


def validate_champion_output(champion_out: str) -> dict:
    """
    Validate that champion model was successfully saved.
    Returns: {"valid": bool, "model_path": str, "size_bytes": int, "errors": [str]}
    """
    validation = {"valid": False, "model_path": None, "size_bytes": 0, "errors": []}
    
    champion_path = Path(champion_out)
    
    if not champion_path.exists():
        validation["errors"].append(f"Champion output directory not created: {champion_out}")
        return validation
    
    # Check for model.pkl
    model_pkl = champion_path / "model.pkl"
    if model_pkl.exists() and model_pkl.is_file():
        size = model_pkl.stat().st_size
        if size > 0:
            validation["valid"] = True
            validation["model_path"] = str(model_pkl)
            validation["size_bytes"] = size
        else:
            validation["errors"].append(f"Champion model.pkl exists but is empty (0 bytes)")
    else:
        validation["errors"].append(f"Champion model.pkl not found in {champion_out}")
    
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--phaseb_model", required=True)
    parser.add_argument("--phasec_model", required=True)
    parser.add_argument("--report_out", required=True)
    parser.add_argument("--champion_out", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("STEP s10: FINAL EVALUATION & CHAMPION SELECTION")
    print("=" * 80)
    _t0 = _time_mod.time()

    # Disable autologging (do NOT change tracking URI — breaks MLflow hierarchy)
    _safe_disable_autolog()

    # 1. VALIDATE INPUTS
    print("\n📋 Validating input paths...")
    validation = validate_input_paths(args)
    
    if validation["errors"]:
        print("  ❌ CRITICAL ERRORS:")
        for err in validation["errors"]:
            print(f"     - {err}")
    
    if validation["warnings"]:
        print("  ⚠️  WARNINGS:")
        for warn in validation["warnings"]:
            print(f"     - {warn}")
    
    if not validation["valid"]:
        print("\n❌ Input validation failed. Cannot proceed with final evaluation.")
        # Still create minimal outputs to prevent pipeline failure
        ensure_output_dirs(args)
        error_report = {
            "status": "failed",
            "validation_errors": validation["errors"],
            "validation_warnings": validation["warnings"]
        }
        with open(args.report_out, "w") as f:
            json.dump(error_report, f, indent=2, cls=NumpyEncoder)
        return
    
    print("  ✅ Input validation passed")

    # 2. ENSURE OUTPUT DIRECTORIES EXIST
    ensure_output_dirs(args)

    # 3. LOAD CONFIG AND DATASET
    print("\n📦 Loading config and dataset...")
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")  # 🔥 CRITICAL FIX
    
    df = pd.read_csv(args.dataset_in, sep=delimiter)  # 🔥 FIXED
    
    print(f"  Task: {task_type}")
    print(f"  Target: {target_col}")
    if task_type == "clustering":
        print(f"  ⚠️ Clustering task - label encoders not applicable")
    
    # 🎯 NEW: COLLECT ALL STAGE METRICS FROM MLFLOW
    print("\n📈 Collecting comprehensive metrics from all pipeline stages...")
    # Fixed: Use correct config key 'azureml.experiment_name' with fallback
    experiment_name = cfg.get("azureml", {}).get("experiment_name") or cfg.get("azure_ml", {}).get("experiment_name_pattern", "mlops_experiment")
    all_stage_metrics = collect_all_stage_metrics(experiment_name)
    
    # Save all collected metrics to outputs/ folder
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics_path = outputs_dir / "all_stages_metrics.json"
    with open(all_metrics_path, "w") as f:
        json.dump(all_stage_metrics, f, indent=2, cls=NumpyEncoder)
    print(f"  ✅ All stage metrics saved: {all_metrics_path}")
    
    # 🎯 NEW: GENERATE PERFORMANCE VISUALIZATIONS
    print("\n📊 Generating performance visualizations...")
    generate_performance_visualizations(all_stage_metrics, outputs_dir, task_type)
    
    # 🎯 NEW: GENERATE FINAL SWEETVIZ REPORT
    print("\n📊 Generating final comprehensive EDA report...")
    generate_comprehensive_sweetviz_report(df, outputs_dir, target_col, task_type)
    
    # Split dataset into features and target
    if task_type != "clustering":
        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' required but not found in dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split into train/test (using test set for final evaluation)
        from sklearn.model_selection import train_test_split
        # Only stratify for classification; regression targets are continuous
        stratify_param = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        print(f"  Test set: {len(X_test):,} samples")
    else:
        # Clustering: no target column
        X_test = df.copy()
        y_test = None
        print(f"  Clustering mode: using full dataset ({len(X_test):,} samples)")
    
    # 4. LOAD MODELS AND LABEL ENCODERS
    print("\n🔧 Loading models from all phases...")
    print(f"  📂 Baseline path: {args.baseline_model}")
    print(f"  📂 Phase B path:  {args.phaseb_model}")
    print(f"  📂 Phase C path:  {args.phasec_model}")

    # List contents of each model directory for debugging
    for label, mpath in [("Baseline", args.baseline_model), ("Phase B", args.phaseb_model), ("Phase C", args.phasec_model)]:
        mp = Path(mpath)
        if mp.is_dir():
            files = list(mp.iterdir())
            print(f"  📁 {label} dir contents: {[f.name for f in files]} ({len(files)} files)")
        elif mp.exists():
            print(f"  📄 {label} is file: {mp.name} ({mp.stat().st_size:,} bytes)")
        else:
            print(f"  ❌ {label} path does not exist")

    baseline, baseline_encoder, baseline_threshold = load_model_and_encoder(args.baseline_model)
    phaseb, phaseb_encoder, phaseb_threshold = load_model_and_encoder(args.phaseb_model)
    phasec, phasec_encoder, phasec_threshold = load_model_and_encoder(args.phasec_model)
    
    print(f"  Baseline: {'✅ Loaded' if baseline else '❌ Failed'}")
    print(f"  Phase B:  {'✅ Loaded' if phaseb else '❌ Failed'}")
    print(f"  Phase C:  {'✅ Loaded' if phasec else '❌ Failed'}")

    # 5. EVALUATE ALL MODELS
    print("\n📊 Evaluating all models...")
    mb = eval_model(baseline, X_test, y_test, task_type, baseline_encoder, threshold=baseline_threshold)

    # Phase B: use variant-preprocessed holdout data if available.
    # s06 saves phaseb_eval_data.csv alongside the champion model — this data
    # was preprocessed with the winning variant's recipe and column-aligned to
    # the model's feature_names_in_.  Using s4 data (baseline-preprocessed)
    # would cause a feature-space mismatch → garbage predictions.
    phaseb_eval_path = Path(args.phaseb_model) / "phaseb_eval_data.csv" if args.phaseb_model else None
    if phaseb_eval_path and phaseb_eval_path.exists() and phaseb is not None:
        try:
            phaseb_eval_df = pd.read_csv(phaseb_eval_path)
            X_test_pb = phaseb_eval_df.drop(columns=[target_col], errors="ignore")
            y_test_pb = phaseb_eval_df[target_col] if target_col in phaseb_eval_df.columns else y_test
            # Safety: align to model expectations
            if hasattr(phaseb, 'feature_names_in_'):
                X_test_pb = X_test_pb.reindex(columns=phaseb.feature_names_in_, fill_value=0)
            pb = eval_model(phaseb, X_test_pb, y_test_pb, task_type, phaseb_encoder, threshold=phaseb_threshold)
            print(f"  ✅ Phase B evaluated on variant-preprocessed holdout ({len(X_test_pb):,} samples)")
        except Exception as _pb_err:
            print(f"  ⚠️  Phase B preprocessed eval failed, falling back to s4 data: {_pb_err}")
            pb = eval_model(phaseb, X_test, y_test, task_type, phaseb_encoder, threshold=phaseb_threshold)
    else:
        pb = eval_model(phaseb, X_test, y_test, task_type, phaseb_encoder, threshold=phaseb_threshold)

    pc = eval_model(phasec, X_test, y_test, task_type, phasec_encoder, threshold=phasec_threshold)
    
    primary_metric = get_primary_metric(task_type)
    if mb and primary_metric in mb:
        print(f"  Baseline: {primary_metric}={mb.get(primary_metric, 'N/A'):.4f}")
    if pb and primary_metric in pb:
        print(f"  Phase B:  {primary_metric}={pb.get(primary_metric, 'N/A'):.4f}")
    if pc and primary_metric in pc:
        print(f"  Phase C:  {primary_metric}={pc.get(primary_metric, 'N/A'):.4f}")

    # ── 5b. LOAD ALL VARIANT RESULTS FROM PHASE B ──────────────
    _variant_leaderboard_df = None
    _all_variant_results = None
    try:
        phaseb_dir = Path(args.phaseb_model).resolve()
        _lb_path = phaseb_dir / "leaderboard.csv"
        _ar_path = phaseb_dir / "all_results.json"
        if _lb_path.is_file():
            _variant_leaderboard_df = pd.read_csv(_lb_path)
            # Sort by primary_metric descending (higher is better for classification/regression)
            _variant_leaderboard_df = _variant_leaderboard_df.sort_values(
                "primary_metric", ascending=False
            ).reset_index(drop=True)
            _variant_leaderboard_df["rank"] = _variant_leaderboard_df.index + 1
            print(f"\n📋 Loaded Phase B leaderboard: {len(_variant_leaderboard_df)} variant results from {_lb_path.name}")
        else:
            print(f"\n⚠️  Phase B leaderboard not found at {_lb_path}")
        if _ar_path.is_file():
            with open(_ar_path) as _arf:
                _all_variant_results = json.load(_arf)
            # Rank all results (including timed_out/failed)
            # Separate successful from failed
            _ok_results = [r for r in _all_variant_results if not r.get("timed_out") and not r.get("failed")]
            _fail_results = [r for r in _all_variant_results if r.get("timed_out") or r.get("failed")]
            # Sort successful by primary_metric desc
            _ok_results.sort(key=lambda r: r.get("metrics", {}).get("primary_metric", -1), reverse=True)
            _fail_results.sort(key=lambda r: r.get("variant_id", ""))
            # Assign ranks
            for _i, _r in enumerate(_ok_results, 1):
                _r["rank"] = _i
            for _i, _r in enumerate(_fail_results, len(_ok_results) + 1):
                _r["rank"] = _i
                _r["rank_note"] = "timed_out" if _r.get("timed_out") else "failed"
            _all_variant_results = _ok_results + _fail_results
            print(f"📋 Loaded all variant results: {len(_all_variant_results)} entries ({len(_ok_results)} OK, {len(_fail_results)} timed_out/failed)")
        else:
            print(f"⚠️  all_results.json not found at {_ar_path}")
    except Exception as _vr_err:
        print(f"⚠️  Could not load variant results (non-fatal): {_vr_err}")

    # 6. SELECT CHAMPION
    print("\n🏆 Selecting champion model...")
    primary_metric = get_primary_metric(task_type)
    
    def primary_score(m):
        if m is None:
            return -np.inf if primary_metric != "davies_bouldin_score" else np.inf
        
        # For davies_bouldin (lower is better), negate it to maintain max-optimization
        if primary_metric == "davies_bouldin_score":
            return -m.get(primary_metric, -np.inf if primary_metric == "davies_bouldin_score" else -np.inf)
        else:
            # For silhouette and other metrics (higher is better)
            return m.get(primary_metric, -np.inf)

    candidates = {
        "baseline": (mb, args.baseline_model),
        "phaseb": (pb, args.phaseb_model),
        "phasec": (pc, args.phasec_model)
    }
    best_key = None
    best_val = -np.inf
    for k, (metrics, path) in candidates.items():
        val = primary_score(metrics)
        if val > best_val:
            best_key, best_val = k, val
    
    # T1: Guard against "all models failed" — no valid champion
    champion_valid = np.isfinite(best_val) and best_key is not None
    if not champion_valid:
        print("  ⚠️  ALL MODELS FAILED — no valid champion (all scores non-finite)")
        print("  ⚠️  Model registration will be skipped downstream")
    else:
        print(f"  ✅ Champion: {best_key} (score={best_val:.4f})")

    # T17: Quality gate — warn if champion score is below minimum quality threshold
    QUALITY_THRESHOLDS = {
        "classification": 0.50,  # balanced_accuracy above random-guess baseline
        "regression": 0.0,       # R² above mean-predictor baseline
        "clustering": 0.0,       # silhouette above zero (better than random)
    }
    quality_threshold = QUALITY_THRESHOLDS.get(task_type, 0.0)
    if champion_valid and best_val < quality_threshold:
        print(f"  ⚠️  T17 QUALITY GATE: Champion score {best_val:.4f} is BELOW threshold "
              f"{quality_threshold} for {task_type}. Registration will proceed but model quality is poor.")

    # ── 6b. SHAP EXPLAINABILITY ─────────────────────────────────
    shap_summary = None
    try:
        import shap
        champion_model_obj = {"baseline": baseline, "phaseb": phaseb, "phasec": phasec}.get(best_key)
        if champion_model_obj is not None and task_type != "clustering":
            print("\n🔍 Computing SHAP feature importance for champion...")
            # Use a sample to keep computation tractable
            _shap_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
            try:
                explainer = shap.TreeExplainer(champion_model_obj)
                shap_values = explainer.shap_values(_shap_sample)
            except Exception:
                explainer = shap.KernelExplainer(
                    champion_model_obj.predict,
                    shap.sample(_shap_sample, min(50, len(_shap_sample)))
                )
                shap_values = explainer.shap_values(_shap_sample)

            # Handle multi-class (list of arrays) vs binary (single array)
            if isinstance(shap_values, list):
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)

            feature_names = list(_shap_sample.columns)
            importance_pairs = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
            shap_summary = [{"feature": f, "mean_abs_shap": round(float(v), 6)} for f, v in importance_pairs[:20]]
            print(f"  ✅ SHAP computed — top feature: {importance_pairs[0][0]} ({importance_pairs[0][1]:.4f})")

            # Save SHAP summary artifact
            shap_path = Path(args.champion_out) / "shap_feature_importance.json"
            shap_path.parent.mkdir(parents=True, exist_ok=True)
            with open(shap_path, "w") as sf:
                json.dump(shap_summary, sf, indent=2, cls=NumpyEncoder)
            print(f"  💾 SHAP saved to {shap_path.name}")
        elif task_type == "clustering":
            print("\n🔍 SHAP: skipped for clustering (no target variable)")
    except ImportError:
        print("\n⚠️  SHAP not installed — skipping explainability (pip install shap)")
    except Exception as shap_err:
        print(f"\n⚠️  SHAP computation failed (non-fatal): {shap_err}")

    # 7. CREATE REPORT
    # Build variant rankings summary for the report
    _variant_rankings_list = []
    if _all_variant_results:
        for _vr in _all_variant_results:
            _vr_entry = {
                "rank": _vr.get("rank"),
                "variant_id": _vr.get("variant_id"),
                "engine": _vr.get("engine"),
                "algorithm": _vr.get("algorithm", "N/A"),
                "primary_metric": _vr.get("metrics", {}).get("primary_metric"),
                "runtime_sec": _vr.get("runtime_sec"),
                "n_features": _vr.get("n_features"),
                "leakage_risk": _vr.get("leakage_risk"),
                "timed_out": _vr.get("timed_out", False),
                "failed": _vr.get("failed", False),
            }
            if _vr.get("rank_note"):
                _vr_entry["rank_note"] = _vr["rank_note"]
            _vr_entry.update({k: v for k, v in _vr.get("metrics", {}).items() if k != "primary_metric"})
            _variant_rankings_list.append(_vr_entry)
    elif _variant_leaderboard_df is not None:
        for _, _row in _variant_leaderboard_df.iterrows():
            _variant_rankings_list.append(_row.to_dict())

    report = {
        "task": task_type,
        "target_column": target_col,
        "test_samples": len(X_test),
        "champion_valid": champion_valid,
        "quality_gate_passed": bool(not champion_valid or best_val >= quality_threshold),
        "quality_threshold": quality_threshold,
        "baseline_metrics": mb,
        "phaseb_metrics": pb,
        "phasec_metrics": pc,
        "selection": {"key": best_key, "score": float(best_val) if np.isfinite(best_val) else None},
        "validation": validation,
        "variant_rankings": _variant_rankings_list,
        "variant_count": {
            "total": len(_variant_rankings_list),
            "successful": sum(1 for v in _variant_rankings_list if not v.get("timed_out") and not v.get("failed")),
            "timed_out": sum(1 for v in _variant_rankings_list if v.get("timed_out")),
            "failed": sum(1 for v in _variant_rankings_list if v.get("failed")),
        },
        "shap_feature_importance": shap_summary,
    }
    
    # Use absolute path resolution for Azure ML outputs
    report_path = Path(args.report_out).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"  ✅ Report saved: {report_path} ({report_path.stat().st_size:,} bytes)")
    
    # 📊 CREATE OUTPUTS FOLDER WITH COMPREHENSIVE FINAL REPORT
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 COMPREHENSIVE FINAL EVALUATION TO outputs/ FOLDER:")
    
    # 1. Save full comparison CSV
    comparison_data = []
    for phase, (metrics, _) in candidates.items():
        if metrics:
            row = {"phase": phase}
            row.update(metrics)
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = outputs_dir / "final_phase_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  ✅ Phase comparison: {comparison_path} ({len(comparison_df)} phases, {comparison_path.stat().st_size:,} bytes)")

    # 1b. Save full variant rankings CSV (ALL variants, not just champions)
    if _variant_rankings_list:
        _vr_df = pd.DataFrame(_variant_rankings_list)
        _vr_path = outputs_dir / "variant_rankings.csv"
        _vr_df.to_csv(_vr_path, index=False)
        print(f"  ✅ Variant rankings: {_vr_path} ({len(_vr_df)} variants, {_vr_path.stat().st_size:,} bytes)")
        # Also save the full all_results.json to outputs for easy access
        if _all_variant_results:
            _avr_path = outputs_dir / "all_variant_results.json"
            with open(_avr_path, "w") as _avrf:
                json.dump(_all_variant_results, _avrf, indent=2, cls=NumpyEncoder)
            print(f"  ✅ All variant results: {_avr_path} ({len(_all_variant_results)} entries)")
        # Print top-10 summary
        _ok_variants = [v for v in _variant_rankings_list if not v.get("timed_out") and not v.get("failed")]
        _to_variants = [v for v in _variant_rankings_list if v.get("timed_out")]
        print(f"\n📊 VARIANT RANKING SUMMARY:")
        print(f"  Total variants trained: {len(_variant_rankings_list)}")
        print(f"  Successful: {len(_ok_variants)}, Timed out: {len(_to_variants)}, Failed: {len(_variant_rankings_list) - len(_ok_variants) - len(_to_variants)}")
        if _ok_variants:
            print(f"  Top-5 by {primary_metric}:")
            for _v in _ok_variants[:5]:
                print(f"    #{_v.get('rank', '?')} {_v.get('variant_id', '?')[:12]:>12s} | {_v.get('engine', '?'):>8s} | {_v.get('algorithm', '?'):>25s} | {primary_metric}={_v.get('primary_metric', 'N/A')}")
    else:
        print("  ⚠️  No variant rankings available to write")

    # 2. Save champion summary
    champion_summary = {
        "champion_phase": best_key,
        "champion_score": float(best_val) if best_val != -np.inf else None,
        "primary_metric": primary_metric,
        "task_type": task_type,
        "test_samples": len(X_test),
        "all_phases": {
            "baseline": mb,
            "phaseb": pb,
            "phasec": pc
        }
    }
    champion_path = outputs_dir / "final_champion_summary.json"
    with open(champion_path, 'w') as f:
        json.dump(champion_summary, f, indent=2, cls=NumpyEncoder)
    print(f"  ✅ Champion summary: {champion_path} ({champion_path.stat().st_size:,} bytes)")
    
    # 3. Copy report to outputs for visibility
    import shutil
    shutil.copy2(report_path, outputs_dir / "final_evaluation_report.json")
    print(f"  ✅ Report copied to outputs: final_evaluation_report.json")
    
    # 4. Copy champion model to outputs/ for easy download
    print(f"\n📦 Copying champion model to outputs/ folder...")
    chosen_path = candidates.get(best_key, (None, None))[1]
    if chosen_path:
        src_model = Path(chosen_path)
        if src_model.exists():
            try:
                if src_model.is_file():
                    shutil.copy2(src_model, outputs_dir / f"final_champion_model_{best_key}.pkl")
                    print(f"  ✅ Champion model copied: final_champion_model_{best_key}.pkl")
                elif src_model.is_dir():
                    # Copy all files from model directory
                    model_files_copied = 0
                    for src_file in src_model.rglob('*'):
                        if src_file.is_file():
                            dest_file = outputs_dir / src_file.name
                            shutil.copy2(src_file, dest_file)
                            model_files_copied += 1
                    print(f"  ✅ Copied {model_files_copied} champion model files to outputs/")
            except Exception as copy_err:
                print(f"  ⚠️  Failed to copy champion model to outputs/: {copy_err}")
        else:
            print(f"  ⚠️  Champion model path does not exist: {src_model}")
    else:
        print(f"  ⚠️  No champion model path available")

    # 8. COPY CHAMPION MODEL AND LABEL ENCODER TO OFFICIAL OUTPUT
    # T2: When no valid champion, create output dir with sentinel file
    if not champion_valid:
        out_dir = Path(args.champion_out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".no_champion").write_text(json.dumps({
            "reason": "all_models_failed", "best_val": str(best_val), "task_type": task_type,
        }))
        print(f"\n⚠️  No valid champion — wrote .no_champion sentinel to {out_dir}")
        chosen_path = None
        chosen_encoder = None
    else:
        print("\n💾 Copying champion model to official output...")
        chosen_path = candidates.get(best_key, (None, None))[1]
        chosen_encoder = None
    
        # Determine which encoder to use based on champion
        if best_key == "baseline":
            chosen_encoder = baseline_encoder
        elif best_key == "phaseb":
            chosen_encoder = phaseb_encoder
        elif best_key == "phasec":
            chosen_encoder = phasec_encoder
    
    if chosen_path:
        src = Path(chosen_path)
        out_dir = Path(args.champion_out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  📂 Champion output path: {out_dir}")
        
        import shutil
        
        # If source is a folder, copy model.pkl and label_encoder.pkl from it
        if src.is_dir():
            model_file = src / "model.pkl"
            encoder_file = src / "label_encoder.pkl"
            
            if model_file.exists():
                shutil.copy(str(model_file), str(out_dir / "model.pkl"))
                print(f"  ✅ Copied model.pkl from folder: {src} ({model_file.stat().st_size:,} bytes)")
            
            # Copy label encoder if it exists
            if encoder_file.exists():
                shutil.copy(str(encoder_file), str(out_dir / "label_encoder.pkl"))
                print(f"  ✅ Copied label_encoder.pkl from folder: {src} ({encoder_file.stat().st_size:,} bytes)")
            elif chosen_encoder is not None:
                # Encoder exists in memory but not in folder, save it
                import joblib
                encoder_path = out_dir / "label_encoder.pkl"
                joblib.dump(chosen_encoder, str(encoder_path))
                print(f"  ✅ Saved label encoder from memory ({encoder_path.stat().st_size:,} bytes)")
        
        # If source is a file, copy it directly (legacy support)
        elif src.exists() and src.suffix == ".pkl":
            shutil.copy(str(src), str(out_dir / "model.pkl"))
            print(f"  ✅ Copied from file: {src} ({src.stat().st_size:,} bytes)")
            
            # Save encoder if available
            if chosen_encoder is not None:
                import joblib
                encoder_path = out_dir / "label_encoder.pkl"
                joblib.dump(chosen_encoder, str(encoder_path))
                print(f"  ✅ Saved label encoder ({encoder_path.stat().st_size:,} bytes)")
    else:
        out_dir = Path(args.champion_out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ⚠️  No valid champion model to copy: {out_dir}")

    # 9. VALIDATE CHAMPION OUTPUT
    print("\n🔍 Validating champion output...")
    output_validation = validate_and_log_outputs(out_dir, "Final Champion")
    
    if output_validation["valid"]:
        print(f"  ✅ Champion output validation passed")
        print(f"     Total size: {output_validation['total_size']:,} bytes in {len(output_validation['files'])} files")
        print(f"     Output path: {out_dir}")
    else:
        print(f"  ❌ Champion output validation failed:")
        for err in output_validation["errors"]:
            print(f"     - {err}")
    
    # Add to report
    report["output_validation"] = output_validation
    report["champion_output_path"] = str(out_dir)

    # 10. CREATE MLFLOW LOGGER AND LOG ALL METRICS TO MLFLOW
    print("\n📈 Creating MLflow logger and logging metrics...")
    logger = create_metrics_logger(
        run_name="s10_final_evaluation",
        tags={"pipeline": "v3_mlops", "phase": "evaluation", "step": "s10"}
    )
    log_metrics_to_mlflow(report, task_type, logger)

    # T9: Log champion validity for Azure ML Studio dashboard filtering
    try:
        logger.log_metric("champion_valid", 1.0 if champion_valid else 0.0)
    except Exception as _cv_err:
        logging.getLogger(__name__).debug("champion_valid metric log failed: %s", _cv_err)

    # 11. SAVE FINAL REPORT WITH OUTPUT VALIDATION
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"  ✅ Final report updated: {report_path}")
    
    print("\n" + "=" * 80)
    print("✅ FINAL EVALUATION COMPLETE")
    print("=" * 80)
    print(f"📄 Report: {report_path}")
    print(f"🏆 Champion: {out_dir}")

    # ── Emit stage signal ──────────────────────────────────────────────
    _elapsed = _time_mod.time() - _t0
    _count_in = 3  # baseline, phaseb, phasec candidates
    _count_out = 1 if best_key else 0
    # Derive delta vs baseline
    _baseline_score = primary_score(mb) if mb else None
    _delta = None
    if _baseline_score is not None and best_val is not None and _baseline_score != -np.inf:
        _delta = round(best_val - _baseline_score, 6)
    try:
        sig = StageSignal(
            stage_name="final_evaluation",
            stage_id="S10",
            task_type=task_type,
            config_name=Path(args.config).name,
            candidate_count_in=_count_in,
            candidate_count_out=_count_out,
            best_score=float(best_val) if best_val != -np.inf else None,
            best_metric_name=primary_metric,
            delta_vs_baseline=_delta,
            compute_time_sec=round(_elapsed, 2),
            recommendation="proceed" if best_key else "stop",
            recommendation_reason=f"Champion: {best_key} ({primary_metric}={best_val:.4f})" if best_key else "No valid champion",
            extra={
                "baseline_metrics": mb,
                "phaseb_metrics": pb,
                "phasec_metrics": pc,
                "champion_phase": best_key,
            },
        )
        write_stage_signal(sig, out_dir="outputs", filename="final_stage_signal.json")
    except Exception as _sig_err:
        print(f"⚠️  Stage signal write failed (non-fatal): {_sig_err}")

    # ── Candidate Ledger: emit comparison rows + merge all stages ──────
    try:
        _ledger_rows = []
        for _phase, _mdict, _model_path in [
            ("baseline", mb, args.baseline_model),
            ("phase_b", pb, args.phaseb_model),
            ("phase_c", pc, args.phasec_model),
        ]:
            _norm = normalize_metrics(task_type, _mdict or {})
            _is_champ = (best_key == _phase.replace("phase_", "phase"))
            _row = make_row(
                stage="final", step_name="s10", engine=_phase,
                candidate_id=f"final_{_phase}",
                task_type=task_type,
                dataset_id=Path(args.config).name,
                status="ok" if _mdict else "failed",
                compute_time_sec=round(_elapsed, 2),
                source_path="src/steps/final_evaluation.py",
                is_stage_best=_is_champ,
                is_final_champion=_is_champ,
                **_norm,
            )
            _ledger_rows.append(_row)
        write_stage_table(
            _ledger_rows,
            csv_path="outputs/s10_candidates.csv",
            parquet_path="outputs/s10_candidates.parquet",
        )

        # Discover all upstream stage ledger CSVs from input paths
        # Search BOTH the input dir itself AND its parent (S06 puts files inside champion_model/)
        _stage_csvs = []
        for _input_dir in [args.baseline_model, args.phaseb_model, args.phasec_model]:
            _resolved = Path(_input_dir).resolve()
            for _search_dir in [_resolved, _resolved.parent]:
                if _search_dir.is_dir():
                    for _csv in _search_dir.glob("s*_candidates.csv"):
                        if _csv.is_file() and str(_csv) not in _stage_csvs:
                            _stage_csvs.append(str(_csv))
        # Also include local outputs
        for _csv in Path("outputs").glob("s*_candidates.csv"):
            if str(_csv) not in _stage_csvs:
                _stage_csvs.append(str(_csv))

        if _stage_csvs:
            merge_ledgers(
                _stage_csvs,
                out_csv="outputs/all_candidates.csv",
                out_parquet="outputs/all_candidates.parquet",
            )
            _summary = build_summary("outputs/all_candidates.csv")
            with open("outputs/all_candidates_summary.json", "w") as _sf:
                json.dump(_summary, _sf, indent=2, cls=NumpyEncoder)
            _md = build_readme_md(_summary)
            with open("outputs/all_candidates_README.md", "w") as _mf:
                _mf.write(_md)
            print(f"📒 Candidate ledger merged: {len(_stage_csvs)} stage files → all_candidates.csv")

            # ── AIM-Tournament: multi-metric ranking + Pareto ──────────
            try:
                _ledger_df = pd.read_csv("outputs/all_candidates.csv")
                # Filter to OK candidates only for ranking
                _ok = _ledger_df[_ledger_df["status"] == "ok"].copy()
                if len(_ok) >= 2:
                    _ok = run_aim_tournament(
                        _ok, task_type, "outputs",
                        k=min(10, len(_ok)),
                        primary_metric=primary_metric,
                    )
                    print(f"🏆 AIM-Tournament: {len(_ok)} candidates ranked")
                else:
                    print("⚠️  AIM-Tournament: <2 OK candidates — skipping ranking")
            except Exception as _aim_err:
                print(f"⚠️  AIM-Tournament failed (non-fatal): {_aim_err}")

            # ── Model Coverage Report ──────────────────────────────────
            try:
                _coverage = build_coverage_report(task_type)
                write_model_coverage(_coverage, "outputs")
            except Exception as _cov_err:
                print(f"⚠️  Model coverage report failed (non-fatal): {_cov_err}")

            # ── Deep Model Breakdown: merge all model_breakdown_*.csv ──
            try:
                _bd_csvs = []
                # Scan upstream step output directories (both input dir and parent)
                for _input_dir in [args.baseline_model, args.phaseb_model, args.phasec_model]:
                    _resolved = Path(_input_dir).resolve()
                    for _search_dir in [_resolved, _resolved.parent]:
                        if _search_dir.is_dir():
                            for _csv in _search_dir.glob("model_breakdown_*.csv"):
                                if _csv.is_file() and str(_csv) not in _bd_csvs:
                                    _bd_csvs.append(str(_csv))
                # Also scan local outputs/ (both naming patterns)
                for _csv in Path("outputs").glob("model_breakdown_*.csv"):
                    if str(_csv) not in _bd_csvs:
                        _bd_csvs.append(str(_csv))
                for _csv in Path("outputs").glob("s*_model_breakdown.csv"):
                    if str(_csv) not in _bd_csvs:
                        _bd_csvs.append(str(_csv))

                if _bd_csvs:
                    _bd_frames = []
                    for _csv_path in _bd_csvs:
                        try:
                            _tmp = pd.read_csv(_csv_path)
                            if not _tmp.empty:
                                _bd_frames.append(_tmp)
                                print(f"  📄 Loaded breakdown: {Path(_csv_path).name} ({len(_tmp)} models)")
                        except Exception as _read_err:
                            print(f"  ⚠️  Could not read {_csv_path}: {_read_err}")

                    if _bd_frames:
                        _all_bd = pd.concat(_bd_frames, ignore_index=True)
                        _all_bd_path = Path("outputs") / "all_models_breakdown.csv"
                        _all_bd.to_csv(_all_bd_path, index=False)
                        print(f"📊 All-models breakdown: {len(_all_bd)} rows → {_all_bd_path}")

                        # Summary statistics
                        print("\n" + "=" * 60)
                        print("📊  DEEP MODEL BREAKDOWN SUMMARY")
                        print("=" * 60)
                        print(f"  Total models evaluated: {len(_all_bd)}")
                        if "engine" in _all_bd.columns:
                            _eng_counts = _all_bd["engine"].value_counts()
                            for _eng, _cnt in _eng_counts.items():
                                print(f"    {_eng:>15s}: {_cnt} models")
                        if "step_name" in _all_bd.columns:
                            _stg_counts = _all_bd["step_name"].value_counts()
                            print(f"  By step:")
                            for _stg, _cnt in _stg_counts.items():
                                print(f"    {_stg:>25s}: {_cnt} models")
                        elif "stage" in _all_bd.columns:
                            _stg_counts = _all_bd["stage"].value_counts()
                            print(f"  By stage:")
                            for _stg, _cnt in _stg_counts.items():
                                print(f"    {_stg:>15s}: {_cnt} models")
                        if "variant" in _all_bd.columns:
                            _var_counts = _all_bd["variant"].value_counts()
                            print(f"  By variant ({len(_var_counts)} unique):")
                            for _var, _cnt in _var_counts.head(10).items():
                                print(f"    {_var:>25s}: {_cnt} models")
                        print("=" * 60)

                        # Log summary to MLflow
                        try:
                            logger.log_metric("total_models_evaluated", len(_all_bd))
                            if "engine" in _all_bd.columns:
                                for _eng, _cnt in _all_bd["engine"].value_counts().items():
                                    logger.log_metric(f"models_{_eng}", int(_cnt))
                        except Exception as _bd_log_err:
                            logging.getLogger(__name__).debug("breakdown summary log_metric failed: %s", _bd_log_err)
                    else:
                        print("⚠️  Model breakdown CSVs found but all empty")
                else:
                    print("ℹ️  No model_breakdown_*.csv files found (breakdown not available)")
            except Exception as _bd_err:
                print(f"⚠️  Model breakdown merge failed (non-fatal): {_bd_err}")
        else:
            print("⚠️  No upstream stage ledger CSVs found to merge")
    except Exception as _ledger_err:
        print(f"⚠️  Candidate ledger merge failed (non-fatal): {_ledger_err}")


if __name__ == "__main__":
    main()
