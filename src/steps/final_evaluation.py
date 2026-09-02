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
from utils.holdout_partition import ROW_ID_COLUMN
from utils.model_bundle import (
    ModelBundle,
    capture_input_schema,
    find_model_bundle,
    load_model_bundle,
    save_model_bundle,
)
from orchestration.contracts import QualityDecision, SplitManifest, canonical_hash
from orchestration.execution_identity import validate_execution_manifest_binding

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
    """Disable autologging without mutating workspace tracking/registry URIs."""
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
    # The Azure ML workspace-provided azureml:// URI is canonical and must stay
    # unchanged so parent/child run lineage remains execution-scoped.


def collect_all_stage_metrics(
    exact_run_ids: dict[str, str],
    execution_id: str | None,
) -> dict:
    """Collect only manifest-bound candidate runs; never scan recent runs."""
    print("📊 Collecting exact manifest-bound MLflow metrics...")
    all_metrics = {
        'preprocessing_stages': {},
        'baseline_models': {},
        'phaseb_recipes': {},
        'phasec_hpo': {},
        'aggregates': {}
    }
    if not execution_id:
        raise RuntimeError("Exact MLflow collection requires execution_id")
    if not exact_run_ids:
        raise RuntimeError("Exact MLflow collection requires candidate run IDs")

    client = mlflow.tracking.MlflowClient()
    for phase, run_id in sorted(exact_run_ids.items()):
        if not run_id:
            raise RuntimeError(f"Missing exact MLflow run ID for {phase}")
        run = client.get_run(run_id)
        tags = dict(run.data.tags)
        actual_execution_id = tags.get("execution_id")
        if actual_execution_id != execution_id:
            raise RuntimeError(
                f"MLflow run {run_id} for {phase} belongs to execution "
                f"{actual_execution_id!r}, expected {execution_id!r}"
            )
        row = {
            "run_id": run_id,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": tags,
        }
        if phase == "baseline":
            all_metrics["baseline_models"][phase] = row
        elif phase == "phaseb":
            all_metrics["phaseb_recipes"][phase] = row
        elif phase == "phasec":
            all_metrics["phasec_hpo"][phase] = row
        all_metrics["aggregates"][phase] = row

    print(f"  ✅ Collected {len(exact_run_ids)} exact candidate runs")
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

    if not Path(args.holdout_in).exists():
        validation["errors"].append(f"Holdout dataset missing: {args.holdout_in}")
        validation["valid"] = False
    else:
        try:
            holdout_df = pd.read_csv(args.holdout_in)
            if holdout_df.empty:
                validation["errors"].append(
                    f"Holdout dataset is empty: {args.holdout_in}"
                )
                validation["valid"] = False
        except Exception as e:
            validation["errors"].append(f"Cannot read holdout dataset: {e}")
            validation["valid"] = False

    if not Path(args.split_manifest_in).is_file():
        validation["errors"].append(
            f"Stage 2 SplitManifest missing: {args.split_manifest_in}"
        )
        validation["valid"] = False
    else:
        try:
            load_split_manifest(args.split_manifest_in)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            validation["errors"].append(
                f"Cannot read Stage 2 SplitManifest: {exc}"
            )
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
            if find_model_bundle(path) is None:
                validation["warnings"].append(
                    f"{phase_name} model folder has no unique exact "
                    f"ModelBundle: {model_path}"
                )
        elif path.suffix != ".pkl":
            validation["warnings"].append(f"{phase_name} model not a .pkl file: {model_path}")
    
    return validation


def ensure_output_dirs(args):
    """Create output directories if they don't exist"""
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.champion_out).mkdir(parents=True, exist_ok=True)


def load_model_and_encoder(path: str):
    """Load only the canonical ModelBundle and its embedded inference policy.
    
    Returns:
        (model, label_encoder, threshold) — threshold is None if not found.
    """
    try:
        bundle_path = find_model_bundle(path)
        if bundle_path is None:
            print(f"  ⚠️  Exact ModelBundle missing or ambiguous: {path}")
            return None, None, None
        bundle = load_model_bundle(bundle_path)
        return bundle, None, bundle.threshold
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


def load_selection_evidence(model_dir: str | Path, phase: str) -> dict:
    """Load training-CV/validation evidence without opening locked-test data."""
    root = Path(model_dir)
    names = (
        ("selection_manifest.json", "model_bundle_manifest.json")
        if phase != "phaseb"
        else ("champion_manifest.json", "selection_manifest.json")
    )
    payload = None
    source_name = None
    for name in names:
        candidate = root / name
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            source_name = name
            break
    if not isinstance(payload, dict):
        return {"phase": phase, "status": "missing", "selection_score": None}
    if payload.get("status") == "skipped_unsupported":
        return {
            "phase": phase,
            "status": "skipped_unsupported",
            "selection_score": None,
            "reason": payload.get("reason"),
        }
    selection_metrics = payload.get("selection_metrics") or {}
    result_metrics = payload.get("metrics") or {}
    evaluation = (
        payload.get("evaluation")
        or result_metrics.get("common_evaluator")
        or {}
    )
    bundle_lineage = (payload.get("model_bundle") or {}).get("lineage") or {}
    bundle_manifest = payload.get("model_bundle") or {}
    raw_lineage = payload.get("lineage") or {}
    execution_manifest = {}
    execution_manifest_path = root / "execution_manifest.json"
    if execution_manifest_path.is_file():
        execution_manifest = json.loads(
            execution_manifest_path.read_text(encoding="utf-8")
        )

    def first(*values):
        return next(
            (
                value
                for value in values
                if value is not None and str(value).strip()
            ),
            None,
        )

    lineage = {
        "execution_id": first(
            payload.get("execution_id"),
            execution_manifest.get("execution_id"),
            raw_lineage.get("execution_id"),
            evaluation.get("execution_id"),
            bundle_lineage.get("execution_id"),
        ),
        "parent_run_id": first(
            payload.get("mlflow_parent_run_id"),
            payload.get("parent_mlflow_run_id"),
            raw_lineage.get("parent_run_id"),
            raw_lineage.get("mlflow_parent_run_id"),
            raw_lineage.get("parent_mlflow_run_id"),
            evaluation.get("mlflow_parent_run_id"),
            bundle_lineage.get("parent_run_id"),
        ),
        "candidate_run_id": first(
            payload.get("mlflow_child_run_id"),
            payload.get("candidate_mlflow_run_id"),
            raw_lineage.get("candidate_run_id"),
            raw_lineage.get("mlflow_child_run_id"),
            raw_lineage.get("candidate_mlflow_run_id"),
            evaluation.get("mlflow_child_run_id"),
            bundle_lineage.get("candidate_run_id"),
        ),
    }
    score = payload.get("selection_score")
    if score is None:
        score = payload.get("primary_metric_value")
    for key in (
        "selection_score",
        "balanced_accuracy",
        "r2",
        "silhouette_score",
    ):
        if score is None:
            score = selection_metrics.get(key)
    return {
        "phase": phase,
        "status": payload.get("status", "success"),
        "selection_score": float(score) if score is not None else None,
        "candidate_id": payload.get("candidate_id") or payload.get("variant_id"),
        "model_bundle_id": payload.get("model_bundle_id")
        or payload.get("bundle_id")
        or bundle_manifest.get("bundle_id"),
        "algorithm": payload.get("algorithm")
        or (payload.get("recipe") or {}).get("algorithm_family"),
        "metric_name": payload.get("metric_name")
        or payload.get("primary_metric_name")
        or selection_metrics.get("primary_metric"),
        "split_fingerprint": payload.get("split_fingerprint")
        or evaluation.get("split_fingerprint")
        or selection_metrics.get("split_fingerprint"),
        "total_folds": payload.get("total_folds")
        or evaluation.get("total_folds")
        or selection_metrics.get("total_folds"),
        "lineage": lineage,
        "source_manifest": source_name,
    }


def validate_selection_lineage(
    evidence: dict[str, dict],
    expected_execution_id: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Fail closed unless every selectable candidate has exact same-run lineage."""
    exact_run_ids: dict[str, str] = {}
    execution_ids: set[str] = set()
    for phase, row in evidence.items():
        score = row.get("selection_score")
        selectable = (
            row.get("status") in {"success", "ok", None}
            and score is not None
            and np.isfinite(float(score))
        )
        if not selectable:
            continue
        lineage = row.get("lineage") or {}
        execution_id = lineage.get("execution_id")
        candidate_run_id = lineage.get("candidate_run_id")
        parent_run_id = lineage.get("parent_run_id")
        missing = [
            name
            for name, value in (
                ("execution_id", execution_id),
                ("parent_run_id", parent_run_id),
                ("candidate_run_id", candidate_run_id),
            )
            if value is None or not str(value).strip()
        ]
        if missing:
            raise RuntimeError(
                f"{phase} selection evidence lacks exact lineage: "
                + ", ".join(missing)
            )
        execution_ids.add(str(execution_id))
        exact_run_ids[phase] = str(candidate_run_id)

    if not exact_run_ids:
        raise RuntimeError("No selectable candidates have exact MLflow lineage")
    if len(execution_ids) != 1:
        raise RuntimeError(
            f"Selection evidence mixes execution IDs: {sorted(execution_ids)}"
        )
    execution_id = next(iter(execution_ids))
    if expected_execution_id and execution_id != str(expected_execution_id):
        raise RuntimeError(
            f"Manifest execution {execution_id!r} does not match current "
            f"execution {expected_execution_id!r}"
        )
    if len(set(exact_run_ids.values())) != len(exact_run_ids):
        raise RuntimeError("Distinct candidates cannot share one child MLflow run")
    return execution_id, exact_run_ids


def validate_selection_comparability(
    evidence: dict[str, dict],
    *,
    expected_metric: str,
    expected_folds: int,
    minimum_candidates: int = 2,
) -> None:
    """Fail closed unless selectable phases use one exact CV contract."""

    if minimum_candidates < 1:
        raise ValueError("minimum_candidates must be at least 1")

    expected_metric_normalized = str(expected_metric).strip().lower().replace(
        " ", "_"
    )
    fingerprints: set[str] = set()
    selectable_count = 0
    for phase, row in evidence.items():
        score = row.get("selection_score")
        selectable = (
            row.get("status") in {"success", "ok", None}
            and score is not None
            and np.isfinite(float(score))
        )
        if not selectable:
            continue
        selectable_count += 1
        metric = str(row.get("metric_name") or "").strip().lower().replace(
            " ", "_"
        )
        if metric != expected_metric_normalized:
            raise RuntimeError(
                f"{phase} selection metric {metric!r} does not match compiled "
                f"metric {expected_metric_normalized!r}"
            )
        try:
            folds = int(row.get("total_folds"))
        except (TypeError, ValueError):
            raise RuntimeError(
                f"{phase} selection evidence lacks a valid total_folds"
            ) from None
        if folds != int(expected_folds):
            raise RuntimeError(
                f"{phase} selection folds {folds} do not match compiled "
                f"fold count {int(expected_folds)}"
            )
        fingerprint = str(row.get("split_fingerprint") or "").strip()
        if not fingerprint:
            raise RuntimeError(
                f"{phase} selection evidence lacks split_fingerprint"
            )
        fingerprints.add(fingerprint)
    if selectable_count < minimum_candidates:
        raise RuntimeError(
            f"Only {selectable_count} selectable candidate(s) have comparable CV "
            f"evidence; at least {minimum_candidates} are required"
        )
    if len(fingerprints) != 1:
        raise RuntimeError(
            "Selectable candidates used different deterministic fold assignments"
        )


def select_champion_from_selection_evidence(
    evidence: dict[str, dict],
) -> tuple[str | None, float | None]:
    """Freeze one champion before any locked-test prediction is made."""
    candidates = []
    for phase, row in evidence.items():
        if row.get("status") not in {"success", "ok", None}:
            continue
        score = row.get("selection_score")
        if score is None or not np.isfinite(float(score)):
            continue
        candidates.append((phase, float(score)))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1])


def make_quality_decision(
    *,
    champion_valid: bool,
    observed_value: float | None,
    threshold: float,
    metric_name: str,
    block_on_quality_fail: bool,
    candidate_id: str,
    evaluated_bundle_hash: str,
) -> dict:
    finite = observed_value is not None and np.isfinite(float(observed_value))
    passed = bool(champion_valid and finite and float(observed_value) >= threshold)
    decision = (
        "pass"
        if passed
        else "block"
        if (not champion_valid or block_on_quality_fail)
        else "warn"
    )
    reasons = []
    if not champion_valid:
        reasons.append("no_valid_champion")
    elif not finite:
        reasons.append("non_finite_locked_test_metric")
    elif not passed:
        reasons.append("locked_test_metric_below_threshold")
    return QualityDecision(
        decision=decision,
        candidate_id=candidate_id,
        evaluated_bundle_hash=evaluated_bundle_hash,
        metric_name=metric_name,
        metric_value=float(observed_value) if finite else None,
        threshold=float(threshold),
        registration_allowed=decision in {"pass", "warn"},
        promotion_aliases=(),
        registration_tags={
            "promotion_allowed": str(decision == "pass").lower(),
            "block_on_quality_fail": str(
                bool(block_on_quality_fail)
            ).lower(),
        },
        reasons=tuple(reasons),
    ).to_dict()


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
                    target_decoder = getattr(model, "target_decoder", None)
                    if target_decoder is not None:
                        preds = target_decoder.inverse_transform(preds)
                    print(f"  🎯 Using optimal threshold={threshold:.4f} (predict_proba)")
                else:
                    # Multiclass: threshold not applicable, fall back to argmax
                    preds = model.predict(X_test)
                    print(f"  ⚠️  Multiclass: ignoring threshold, using model.predict()")
            else:
                preds = model.predict(X_test)
            
            # Compare in the original target domain when the immutable bundle
            # owns a decoder. Legacy numeric-output models may still provide a
            # separate encoder, in which case only y_test is transformed.
            y_test_encoded = y_test
            target_decoder = getattr(model, "target_decoder", None)
            if target_decoder is not None:
                y_test_encoded = np.asarray(y_test)
            elif label_encoder is not None:
                # Model was trained with encoded labels, encode y_test for comparison
                y_test_encoded = label_encoder.transform(y_test)
                print(f"  ✅ Encoded y_test using saved label encoder")
            elif (
                (pd.api.types.is_string_dtype(y_test) or y_test.dtype == "object")
                and np.asarray(preds).dtype.kind not in {"O", "U", "S"}
            ):
                # Legacy numeric predictions with original-domain test labels.
                from sklearn.preprocessing import LabelEncoder

                temp_encoder = LabelEncoder()
                y_test_encoded = temp_encoder.fit_transform(y_test)
                print("  ⚠️  Encoded original labels for legacy numeric predictions")
            
            metrics = {
                "accuracy": float(accuracy_score(y_test_encoded, preds)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_encoded, preds)),
            }
            
            # Detect binary vs multiclass for correct averaging
            n_classes = len(np.unique(y_test_encoded))
            numeric_binary = (
                n_classes == 2
                and set(np.unique(y_test_encoded)).issubset({0, 1})
            )
            avg_strategy = "binary" if numeric_binary else "weighted"
            
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
            # Clustering models (KMeans, DBSCAN, …) require purely numeric input.
            # Select numeric columns and cast to float64 to match training-time
            # behaviour in stage5_pycaret_train.py (which also casts to float64).
            # Non-numeric columns (e.g. residual object cols surviving stage4)
            # would cause model.predict() to raise, returning None and -inf score.
            X_eval = X_test.select_dtypes(include=[np.number]).astype(np.float64)
            if X_eval.shape[1] == 0:
                print("  ❌ No numeric features available for clustering evaluation")
                return None
            # Align to model's expected feature set when available
            if hasattr(model, 'feature_names_in_'):
                X_eval = X_eval.reindex(columns=model.feature_names_in_, fill_value=0.0)
                X_eval = X_eval.astype(np.float64)
            preds = model.predict(X_eval)

            # Only compute silhouette if we have more than 1 cluster
            n_clusters = len(np.unique(preds))
            metrics = {}

            if n_clusters > 1:
                # 🔥 FIX: Sample data for clustering metrics to prevent OOM on large datasets
                # silhouette_score is O(n²) — 541K rows will exhaust 16 GB RAM
                _CLUSTER_EVAL_CAP = 10_000
                n_total = len(X_eval)
                if n_total > _CLUSTER_EVAL_CAP:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(n_total, size=_CLUSTER_EVAL_CAP, replace=False)
                    X_sample = X_eval.iloc[idx] if hasattr(X_eval, 'iloc') else X_eval[idx]
                    preds_sample = preds[idx]
                    print(f"  📊 Sampling {_CLUSTER_EVAL_CAP:,}/{n_total:,} rows for clustering metrics")
                else:
                    X_sample = X_eval
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
    
    bundle_path = find_model_bundle(champion_path)
    if bundle_path is not None:
        size = bundle_path.stat().st_size
        if size > 0:
            validation["valid"] = True
            validation["model_path"] = str(bundle_path)
            validation["size_bytes"] = size
        else:
            validation["errors"].append(
                "Champion ModelBundle exists but is empty"
            )
    else:
        validation["errors"].append(
            f"Exact champion ModelBundle not found in {champion_out}"
        )
    
    return validation


def enforce_input_validation(args, validation: dict) -> None:
    """Write diagnostics and fail the component when required inputs are invalid."""
    if validation["valid"]:
        return
    ensure_output_dirs(args)
    error_report = {
        "status": "failed",
        "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"],
    }
    with open(args.report_out, "w") as report_file:
        json.dump(error_report, report_file, indent=2, cls=NumpyEncoder)
    raise RuntimeError("Input validation failed; final evaluation was not performed")


def assert_matching_row_identity(
    candidate: pd.Series,
    canonical: pd.Series,
) -> None:
    """Require exact canonical holdout row identity and ordering."""
    if candidate.isna().any():
        raise ValueError("Phase B evaluation row identities are incomplete")
    candidate_text = candidate.astype(str).reset_index(drop=True)
    canonical_text = canonical.astype(str).reset_index(drop=True)
    pd.testing.assert_series_equal(
        candidate_text,
        canonical_text,
        check_names=False,
    )


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load and validate the exact immutable Stage 2 split contract."""
    return SplitManifest.from_json(Path(path).read_text(encoding="utf-8"))


def validate_locked_holdout_identity(
    split_manifest: SplitManifest,
    holdout_row_ids: pd.Series,
    *,
    task_type: str,
) -> dict:
    """Fail closed unless the S10 holdout is the Stage 2 locked partition."""
    if split_manifest.task_type != task_type:
        raise ValueError(
            "Stage 2 SplitManifest task type does not match final evaluation: "
            f"{split_manifest.task_type!r} != {task_type!r}"
        )
    if holdout_row_ids.isna().any():
        raise ValueError("Declared holdout row identities must be complete")
    canonical_ids = holdout_row_ids.astype(str).reset_index(drop=True)
    if not canonical_ids.is_unique:
        raise ValueError("Declared holdout row identities must be unique")
    if len(canonical_ids) != split_manifest.test_count:
        raise ValueError(
            "Declared holdout row count does not match Stage 2 SplitManifest: "
            f"{len(canonical_ids)} != {split_manifest.test_count}"
        )
    actual_test_ids_hash = canonical_hash(canonical_ids.tolist())
    if actual_test_ids_hash != split_manifest.test_ids_hash:
        raise ValueError(
            "Declared holdout identity hash does not match Stage 2 SplitManifest"
        )
    return {
        "split_id": split_manifest.split_id,
        "data_version": split_manifest.data_version,
        "test_count": split_manifest.test_count,
        "test_ids_hash": split_manifest.test_ids_hash,
    }


def bind_selected_candidate_to_source_bundle(
    selection_evidence: dict,
    source_bundle: ModelBundle,
) -> dict:
    """Prove that the selected manifest names the exact loaded source bundle."""
    selected_candidate_id = str(
        selection_evidence.get("candidate_id") or ""
    ).strip()
    if not selected_candidate_id:
        raise RuntimeError("Selected candidate manifest lacks candidate_id")
    source_candidate_id = str(source_bundle.candidate_id).strip()
    if selected_candidate_id != source_candidate_id:
        raise RuntimeError(
            "Selected candidate manifest candidate_id does not match source "
            f"ModelBundle: {selected_candidate_id!r} != {source_candidate_id!r}"
        )

    source_bundle_id = source_bundle.bundle_id
    declared_bundle_id = str(
        selection_evidence.get("model_bundle_id") or ""
    ).strip()
    if declared_bundle_id and declared_bundle_id != source_bundle_id:
        raise RuntimeError(
            "Selected candidate manifest bundle identity does not match source "
            f"ModelBundle: {declared_bundle_id!r} != {source_bundle_id!r}"
        )
    return {
        "candidate_id": source_candidate_id,
        "source_bundle_id": source_bundle_id,
        "declared_bundle_id": declared_bundle_id or None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--holdout_in", required=True)
    parser.add_argument("--split_manifest_in", required=True)
    parser.add_argument("--execution_manifest_in", required=True)
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
        enforce_input_validation(args, validation)
    
    print("  ✅ Input validation passed")

    # 2. ENSURE OUTPUT DIRECTORIES EXIST
    ensure_output_dirs(args)

    # 3. LOAD CONFIG AND DATASET
    print("\n📦 Loading config and dataset...")
    import yaml
    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    from orchestration.config_compiler import compile_config

    cfg = compile_config(raw_cfg, source_name=Path(args.config).name)
    execution_manifest = validate_execution_manifest_binding(
        args.execution_manifest_in,
        cfg,
    )
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")  # 🔥 CRITICAL FIX

    df = pd.read_csv(args.dataset_in, sep=delimiter)
    df_holdout = pd.read_csv(args.holdout_in, sep=delimiter)
    split_manifest = load_split_manifest(args.split_manifest_in)
    if df.empty or df_holdout.empty:
        raise ValueError("Training and holdout datasets must both contain rows")
    if ROW_ID_COLUMN not in df_holdout.columns:
        raise ValueError("Declared holdout is missing canonical row identities")
    canonical_identity_raw = df_holdout.pop(ROW_ID_COLUMN)
    if canonical_identity_raw.isna().any():
        raise ValueError("Declared holdout row identities must be complete")
    canonical_holdout_identity = canonical_identity_raw.astype(str)
    if not canonical_holdout_identity.is_unique:
        raise ValueError("Declared holdout row identities must be complete and unique")
    if list(df.columns) != list(df_holdout.columns):
        raise ValueError("Training and holdout dataset schemas must match exactly")
    split_binding = validate_locked_holdout_identity(
        split_manifest,
        canonical_holdout_identity,
        task_type=task_type,
    )
    _holdout_source = "stage2_split_manifest_bound_component_input"
    print(
        f"   ✅ Loaded declared holdout input ({len(df_holdout):,} rows) "
        "for honest evaluation"
    )
    
    print(f"  Task: {task_type}")
    print(f"  Target: {target_col}")
    if task_type == "clustering":
        print(f"  ⚠️ Clustering task - label encoders not applicable")
    
    _current_execution_id = execution_manifest.execution_id
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Training/reference data is safe for diagnostics; the locked test is not.
    print("\n📊 Generating final comprehensive EDA report...")
    generate_comprehensive_sweetviz_report(df, outputs_dir, target_col, task_type)
    
    # Split dataset into features and target
    if task_type != "clustering":
        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' required but not found in dataset")

        X_test = df_holdout.drop(columns=[target_col])
        y_test = df_holdout[target_col]
        X_train = df.drop(columns=[target_col])
        y_train = df[target_col]
        print(f"  ✅ Test set (declared holdout): {len(X_test):,} samples — honest")
    else:
        # Clustering: no target column
        X_test = df_holdout.copy()
        y_test = None
        X_train = df.copy()
        y_train = None
        print(f"  Clustering mode: using holdout dataset ({len(X_test):,} samples)")
    
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
    # Freeze the champion from CV/validation evidence before touching the
    # locked test. Only the frozen bundle receives one final evaluation.
    selection_evidence = {
        "baseline": load_selection_evidence(args.baseline_model, "baseline"),
        "phaseb": load_selection_evidence(args.phaseb_model, "phaseb"),
        "phasec": load_selection_evidence(args.phasec_model, "phasec"),
    }
    _execution_id, exact_candidate_run_ids = validate_selection_lineage(
        selection_evidence,
        expected_execution_id=_current_execution_id,
    )
    validate_selection_comparability(
        selection_evidence,
        expected_metric=get_primary_metric(task_type),
        expected_folds=(
            1 if task_type == "clustering" else int(cfg["split"]["cv_folds"])
        ),
        minimum_candidates=int(
            cfg["metrics"]["min_comparable_candidates"]
        ),
    )
    all_stage_metrics = collect_all_stage_metrics(
        exact_candidate_run_ids,
        _execution_id,
    )
    all_metrics_path = outputs_dir / "all_stages_metrics.json"
    with open(all_metrics_path, "w") as f:
        json.dump(all_stage_metrics, f, indent=2, cls=NumpyEncoder)
    print(f"  ✅ Exact stage metrics saved: {all_metrics_path}")
    generate_performance_visualizations(
        all_stage_metrics,
        outputs_dir,
        task_type,
    )
    best_key, best_val = select_champion_from_selection_evidence(
        selection_evidence
    )
    champion_valid = best_key is not None and best_val is not None
    if not champion_valid:
        raise RuntimeError("No candidate has complete CV/validation evidence")
    model_by_phase = {
        "baseline": (baseline, baseline_encoder, baseline_threshold),
        "phaseb": (phaseb, phaseb_encoder, phaseb_threshold),
        "phasec": (phasec, phasec_encoder, phasec_threshold),
    }
    champion_model, champion_encoder, champion_threshold = model_by_phase[best_key]
    if champion_model is None:
        raise RuntimeError(f"Frozen {best_key} champion model could not be loaded")
    source_bundle_binding = bind_selected_candidate_to_source_bundle(
        selection_evidence[best_key],
        champion_model,
    )

    print(
        f"\n🔒 Champion frozen before locked test: {best_key} "
        f"(selection_score={best_val:.4f})"
    )
    # Every phase exposes the same raw-input ModelBundle contract.  The locked
    # test therefore enters the frozen champion exactly once, here.
    locked_test_metrics = eval_model(
        champion_model,
        X_test,
        y_test,
        task_type,
        champion_encoder,
        threshold=champion_threshold,
    )
    if locked_test_metrics is None:
        raise RuntimeError("Frozen champion failed locked-test evaluation")
    mb = selection_evidence["baseline"]
    pb = selection_evidence["phaseb"]
    pc = selection_evidence["phasec"]
    {"baseline": mb, "phaseb": pb, "phasec": pc}[best_key][
        "locked_test_metrics"
    ] = locked_test_metrics
    
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

    # 6. CHAMPION IS ALREADY FROZEN FROM SELECTION EVIDENCE
    print("\n🏆 Using frozen champion model...")
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
    print(f"  ✅ Champion: {best_key} (selection_score={best_val:.4f})")

    # T17: Quality gate — configurable thresholds via cfg["registry"]["min_quality"].
    # By default warn-only (block_on_quality_fail=False); set to true in config
    # to hard-block registration when champion score is below threshold.
    DEFAULT_QUALITY_THRESHOLDS = {
        "classification": 0.50,  # balanced_accuracy — above random-guess baseline
        "regression": 0.0,       # R² — above mean-predictor baseline
        "clustering": 0.0,       # silhouette — above zero (any cluster separation)
    }
    _registry_cfg = cfg.get("registry", {}) if isinstance(cfg, dict) else {}
    _min_q = _registry_cfg.get("min_quality", {}) or {}
    quality_threshold = float(_min_q.get(task_type, DEFAULT_QUALITY_THRESHOLDS.get(task_type, 0.0)))
    block_on_quality_fail = bool(_registry_cfg.get("block_on_quality_fail", False))
    final_test_score = locked_test_metrics.get(primary_metric)
    _finite_quality_value = (
        final_test_score is not None
        and np.isfinite(float(final_test_score))
    )
    quality_gate_passed = bool(
        champion_valid
        and _finite_quality_value
        and float(final_test_score) >= quality_threshold
    )
    quality_decision = (
        "pass"
        if quality_gate_passed
        else "block"
        if (not champion_valid or block_on_quality_fail)
        else "warn"
    )
    if quality_decision != "pass":
        if not champion_valid:
            print(f"  ❌ T17 QUALITY GATE FAIL: no valid champion (registration will be blocked)")
        else:
            print(f"  ⚠️ T17 QUALITY {quality_decision.upper()}: locked-test score {final_test_score} < threshold "
                  f"{quality_threshold} for {task_type}")
    else:
        print(f"  ✅ T17 QUALITY GATE PASS: locked-test score {final_test_score:.4f} ≥ threshold {quality_threshold}")

    # ── 6b. SHAP EXPLAINABILITY ─────────────────────────────────
    shap_summary = None
    try:
        import shap
        champion_model_obj = {"baseline": baseline, "phaseb": phaseb, "phasec": phasec}.get(best_key)
        if champion_model_obj is not None and task_type != "clustering":
            print("\n🔍 Computing SHAP feature importance for champion...")
            # Use a sample to keep computation tractable
            _shap_sample = X_train.sample(
                n=min(200, len(X_train)),
                random_state=42,
            )
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
        "schema_version": 2,
        "task": task_type,
        "target_column": target_col,
        "test_samples": len(X_test),
        "champion_valid": champion_valid,
        "quality_gate_passed": quality_gate_passed,
        # Bound to the evaluated bundle below before the component succeeds.
        "quality_decision": None,
        "quality_threshold": quality_threshold,
        "block_on_quality_fail": block_on_quality_fail,
        "holdout_source": _holdout_source,
        "split_manifest": split_binding,
        "source_model_bundle": source_bundle_binding,
        "execution_manifest": execution_manifest.to_dict(),
        "baseline_metrics": mb,
        "phaseb_metrics": pb,
        "phasec_metrics": pc,
        "selection_evidence": selection_evidence,
        "selection": {
            "key": best_key,
            "score": float(best_val) if np.isfinite(best_val) else None,
            "source": "cross_validation_or_validation",
            "locked_test_used_for_selection": False,
            "candidate_id": source_bundle_binding["candidate_id"],
            "source_bundle_id": source_bundle_binding["source_bundle_id"],
        },
        "final_test": {
            "evaluated_once": True,
            "candidate_phase": best_key,
            "candidate_id": source_bundle_binding["candidate_id"],
            "source_bundle_id": source_bundle_binding["source_bundle_id"],
            "metrics": locked_test_metrics,
            "row_count": len(X_test),
        },
        "lineage": {
            "execution_id": _execution_id,
            "config_hash": execution_manifest.config_hash,
            "code_sha": execution_manifest.code_sha,
            "split_id": split_binding["split_id"],
            "source_bundle_id": source_bundle_binding["source_bundle_id"],
            "source_candidate_id": source_bundle_binding["candidate_id"],
            "parent_run_id": selection_evidence[best_key]
            .get("lineage", {})
            .get("parent_run_id"),
            "final_evaluation_run_id": os.getenv("AZUREML_RUN_ID"),
            "candidate_lineage": selection_evidence[best_key].get("lineage", {}),
        },
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
    
    # 4. Copy only the exact bundle artifacts for operator visibility.
    print("\n📦 Copying exact champion bundle to outputs/ folder...")
    chosen_path = candidates.get(best_key, (None, None))[1]
    source_bundle_path = (
        find_model_bundle(chosen_path) if chosen_path else None
    )
    if source_bundle_path is None:
        raise RuntimeError("Frozen champion has no unique exact ModelBundle")
    source_manifest_path = (
        source_bundle_path.parent / "model_bundle_manifest.json"
    )
    shutil.copy2(
        source_bundle_path,
        outputs_dir / f"{best_key}_model_bundle.pkl",
    )
    if source_manifest_path.is_file():
        shutil.copy2(
            source_manifest_path,
            outputs_dir / f"{best_key}_model_bundle_manifest.json",
        )

    # 8. Materialize one evaluated ModelBundle as the official output.
    out_dir = Path(args.champion_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not champion_valid:
        (out_dir / ".no_champion").write_text(json.dumps({
            "reason": "all_models_failed", "best_val": str(best_val), "task_type": task_type,
        }))
        print(f"\n⚠️  No valid champion — wrote .no_champion sentinel to {out_dir}")

    if champion_valid:
        source_bundle = load_model_bundle(source_bundle_path)
        reloaded_source_binding = bind_selected_candidate_to_source_bundle(
            selection_evidence[best_key],
            source_bundle,
        )
        if reloaded_source_binding != source_bundle_binding:
            raise RuntimeError(
                "Source ModelBundle identity changed during final evaluation"
            )
        evaluated_lineage = {
            **dict(source_bundle.lineage),
            **report["lineage"],
        }
        evaluated_bundle = ModelBundle(
            estimator=source_bundle.estimator,
            preprocessing=source_bundle.preprocessing,
            target_decoder=source_bundle.target_decoder,
            task_type=task_type,
            candidate_id=source_bundle.candidate_id,
            input_schema=source_bundle.input_schema,
            recipe=source_bundle.recipe,
            selection_metrics={
                "metric_name": primary_metric,
                "selection_score": best_val,
                "source": "cross_validation_or_validation",
            },
            final_test_metrics=locked_test_metrics,
            environment={
                **dict(source_bundle.environment),
                "code_sha": execution_manifest.code_sha,
                "environment_hashes": dict(
                    execution_manifest.environment_hashes
                ),
            },
            lineage=evaluated_lineage,
            dependencies=source_bundle.dependencies,
            threshold=champion_threshold,
            labels=source_bundle.labels,
            signature=source_bundle.signature
            or {
                "inputs": list(capture_input_schema(X_train)["column_order"]),
                "outputs": ["prediction"],
            },
            input_example=(
                source_bundle.input_example
                if source_bundle.input_example is not None
                else X_train.head(1).to_dict(orient="records")
            ),
        )
        report["model_bundle"] = save_model_bundle(
            evaluated_bundle,
            out_dir,
        )
        quality_decision_record = make_quality_decision(
            champion_valid=champion_valid,
            observed_value=final_test_score,
            threshold=quality_threshold,
            metric_name=primary_metric,
            block_on_quality_fail=block_on_quality_fail,
            candidate_id=evaluated_bundle.candidate_id,
            evaluated_bundle_hash=report["model_bundle"]["bundle_id"],
        )
        report["quality_decision"] = quality_decision_record

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
        tags={
            "pipeline": "v3_mlops",
            "phase": "evaluation",
            "step": "s10",
            "execution_id": str(_execution_id or ""),
        }
    )
    log_metrics_to_mlflow(report, task_type, logger)

    # T9: Log champion validity for Azure ML Studio dashboard filtering
    try:
        logger.log_metric("champion_valid", 1.0 if champion_valid else 0.0)
        logger.log_metric("quality_gate_passed", 1.0 if quality_gate_passed else 0.0)
    except Exception as _cv_err:
        logging.getLogger(__name__).debug("champion_valid metric log failed: %s", _cv_err)

    # 11. SAVE FINAL REPORT WITH OUTPUT VALIDATION
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    shutil.copy2(report_path, outputs_dir / "final_evaluation_report.json")
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

    # ── Agent 2: STRICT QUALITY GATE — block downstream registration ──
    if not quality_gate_passed and block_on_quality_fail:
        print("\n" + "=" * 80)
        print("🚫 BLOCKING: quality gate failed and registry.block_on_quality_fail=True")
        print(f"   task_type={task_type}  threshold={quality_threshold}  champion_score={best_val}")
        print("   Set registry.block_on_quality_fail=False in config to override.")
        print("=" * 80)
        sys.exit(2)


if __name__ == "__main__":
    main()
