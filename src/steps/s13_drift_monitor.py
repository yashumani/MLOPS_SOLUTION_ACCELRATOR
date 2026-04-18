"""
Stage 13: Drift Monitor — Baseline Profile & Retraining Cadence

Computes per-feature baseline statistics and PSI self-check on the
processed training dataset. Outputs a drift baseline artifact and
a retraining cadence recommendation.

This is a TRAINING-TIME step, not a production monitor. It:
  1. Generates per-feature statistics (baseline for future production monitoring)
  2. Validates the PSI detector via train/test self-check (expects PSI ≈ 0)
  3. Recommends retraining cadence based on dataset characteristics

Inputs:
  - config_name: Config YAML filename (for task_type, target_column, dataset_name)
  - dataset_in: Processed training CSV from s4 (feature-engineered)
  - final_report: Champion manifest JSON from s10 (champion info + metrics)
  - registry_info: Model registration JSON from s12 (registration status)

Outputs:
  - drift_report: JSON with PSI scores, baseline stats, cadence recommendation
  - drift_baseline: Folder with per-feature baseline statistics (for production use)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.drift_detector import (
    compute_feature_psi,
    compute_baseline_statistics,
    compute_stability_score,
    determine_retraining_cadence,
    classify_feature_drift,
    compute_feature_volatility,
    PSI_GREEN,
    PSI_YELLOW,
)

# Evidently for baseline-comparison drift detection
try:
    # evidently 0.4.x (compatible with fsspec<=2023.10.0 / azureml-fsspec)
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import (
        ColumnDriftMetric,
        DatasetDriftMetric,
    )
    HAS_EVIDENTLY = True
except ImportError:
    try:
        # evidently >=0.5 moved old API under .legacy
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.metrics import (
            ColumnDriftMetric,
            DatasetDriftMetric,
        )
        HAS_EVIDENTLY = True
    except ImportError:
        HAS_EVIDENTLY = False

# Optional: metrics logger (T11 pattern)
try:
    from utils.azureml_metrics_logger import create_metrics_logger
except ImportError:
    create_metrics_logger = None

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
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
        return super().default(obj)


def _safe_disable_autolog():
    """Disable autolog + convert azureml:// tracking URI to https://."""
    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if uri.startswith("azureml://"):
        mlflow.set_tracking_uri(uri.replace("azureml://", "https://"))
        logger.info("🔗 MLflow tracking URI converted to HTTPS")


def _load_config(config_name: str) -> dict:
    """Load config YAML from configs/ directory."""
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    config_path = config_dir / config_name
    if not config_path.exists():
        # Try without .yml extension
        config_path = config_dir / f"{config_name}.yml"
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _load_json_safe(path: str, label: str) -> dict:
    """Load JSON file with error handling."""
    try:
        p = Path(path)
        if p.is_dir():
            # Check for common JSON files in directory
            for name in ["registry_info.json", "final_report.json", "report.json"]:
                candidate = p / name
                if candidate.exists():
                    with open(candidate) as f:
                        return json.load(f)
            # Try first .json file
            json_files = list(p.glob("*.json"))
            if json_files:
                with open(json_files[0]) as f:
                    return json.load(f)
            logger.warning(f"{label}: directory has no JSON files: {path}")
            return {}
        if not p.exists():
            logger.warning(f"{label}: file not found: {path}")
            return {}
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"{label}: failed to load {path}: {e}")
        return {}


def _detect_delimiter(path: str) -> str:
    """Detect CSV delimiter by inspection."""
    try:
        with open(path, "r") as f:
            first_line = f.readline()
        if "\t" in first_line and "," not in first_line:
            return "\t"
        if ";" in first_line and first_line.count(";") > first_line.count(","):
            return ";"
    except Exception:
        pass
    return ","


def _load_previous_baseline(baseline_path: str) -> dict:
    """Load baseline artifacts from a previous pipeline run.

    Returns dict with 'reference_data' (DataFrame) and 'metadata' (dict),
    or empty dict if baseline not found/invalid.
    """
    bp = Path(baseline_path)
    if not bp.exists() or not bp.is_dir():
        return {}

    ref_dist_file = bp / "reference_distributions.json"
    baseline_file = bp / "feature_baseline.json"
    ref_csv = bp / "reference_data.csv"

    if not baseline_file.exists():
        logger.warning(f"Baseline metadata not found: {baseline_file}")
        return {}

    try:
        with open(baseline_file) as f:
            metadata = json.load(f)

        # If a reference CSV was saved, load it directly
        reference_data = None
        if ref_csv.exists():
            reference_data = pd.read_csv(ref_csv)

        ref_distributions = {}
        if ref_dist_file.exists():
            with open(ref_dist_file) as f:
                ref_distributions = json.load(f)

        return {
            "metadata": metadata,
            "reference_data": reference_data,
            "reference_distributions": ref_distributions,
        }
    except Exception as e:
        logger.warning(f"Failed to load previous baseline: {e}")
        return {}


def _run_evidently_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                         target_column: str = None) -> dict:
    """Run Evidently drift detection comparing reference vs current data.

    Returns dict with drift results or empty dict if Evidently unavailable.
    """
    if not HAS_EVIDENTLY:
        logger.warning("Evidently not installed — skipping visual drift report")
        return {}

    try:
        # Align columns
        common_cols = sorted(set(reference_df.columns) & set(current_df.columns))
        ref = reference_df[common_cols].copy()
        cur = current_df[common_cols].copy()

        # Feature drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        result = report.as_dict()

        drift_info = {}
        for metric_result in result.get("metrics", []):
            metric_id = metric_result.get("metric", "")
            mr = metric_result.get("result", {})

            if metric_id == "DatasetDriftMetric":
                drift_info["dataset_drift"] = mr.get("dataset_drift", False)
                drift_info["share_of_drifted_columns"] = mr.get(
                    "share_of_drifted_columns", 0.0
                )
                drift_info["number_of_drifted_columns"] = mr.get(
                    "number_of_drifted_columns", 0
                )
                drift_info["number_of_columns"] = mr.get("number_of_columns", 0)

            if metric_id == "DataDriftTable":
                drifted_cols = []
                col_drifts = mr.get("drift_by_columns", {})
                for col, info in col_drifts.items():
                    if info.get("drift_detected", False):
                        drifted_cols.append({
                            "column": col,
                            "drift_score": round(info.get("drift_score", 0), 6),
                            "stattest_name": info.get("stattest_name", ""),
                        })
                drift_info["drifted_columns"] = drifted_cols

        return drift_info

    except Exception as e:
        logger.warning(f"Evidently drift detection failed: {e}")
        return {}


def _run_concept_drift(final_report: dict, baseline_metadata: dict,
                       task_type: str) -> dict:
    """Check concept drift by comparing current vs baseline metrics."""
    concept = {"detected": False, "metric_name": "", "current": None,
               "baseline": None, "drop": 0.0}

    # Get current metric from final_report
    selection = final_report.get("selection", {})
    current_score = selection.get("score")
    if current_score is None:
        current_score = final_report.get("primary_metric_value")

    # Get baseline metric from previous run
    prev_score = baseline_metadata.get("champion_metric")

    if current_score is None or prev_score is None:
        return concept

    metric_name = "balanced_accuracy" if task_type == "classification" else "r2_score"
    drop = float(prev_score) - float(current_score)
    threshold = 0.05

    concept["metric_name"] = metric_name
    concept["current"] = round(float(current_score), 4)
    concept["baseline"] = round(float(prev_score), 4)
    concept["drop"] = round(drop, 4)
    concept["detected"] = drop > threshold

    return concept


def run_drift_monitor(args):
    """Main drift monitoring logic."""
    start_time = time.time()
    _safe_disable_autolog()

    # ── Load configuration ──────────────────────────────────────
    config = _load_config(args.config_name)
    task_type = config.get("task_type", "classification")
    dataset_cfg = config.get("dataset", {})
    target_column = dataset_cfg.get("target_column", None)
    dataset_name = dataset_cfg.get("name", args.config_name.replace(".yml", ""))
    execution_id = f"s13_{dataset_name}_{int(time.time())}"

    logger.info(f"═══ s13 Drift Monitor: {dataset_name} ({task_type}) ═══")
    logger.info(f"  Execution ID: {execution_id}")

    # ── Load upstream artifacts ─────────────────────────────────
    final_report = _load_json_safe(args.final_report, "final_report")
    registry_info = (
        _load_json_safe(args.registry_info, "registry_info")
        if args.registry_info else {}
    )

    # ── Load dataset ────────────────────────────────────────────
    delimiter = _detect_delimiter(args.dataset_in)
    df = pd.read_csv(args.dataset_in, sep=delimiter)
    logger.info(f"  Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Separate features from target ───────────────────────────
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    elif task_type == "clustering":
        X = df.copy()
        y = None
        target_column = None
    else:
        # Fallback: assume last column is target
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logger.warning(f"  Target column not in config; using last column: {target_column}")

    feature_cols = list(X.columns)
    n_rows, n_features = X.shape
    logger.info(f"  Features: {n_features}, Target: {target_column or 'none (clustering)'}")

    # ── Train/Test split (matches s10: 80/20, seed=42) ──────────
    from sklearn.model_selection import train_test_split

    if task_type == "classification" and y is not None:
        stratify_param = y
    else:
        stratify_param = None

    if y is not None:
        X_ref, X_test, y_ref, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
    else:
        # Clustering: no target
        X_ref, X_test = train_test_split(X, test_size=0.2, random_state=42)

    logger.info(f"  Reference: {X_ref.shape[0]} rows, Test: {X_test.shape[0]} rows")

    # ── Compute per-feature PSI (self-check) ────────────────────
    logger.info("  Computing per-feature PSI (self-check)...")
    psi_scores = compute_feature_psi(X_ref, X_test, n_bins=10)

    overall_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
    max_psi = float(np.max(list(psi_scores.values()))) if psi_scores else 0.0
    max_psi_feature = max(psi_scores, key=psi_scores.get) if psi_scores else "none"

    drifted_features = [
        {"feature": f, "psi": round(p, 6), "severity": classify_feature_drift(p)}
        for f, p in psi_scores.items()
        if p >= PSI_GREEN
    ]

    # Self-check should show minimal drift (same dataset split)
    self_check_status = "PASS" if overall_psi < PSI_GREEN else "WARN"
    if self_check_status == "WARN":
        logger.warning(f"  ⚠️ Self-check PSI elevated: {overall_psi:.4f} (expected < {PSI_GREEN})")
    else:
        logger.info(f"  ✅ Self-check PSI: {overall_psi:.6f} (PASS)")

    logger.info(f"  Max feature PSI: {max_psi_feature} = {max_psi:.6f}")

    # ── Compute baseline statistics ─────────────────────────────
    logger.info("  Computing baseline statistics...")
    baseline_stats = compute_baseline_statistics(X, feature_cols)

    # ── Compute feature volatility ──────────────────────────────
    feature_vol = compute_feature_volatility(X)

    # ── Compute class imbalance ratio ───────────────────────────
    imbalance_ratio = None
    if task_type == "classification" and y is not None:
        vc = y.value_counts()
        if len(vc) >= 2:
            imbalance_ratio = float(vc.min() / vc.max())

    # ── Compute stability score & cadence ───────────────────────
    stability_score, stability_components = compute_stability_score(
        psi_scores=psi_scores,
        n_rows=n_rows,
        n_features=n_features,
        imbalance_ratio=imbalance_ratio,
        feature_volatility=feature_vol,
    )
    cadence_name, cadence_days, cadence_rationale = determine_retraining_cadence(stability_score)

    logger.info(f"  Stability score: {stability_score}/100")
    logger.info(f"  Recommended cadence: {cadence_name} (every {cadence_days} days)")

    # ── Extract champion info ───────────────────────────────────
    champion_info = {}
    # From final_report
    champion_info["algorithm"] = (
        final_report.get("algorithm")
        or final_report.get("selection", {}).get("algorithm")
        or "unknown"
    )
    champion_info["primary_metric"] = final_report.get("selection", {}).get("score")
    champion_info["phase"] = final_report.get("selection", {}).get("key", "unknown")

    # From registry_info
    champion_info["registered"] = not registry_info.get("registration_skipped", False)
    champion_info["model_name"] = registry_info.get("model_name", "unknown")
    champion_info["model_version"] = registry_info.get("version", "0")

    # ── Comparison drift (vs previous baseline) ─────────────────
    comparison_drift = {"available": False}
    if getattr(args, "baseline_in", None):
        logger.info("  Loading previous baseline for comparison drift...")
        prev = _load_previous_baseline(args.baseline_in)
        if prev and prev.get("metadata"):
            comparison_drift["available"] = True
            prev_meta = prev["metadata"]
            logger.info(
                f"  Previous baseline: {prev_meta.get('dataset_name', '?')} "
                f"({prev_meta.get('n_rows', '?')} rows)"
            )

            # Evidently feature drift (reference vs current)
            ref_data = prev.get("reference_data")
            if ref_data is not None and not ref_data.empty:
                logger.info("  Running Evidently comparison drift...")
                evidently_result = _run_evidently_drift(ref_data, X, target_column)
                comparison_drift["evidently"] = evidently_result
                if evidently_result.get("dataset_drift"):
                    logger.warning("  ⚠️ Evidently: DATASET DRIFT DETECTED")
                elif evidently_result:
                    logger.info("  ✅ Evidently: No dataset drift")
            else:
                logger.info("  No reference CSV in baseline — skipping Evidently comparison")

            # Concept drift (metric comparison)
            concept_result = _run_concept_drift(final_report, prev_meta, task_type)
            comparison_drift["concept_drift"] = concept_result
            if concept_result.get("detected"):
                logger.warning(
                    f"  ⚠️ Concept drift: {concept_result['metric_name']} "
                    f"dropped from {concept_result['baseline']} to {concept_result['current']}"
                )
        else:
            logger.info("  Previous baseline empty or invalid — skipping comparison")

    # ── Warnings ────────────────────────────────────────────────
    warnings = []
    if self_check_status == "WARN":
        warnings.append(
            f"Self-check PSI ({overall_psi:.4f}) exceeds green threshold ({PSI_GREEN}). "
            "This may indicate high variance features or small dataset."
        )
    if n_rows < 500:
        warnings.append(f"Small dataset ({n_rows} rows). PSI may be unreliable.")
    if n_features > n_rows * 0.5:
        warnings.append(
            f"High dimensionality ({n_features} features vs {n_rows} rows). "
            "Consider feature reduction for more stable drift monitoring."
        )
    if comparison_drift.get("concept_drift", {}).get("detected"):
        cd = comparison_drift["concept_drift"]
        warnings.append(
            f"Concept drift detected: {cd['metric_name']} dropped "
            f"from {cd['baseline']} to {cd['current']} (Δ={cd['drop']})."
        )
    if comparison_drift.get("evidently", {}).get("dataset_drift"):
        n_drifted_ev = comparison_drift["evidently"].get("number_of_drifted_columns", 0)
        warnings.append(
            f"Dataset drift detected by Evidently: {n_drifted_ev} columns drifted."
        )

    # ── Build report ────────────────────────────────────────────
    drift_report = {
        "execution_id": execution_id,
        "config_name": args.config_name,
        "task_type": task_type,
        "dataset_name": dataset_name,
        "n_rows": n_rows,
        "n_features": n_features,
        "target_column": target_column,
        "detector": "psi",
        "self_check": {
            "method": "train_test_split_80_20_seed_42",
            "overall_psi": round(overall_psi, 6),
            "max_feature_psi": round(max_psi, 6),
            "max_feature_name": max_psi_feature,
            "drifted_features": drifted_features,
            "n_drifted": len(drifted_features),
            "status": self_check_status,
        },
        "feature_psi_scores": {f: round(p, 6) for f, p in sorted(psi_scores.items(), key=lambda x: -x[1])},
        "stability_assessment": {
            "stability_score": stability_score,
            "components": stability_components,
            "recommended_cadence": cadence_name,
            "recommended_days": cadence_days,
            "rationale": cadence_rationale,
        },
        "champion_info": champion_info,
        "comparison_drift": comparison_drift,
        "warnings": warnings,
        "runtime_seconds": round(time.time() - start_time, 2),
    }

    # ── Write drift_report output ───────────────────────────────
    report_path = Path(args.drift_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2, cls=NumpyEncoder)
    logger.info(f"  📄 Drift report → {report_path}")

    # ── Write drift_baseline output ─────────────────────────────
    baseline_dir = Path(args.drift_baseline)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Save feature baseline statistics
    baseline_artifact = {
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target_column": target_column,
        "n_rows": n_rows,
        "n_features": n_features,
        "feature_statistics": baseline_stats,
        "champion_metric": champion_info.get("primary_metric"),
        "champion_algorithm": champion_info.get("algorithm"),
        "psi_bins": 10,
        "reference_split": "train_80pct_seed_42",
    }
    with open(baseline_dir / "feature_baseline.json", "w") as f:
        json.dump(baseline_artifact, f, indent=2, cls=NumpyEncoder)

    # Save compact reference distributions for each numeric feature
    # (bin edges + counts for PSI recomputation in production)
    reference_distributions = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X_ref[col]):
            vals = X_ref[col].dropna().values
            if len(vals) > 0 and vals.min() != vals.max():
                counts, bin_edges = np.histogram(vals, bins=10)
                reference_distributions[col] = {
                    "type": "numeric",
                    "bin_edges": [float(b) for b in bin_edges],
                    "counts": [int(c) for c in counts],
                    "total": int(len(vals)),
                }
        else:
            vc = X_ref[col].value_counts()
            reference_distributions[col] = {
                "type": "categorical",
                "category_counts": {str(k): int(v) for k, v in vc.items()},
                "total": int(len(X_ref[col].dropna())),
            }

    with open(baseline_dir / "reference_distributions.json", "w") as f:
        json.dump(reference_distributions, f, indent=2, cls=NumpyEncoder)

    # Save reference data CSV for Evidently comparison in next run
    X_ref.to_csv(baseline_dir / "reference_data.csv", index=False)

    logger.info(f"  📁 Baseline artifacts → {baseline_dir}")

    # ── Log metrics to MLflow ───────────────────────────────────
    try:
        mlflow.log_metric("overall_psi", overall_psi)
        mlflow.log_metric("max_feature_psi", max_psi)
        mlflow.log_metric("stability_score", stability_score)
        mlflow.log_metric("recommended_days", cadence_days)
        mlflow.log_metric("n_features_monitored", n_features)
        mlflow.log_metric("n_drifted_features", len(drifted_features))
        mlflow.log_param("detector", "psi")
        mlflow.log_param("cadence", cadence_name)
        mlflow.log_param("self_check_status", self_check_status)
        mlflow.log_param("dataset_name", dataset_name)
        if comparison_drift.get("available"):
            ev = comparison_drift.get("evidently", {})
            if ev:
                mlflow.log_metric("evidently_dataset_drift",
                                  1 if ev.get("dataset_drift") else 0)
                mlflow.log_metric("evidently_drifted_share",
                                  ev.get("share_of_drifted_columns", 0.0))
            cd = comparison_drift.get("concept_drift", {})
            if cd.get("detected") is not None:
                mlflow.log_metric("concept_drift_detected",
                                  1 if cd.get("detected") else 0)
                if cd.get("drop") is not None:
                    mlflow.log_metric("concept_drift_drop", cd["drop"])
            mlflow.log_param("baseline_comparison", "true")
        else:
            mlflow.log_param("baseline_comparison", "false")
        logger.info("  📊 Metrics logged to MLflow")
    except Exception as e:
        logger.warning(f"  MLflow logging failed (non-fatal): {e}")

    # ── Summary ─────────────────────────────────────────────────
    logger.info("═══ s13 Drift Monitor Summary ═══")
    logger.info(f"  Dataset: {dataset_name} ({task_type})")
    logger.info(f"  Self-check: {self_check_status} (PSI={overall_psi:.6f})")
    logger.info(f"  Stability: {stability_score}/100 → {cadence_name} ({cadence_days}d)")
    logger.info(f"  Champion: {champion_info.get('algorithm', '?')} (registered={champion_info.get('registered', '?')})")
    if comparison_drift.get("available"):
        ev = comparison_drift.get("evidently", {})
        cd = comparison_drift.get("concept_drift", {})
        logger.info(f"  Comparison drift: Evidently={ev.get('dataset_drift', 'N/A')}, "
                     f"Concept={cd.get('detected', 'N/A')}")
    else:
        logger.info("  Comparison drift: no previous baseline")
    if warnings:
        for w in warnings:
            logger.warning(f"  ⚠️ {w}")
    logger.info(f"  Runtime: {time.time() - start_time:.1f}s")

    return 0


def main():
    parser = argparse.ArgumentParser(description="s13 Drift Monitor — Baseline & Cadence")
    parser.add_argument("--config_name", type=str, required=True, help="Config YAML filename")
    parser.add_argument("--dataset_in", type=str, required=True, help="Processed dataset CSV (from s4)")
    parser.add_argument("--final_report", type=str, required=True, help="Final evaluation report JSON (from s10)")
    parser.add_argument("--registry_info", type=str, default=None, help="Model registry info JSON (from s12, optional)")
    parser.add_argument("--baseline_in", type=str, default=None, help="Previous drift baseline folder (optional)")
    parser.add_argument("--drift_report", type=str, required=True, help="Output: drift report JSON")
    parser.add_argument("--drift_baseline", type=str, required=True, help="Output: drift baseline folder")
    args = parser.parse_args()

    sys.exit(run_drift_monitor(args))


if __name__ == "__main__":
    main()
