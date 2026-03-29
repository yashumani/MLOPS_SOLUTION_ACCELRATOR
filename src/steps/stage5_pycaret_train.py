import argparse
import json
import time as _time_mod
from pathlib import Path
import sys
import shutil

import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.azureml_metrics_logger import (
    create_metrics_logger, ensure_outputs_dir, safe_write_json, safe_copy
)
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_candidate_artifacts,
    write_stage_table,
)
from utils.model_universe import get_model_list, build_coverage_report, write_model_coverage
from utils.model_universe import build_pycaret_breakdown, write_model_breakdown

import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_primary_metric(task_type: str) -> str:
    """Return the PyCaret sort metric name per task type."""
    return {"classification": "AUC", "regression": "R2", "clustering": "Silhouette"}.get(task_type, "AUC")


def _optimal_threshold_f1(y_true, y_proba, pos_label=1) -> tuple:
    """Find the probability threshold that maximises F1 for *pos_label*."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, pos_label=pos_label, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return round(best_t, 2), round(best_f1, 4)


def main():
    _t0 = _time_mod.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    # 🔥 FIX: Convert azureml:// to https:// to avoid model registry errors
    _mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if _mlflow_uri.startswith("azureml://"):
        import mlflow
        mlflow.set_tracking_uri(_mlflow_uri.replace("azureml://", "https://"))
        print("🔗 MLflow tracking URI converted to HTTPS")

    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")

    df = pd.read_csv(args.dataset_in, sep=delimiter)

    # Validate target column (required for classification/regression)
    if task_type != "clustering":
        if not target_col:
            raise ValueError(f"Target column required for {task_type} task but not specified in config")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing in dataset for PyCaret training")

    # Prepare output directories
    outputs_dir = ensure_outputs_dir()
    model_dir = Path(args.model_out).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train via PyCaret
    metrics = {}
    manifest = {"engine": "pycaret", "models": []}

    try:
        if task_type == "classification":
            from pycaret.classification import setup, compare_models, pull, save_model, predict_model
            from sklearn.metrics import (
                classification_report, confusion_matrix,
                roc_auc_score, average_precision_score,
            )

            print(f"\nPyCaret Classification Baseline - IMBALANCE-AWARE")
            print(f"   Dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

            # Detect imbalance
            target_counts = df[target_col].value_counts()
            imbalance_ratio = float(target_counts.min() / target_counts.max())
            print(f"   Target distribution: {target_counts.to_dict()}")
            print(f"   Imbalance ratio: {imbalance_ratio:.3f}")

            # 🔥 FIX (A3): Double-SMOTE guard
            # If an upstream recipe (Stage 3) already applied SMOTE, the target distribution
            # will be near-balanced (ratio ~1.0) and this threshold naturally prevents double-SMOTE.
            # Explicit guard: if ratio >= 0.8 AND we'd normally expect imbalance, SMOTE was already applied.
            use_fix_imbalance = imbalance_ratio < 0.35
            sort_metric = "AUC"

            if use_fix_imbalance:
                if imbalance_ratio >= 0.8:
                    # Data looks balanced despite originating from an imbalanced dataset —
                    # upstream recipe likely applied SMOTE already. Skip PyCaret's SMOTE.
                    print(f"   ⚠️  Ratio {imbalance_ratio:.3f} ≥ 0.8 — likely upstream SMOTE already applied, skipping fix_imbalance")
                    use_fix_imbalance = False
                else:
                    print(f"   Imbalance detected (ratio {imbalance_ratio:.3f} < 0.35) -> fix_imbalance=True (SMOTE)")
            else:
                print(f"   Balanced enough (ratio {imbalance_ratio:.3f}) -> no SMOTE")

            # Adaptive fold count: 3 for large datasets, 5 for small
            _n_folds = 3 if df.shape[0] > 50_000 else 5
            print(f"   CV folds: {_n_folds} (adaptive; rows={df.shape[0]:,})")

            _include = get_model_list("classification", "pycaret") or None
            print(f"   Models : {len(_include) if _include else 'ALL'} from MODEL_UNIVERSE")

            setup(
                data=df,
                target=target_col,
                session_id=42,
                verbose=False,
                log_experiment=False,
                fix_imbalance=use_fix_imbalance,
                fold_strategy="stratifiedkfold",
                fold=_n_folds,
            )

            best = compare_models(sort=sort_metric, n_select=1,
                                  include=_include, turbo=True)
            leaderboard = pull()

            # ── Extract ALL CV metrics from PyCaret leaderboard ──
            _PYCARET_CV_MAP = {
                "Accuracy": "cv_accuracy", "AUC": "cv_auc",
                "Recall": "cv_recall", "Prec.": "cv_precision",
                "F1": "cv_f1", "Kappa": "cv_kappa", "MCC": "cv_mcc",
            }
            for _pc_col, _cv_key in _PYCARET_CV_MAP.items():
                if _pc_col in leaderboard.columns:
                    metrics[_cv_key] = round(float(leaderboard[_pc_col].iloc[0]), 4)
            print(f"   CV metrics extracted: {[k for k in metrics if k.startswith('cv_')]}")

            # --- Threshold tuning for imbalanced classification ---
            try:
                preds_df = predict_model(best, data=df)
                score_col = "prediction_score" if "prediction_score" in preds_df.columns else "Score"
                y_true = df[target_col]
                y_proba = preds_df[score_col] if score_col in preds_df.columns else None

                if y_proba is not None:
                    pos_label = target_counts.idxmin()
                    auc = roc_auc_score(y_true, y_proba)
                    pr_auc = average_precision_score(y_true, y_proba, pos_label=pos_label)
                    opt_thresh, opt_f1 = _optimal_threshold_f1(y_true, y_proba, pos_label)

                    y_pred_tuned = (y_proba >= opt_thresh).astype(int)
                    cr = classification_report(y_true, y_pred_tuned, output_dict=True, zero_division=0)
                    cm = confusion_matrix(y_true, y_pred_tuned)

                    metrics["auc"] = round(auc, 4)
                    metrics["pr_auc"] = round(pr_auc, 4)
                    metrics["optimal_threshold"] = opt_thresh
                    metrics["f1_at_optimal_threshold"] = opt_f1
                    pos_report = cr.get(str(pos_label), cr.get("weighted avg", {}))
                    metrics["recall"] = round(float(pos_report.get("recall", 0)), 4)
                    metrics["precision"] = round(float(pos_report.get("precision", 0)), 4)
                    metrics["confusion_matrix"] = cm.tolist()

                    print(f"   AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
                    print(f"   Optimal threshold: {opt_thresh} -> F1={opt_f1:.4f}")
                    print(f"   Confusion matrix (threshold={opt_thresh}):\n{cm}")
                else:
                    print("   prediction_score column not found; skipping threshold tuning")
            except Exception as thr_err:
                print(f"   Threshold tuning failed (non-fatal): {thr_err}")

            metrics["imbalance_ratio"] = round(imbalance_ratio, 4)
            metrics["fix_imbalance_applied"] = use_fix_imbalance
            metrics["sort_metric"] = sort_metric

        elif task_type == "regression":
            from pycaret.regression import setup, compare_models, pull, save_model

            _n_folds = 3 if df.shape[0] > 50_000 else 5
            _include = get_model_list("regression", "pycaret") or None
            print(f"\nPyCaret Regression Baseline")
            print(f"   Dataset : {df.shape[0]:,} rows × {df.shape[1]} cols")
            print(f"   Models  : {len(_include) if _include else 'ALL'} from MODEL_UNIVERSE")
            print(f"   CV folds: {_n_folds} (adaptive)")
            setup(data=df, target=target_col, session_id=42, verbose=False,
                  log_experiment=False, fold=_n_folds)
            best = compare_models(sort="R2", n_select=1,
                                  include=_include, turbo=True)
            leaderboard = pull()

            metrics["r2"] = float(leaderboard.iloc[0]["R2"]) if "R2" in leaderboard.columns else None
            metrics["rmse"] = float(leaderboard.iloc[0]["RMSE"]) if "RMSE" in leaderboard.columns else None
            metrics["mae"] = float(leaderboard.iloc[0]["MAE"]) if "MAE" in leaderboard.columns else None

        elif task_type == "clustering":
            from pycaret.clustering import setup, create_model, pull, save_model
            from sklearn.metrics import silhouette_score, davies_bouldin_score

            print(f"\nPyCaret Clustering Baseline")
            # Cast numeric columns to float64 to prevent dtype mismatch errors
            _numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[_numeric_cols] = df[_numeric_cols].astype(np.float64)
            print(f"   Cast {len(_numeric_cols)} numeric cols to float64")
            
            setup(data=df, session_id=42, verbose=False, log_experiment=False)
            best = create_model("kmeans")
            leaderboard = pull()

            predictions = best.predict(df)
            sil = silhouette_score(df.select_dtypes(include=[np.number]).astype(np.float64), predictions)
            db = davies_bouldin_score(df.select_dtypes(include=[np.number]).astype(np.float64), predictions)
            metrics["silhouette_score"] = round(sil, 4)
            metrics["davies_bouldin_score"] = round(db, 4)
            print(f"   silhouette={sil:.4f}, davies_bouldin={db:.4f}")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # Common metadata
        metrics["best_model"] = str(best)
        metrics["leaderboard"] = leaderboard.to_dict() if hasattr(leaderboard, "to_dict") else {}
        manifest["best_model_name"] = str(best)
        manifest["leaderboard_columns"] = leaderboard.columns.tolist() if hasattr(leaderboard, "columns") else []
        manifest["rows"] = int(leaderboard.shape[0]) if hasattr(leaderboard, "shape") else 0

        # Save model
        model_path = model_dir / "model"
        save_model(best, str(model_path))
        print(f"\nModel saved: {model_path}")

        # Save threshold info alongside model for downstream steps (s10)
        if task_type == "classification" and metrics.get("optimal_threshold") is not None:
            threshold_info = {
                "optimal_threshold": metrics["optimal_threshold"],
                "f1_at_threshold": metrics.get("f1_at_optimal_threshold"),
                "source": "s5a_baseline_pycaret",
            }
            safe_write_json(model_dir / "threshold_info.json", threshold_info)
            print(f"   💾 Threshold info saved: {model_dir / 'threshold_info.json'}")

        # ----- OUTPUTS (Source of Truth for Azure ML Studio) -----
        leaderboard.to_csv(outputs_dir / "stage5a_baseline_pycaret_leaderboard.csv", index=False)
        safe_write_json(outputs_dir / "stage5a_baseline_pycaret_top10.json",
                        leaderboard.head(10).to_dict(orient="records"))
        if (model_dir / "model.pkl").exists():
            safe_copy(model_dir / "model.pkl", outputs_dir / "stage5a_best_model.pkl")

        # 📊 Deep model-level breakdown
        primary_metric = get_primary_metric(task_type)
        try:
            breakdown_df = build_pycaret_breakdown(
                leaderboard,
                step_name="s05a",
                phase="baseline",
                variant="baseline",
                task_type=task_type,
                metric_col=primary_metric,
            )
            write_model_breakdown(breakdown_df, str(outputs_dir), "s05a_model_breakdown.csv")
            print(f"   📊 Model breakdown: {len(breakdown_df)} models written to s05a_model_breakdown.csv")
        except Exception as _bd_err:
            print(f"   ⚠️  Model breakdown failed (non-fatal): {_bd_err}")

        best_metric_val = None
        if task_type != "clustering" and primary_metric in leaderboard.columns:
            best_metric_val = float(leaderboard.iloc[0][primary_metric])

        # ── Store scoring data in manifest for cross-engine comparison (s5z) ──
        if task_type == "clustering":
            manifest["best_metric"] = metrics.get("silhouette_score")
            manifest["metric_name"] = "silhouette_score"
            manifest["silhouette_score"] = metrics.get("silhouette_score")
        else:
            manifest["best_metric"] = best_metric_val
            manifest["metric_name"] = primary_metric

        summary = {
            "stage": "5a_baseline_pycaret", "engine": "pycaret",
            "task_type": task_type,
            "models_compared": int(leaderboard.shape[0]),
            "best_model": str(best),
            "best_metric_value": best_metric_val,
            "sort_metric": primary_metric,
            "dataset_shape": list(df.shape),
        }
        if task_type == "classification":
            summary["imbalance_ratio"] = metrics.get("imbalance_ratio")
            summary["optimal_threshold"] = metrics.get("optimal_threshold")
            summary["auc"] = metrics.get("auc")
            summary["pr_auc"] = metrics.get("pr_auc")
            summary["f1_at_threshold"] = metrics.get("f1_at_optimal_threshold")
        safe_write_json(outputs_dir / "stage5a_baseline_pycaret_summary.json", summary)

        # ── Deep Model-Level Breakdown ────────────────────────────────────
        try:
            breakdown_rows = []
            for idx, row_data in leaderboard.iterrows():
                bd_row = {
                    "model_name": str(idx) if isinstance(idx, str) else str(row_data.get("Model", idx)),
                    "engine": "pycaret",
                    "variant": "baseline",
                    "stage": "s05a",
                    "task_type": task_type,
                }
                # Extract all metric columns from leaderboard
                for col in leaderboard.columns:
                    if col != "Model":
                        try:
                            bd_row[col.lower()] = float(row_data[col])
                        except (ValueError, TypeError):
                            bd_row[col.lower()] = str(row_data[col])
                breakdown_rows.append(bd_row)
            if breakdown_rows:
                bd_df = pd.DataFrame(breakdown_rows)
                bd_df.to_csv(outputs_dir / "model_breakdown_s05a.csv", index=False)
                print(f"📊 Model breakdown: {len(breakdown_rows)} models → outputs/model_breakdown_s05a.csv")
        except Exception as _bd_err:
            print(f"⚠️  Model breakdown write failed (non-fatal): {_bd_err}")

        print(f"\noutputs/ artefacts written")

    except Exception as e:
        print(f"\nPyCaret training error: {e}")
        metrics["error"] = str(e)
        manifest["error"] = str(e)
        (model_dir / ".error").write_text(str(e))
        safe_write_json(outputs_dir / "stage5a_error.json", {"error": str(e)})

    # Write required pipeline outputs
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, default=str)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, default=str)

    # MLflow logging (metrics/params only - artifacts already in outputs/)
    logger = create_metrics_logger(
        run_name="s05a_baseline_pycaret",
        tags={"pipeline": "v3_mlops", "phase": "baseline", "step": "s05a"},
    )
    try:
        logger.log_param("task_type", task_type)
        logger.log_param("best_model", str(metrics.get("best_model", "unknown"))[:500])
        logger.log_metric("dataset_rows", int(df.shape[0]))
        logger.log_metric("dataset_cols", int(df.shape[1]))

        if task_type == "classification":
            for k in ("auc", "pr_auc", "recall", "precision", "f1_at_optimal_threshold",
                       "optimal_threshold", "imbalance_ratio"):
                if k in metrics and metrics[k] is not None:
                    logger.log_metric(k, float(metrics[k]))
            logger.log_param("fix_imbalance", str(metrics.get("fix_imbalance_applied")))
            logger.log_param("sort_metric", metrics.get("sort_metric", "AUC"))
        elif task_type == "regression":
            for k in ("r2", "rmse", "mae"):
                if k in metrics and metrics[k] is not None:
                    logger.log_metric(k, float(metrics[k]))
        elif task_type == "clustering":
            for k in ("silhouette_score", "davies_bouldin_score"):
                if k in metrics:
                    logger.log_metric(k, float(metrics[k]))

        if "error" in metrics:
            logger.log_param("pycaret_error", metrics["error"][:500])
    except Exception as mlflow_err:
        print(f"MLflow logging failed (non-fatal): {mlflow_err}")

    logger.end_run()

    # ── Candidate Ledger ──────────────────────────────────────────────
    try:
        _elapsed = _time_mod.time() - _t0
        _norm = normalize_metrics(task_type, metrics)
        _status = "failed" if "error" in metrics else "ok"
        row = make_row(
            stage="baseline", step_name="s05a", engine="pycaret",
            candidate_id=f"pycaret_{metrics.get('best_model', 'unknown')}",
            task_type=task_type,
            dataset_id=Path(args.dataset_in).stem,
            status=_status,
            failure_reason=metrics.get("error", ""),
            compute_time_sec=round(_elapsed, 2),
            source_path="src/steps/stage5_pycaret_train.py",
            recipe_name="baseline",
            is_stage_best=True,
            **_norm,
        )
        write_candidate_artifacts(
            "outputs", row,
            inputs_dict={"engine": "pycaret", "task_type": task_type, "sort_metric": get_primary_metric(task_type)},
            metrics_dict=metrics,
        )
        write_stage_table(
            [row],
            csv_path="outputs/s05a_candidates.csv",
            parquet_path="outputs/s05a_candidates.parquet",
        )
    except Exception as _ledger_err:
        print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")

    print("\n" + "=" * 80)
    print("✅ STEP S05a COMPLETE")
    print("=" * 80)

    # ── Model Coverage Report ─────────────────────────────────────────
    try:
        coverage = build_coverage_report(task_type)
        write_model_coverage(coverage, "outputs")
    except Exception as _cov_err:
        print(f"⚠️  Model coverage report failed (non-fatal): {_cov_err}")


if __name__ == "__main__":
    main()
