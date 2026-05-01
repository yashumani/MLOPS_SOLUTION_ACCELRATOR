import argparse
import json
import logging
import time as _time_mod
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Add src to path for imports (single canonical insertion at front of sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.azureml_metrics_logger import (
    create_metrics_logger, ensure_outputs_dir, safe_write_json, safe_copy, safe_dict_get
)
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_candidate_artifacts,
    write_stage_table,
)
from utils.model_universe import get_model_list
from utils.model_universe import build_flaml_breakdown, write_model_breakdown
import os


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

    print("=" * 80)
    print("STEP S05b: BASELINE — FLAML")
    print("=" * 80)

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")  # 🔥 CRITICAL FIX
    time_budget = cfg.get("phases", {}).get("phase_a_baseline", {}).get("flaml_config", {}).get("time_budget", 120)

    # 🔥 Agent 1: prefer sibling train.csv (holdout-leak-safe).
    _ds_path = Path(args.dataset_in)
    _train_sibling = _ds_path.parent / "train.csv"
    if _train_sibling.exists() and _train_sibling.stat().st_size > 0:
        df = pd.read_csv(_train_sibling, sep=delimiter)
        print(f"   ✅ Loaded sibling train.csv ({len(df):,} rows) — holdout isolated")
    else:
        df = pd.read_csv(args.dataset_in, sep=delimiter)
        print(f"   ⚠️ No sibling train.csv — using combined dataset ({len(df):,} rows)")

    outputs_dir = ensure_outputs_dir()

    metrics = {}
    manifest = {"engine": "flaml", "models": []}

    # Skip FLAML for clustering (not supported by FLAML AutoML)
    if task_type == "clustering":
        print("FLAML AutoML does not support clustering task type; skipping s05b_baseline_flaml")
        metrics["status"] = "skipped"
        metrics["reason"] = "FLAML does not support clustering"
        manifest["status"] = "skipped"
        manifest["reason"] = "FLAML does not support clustering; use PyCaret clustering only"

        # Write valid outputs
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f)
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f)

        # Create empty model folder
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".skipped").write_text("FLAML does not support clustering")

        safe_write_json(outputs_dir / "stage5b_skipped.json",
                        {"status": "skipped", "reason": "FLAML does not support clustering"})

        # Create logger for this early exit and log skip
        _logger = create_metrics_logger(
            run_name="s05b_baseline_flaml",
            tags={"pipeline": "v3_mlops", "phase": "baseline", "step": "s05b"},
        )
        try:
            _logger.log_param("task_type", task_type)
            _logger.log_param("flaml_status", "skipped")
        except Exception as e:
            logger.warning("MLflow skip-path log_param failed: %s", e)
        _logger.end_run()
        return  # Exit early for clustering

    # Classification/Regression: validate target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in dataset for FLAML training")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Hold out 20% for honest evaluation (avoid train=eval leak)
    stratify_col = y if task_type == "classification" else None
    _seed = int(cfg.get("random_seed", 42))
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=_seed, stratify=stratify_col
    )

    try:
        from flaml import AutoML
        from sklearn.metrics import (
            accuracy_score, r2_score, roc_auc_score,
            f1_score, precision_score, recall_score,
            average_precision_score, confusion_matrix,
            cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
        )

        automl = AutoML()
        task = "classification" if task_type == "classification" else "regression"

        # Use AUC for classification to avoid majority-class-only models
        if task == "classification":
            metric = "roc_auc"
        else:
            metric = "r2"

        print(f"\nFLAML {task} baseline  metric={metric}  budget={time_budget}s")
        _flaml_models = get_model_list(task_type, "flaml")
        automl.fit(
            X_train=X_train, y_train=y_train,
            task=task, metric=metric,
            time_budget=time_budget,
            log_file_name="flaml.log",
            estimator_list=_flaml_models if _flaml_models else None,
        )

        best_estimator = automl.best_estimator
        best_config = automl.best_config

        # Compute explicit evaluation metrics on HOLDOUT (not training data)
        preds = automl.predict(X_holdout)
        if task == "classification":
            acc = float(accuracy_score(y_holdout, preds))
            bal_acc = float(balanced_accuracy_score(y_holdout, preds))
            f1 = float(f1_score(y_holdout, preds, average="weighted", zero_division=0))
            prec = float(precision_score(y_holdout, preds, average="weighted", zero_division=0))
            rec = float(recall_score(y_holdout, preds, average="weighted", zero_division=0))
            cm = confusion_matrix(y_holdout, preds).tolist()
            metrics["accuracy"] = round(acc, 4)
            metrics["balanced_accuracy"] = round(bal_acc, 4)
            metrics["f1"] = round(f1, 4)
            metrics["precision"] = round(prec, 4)
            metrics["recall"] = round(rec, 4)
            metrics["kappa"] = round(float(cohen_kappa_score(y_holdout, preds)), 4)
            metrics["mcc"] = round(float(matthews_corrcoef(y_holdout, preds)), 4)
            metrics["confusion_matrix"] = cm

            # Try to get AUC / PR-AUC from predict_proba
            try:
                proba = automl.predict_proba(X_holdout)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    auc = float(roc_auc_score(y_holdout, proba[:, 1]))
                    pr_auc = float(average_precision_score(y_holdout, proba[:, 1]))
                else:
                    auc = float(roc_auc_score(y_holdout, proba, multi_class="ovr", average="weighted"))
                    pr_auc = None
                metrics["auc"] = round(auc, 4)
                if pr_auc is not None:
                    metrics["pr_auc"] = round(pr_auc, 4)
            except Exception as e:
                logger.warning("FLAML predict_proba/AUC computation failed: %s", e)

            score = bal_acc
            metric_name = "balanced_accuracy"
            print(f"   AUC={metrics.get('auc','N/A')} | F1={f1:.4f} | Acc={acc:.4f} | BalancedAcc={bal_acc:.4f}")
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            score = float(r2_score(y_holdout, preds))
            mse_val = float(mean_squared_error(y_holdout, preds))
            rmse_val = float(np.sqrt(mse_val))
            mae_val = float(mean_absolute_error(y_holdout, preds))
            metric_name = "r2"
            metrics["r2"] = round(score, 4)
            metrics["rmse"] = round(rmse_val, 4)
            metrics["mae"] = round(mae_val, 4)
            metrics["mse"] = round(mse_val, 4)
            print(f"   R2={score:.4f} | RMSE={rmse_val:.4f} | MAE={mae_val:.4f}")

        metrics["best_estimator"] = best_estimator
        metrics["best_config"] = best_config
        metrics["best_metric"] = round(score, 4) if score is not None else None
        metrics["metric_name"] = metric_name
        manifest["best_estimator"] = best_estimator
        manifest["best_config"] = best_config
        manifest["best_metric"] = round(score, 4) if score is not None else None
        manifest["metric_name"] = metric_name
        # Store primary metric separately for normalized cross-engine comparison
        if task == "classification" and metrics.get("balanced_accuracy") is not None:
            manifest["accuracy"] = round(acc, 4)
            manifest["balanced_accuracy"] = metrics["balanced_accuracy"]

        # Save model
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"

        try:
            import joblib
            joblib.dump(automl.model, model_path)
            print(f"Model saved: {model_path}")
        except Exception as save_err:
            print(f"Model save failed: {save_err}")
            (model_dir / ".error").write_text(str(save_err))

        # ----- OUTPUTS (Source of Truth for Azure ML Studio) -----
        try:
            # Build safe iterations table from config_history
            iterations_data = []
            if hasattr(automl, "config_history") and automl.config_history:
                for iter_key, cfg_val in automl.config_history.items():
                    if isinstance(cfg_val, dict):
                        learner = cfg_val.get("learner", cfg_val.get("ml", {}).get("learner", "unknown") if isinstance(cfg_val.get("ml"), dict) else "unknown")
                    else:
                        learner = str(cfg_val)
                    iterations_data.append({
                        "iteration": int(iter_key) if isinstance(iter_key, (int, float)) else 0,
                        "learner": learner,
                        "config": str(cfg_val)[:500],
                    })
            if not iterations_data:
                iterations_data.append({
                    "iteration": 0,
                    "learner": best_estimator if isinstance(best_estimator, str) else "unknown",
                    "config": str(best_config)[:500],
                })

            iterations_df = pd.DataFrame(iterations_data)
            iterations_df.to_csv(outputs_dir / "stage5b_baseline_flaml_iterations.csv", index=False)

            if model_path.exists():
                safe_copy(model_path, outputs_dir / "stage5b_best_model.pkl")

            summary = {
                "stage": "5b_baseline_flaml", "engine": "flaml",
                "task_type": task_type,
                "metric_optimized": metric,
                "total_iterations": len(iterations_data),
                "best_estimator": best_estimator,
                "best_metric_value": metrics.get("best_metric"),
                "dataset_shape": list(df.shape),
            }
            if task_type == "classification":
                summary["auc"] = metrics.get("auc")
                summary["f1"] = metrics.get("f1")
                summary["recall"] = metrics.get("recall")
            safe_write_json(outputs_dir / "stage5b_baseline_flaml_summary.json", summary)

            # ── Deep Model-Level Breakdown (shared utility) ─────────────────
            try:
                breakdown_df = build_flaml_breakdown(
                    automl,
                    step_name="s05b",
                    phase="baseline",
                    variant="baseline",
                    task_type=task_type,
                    metric_name=metrics.get("metric_name", metric),
                    best_metric_value=metrics.get("best_metric"),
                )
                write_model_breakdown(breakdown_df, str(outputs_dir), "s05b_model_breakdown.csv")
                print(f"📊 Model breakdown: {len(breakdown_df)} entries → outputs/s05b_model_breakdown.csv")
            except Exception as _bd_err:
                print(f"⚠️  Model breakdown write failed (non-fatal): {_bd_err}")

            print(f"\noutputs/ artefacts written")
        except Exception as out_err:
            print(f"outputs/ creation error (non-fatal): {out_err}")
            safe_write_json(outputs_dir / "stage5b_error.json", {"error": str(out_err)})

    except Exception as e:
        print(f"\nFLAML training error: {e}")
        metrics["error"] = str(e)
        manifest["error"] = str(e)
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".error").write_text(str(e))
        safe_write_json(outputs_dir / "stage5b_error.json", {"error": str(e)})

    # Write required pipeline outputs
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, default=str)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, default=str)

    # MLflow logging (metrics/params only)
    logger = create_metrics_logger(
        run_name="s05b_baseline_flaml",
        tags={"pipeline": "v3_mlops", "phase": "baseline", "step": "s05b"},
    )
    try:
        logger.log_param("task_type", task_type)
        logger.log_param("target_column", str(target_col))
        logger.log_param("metric_optimized", metric if "error" not in metrics else "N/A")
        logger.log_metric("dataset_rows", int(df.shape[0]))
        logger.log_metric("dataset_cols", int(df.shape[1] - 1))

        if "best_estimator" in metrics:
            logger.log_param("best_estimator", str(metrics["best_estimator"])[:500])

        if task_type == "classification":
            for k in ("auc", "pr_auc", "f1", "precision", "recall", "accuracy"):
                if k in metrics and metrics[k] is not None:
                    logger.log_metric(k, float(metrics[k]))
        elif task_type == "regression":
            for k in ("r2", "rmse", "mae", "mse"):
                if k in metrics and metrics[k] is not None:
                    logger.log_metric(k, float(metrics[k]))

        if "best_metric" in metrics and metrics["best_metric"] is not None:
            logger.log_metric("best_metric", float(metrics["best_metric"]))
        if "error" in metrics:
            logger.log_param("flaml_error", str(metrics["error"])[:500])
    except Exception as mlflow_err:
        print(f"MLflow logging failed (non-fatal): {mlflow_err}")

    logger.end_run()

    # ── Candidate Ledger ──────────────────────────────────────────────────
    try:
        _elapsed = _time_mod.time() - _t0
        _status = "skipped" if metrics.get("status") == "skipped" else ("failed" if "error" in metrics else "ok")
        _norm = normalize_metrics(task_type, metrics)
        row = make_row(
            stage="baseline", step_name="s05b", engine="flaml",
            candidate_id=f"flaml_{metrics.get('best_estimator', 'none')}",
            task_type=task_type,
            dataset_id=Path(args.dataset_in).stem,
            status=_status,
            failure_reason=metrics.get("error") or metrics.get("reason", ""),
            compute_time_sec=round(_elapsed, 2),
            source_path="src/steps/stage5_flaml_train.py",
            recipe_name="baseline",
            is_stage_best=True,
            **_norm,
        )
        write_candidate_artifacts(
            "outputs", row,
            inputs_dict={"engine": "flaml", "task_type": task_type, "time_budget": time_budget},
            metrics_dict=metrics,
        )
        write_stage_table(
            [row],
            csv_path="outputs/s05b_candidates.csv",
            parquet_path="outputs/s05b_candidates.parquet",
        )
    except Exception as _ledger_err:
        print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")

    print("\n" + "=" * 80)
    print("✅ STEP S05b COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
