import argparse
import json
import logging
import time as _time_mod
from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
from utils.common_evaluator import EvaluationSpec, evaluate_candidate
from utils.phasea_model_bundle import (
    PhaseABundleError,
    build_phasea_evaluation_pipeline,
    fit_discovery_preprocessor,
    fit_save_phasea_bundle,
    load_baseline_recipe,
    load_phasea_split_manifest,
    phasea_candidate_id,
    split_raw_training_frame,
)
import os


def main():
    _t0 = _time_mod.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("STEP S05b: BASELINE — FLAML")
    print("=" * 80)

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")  # 🔥 CRITICAL FIX
    _baseline_cfg = cfg.get("phases", {}).get("phase_a_baseline", {}) or {}
    time_budget = min(
        600.0,
        float(
            _baseline_cfg.get("flaml_config", {}).get(
                "time_budget",
                _baseline_cfg.get("candidate_engine_timeout_seconds", 600),
            )
        ),
    )
    _cv_folds = int(_baseline_cfg.get("cv_folds", 5))
    _execution_id = (
        os.getenv("MLOPS_EXECUTION_ID")
        or os.getenv("AZUREML_ROOT_RUN_ID")
        or os.getenv("AZUREML_RUN_ID")
    )
    _seed = int(cfg.get("random_seed", 42))

    _raw_df = pd.read_csv(args.dataset_in, sep=delimiter)
    _split_manifest = load_phasea_split_manifest(
        args.split_manifest,
        task_type=task_type,
        train_count=len(_raw_df),
        random_seed=_seed,
    )
    _phasea_recipe = load_baseline_recipe(
        Path(__file__).resolve().parents[2],
        task_type,
    )
    _raw_features, _raw_target = split_raw_training_frame(
        _raw_df,
        task_type=task_type,
        target_column=target_col,
    )
    print(
        f"   Loaded Stage 2 raw train ({len(_raw_df):,} rows); "
        f"split_id={_split_manifest.split_id[:12]}"
    )

    outputs_dir = ensure_outputs_dir()
    model_dir = Path(args.model_out).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    manifest = {
        "schema_version": 2,
        "engine": "flaml",
        "task_type": task_type,
        "models": [],
        "split_id": _split_manifest.split_id,
        "raw_input_bundle_eligible": False,
        "status": "pending",
    }

    # Skip FLAML for clustering (not supported by FLAML AutoML)
    if task_type == "clustering":
        print("FLAML AutoML does not support clustering task type; skipping s05b_baseline_flaml")
        metrics["status"] = "skipped"
        metrics["reason"] = "FLAML does not support clustering"
        manifest["status"] = "skipped_unsupported"
        manifest["reason"] = "FLAML does not support clustering; use PyCaret clustering only"
        manifest["eligibility_reason"] = "clustering_is_pycaret_only"

        # Write valid outputs
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f)
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f)

        # Create empty model folder
        (model_dir / ".skipped").write_text("FLAML does not support clustering")

        safe_write_json(outputs_dir / "stage5b_skipped.json",
                        {"status": "skipped", "reason": "FLAML does not support clustering"})

        # Create logger for this early exit and log skip
        _logger = create_metrics_logger(
            run_name="s05b_baseline_flaml",
            tags={
                "pipeline": "v3_mlops",
                "phase": "baseline",
                "step": "s05b",
                "execution_id": str(_execution_id or ""),
            },
        )
        try:
            _logger.log_param("task_type", task_type)
            _logger.log_param("flaml_status", "skipped")
        except Exception as e:
            logger.warning("MLflow skip-path log_param failed: %s", e)
        _logger.end_run()
        return  # Exit early for clustering

    X, _ = fit_discovery_preprocessor(
        _raw_features,
        _raw_target,
        recipe=_phasea_recipe,
        random_seed=_seed,
    )
    y = _raw_target

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
            X_train=X, y_train=y,
            task=task, metric=metric,
            time_budget=time_budget,
            log_file_name="flaml.log",
            estimator_list=_flaml_models if _flaml_models else None,
            seed=_seed,
        )

        best_estimator = automl.best_estimator
        best_config = automl.best_config

        # Refit/evaluate the selected estimator on the same deterministic folds
        # used for PyCaret.  No holdout or locked-test rows enter selection.
        _candidate_model = getattr(automl.model, "estimator", automl.model)
        _candidate_id = phasea_candidate_id(
            "flaml",
            _candidate_model,
            _phasea_recipe,
        )
        _evaluation_model = build_phasea_evaluation_pipeline(
            _candidate_model,
            recipe=_phasea_recipe,
            task_type=task_type,
            random_seed=_seed,
        )
        _evidence = evaluate_candidate(
            _evaluation_model,
            _raw_features,
            _raw_target,
            candidate_id=_candidate_id,
            engine="flaml",
            spec=EvaluationSpec(
                task_type=task_type,
                seed=_seed,
                folds=_cv_folds,
                timeout_seconds=time_budget,
                execution_id=_execution_id,
            ),
            mlflow_parent_run_id=_execution_id,
            mlflow_child_run_id=os.getenv("AZUREML_RUN_ID"),
        )
        metrics["evaluation"] = _evidence.to_dict()
        metrics["status"] = _evidence.status
        metrics["selection_score"] = _evidence.selection_score
        metrics["metric_name"] = _evidence.primary_metric
        metrics.update(_evidence.metrics)
        score = _evidence.selection_score
        metric_name = _evidence.primary_metric

        metrics["best_estimator"] = best_estimator
        metrics["best_config"] = best_config
        metrics["best_metric"] = round(score, 4) if score is not None else None
        metrics["metric_name"] = metric_name
        manifest["best_estimator"] = best_estimator
        manifest["best_config"] = best_config
        manifest["best_metric"] = round(score, 4) if score is not None else None
        manifest["metric_name"] = metric_name
        manifest.update(
            {
                "schema_version": 2,
                "candidate_id": _candidate_id,
                "status": _evidence.status,
                "evaluation": _evidence.to_dict(),
                "selection_score": _evidence.selection_score,
                "execution_id": _execution_id,
                "mlflow_parent_run_id": _execution_id,
                "mlflow_child_run_id": os.getenv("AZUREML_RUN_ID"),
            }
        )
        if _evidence.status == "success" and _evidence.selection_score is not None:
            try:
                _bundle_artifact = fit_save_phasea_bundle(
                    _candidate_model,
                    _raw_features,
                    _raw_target,
                    task_type=task_type,
                    engine="flaml",
                    candidate_id=_candidate_id,
                    recipe=_phasea_recipe,
                    evidence=_evidence,
                    split_manifest=_split_manifest,
                    output_dir=model_dir,
                    random_seed=_seed,
                    execution_id=_execution_id,
                    mlflow_parent_run_id=_execution_id,
                    mlflow_child_run_id=os.getenv("AZUREML_RUN_ID"),
                )
                manifest.update(
                    {
                        "status": "success",
                        "raw_input_bundle_eligible": True,
                        "eligibility_reason": "verified_raw_input_bundle",
                        "model_bundle": dict(_bundle_artifact.manifest),
                        "bundle_smoke_test": dict(_bundle_artifact.smoke_test),
                    }
                )
            except PhaseABundleError as bundle_error:
                manifest.update(
                    {
                        "status": "ineligible_raw_bundle",
                        "raw_input_bundle_eligible": False,
                        "eligibility_reason": str(bundle_error),
                    }
                )
        else:
            manifest.update(
                {
                    "status": _evidence.status,
                    "raw_input_bundle_eligible": False,
                    "eligibility_reason": (
                        _evidence.failure_reason
                        or "common_evaluator_evidence_not_selectable"
                    ),
                }
            )
        # Store primary metric separately for normalized cross-engine comparison
        if task == "classification" and metrics.get("balanced_accuracy") is not None:
            manifest["accuracy"] = metrics.get("accuracy")
            manifest["balanced_accuracy"] = metrics["balanced_accuracy"]

        # Save model
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
                "dataset_shape": list(_raw_df.shape),
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
        if manifest.get("raw_input_bundle_eligible") is not True:
            manifest["status"] = "failure"
            manifest["raw_input_bundle_eligible"] = False
            manifest["eligibility_reason"] = f"phasea_training_failed: {e}"
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
        tags={
            "pipeline": "v3_mlops",
            "phase": "baseline",
            "step": "s05b",
            "execution_id": str(_execution_id or ""),
        },
    )
    try:
        logger.log_param("task_type", task_type)
        logger.log_param("target_column", str(target_col))
        logger.log_param("metric_optimized", metric if "error" not in metrics else "N/A")
        logger.log_metric("dataset_rows", int(_raw_df.shape[0]))
        logger.log_metric("dataset_cols", int(_raw_features.shape[1]))

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
        _status = (
            "skipped"
            if metrics.get("status") == "skipped"
            else (
                "ok"
                if manifest.get("raw_input_bundle_eligible") is True
                else "failed"
            )
        )
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
