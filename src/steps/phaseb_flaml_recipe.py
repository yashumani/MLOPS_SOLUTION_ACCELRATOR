import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mlflow

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--recipe_name", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).resolve().parents[2]
    recipe_path = root / "configs" / "recipes" / args.recipe_name
    if not recipe_path.is_file():
        raise FileNotFoundError(
            f"Recipe not found at {recipe_path}. Recipes must come from uploaded code (configs/recipes) and not workspaceblobstore."
        )
    with open(recipe_path, "r") as f:
        recipe = yaml.safe_load(f)
    import os
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")  # 🔥 CRITICAL FIX
    time_budget = cfg.get("phases", {}).get("phase_b_recipes", {}).get("flaml_config", {}).get("time_budget", 120)
    # Hard cap: Phase B variants should never exceed 10 min per engine
    _MAX_FLAML_VARIANT_BUDGET = 600
    if time_budget > _MAX_FLAML_VARIANT_BUDGET:
        print(f"⚠️  FLAML time_budget {time_budget}s exceeds Phase B cap; clamped to {_MAX_FLAML_VARIANT_BUDGET}s")
        time_budget = _MAX_FLAML_VARIANT_BUDGET
    
    # Validate recipe compatibility with task_type
    recipe_task = recipe.get("task_type")
    if recipe_task and recipe_task != task_type:
        print(f"⚠️ WARNING: Recipe task_type '{recipe_task}' does not match config task_type '{task_type}'")
        print(f"⚠️ Proceeding anyway, but results may be suboptimal")
    
    # Check for classification-specific techniques in non-classification tasks
    if task_type != "classification":
        imbalance_method = recipe.get("stage3_preprocessing", {}).get("imbalance_handling", {}).get("method")
        if imbalance_method and imbalance_method.lower() in ["smote", "adasyn", "smoteenn", "smotetomek"]:
            print(f"⚠️ WARNING: Recipe contains SMOTE/resampling ('{imbalance_method}') which is classification-only")
            print(f"⚠️ SKIPPING imbalance handling for {task_type} task")
            # Override recipe to skip SMOTE
            recipe.setdefault("stage3_preprocessing", {}).setdefault("imbalance_handling", {})["method"] = "none"

    df = pd.read_csv(args.dataset_in, sep=delimiter)  # 🔥 FIXED

    # Disable autologging (do NOT change tracking URI — breaks MLflow hierarchy)
    try:
        mlflow.sklearn.autolog(disable=True)
    except Exception:
        pass
    # Set local model registry to avoid unsupported azureml:// registry errors
    os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")

    metrics = {}
    manifest = {"engine": "flaml", "recipe": recipe.get("recipe_name"), "models": []}

    # Skip FLAML for clustering (not supported by FLAML AutoML)
    if task_type == "clustering":
        print("ℹ️ FLAML AutoML does not support clustering task type; skipping Phase B FLAML recipe")
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
        
        # Log to MLflow
        try:
            logger.log_param("task_type", task_type)
            logger.log_param("flaml_status", "skipped")
            logger.log_dict(metrics, "flaml_recipe_metrics.json")
            logger.log_dict(manifest, "flaml_recipe_manifest.json")
        except Exception as mlflow_err:
            print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")
        
        return  # Exit early for clustering

    # Classification/Regression: validate target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in dataset for FLAML recipe training")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    try:
        from flaml import AutoML
        from sklearn.metrics import accuracy_score, r2_score
        
        automl = AutoML()
        task = "classification" if task_type == "classification" else "regression"
        metric = "accuracy" if task == "classification" else "r2"
        automl.fit(X_train=X, y_train=y, task=task, metric=metric, time_budget=time_budget, log_file_name="flaml_recipe.log")
        best_estimator = automl.best_estimator
        best_config = automl.best_config
        best_metric = automl.best_loss if task == "classification" else automl.best_metric
        
        # Compute a higher-is-better score explicitly
        try:
            preds = automl.predict(X)
            if task == "classification":
                score = float(accuracy_score(y, preds))
            else:
                score = float(r2_score(y, preds))
        except Exception as metric_err:
            score = None
            print(f"⚠️  Metric computation failed: {metric_err}")
        
        metrics["best_estimator"] = str(best_estimator)
        metrics["best_config"] = str(best_config)
        metrics["best_metric"] = best_metric
        if score is not None:
            metrics["computed_score"] = score
        manifest["best_estimator"] = str(best_estimator)
        manifest["best_config"] = str(best_config)
        manifest["best_metric"] = best_metric
        if score is not None:
            manifest["computed_score"] = score
        
        # Save model
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        
        try:
            import joblib
            model_obj = automl.model.estimator
            joblib.dump(model_obj, model_path)
            print(f"✅ Model saved: {model_path} ({model_path.stat().st_size:,} bytes)")
        except Exception as save_err:
            print(f"❌ Model save failed: {save_err}")
            (model_dir / ".error").write_text(str(save_err))
    except Exception as e:
        metrics["error"] = str(e)
        manifest["error"] = str(e)
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".error").write_text(str(e))
    
    # 📊 CREATE OUTPUTS FOLDER FOR RECIPE-SPECIFIC TRACKING (ALWAYS RUN - OUTSIDE TRY BLOCK)
    import shutil
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 RECIPE-SPECIFIC LOGGING TO outputs/ FOLDER:")
    
    # Only create outputs if training succeeded (no error in metrics)
    if "error" not in metrics:
        try:
            recipe_name = recipe.get('recipe_name', 'unknown').replace('/', '_').replace(' ', '_')
            
            # 1. 📊 ENHANCED: Export FLAML iterations with ALL available metrics
            iterations_data = []
            per_estimator_stats = {}  # Track per-estimator performance
            
            print(f"\n   📊 Extracting FLAML iteration history...")
            
            if hasattr(automl, 'config_history') and automl.config_history:
                total_trials = len(automl.config_history)
                print(f"      📋 Total trials in config_history: {total_trials}")
                
                skipped_trials = 0
                processed_trials = 0
                
                for trial_id, value in automl.config_history.items():
                    # 🔥 FIX: FLAML 2.2.0 has inconsistent config_history structure
                    # Handle multiple formats: tuple(config, result), dict(result), or int(failed)
                    
                    config = None
                    result = None
                    
                    if isinstance(value, tuple):
                        if len(value) == 2:
                            config, result = value
                        else:
                            print(f"      ⚠️  Trial {trial_id}: Unexpected tuple length {len(value)}, skipping")
                            skipped_trials += 1
                            continue
                    elif isinstance(value, dict):
                        # Legacy format: value is result dict only
                        config = {}
                        result = value
                    elif isinstance(value, (int, type(None))):
                        # Failed trial (just trial_id or None)
                        print(f"      ⚠️  Trial {trial_id}: Failed trial (value={value}), skipping")
                        skipped_trials += 1
                        continue
                    else:
                        print(f"      ⚠️  Trial {trial_id}: Unknown type {type(value)}, skipping")
                        skipped_trials += 1
                        continue
                    
                    # Extract learner name safely
                    learner = config.get('learner', 'unknown') if isinstance(config, dict) else 'unknown'
                    
                    # Extract all available metrics from result dict
                    iteration_metrics = {
                        'iteration': trial_id,
                        'learner': learner,
                        'train_loss': result.get('train_loss') if isinstance(result, dict) else None,
                        'val_loss': result.get('val_loss') if isinstance(result, dict) else None,
                        'wall_clock_time': result.get('wall_clock_time') if isinstance(result, dict) else None,
                        'pred_time': result.get('pred_time') if isinstance(result, dict) else None,
                        'total_search_time': result.get('total_search_time') if isinstance(result, dict) else None,
                    }
                    
                    # Add hyperparameters from config if available
                    if isinstance(config, dict):
                        # Common hyperparameters across estimators
                        for hp in ['n_estimators', 'max_depth', 'learning_rate', 'min_child_samples', 
                                   'num_leaves', 'max_bin', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
                            if hp in config:
                                iteration_metrics[f'hp_{hp}'] = config[hp]
                    
                    iterations_data.append(iteration_metrics)
                    processed_trials += 1
                    
                    # Track per-estimator statistics
                    if learner not in per_estimator_stats:
                        per_estimator_stats[learner] = {
                            'count': 0,
                            'best_val_loss': float('inf'),
                            'worst_val_loss': float('-inf'),
                            'total_time': 0.0
                        }
                    
                    per_estimator_stats[learner]['count'] += 1
                    if isinstance(result, dict) and result.get('val_loss') is not None:
                        val_loss = result['val_loss']
                        per_estimator_stats[learner]['best_val_loss'] = min(
                            per_estimator_stats[learner]['best_val_loss'], val_loss
                        )
                        per_estimator_stats[learner]['worst_val_loss'] = max(
                            per_estimator_stats[learner]['worst_val_loss'], val_loss
                        )
                    if isinstance(result, dict) and result.get('wall_clock_time') is not None:
                        per_estimator_stats[learner]['total_time'] += result['wall_clock_time']
            
                print(f"      ✅ Processed: {processed_trials}/{total_trials} trials")
                if skipped_trials > 0:
                    print(f"      ⚠️  Skipped: {skipped_trials} failed/malformed trials")
            else:
                print(f"      ⚠️  No config_history found in AutoML object")
            
            if iterations_data:
                iterations_df = pd.DataFrame(iterations_data)
                iterations_path = outputs_dir / f"phaseb_{recipe_name}_flaml_iterations.csv"
                iterations_df.to_csv(iterations_path, index=False)
                print(f"  ✅ Recipe iterations: {iterations_path} ({len(iterations_df)} iterations, {len(per_estimator_stats)} estimators)")
                
                # Save per-estimator summary
                estimator_summary_path = outputs_dir / f"phaseb_{recipe_name}_estimator_summary.json"
                with open(estimator_summary_path, 'w') as f:
                    json.dump(per_estimator_stats, f, indent=2)
                print(f"  ✅ Estimator summary: {estimator_summary_path}")
                
                # Print estimator statistics
                print(f"\n  📈 Per-Estimator Statistics:")
                for est, stats in sorted(per_estimator_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                    best_loss = f"{stats['best_val_loss']:.4f}" if stats['best_val_loss'] != float('inf') else 'N/A'
                    print(f"     {est:15s} | Trials: {stats['count']:3d} | Best Loss: {best_loss} | Time: {stats['total_time']:.1f}s")
                
            else:
                print(f"  ⚠️  No FLAML iteration history available for recipe {recipe_name}")
            
            # 2. Copy model to outputs (if exists)
            model_dir = Path(args.model_out).resolve()
            model_path = model_dir / "model.pkl"
            if model_path.exists():
                shutil.copy2(model_path, outputs_dir / f'phaseb_{recipe_name}_best_model.pkl')
                print(f"  ✅ Best model copied: phaseb_{recipe_name}_best_model.pkl")
            else:
                print(f"  ⚠️  Model file not found, skipping copy")
            
            # 3. Enhanced Recipe Summary with Statistics
            recipe_summary = {
                "recipe": recipe_name,
                "engine": "flaml",
                "total_iterations": len(iterations_df) if iterations_data else 0,
                "best_estimator": str(best_estimator),
                "best_metric_value": float(score) if score is not None else None,
                "time_budget_seconds": time_budget,
                "actual_time_seconds": automl.time_taken_best_iter if hasattr(automl, 'time_taken_best_iter') else None,
                "estimators_tried": list(per_estimator_stats.keys()) if per_estimator_stats else [],
                "estimator_statistics": per_estimator_stats
            }
            
            # Add best config details if available
            if hasattr(automl, 'best_config') and automl.best_config:
                recipe_summary["best_config"] = str(automl.best_config)
            
            summary_path = outputs_dir / f"phaseb_{recipe_name}_flaml_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(recipe_summary, f, indent=2, default=str)
            print(f"  ✅ Recipe summary: {summary_path}")

            # ── Deep Model-Level Breakdown (shared utility) ────────────
            try:
                from utils.model_universe import build_flaml_breakdown, write_model_breakdown
                breakdown_df = build_flaml_breakdown(
                    automl,
                    step_name=f"phaseb_{recipe_name}",
                    phase="phase_b",
                    variant=recipe_name,
                    task_type=task_type,
                    metric_name=metrics.get("metric_name", "unknown"),
                    best_metric_value=metrics.get("best_metric"),
                )
                bd_path = write_model_breakdown(
                    breakdown_df, str(outputs_dir),
                    f"model_breakdown_phaseb_{recipe_name}_flaml.csv",
                )
                print(f"  📊 Model breakdown: {len(breakdown_df)} entries → {Path(bd_path).name}")
            except Exception as _bd_err:
                print(f"  ⚠️  Model breakdown write failed (non-fatal): {_bd_err}")
        except Exception as outputs_err:
            print(f"⚠️  Failed to create outputs folder contents: {outputs_err}")
            # Still create a summary with error info
            error_summary = {"recipe": recipe.get('recipe_name', 'unknown'), "error": str(outputs_err)}
            with open(outputs_dir / f"phaseb_recipe_error.json", 'w') as f:
                json.dump(error_summary, f, indent=2)
    else:
        # Training failed, create minimal error output
        error_summary = {"recipe": recipe.get('recipe_name', 'unknown'), "error": metrics.get("error", "Unknown error")}
        with open(outputs_dir / "phaseb_recipe_error.json", 'w') as f:
            json.dump(error_summary, f, indent=2)
        print(f"  ⚠️  Training failed, created error summary in outputs/")

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f)

    # Start MLflow run
    # Create dual logger for MLflow and Azure ML
    logger = create_metrics_logger(
        run_name="s06b_phaseb_flaml",
        tags={"pipeline": "v3_mlops", "phase": "phaseb", "step": "s06b"}
    )

    # 🔥 ENTERPRISE-LEVEL RECIPE & FLAML LOGGING
    print(f"\n📋 LOGGING RECIPE & FLAML DETAILS TO MLFLOW:")
    
    # Write recipe artifacts to outputs/ (Azure ML auto-uploads, MLflow artifact logging not supported with azureml://)
    try:
        flaml_metrics = {
            "engine": "flaml",
            "recipe": recipe.get("recipe_name", args.recipe_name),
            "best_estimator": str(best_estimator) if 'best_estimator' in locals() else None,
            "best_metric": float(metrics.get("best_metric", 0)) if "best_metric" in metrics else None,
            "total_iterations": len(automl.config_history) if hasattr(automl, 'config_history') else 0
        }
        flaml_manifest = manifest.copy()
        
        with open(outputs_dir / "flaml_recipe_metrics.json", "w") as f:
            json.dump(flaml_metrics, f, indent=2)
        with open(outputs_dir / "flaml_recipe_manifest.json", "w") as f:
            json.dump(flaml_manifest, f, indent=2)
        with open(outputs_dir / f"recipe_{args.recipe_name.replace('.yml', '.json').replace('.yaml', '.json')}", "w") as f:
            json.dump(recipe, f, indent=2)
        
        print(f"   ✅ Recipe artifacts saved to outputs/ folder (Azure ML will auto-upload)")
    except Exception as artifact_err:
        print(f"   ⚠️  Failed to save recipe artifacts: {artifact_err}")
    
    # Log recipe parameters to MLflow (parameters work with azureml:// scheme)
    try:
        # Recipe identification
        logger.log_param("recipe", recipe.get("recipe_name", args.recipe_name.split('/')[-1].replace('.yml', '')))  # 🚀 VSS: Log variant ID
        logger.log_param("recipe_name", recipe.get("recipe_name", args.recipe_name))
        logger.log_param("recipe_version", recipe.get("version", "1.0"))
        logger.log_param("recipe_description", recipe.get("description", "No description"))
        logger.log_param("recipe_file", args.recipe_name)
        
        # Stage 3 preprocessing techniques from recipe
        stage3 = recipe.get("stage3_preprocessing", {})
        if stage3:
            # Imputation
            imputation = stage3.get("imputation", {})
            logger.log_param("recipe_imputation_method", imputation.get("method", "none"))
            if imputation.get("n_neighbors"):
                logger.log_param("recipe_imputation_n_neighbors", int(imputation["n_neighbors"]))
            
            # Imbalance handling
            imbalance = stage3.get("imbalance_handling", {})
            logger.log_param("recipe_imbalance_method", imbalance.get("method", "none"))
            if imbalance.get("sampling_strategy"):
                logger.log_param("recipe_sampling_strategy", str(imbalance["sampling_strategy"]))
            
            # Encoding
            encoding = stage3.get("encoding", {})
            logger.log_param("recipe_encoding_method", encoding.get("categorical_method", "onehot"))
            logger.log_param("recipe_handle_unknown", encoding.get("handle_unknown", "error"))
            
            # Scaling
            scaling = stage3.get("scaling", {})
            logger.log_param("recipe_scaling_method", scaling.get("method", "standard"))
        
        # Stage 4 feature engineering from recipe
        stage4 = recipe.get("stage4_feature_engineering", {})
        if stage4:
            feature_sel = stage4.get("feature_selection", {})
            logger.log_param("recipe_feature_selection", feature_sel.get("method", "none"))
            if feature_sel.get("k_features"):
                logger.log_param("recipe_k_features", int(feature_sel["k_features"]))
        
        # FLAML-specific parameters
        logger.log_param("engine", "flaml")
        logger.log_param("flaml_time_budget", int(time_budget))
        logger.log_param("task_type", task_type)
        logger.log_param("target_column", target_col)
        logger.log_param("flaml_task", "classification" if task_type == "classification" else "regression")
        logger.log_param("flaml_metric", "accuracy" if task_type == "classification" else "r2")
        
        # Model results
        if "best_metric" in metrics:
            logger.log_metric("best_metric", float(metrics["best_metric"]))
        if "computed_score" in metrics:
            logger.log_metric("computed_score", float(metrics["computed_score"]))
        if "error" not in metrics:
            logger.log_param("best_estimator", metrics.get("best_estimator", "unknown"))
            
        # Dataset metrics
        logger.log_metric("dataset_rows", int(X.shape[0]))
        logger.log_metric("dataset_cols", int(X.shape[1]))
        
        # FLAML iteration metrics
        if "error" not in metrics and hasattr(automl, 'config_history') and automl.config_history:
            logger.log_metric("flaml_total_iterations", len(automl.config_history))
        
        # Artifacts already saved to outputs/ folder (MLflow artifact logging not supported with azureml://)
        # See lines 245-260 for outputs/ folder writes
        
        if "error" in metrics:
            logger.log_param("flaml_recipe_error", metrics["error"])
        
        print(f"   ✅ Recipe: {recipe.get('recipe_name')}")
        print(f"   ✅ Imputation: {stage3.get('imputation', {}).get('method', 'none')}")
        print(f"   ✅ Imbalance: {stage3.get('imbalance_handling', {}).get('method', 'none')}")
        print(f"   ✅ Encoding: {stage3.get('encoding', {}).get('categorical_method', 'onehot')}")
        print(f"   ✅ Scaling: {stage3.get('scaling', {}).get('method', 'standard')}")
        print(f"   ✅ Feature Selection: {stage4.get('feature_selection', {}).get('method', 'none')}")
        print(f"   ✅ FLAML Budget: {time_budget}s, Iterations: {len(automl.config_history) if 'error' not in metrics and hasattr(automl, 'config_history') else 0}")
            
    except Exception as mlflow_err:
        print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")

    # End logging
    logger.end_run()


if __name__ == "__main__":
    main()
