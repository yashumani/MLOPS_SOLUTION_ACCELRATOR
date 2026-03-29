import argparse
import json
import time
from pathlib import Path
import sys
import pandas as pd

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.stage_signals import StageSignal, write_stage_signal
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_stage_table,
)


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


def get_primary_metric(task_type: str) -> str:
    """Return the primary metric column/field name for a task type.
    
    Args:
        task_type: "classification", "regression", or "clustering"
    
    Returns:
        Metric name: "Accuracy", "R2", "silhouette_score"
    """
    if task_type == "classification":
        return "Accuracy"
    elif task_type == "regression":
        return "R2"
    elif task_type == "clustering":
        return "silhouette_score"
    else:
        # Default to classification for unknown tasks
        return "Accuracy"


def load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def best_score_from_manifest(manifest, task: str = "classification", recipe_key: str = ""):
    """Extract best score from manifest, return (score, reason)"""
    if not manifest:
        return None, f"{recipe_key}: manifest missing"
    
    engine = manifest.get("engine", "unknown")
    status = manifest.get("status")
    
    # Check if engine is skipped (e.g., FLAML for clustering)
    if status == "skipped":
        return None, f"{recipe_key} {engine} skipped: {manifest.get('reason', 'not available')}"
    
    primary_metric = get_primary_metric(task)
    
    # FLAML: best_metric field
    if engine == "flaml":
        score = manifest.get("best_metric")
        if score is not None:
            try:
                return float(score), f"{recipe_key} FLAML best_metric={score}"
            except (ValueError, TypeError):
                return None, f"{recipe_key} FLAML best_metric not numeric: {score}"
        return None, f"{recipe_key} FLAML missing best_metric"
    
    # PyCaret: try leaderboard first, then metrics_dict
    if engine == "pycaret":
        # For clustering, check direct metric fields first
        if task == "clustering":
            score = manifest.get("silhouette_score")
            if score is not None:
                try:
                    return float(score), f"{recipe_key} PyCaret silhouette_score={score}"
                except (ValueError, TypeError):
                    pass
        
        # Try leaderboard structure
        lb = manifest.get("leaderboard")
        if lb and isinstance(lb, dict):
            if primary_metric in lb and isinstance(lb[primary_metric], dict):
                try:
                    first_key = sorted(lb[primary_metric].keys())[0]
                    score = float(lb[primary_metric][first_key])
                    return score, f"{recipe_key} PyCaret leaderboard[{primary_metric}]={score}"
                except (ValueError, TypeError, KeyError, IndexError):
                    pass
        
        # Try best_metric field
        score = manifest.get("best_metric")
        if score is not None:
            try:
                return float(score), f"{recipe_key} PyCaret best_metric={score}"
            except (ValueError, TypeError):
                pass
        
        # Try metrics_dict
        for key in ["metrics_dict", "top_model_metrics", "best_model_metrics"]:
            metrics_dict = manifest.get(key)
            if metrics_dict and isinstance(metrics_dict, dict):
                if primary_metric in metrics_dict:
                    try:
                        score = float(metrics_dict[primary_metric])
                        return score, f"{recipe_key} PyCaret {key}[{primary_metric}]={score}"
                    except (ValueError, TypeError):
                        pass
        
        return None, f"{recipe_key} PyCaret: no {primary_metric} metric found"
    
    return None, f"{recipe_key} Unknown engine: {engine}"


def main():
    _t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--r1_pycaret_manifest", required=True)
    parser.add_argument("--r1_pycaret_model", required=True)
    parser.add_argument("--r1_flaml_manifest", required=True)
    parser.add_argument("--r1_flaml_model", required=True)
    parser.add_argument("--r2_pycaret_manifest", required=True)
    parser.add_argument("--r2_pycaret_model", required=True)
    parser.add_argument("--r2_flaml_manifest", required=True)
    parser.add_argument("--r2_flaml_model", required=True)
    parser.add_argument("--report_out", required=True)
    parser.add_argument("--champion_out", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("STEP S07z: AGGREGATE PHASE B")
    print("=" * 80)

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"

    manifests = {
        "r1_pycaret": load_json(args.r1_pycaret_manifest),
        "r1_flaml": load_json(args.r1_flaml_manifest),
        "r2_pycaret": load_json(args.r2_pycaret_manifest),
        "r2_flaml": load_json(args.r2_flaml_manifest),
    }
    scores = {}
    reasons = {}
    for k, v in manifests.items():
        score, reason = best_score_from_manifest(v, task=task_type, recipe_key=k)
        scores[k] = score
        reasons[k] = reason
    
    # Select best key
    best_key = None
    best_val = None
    for k, v in scores.items():
        if v is None:
            print(f"  ⚠️  {reasons[k]}")
            continue
        if best_val is None or v > best_val:
            best_key, best_val = k, v
    
    print(f"🏆 Phase B Aggregate: Selected {best_key} | Score: {best_val} | Reason: {reasons.get(best_key, 'N/A')}")

    # Map to model path
    model_map = {
        "r1_pycaret": Path(args.r1_pycaret_model),
        "r1_flaml": Path(args.r1_flaml_model),
        "r2_pycaret": Path(args.r2_pycaret_model),
        "r2_flaml": Path(args.r2_flaml_model),
    }
    champion_path = model_map.get(best_key)

    report = {
        "task": task_type,
        "scores": scores,
        "selection": {"key": best_key, "score": best_val},
        "model_copied": False,
    }
    
    # 🔥 NEW: Save champion metadata for Phase C HPO
    champion_metadata = {
        "champion_key": best_key,
        "champion_score": best_val,
        "task_type": task_type,
        "algorithm": "unknown",
        "recipe": "unknown",
        "engine": "unknown",
        "data_path": args.r1_pycaret_model.replace("/model_out", "/train_data_out") if best_key else None
    }
    
    # Extract algorithm and recipe from champion manifest
    if best_key and manifests.get(best_key):
        champion_manifest = manifests[best_key]
        champion_metadata["algorithm"] = champion_manifest.get("best_model_name") or champion_manifest.get("best_estimator") or "unknown"
        champion_metadata["recipe"] = champion_manifest.get("recipe", "unknown")
        champion_metadata["engine"] = champion_manifest.get("engine", "unknown")
    
    # Save champion manifest for Phase C
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    champion_manifest_path = outputs_dir / "phaseb_champion_manifest.json"
    with open(champion_manifest_path, 'w') as f:
        json.dump(champion_metadata, f, indent=2)
    print(f"\n🏆 CHAMPION METADATA FOR PHASE C:")
    print(f"  Algorithm: {champion_metadata['algorithm']}")
    print(f"  Recipe: {champion_metadata['recipe']}")
    print(f"  Engine: {champion_metadata['engine']}")
    print(f"  Score: {champion_metadata['champion_score']}")
    print(f"  ✅ Saved to: {champion_manifest_path}")
    
    # 🔥 ENTERPRISE-LEVEL AGGREGATE LOGGING
    print(f"\n📊 LOGGING PHASE B AGGREGATION TO MLFLOW:")
    from utils.azureml_metrics_logger import create_metrics_logger
    logger = create_metrics_logger(
        run_name="s08z_aggregate_phaseb",
        tags={"pipeline": "v3_mlops", "phase": "phaseb", "step": "s08z"}
    )
    
    try:
        # Overall Phase B summary
        logger.log_param("phase", "B")
        logger.log_param("task_type", task_type)
        logger.log_param("total_recipes_evaluated", 2)
        logger.log_param("total_engines_evaluated", 4)
        
        # Recipe names (extract from manifests)
        r1_recipe_name = manifests.get("r1_pycaret", {}).get("recipe", "unknown")
        r2_recipe_name = manifests.get("r2_pycaret", {}).get("recipe", "unknown")
        logger.log_param("recipe_1_name", r1_recipe_name)
        logger.log_param("recipe_2_name", r2_recipe_name)
        
        # Log all scores
        for key, score in scores.items():
            if score is not None:
                logger.log_metric(f"score_{key}", float(score))
            logger.log_param(f"reason_{key}", reasons.get(key, "N/A"))
        
        # Champion selection
        logger.log_param("champion_key", str(best_key))
        logger.log_metric("champion_score", float(best_val) if best_val is not None else 0.0)
        logger.log_param("champion_reason", reasons.get(best_key, "N/A"))
        
        # Recipe comparison summary
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        if len(valid_scores) > 0:
            logger.log_metric("valid_models_count", len(valid_scores))
            logger.log_metric("failed_models_count", 4 - len(valid_scores))
            logger.log_metric("best_score", float(max(valid_scores.values())))
            logger.log_metric("worst_score", float(min(valid_scores.values())))
            logger.log_metric("score_range", float(max(valid_scores.values()) - min(valid_scores.values())))
        
        # Log full report
        logger.log_dict(report, "aggregate_phaseb_report.json")
        
        print(f"   ✅ Recipes: {r1_recipe_name}, {r2_recipe_name}")
        print(f"   ✅ Valid models: {len(valid_scores)}/4")
        print(f"   ✅ Champion: {best_key} (score: {best_val})")
        
        logger.end_run()
        
    except Exception as agg_log_err:
        print(f"⚠️  Aggregate logging failed (non-fatal): {agg_log_err}")
        try:
            logger.end_run()
        except:
            pass
    
    # Use absolute path resolution for Azure ML outputs
    report_path = Path(args.report_out).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Report saved: {report_path} ({report_path.stat().st_size:,} bytes)")
    
    # 📊 CREATE OUTPUTS FOLDER WITH RECIPE COMPARISON TABLE
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 RECIPE COMPARISON LOGGING TO outputs/ FOLDER:")
    
    # Build comparison table using the already loaded manifests and scores
    comparison_data = []
    
    # Recipe 1 - PyCaret
    if scores.get('r1_pycaret') is not None:
        comparison_data.append({
            'recipe_engine': 'recipe1_pycaret',
            'recipe_file': manifests['r1_pycaret'].get('recipe', 'unknown') if manifests['r1_pycaret'] else 'unknown',
            'best_model': manifests['r1_pycaret'].get('best_model_name', 'unknown') if manifests['r1_pycaret'] else 'unknown',
            'metric_value': scores['r1_pycaret'],
            'models_trained': manifests['r1_pycaret'].get('rows', 0) if manifests['r1_pycaret'] else 0,
            'status': 'completed'
        })
    
    # Recipe 1 - FLAML
    if scores.get('r1_flaml') is not None:
        comparison_data.append({
            'recipe_engine': 'recipe1_flaml',
            'recipe_file': manifests['r1_flaml'].get('recipe', 'unknown') if manifests['r1_flaml'] else 'unknown',
            'best_model': manifests['r1_flaml'].get('best_estimator', 'unknown') if manifests['r1_flaml'] else 'unknown',
            'metric_value': scores['r1_flaml'],
            'models_trained': 'N/A',
            'status': 'completed'
        })
    
    # Recipe 2 - PyCaret
    if scores.get('r2_pycaret') is not None:
        comparison_data.append({
            'recipe_engine': 'recipe2_pycaret',
            'recipe_file': manifests['r2_pycaret'].get('recipe', 'unknown') if manifests['r2_pycaret'] else 'unknown',
            'best_model': manifests['r2_pycaret'].get('best_model_name', 'unknown') if manifests['r2_pycaret'] else 'unknown',
            'metric_value': scores['r2_pycaret'],
            'models_trained': manifests['r2_pycaret'].get('rows', 0) if manifests['r2_pycaret'] else 0,
            'status': 'completed'
        })
    
    # Recipe 2 - FLAML
    if scores.get('r2_flaml') is not None:
        comparison_data.append({
            'recipe_engine': 'recipe2_flaml',
            'recipe_file': manifests['r2_flaml'].get('recipe', 'unknown') if manifests['r2_flaml'] else 'unknown',
            'best_model': manifests['r2_flaml'].get('best_estimator', 'unknown') if manifests['r2_flaml'] else 'unknown',
            'metric_value': scores['r2_flaml'],
            'models_trained': 'N/A',
            'status': 'completed'
        })
    
    # Save comparison table
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = outputs_dir / "phaseb_recipe_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  ✅ Recipe comparison: {comparison_path} ({len(comparison_df)} recipes, {comparison_path.stat().st_size:,} bytes)")
        
        # Identify best recipe
        best_idx = comparison_df['metric_value'].idxmax()
        best_recipe = comparison_df.loc[best_idx]
        print(f"  🏆 Best recipe: {best_recipe['recipe_engine']} with {best_recipe['metric_value']:.4f}")
        
        # Save best recipe info
        best_info = {
            'best_recipe_engine': best_recipe['recipe_engine'],
            'best_metric': float(best_recipe['metric_value']),
            'best_model': best_recipe['best_model'],
            'all_recipes_count': len(comparison_df)
        }
        best_path = outputs_dir / "phaseb_best_recipe.json"
        with open(best_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        print(f"  ✅ Best recipe info: {best_path}")

    if champion_path and champion_path.exists():
        import shutil
        try:
            # Azure ML-safe copy with absolute path resolution
            output_path = Path(args.champion_out).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"  📂 Champion output path: {output_path}")
            
            # Copy all files recursively
            copied_count = 0
            if champion_path.is_file():
                # Single file model
                dest = output_path / champion_path.name
                shutil.copy2(champion_path, dest)
                copied_count = 1
                print(f"  ✅ Copied model file: {champion_path.name} ({champion_path.stat().st_size:,} bytes)")
            elif champion_path.is_dir():
                # Directory with model files - copy recursively
                for src_file in champion_path.rglob('*'):
                    if src_file.is_file():
                        rel_path = src_file.relative_to(champion_path)
                        dest_file = output_path / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dest_file)
                        copied_count += 1
                print(f"  ✅ Copied {copied_count} model files to {output_path}")
            
            report["model_copied"] = True
            report["files_copied"] = copied_count
            
            # Validate outputs immediately after copying
            print("\n🔍 Validating champion model output...")
            validation = validate_and_log_outputs(output_path, "Champion Model")
            report["output_validation"] = validation
            
            if validation["valid"]:
                print(f"  ✅ Output validation passed: {validation['total_size']:,} bytes in {len(validation['files'])} files")
            else:
                print(f"  ❌ Output validation failed:")
                for err in validation["errors"]:
                    print(f"     - {err}")
        except Exception as e:
            print(f"  ❌ Error copying model: {e}")
            import traceback
            traceback.print_exc()
            report["model_copy_error"] = str(e)
    else:
        # Create empty output folder
        output_path = Path(args.champion_out).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"  ⚠️  Champion path missing or doesn't exist: {champion_path}")
    
    # Update report with final status (absolute path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Final report updated: {report_path}")

    # ── Emit stage signal ──────────────────────────────────────────────
    _elapsed = time.time() - _t0
    _count_in = 4  # 2 recipes × 2 engines
    _valid = len({k for k, v in scores.items() if v is not None})
    _count_out = 1 if best_key else 0
    _primary_metric = get_primary_metric(task_type)
    _baseline_score = None  # Could be loaded from prior stage signal if available
    _delta = None
    try:
        sig = StageSignal(
            stage_name="phaseb_aggregate",
            stage_id="S07z",
            task_type=task_type,
            config_name=Path(args.config).name,
            candidate_count_in=_count_in,
            candidate_count_out=_count_out,
            best_score=best_val,
            best_metric_name=_primary_metric,
            delta_vs_baseline=_delta,
            failure_rate=(_count_in - _valid) / max(_count_in, 1),
            failure_count=_count_in - _valid,
            total_count=_count_in,
            compute_time_sec=round(_elapsed, 2),
            topk_gap=round(max(v for v in scores.values() if v is not None) - min(v for v in scores.values() if v is not None), 6) if _valid >= 2 else None,
            recommendation="proceed" if best_val is not None else "stop",
            recommendation_reason=reasons.get(best_key, "") if best_key else "No valid scores",
        )
        write_stage_signal(sig, out_dir="outputs", filename="phaseb_stage_signal.json")
    except Exception as _sig_err:
        print(f"⚠️  Stage signal write failed (non-fatal): {_sig_err}")

    # ── Candidate Ledger ──────────────────────────────────────────────────
    try:
        _ledger_rows = []
        _primary_metric = get_primary_metric(task_type)
        for _key, _val in scores.items():
            _norm = normalize_metrics(task_type, {_primary_metric: _val} if _val is not None else {})
            _is_best = (_key == best_key)
            _row = make_row(
                stage="phase_b", step_name="s07z", engine=_key,
                candidate_id=f"phaseb_{_key}",
                task_type=task_type,
                dataset_id=Path(args.config).name,
                status="ok" if _val is not None else "failed",
                failure_reason=reasons.get(_key, "") if _val is None else "",
                compute_time_sec=round(_elapsed, 2),
                source_path="src/steps/aggregate_phaseb.py",
                recipe_name=_key,
                is_stage_best=_is_best,
                **_norm,
            )
            _ledger_rows.append(_row)
        write_stage_table(
            _ledger_rows,
            csv_path="outputs/s07z_candidates.csv",
            parquet_path="outputs/s07z_candidates.parquet",
        )
    except Exception as _ledger_err:
        print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")


if __name__ == "__main__":
    main()
