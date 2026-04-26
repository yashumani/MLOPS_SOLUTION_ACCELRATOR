import argparse
import json
import logging
import time
from pathlib import Path
import sys

# Ensure src/ on path (single canonical insertion at front of sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.stage_signals import StageSignal, write_stage_signal
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_stage_table,
)

# Module-level logger for diagnostic/debug messages.
logger = logging.getLogger(__name__)


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
        Metric name: "balanced_accuracy", "R2", "silhouette_score"
    """
    if task_type == "classification":
        return "balanced_accuracy"
    elif task_type == "regression":
        return "R2"
    elif task_type == "clustering":
        return "silhouette_score"
    else:
        # Default to classification for unknown tasks
        return "balanced_accuracy"


def load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        # Preserve None contract for callers; log so failures are not silent.
        logger.warning(f"JSON parse error in aggregate_baseline (path={path}): {e}")
        return None


def select_champion(pycaret_manifest, flaml_manifest, task: str = "classification", ts_manifest=None):
    """Select best model from pycaret, flaml, and (optionally) timeseries baselines.
    
    Returns: {"source": "pycaret"|"flaml"|"timeseries"|None, "score": float|None, "reason": str}
    """
    best = {"source": None, "score": None, "reason": ""}
    primary_metric = get_primary_metric(task)
    
    def extract_score(m, engine_name):
        """Extract best score from manifest, return (score, reason) or (None, reason)"""
        if not m:
            return None, f"{engine_name} manifest missing"
        
        engine = m.get("engine", "unknown")
        status = m.get("status")
        
        # Check if engine is skipped (e.g., FLAML for clustering)
        if status == "skipped":
            return None, f"{engine_name} skipped: {m.get('reason', 'not available')}"
        
        # FLAML: prefer primary metric for apples-to-apples cross-engine comparison
        if engine == "flaml":
            # Try primary_metric field first (e.g., "accuracy" for classification)
            primary_lower = primary_metric.lower()
            for key in [primary_lower, primary_metric]:
                score = m.get(key)
                if score is not None:
                    try:
                        return float(score), f"FLAML {key}={score}"
                    except (ValueError, TypeError):
                        continue
            # Fallback: best_metric (may be AUC or other optimization target)
            score = m.get("best_metric")
            if score is not None:
                try:
                    return float(score), f"FLAML best_metric={score}"
                except (ValueError, TypeError):
                    return None, f"FLAML best_metric not numeric: {score}"
            return None, "FLAML missing best_metric"
        
        # PyCaret: try leaderboard first, then metrics_dict, then clustering metrics
        if engine == "pycaret":
            # For clustering, check direct metric fields first
            if task == "clustering":
                score = m.get("silhouette_score")
                if score is not None:
                    try:
                        return float(score), f"PyCaret silhouette_score={score}"
                    except (ValueError, TypeError):
                        pass
            
            # Try leaderboard structure (dict of columns)
            lb = m.get("leaderboard")
            if lb and isinstance(lb, dict):
                if primary_metric in lb and isinstance(lb[primary_metric], dict):
                    try:
                        models = lb[primary_metric]
                        best_key = max(models.keys(), key=lambda k: float(models[k]))
                        score = float(models[best_key])
                        return score, f"PyCaret leaderboard[{primary_metric}]={score}"
                    except (ValueError, TypeError, KeyError, IndexError):
                        pass

            # Try direct manifest field (used for balanced_accuracy and normalized metrics)
            score = m.get(primary_metric)
            if score is not None:
                try:
                    return float(score), f"PyCaret {primary_metric}={score}"
                except (ValueError, TypeError):
                    pass
            
            # Try best_metric field (some versions use this)
            score = m.get("best_metric")
            if score is not None:
                try:
                    return float(score), f"PyCaret best_metric={score}"
                except (ValueError, TypeError):
                    pass
            
            # Try metrics_dict or top_model_metrics
            for key in ["metrics_dict", "top_model_metrics", "best_model_metrics"]:
                metrics_dict = m.get(key)
                if metrics_dict and isinstance(metrics_dict, dict):
                    if primary_metric in metrics_dict:
                        try:
                            score = float(metrics_dict[primary_metric])
                            return score, f"PyCaret {key}[{primary_metric}]={score}"
                        except (ValueError, TypeError):
                            pass
            
            return None, f"PyCaret: no {primary_metric} metric found"
        
        return None, f"Unknown engine: {engine}"
    
    ps, ps_reason = extract_score(pycaret_manifest, "PyCaret")
    fs, fs_reason = extract_score(flaml_manifest, "FLAML")
    # K1: timeseries baseline (optional). Status "skipped" is normal for non-TS tasks.
    ts_score, ts_reason = (None, "timeseries manifest not provided")
    if ts_manifest is not None:
        if ts_manifest.get("status") == "skipped":
            ts_reason = f"Timeseries skipped: {ts_manifest.get('reason', 'not applicable')}"
        else:
            for _key in ("best_metric", "score", "primary_metric_value"):
                _v = ts_manifest.get(_key)
                if _v is not None:
                    try:
                        ts_score = float(_v)
                        ts_reason = f"Timeseries {_key}={_v}"
                        break
                    except (ValueError, TypeError):
                        continue
            if ts_score is None:
                ts_reason = "Timeseries manifest present but no numeric score"
    
    # Select best (higher is better for accuracy, r2, silhouette_score; for forecasting,
    # timeseries baseline is preferred when present and scored).
    candidates = []
    if ps is not None:
        candidates.append(("pycaret", ps, ps_reason))
    if fs is not None:
        candidates.append(("flaml", fs, fs_reason))
    if ts_score is not None:
        # For forecasting tasks (e.g. lower error is better), task-specific scoring
        # would invert the comparison. For now treat ts_score as already-normalized
        # (timeseries step is responsible for emitting a higher-is-better metric).
        candidates.append(("timeseries", ts_score, ts_reason))
    
    if candidates:
        candidates.sort(key=lambda c: c[1], reverse=True)
        best["source"], best["score"], best["reason"] = candidates[0]
    else:
        best["reason"] = (
            f"No valid scores found. PyCaret: {ps_reason}. FLAML: {fs_reason}. "
            f"Timeseries: {ts_reason}"
        )
    
    return best


def main():
    _t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pycaret_manifest", required=True)
    parser.add_argument("--pycaret_model", required=True)
    parser.add_argument("--flaml_manifest", required=True)
    parser.add_argument("--flaml_model", required=True)
    # K1: optional timeseries baseline (skipped internally for non-TS tasks)
    parser.add_argument("--ts_manifest", required=False, default=None)
    parser.add_argument("--ts_model", required=False, default=None)
    parser.add_argument("--report_out", required=True)
    parser.add_argument("--champion_out", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("STEP S05z: AGGREGATE BASELINE")
    print("=" * 80)

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    config_name = Path(args.config).name

    pman = load_json(args.pycaret_manifest)
    fman = load_json(args.flaml_manifest)
    # K1: load timeseries manifest if provided (optional)
    tman = load_json(args.ts_manifest) if args.ts_manifest else None

    champion = select_champion(pman, fman, task=task_type, ts_manifest=tman)
    print(f"🏆 Aggregate Baseline: Selected {champion['source']} | Score: {champion['score']} | Reason: {champion.get('reason', 'N/A')}")
    
    # Choose model accordingly (both are folders containing model files)
    model_src = None
    flaml_path = Path(args.flaml_model)
    pycaret_path = Path(args.pycaret_model)
    
    # Check if model folders exist and contain files (are not empty)
    def is_valid_model_folder(p):
        if not p.exists():
            print(f"    Model path does not exist: {p}")
            return False
        if p.is_file():  # Single file model
            sz = p.stat().st_size
            valid = sz > 0
            print(f"    Model file exists: {p} (size: {sz} bytes) {'✅' if valid else '❌'}")
            return valid
        if p.is_dir():  # Folder with model files
            contents = list(p.iterdir())
            valid = len(contents) > 0
            print(f"    Model folder exists: {p} (files: {len(contents)}) {'✅' if valid else '❌'}")
            if contents:
                for item in contents[:3]:
                    print(f"        - {item.name}")
                if len(contents) > 3:
                    print(f"        ... and {len(contents)-3} more")
            return valid
        return False
    
    if champion["source"] == "flaml" and is_valid_model_folder(flaml_path):
        model_src = flaml_path
        print(f"  ✅ Using FLAML model from {model_src}")
    elif champion["source"] == "pycaret" and is_valid_model_folder(pycaret_path):
        model_src = pycaret_path
        print(f"  ✅ Using PyCaret model from {model_src}")
    elif champion["source"] == "timeseries" and args.ts_model and is_valid_model_folder(Path(args.ts_model)):
        # K1: timeseries champion
        model_src = Path(args.ts_model)
        print(f"  ✅ Using Timeseries model from {model_src}")
    else:
        # fallback: prefer pycaret model
        if is_valid_model_folder(pycaret_path):
            model_src = pycaret_path
            print(f"  ⚠️  Fallback to PyCaret model from {model_src}")
        elif is_valid_model_folder(flaml_path):
            model_src = flaml_path
            print(f"  ⚠️  Fallback to FLAML model from {model_src}")
        else:
            print(f"  ❌ No valid model found! PyCaret: {pycaret_path.exists()}, FLAML: {flaml_path.exists()}")

    report = {
        "task": task_type,
        "selection": {
            "source": champion["source"],
            "score": champion["score"],
            "reason": champion.get("reason", "N/A")
        },
        "pycaret_manifest_present": pman is not None,
        "flaml_manifest_present": fman is not None,
        "model_copied": False,
    }
    
    # Use absolute path resolution for Azure ML outputs
    report_path = Path(args.report_out).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Report saved: {report_path} ({report_path.stat().st_size:,} bytes)")

    if model_src:
        import shutil
        try:
            # Azure ML-safe copy with absolute path resolution
            champion_path = Path(args.champion_out).resolve()
            champion_path.mkdir(parents=True, exist_ok=True)
            print(f"  📂 Champion output path: {champion_path}")
            
            # Copy all files recursively
            copied_count = 0
            if model_src.is_file():
                # Single file model
                dest = champion_path / model_src.name
                shutil.copy2(model_src, dest)
                copied_count = 1
                print(f"  ✅ Copied model file: {model_src.name} ({model_src.stat().st_size:,} bytes)")
            elif model_src.is_dir():
                # Directory with model files - copy recursively
                for src_file in model_src.rglob('*'):
                    if src_file.is_file():
                        rel_path = src_file.relative_to(model_src)
                        dest_file = champion_path / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dest_file)
                        copied_count += 1
                print(f"  ✅ Copied {copied_count} model files to {champion_path}")
            
            report["model_copied"] = True
            report["files_copied"] = copied_count
            
            # Validate outputs immediately after copying
            print("\n🔍 Validating champion model output...")
            validation = validate_and_log_outputs(champion_path, "Champion Model")
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
        # Create empty output folder if no model selected
        champion_path = Path(args.champion_out).resolve()
        champion_path.mkdir(parents=True, exist_ok=True)
        print(f"  ℹ️  Created empty champion folder (no model found): {champion_path}")
    
    # Update report with final status (absolute path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Final report updated: {report_path}")

    # ── Emit stage signal ──────────────────────────────────────────────
    _elapsed = time.time() - _t0
    _count_in = 2  # pycaret + flaml
    _count_out = 1 if champion["source"] else 0
    _primary_metric = get_primary_metric(task_type)
    try:
        sig = StageSignal(
            stage_name="baseline_aggregate",
            stage_id="S05z",
            task_type=task_type,
            config_name=config_name,
            candidate_count_in=_count_in,
            candidate_count_out=_count_out,
            best_score=champion["score"],
            best_metric_name=_primary_metric,
            failure_rate=(_count_in - _count_out) / max(_count_in, 1),
            failure_count=_count_in - _count_out,
            total_count=_count_in,
            compute_time_sec=round(_elapsed, 2),
            recommendation="proceed" if champion["score"] is not None else "stop",
            recommendation_reason=champion.get("reason", ""),
        )
        write_stage_signal(sig, out_dir="outputs", filename="baseline_stage_signal.json")
    except Exception as _sig_err:
        print(f"⚠️  Stage signal write failed (non-fatal): {_sig_err}")

    # ── Candidate Ledger ──────────────────────────────────────────────────
    try:
        _ledger_rows = []
        _primary_metric = get_primary_metric(task_type)
        for _eng, _man in [("pycaret", pman), ("flaml", fman)]:
            _mdict = {}
            if _man and isinstance(_man, dict):
                # Gather all numeric metrics from manifest
                for _k, _v in _man.items():
                    try:
                        float(_v); _mdict[_k] = _v
                    except (ValueError, TypeError):
                        pass
            _norm = normalize_metrics(task_type, _mdict)
            _st = "skipped" if (_man or {}).get("status") == "skipped" else "ok"
            _is_best = (champion.get("source") == _eng)
            _row = make_row(
                stage="baseline", step_name="s05z", engine=_eng,
                candidate_id=f"{_eng}_baseline",
                task_type=task_type,
                dataset_id=config_name,
                status=_st,
                compute_time_sec=round(_elapsed, 2),
                source_path="src/steps/aggregate_baseline.py",
                recipe_name="baseline",
                is_stage_best=_is_best,
                **_norm,
            )
            _ledger_rows.append(_row)
        write_stage_table(
            _ledger_rows,
            csv_path="outputs/s05z_candidates.csv",
            parquet_path="outputs/s05z_candidates.parquet",
        )
    except Exception as _ledger_err:
        print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")


if __name__ == "__main__":
    main()
