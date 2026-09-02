import argparse
import json
import time
from pathlib import Path
import sys

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


def load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def has_exact_model_bundle(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "model_bundle.pkl").is_file()
        and (path / "model_bundle.pkl").stat().st_size > 0
        and (path / "model_bundle_manifest.json").is_file()
        and (path / "model_bundle_manifest.json").stat().st_size > 0
    )


def main():
    _t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # K7 fix: YAML passes --hpo_metrics; accept both names for backward-compat.
    parser.add_argument("--hpo_metrics", "--hpo_metrics_json",
                        dest="hpo_metrics", required=True)
    parser.add_argument("--optimized_model", required=True)
    parser.add_argument("--report_out", required=True)
    parser.add_argument("--champion_out", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("STEP S09: AGGREGATE PHASE C")
    print("=" * 80)

    metrics = load_json(args.hpo_metrics) or {}
    report = {
        "phase": "C",
        "selection": {
            "score": metrics.get("best_score"),
            "params": metrics.get("best_params")
        },
        "model_copied": False,
    }
    
    # Use absolute path resolution for Azure ML outputs
    report_path = Path(args.report_out).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Report saved: {report_path} ({report_path.stat().st_size:,} bytes)")
    
    # 📊 CREATE OUTPUTS FOLDER FOR AZURE ML STUDIO VISIBILITY
    import shutil
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 PHASE C AGGREGATE TO outputs/ FOLDER:")
    
    # 1. Copy report to outputs
    shutil.copy2(report_path, outputs_dir / "phasec_aggregate_report.json")
    print(f"  ✅ Aggregate report copied: phasec_aggregate_report.json")
    
    # 2. Save champion summary
    champion_summary = {
        "phase": "C",
        "best_score": metrics.get("best_score"),
        "best_params": metrics.get("best_params"),
        "optimizer": "optuna"
    }
    with open(outputs_dir / "phasec_champion_summary.json", 'w') as f:
        json.dump(champion_summary, f, indent=2)
    print(f"  ✅ Champion summary: phasec_champion_summary.json")

    src = Path(args.optimized_model)
    print(f"🏆 Phase C Aggregate: HPO best_score={metrics.get('best_score')}")
    
    if metrics.get("status") == "success" and has_exact_model_bundle(src):
        import shutil
        try:
            # Azure ML-safe copy with absolute path resolution
            output_path = Path(args.champion_out).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"  📂 Champion output path: {output_path}")
            
            # Copy all files recursively
            copied_count = 0
            if src.is_file():
                # Single file model
                dest = output_path / src.name
                shutil.copy2(src, dest)
                copied_count = 1
                print(f"  ✅ Copied model file: {src.name} ({src.stat().st_size:,} bytes)")
            elif src.is_dir():
                # Directory with model files - copy recursively
                for src_file in src.rglob('*'):
                    if src_file.is_file():
                        rel_path = src_file.relative_to(src)
                        dest_file = output_path / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dest_file)
                        copied_count += 1
                print(f"  ✅ Copied {copied_count} model files to {output_path}")
            
            report["model_copied"] = True
            report["files_copied"] = copied_count
            (output_path / "selection_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": metrics.get("status", "success"),
                        "phase": "phasec",
                        "candidate_id": (
                            metrics.get("candidate_id")
                            or (metrics.get("model_bundle") or {}).get("candidate_id")
                        ),
                        "algorithm": metrics.get("algorithm"),
                        "selection_score": metrics.get("best_score"),
                        "metric_name": metrics.get("selection_metric"),
                        "selection_evidence": metrics.get("selection_evidence"),
                        "split_fingerprint": metrics.get(
                            "split_fingerprint"
                        ),
                        "total_folds": metrics.get("total_folds"),
                        "same_family": metrics.get("same_family"),
                        "execution_id": metrics.get("execution_id"),
                        "mlflow_parent_run_id": metrics.get(
                            "mlflow_parent_run_id"
                        ),
                        "mlflow_child_run_id": metrics.get(
                            "mlflow_child_run_id"
                        ),
                        "lineage": (
                            (metrics.get("model_bundle") or {}).get("lineage")
                            or {
                                "execution_id": metrics.get("execution_id"),
                                "parent_run_id": metrics.get(
                                    "mlflow_parent_run_id"
                                ),
                                "candidate_run_id": metrics.get(
                                    "mlflow_child_run_id"
                                ),
                            }
                        ),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            
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
            if not has_exact_model_bundle(output_path):
                (output_path / ".no_model").write_text(
                    "Source had no exact ModelBundle"
                )
                print("  ⚠️  No exact ModelBundle — wrote .no_model sentinel")
                report["model_copied"] = False
        except Exception as e:
            print(f"  ❌ Error copying model: {e}")
            import traceback
            traceback.print_exc()
            report["model_copy_error"] = str(e)
    else:
        # T4: Create output folder with sentinel file so downstream knows no model was produced
        output_path = Path(args.champion_out).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / ".no_model").write_text(
            "HPO produced no exact successful ModelBundle"
        )
        print(
            f"  ⚠️  HPO output is skipped/incomplete: {src} "
            "— wrote .no_model sentinel"
        )
    
    # Update report with final status (absolute path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Final report updated: {report_path}")

    # ── Emit stage signal ──────────────────────────────────────────────
    _elapsed = time.time() - _t0
    _hpo_score = metrics.get("best_score")
    try:
        import yaml
        with open(args.config, "r") as f:
            _cfg = yaml.safe_load(f)
        _task = _cfg.get("task_type", "classification")
    except Exception:
        _task = "classification"
    try:
        sig = StageSignal(
            stage_name="phasec_aggregate",
            stage_id="S09",
            task_type=_task,
            config_name=Path(args.config).name,
            candidate_count_in=1,
            candidate_count_out=1 if _hpo_score is not None else 0,
            best_score=float(_hpo_score) if _hpo_score is not None else None,
            best_metric_name="best_score",
            compute_time_sec=round(_elapsed, 2),
            recommendation="proceed" if _hpo_score is not None else "stop",
            recommendation_reason="HPO optimised model ready" if _hpo_score else "HPO produced no score",
        )
        write_stage_signal(sig, out_dir="outputs", filename="phasec_stage_signal.json")
    except Exception as _sig_err:
        print(f"⚠️  Stage signal write failed (non-fatal): {_sig_err}")

    # ── Candidate Ledger ──────────────────────────────────────────────────
    try:
        _norm = normalize_metrics(_task, {"best_score": _hpo_score} if _hpo_score is not None else {})
        _row = make_row(
            stage="phase_c", step_name="s09", engine="optuna",
            candidate_id="phasec_hpo_champion",
            task_type=_task,
            dataset_id=Path(args.config).name,
            status="ok" if _hpo_score is not None else "failed",
            compute_time_sec=round(_elapsed, 2),
            source_path="src/steps/aggregate_phasec.py",
            recipe_name=metrics.get("algorithm", "optuna_hpo"),
            is_stage_best=True,
            params_json=json.dumps(metrics.get("best_params", {}), default=str),
            **_norm,
        )
        write_stage_table(
            [_row],
            csv_path="outputs/s09_candidates.csv",
            parquet_path="outputs/s09_candidates.parquet",
        )
    except Exception as _ledger_err:
        print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")


if __name__ == "__main__":
    main()
