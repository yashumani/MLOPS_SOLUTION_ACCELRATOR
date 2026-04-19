import argparse
import json
from pathlib import Path
import sys

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


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
        # Try leaderboard structure
        lb = manifest.get("leaderboard")
        if lb and isinstance(lb, dict):
            col = "Accuracy" if task == "classification" else "R2"
            if col in lb and isinstance(lb[col], dict):
                try:
                    first_key = sorted(lb[col].keys())[0]
                    score = float(lb[col][first_key])
                    return score, f"{recipe_key} PyCaret leaderboard[{col}]={score}"
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
                metric_field = "Accuracy" if task == "classification" else "R2"
                if metric_field in metrics_dict:
                    try:
                        score = float(metrics_dict[metric_field])
                        return score, f"{recipe_key} PyCaret {key}[{metric_field}]={score}"
                    except (ValueError, TypeError):
                        pass
        
        return None, f"{recipe_key} PyCaret: no leaderboard/metrics found"
    
    return None, f"{recipe_key} Unknown engine: {engine}"


def main():
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
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    if champion_path and champion_path.exists():
        import shutil
        try:
            # Check if it's a directory or file and copy accordingly
            # Use dirs_exist_ok=True because Azure ML pre-creates output directories as mount points
            output_path = Path(args.champion_out)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if champion_path.is_dir():
                # Copy folder structure
                shutil.copytree(champion_path, output_path, dirs_exist_ok=True)
                print(f"  ✅ Champion model (folder) copied to {output_path}")
            else:
                # Copy single file
                shutil.copy2(champion_path, output_path)
                print(f"  ✅ Champion model (file) copied to {output_path}")
            report["model_copied"] = True
        except Exception as e:
            print(f"  ❌ Error copying model: {e}")
            report["model_copy_error"] = str(e)
    else:
        # Create empty output folder
        Path(args.champion_out).mkdir(parents=True, exist_ok=True)
        print(f"  ⚠️  Champion path missing or doesn't exist: {champion_path}")
    
    # Update report with final status
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
