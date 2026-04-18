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


def select_champion(pycaret_manifest, flaml_manifest, task: str = "classification"):
    """Select best model from pycaret and flaml baselines.
    
    Returns: {"source": "pycaret"|"flaml"|None, "score": float|None, "reason": str}
    """
    best = {"source": None, "score": None, "reason": ""}
    
    def extract_score(m, engine_name):
        """Extract best score from manifest, return (score, reason) or (None, reason)"""
        if not m:
            return None, f"{engine_name} manifest missing"
        
        engine = m.get("engine", "unknown")
        
        # FLAML: best_metric field
        if engine == "flaml":
            score = m.get("best_metric")
            if score is not None:
                try:
                    return float(score), f"FLAML best_metric={score}"
                except (ValueError, TypeError):
                    return None, f"FLAML best_metric not numeric: {score}"
            return None, "FLAML missing best_metric"
        
        # PyCaret: try leaderboard first, then metrics_dict
        if engine == "pycaret":
            # Try leaderboard structure (dict of columns)
            lb = m.get("leaderboard")
            if lb and isinstance(lb, dict):
                col = "Accuracy" if task == "classification" else "R2"
                if col in lb and isinstance(lb[col], dict):
                    try:
                        first_key = sorted(lb[col].keys())[0]
                        score = float(lb[col][first_key])
                        return score, f"PyCaret leaderboard[{col}]={score}"
                    except (ValueError, TypeError, KeyError, IndexError):
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
                    metric_field = "Accuracy" if task == "classification" else "R2"
                    if metric_field in metrics_dict:
                        try:
                            score = float(metrics_dict[metric_field])
                            return score, f"PyCaret {key}[{metric_field}]={score}"
                        except (ValueError, TypeError):
                            pass
            
            return None, "PyCaret: no leaderboard/metrics found"
        
        return None, f"Unknown engine: {engine}"
    
    ps, ps_reason = extract_score(pycaret_manifest, "PyCaret")
    fs, fs_reason = extract_score(flaml_manifest, "FLAML")
    
    # Select best (higher is better for both accuracy and FLAML auto-metric)
    if fs is not None and (ps is None or fs >= ps):
        best["source"] = "flaml"
        best["score"] = fs
        best["reason"] = fs_reason
    elif ps is not None:
        best["source"] = "pycaret"
        best["score"] = ps
        best["reason"] = ps_reason
    else:
        best["reason"] = f"No valid scores found. PyCaret: {ps_reason}. FLAML: {fs_reason}"
    
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pycaret_manifest", required=True)
    parser.add_argument("--pycaret_model", required=True)
    parser.add_argument("--flaml_manifest", required=True)
    parser.add_argument("--flaml_model", required=True)
    parser.add_argument("--report_out", required=True)
    parser.add_argument("--champion_out", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"

    pman = load_json(args.pycaret_manifest)
    fman = load_json(args.flaml_manifest)

    champion = select_champion(pman, fman, task=task_type)
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
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    if model_src:
        import shutil
        try:
            # Copy model folder (source) to champion output folder
            champion_path = Path(args.champion_out)
            if champion_path.exists():
                shutil.rmtree(champion_path)
            shutil.copytree(model_src, champion_path)
            report["model_copied"] = True
            print(f"  ✅ Champion model copied to {champion_path}")
        except Exception as e:
            print(f"  ❌ Error copying model: {e}")
            report["model_copy_error"] = str(e)
    else:
        # Create empty output folder if no model selected
        Path(args.champion_out).mkdir(parents=True, exist_ok=True)
        print(f"  ℹ️  Created empty champion folder (no model found)")
    
    # Update report with final status
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
