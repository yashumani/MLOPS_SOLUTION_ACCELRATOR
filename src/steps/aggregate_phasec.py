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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hpo_metrics_json", required=True)
    parser.add_argument("--optimized_model", required=True)
    parser.add_argument("--report_out", required=True)
    parser.add_argument("--champion_out", required=True)
    args = parser.parse_args()

    metrics = load_json(args.hpo_metrics_json) or {}
    report = {
        "phase": "C",
        "selection": {
            "score": metrics.get("best_score"),
            "params": metrics.get("best_params")
        },
        "model_copied": False,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    src = Path(args.optimized_model)
    print(f"🏆 Phase C Aggregate: HPO best_score={metrics.get('best_score')}")
    
    if src.exists():
        import shutil
        try:
            # Handle both file and folder cases
            output_path = Path(args.champion_out)
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
            
            if src.is_dir():
                # Copy folder structure
                shutil.copytree(src, output_path)
                print(f"  ✅ Champion model (folder) copied to {output_path}")
            else:
                # Copy single file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, output_path)
                print(f"  ✅ Champion model (file) copied to {output_path}")
            report["model_copied"] = True
        except Exception as e:
            print(f"  ❌ Error copying model: {e}")
            report["model_copy_error"] = str(e)
    else:
        # Create empty output folder
        Path(args.champion_out).mkdir(parents=True, exist_ok=True)
        print(f"  ⚠️  HPO model path missing: {src}")
    
    # Update report with final status
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
