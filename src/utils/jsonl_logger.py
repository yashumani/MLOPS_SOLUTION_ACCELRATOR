"""
JSONL Metric and Parameter Logger for Azure ML
Works with command jobs and pipeline components
Azure ML automatically captures files from outputs/ directory

This replaces MLflow logging for compatibility with command jobs.
Uses JSONL format (one JSON object per line) for easy parsing.
"""

import json as json_lib
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def _atomic_append_jsonl(path: Path, obj: dict) -> None:
    """Append one JSONL record via same-directory temp file + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json_lib.dumps(obj) + "\n"
    existing = path.read_text() if path.exists() else ""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as tmp_file:
            tmp_file.write(existing)
            tmp_file.write(line)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class AzureMLJSONLLogger:
    """
    Safe JSONL logger for Azure ML command jobs and pipeline components
    Logs metrics and parameters to outputs/ directory
    """
    
    def __init__(self, metrics_file: str = "outputs/metrics.jsonl", 
                 params_file: str = "outputs/params.jsonl"):
        """Initialize logger with output file paths."""
        self.metrics_file = Path(metrics_file)
        self.params_file = Path(params_file)
        
        # Create outputs directory
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.params_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric to metrics.jsonl
        
        Args:
            name: Metric name
            value: Metric value (float/int)
            step: Optional step number (not used but kept for API compatibility)
        """
        try:
            metric_obj = {"name": name, "value": float(value)}
            if step is not None:
                metric_obj["step"] = step
            
            _atomic_append_jsonl(self.metrics_file, metric_obj)
        except Exception as e:
            print(f"⚠️  Failed to log metric '{name}': {e}")
    
    def log_param(self, name: str, value: Any) -> None:
        """
        Log a parameter to params.jsonl
        
        Args:
            name: Parameter name
            value: Parameter value (converted to string)
        """
        try:
            param_obj = {"name": name, "value": str(value)}
            
            _atomic_append_jsonl(self.params_file, param_obj)
        except Exception as e:
            print(f"⚠️  Failed to log param '{name}': {e}")
    
    def log_metrics(self, metrics_dict: dict) -> None:
        """
        Log multiple metrics from a dictionary
        
        Args:
            metrics_dict: Dictionary of {metric_name: metric_value}
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value)
    
    def log_params(self, params_dict: dict) -> None:
        """
        Log multiple parameters from a dictionary
        
        Args:
            params_dict: Dictionary of {param_name: param_value}
        """
        for name, value in params_dict.items():
            self.log_param(name, value)
    
    def log_artifact(self, artifact_path: str) -> None:
        """
        Log artifact path (Azure ML auto-captures outputs/ directory)
        This is a no-op in JSONL logging but kept for API compatibility
        
        Args:
            artifact_path: Path to artifact file (should be in outputs/)
        """
        print(f"📁 Artifact will be auto-captured: {artifact_path}")
    
    def end_run(self) -> None:
        """
        End logging session
        This is a no-op in JSONL logging but kept for API compatibility
        """
        pass


# Convenience functions for backward compatibility with v4_main_simple pattern

def log_metric_to_azureml(name: str, value: Any) -> None:
    """
    Log a metric to outputs/metrics.jsonl (convenience function)
    Azure ML automatically captures files from outputs/
    """
    metrics_dir = Path("outputs")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        _atomic_append_jsonl(metrics_dir / "metrics.jsonl", {"name": name, "value": value})
    except Exception as e:
        print(f"⚠️  Failed to log metric '{name}': {e}")


def log_param_to_azureml(name: str, value: Any) -> None:
    """
    Log a parameter to outputs/params.jsonl (convenience function)
    Azure ML automatically captures files from outputs/
    """
    params_dir = Path("outputs")
    params_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        _atomic_append_jsonl(params_dir / "params.jsonl", {"name": name, "value": str(value)})
    except Exception as e:
        print(f"⚠️  Failed to log param '{name}': {e}")
