import os
import time
import hashlib
from typing import Dict, Optional

import mlflow


def generate_execution_id(dataset_name: str, task_type: str, preset: str, config_bytes: bytes) -> str:
    """Deterministic short execution_id based on key elements + timestamp."""
    timestamp = str(int(time.time()))
    payload = f"{dataset_name}|{task_type}|{preset}|{timestamp}|".encode() + config_bytes
    return hashlib.sha1(payload).hexdigest()[:12]


def start_parent_run(experiment_name: str, tags: Optional[Dict[str, str]] = None):
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(tags=tags or {})
    return run


def start_child_run(parent_run_id: str, tags: Optional[Dict[str, str]] = None):
    return mlflow.start_run(run_id=None, tags=tags or {}, nested=True)


def set_tags(tags: Dict[str, str]):
    for k, v in tags.items():
        mlflow.set_tag(k, v)


def log_dict(name: str, data: Dict):
    mlflow.log_dict(data, name)


def end_run():
    mlflow.end_run()
