"""
Stage Signals — Shared signal framework for bounded-tournament pipeline stages.

Every major pipeline stage (baseline aggregate, Phase B, Phase C, final evaluation)
emits a `stage_signal.json` into its outputs/ folder. This module provides:

  - StageSignal dataclass with required + optional fields
  - write_stage_signal()  — atomic JSON write
  - load_stage_signal()   — read back from file
  - summarize_signals()   — merge multiple signals into a pandas DataFrame

Signal Spec (required fields):
  stage_name, stage_id, task_type, config_name, timestamp,
  candidate_count_in, candidate_count_out,
  best_score, best_metric_name,
  delta_vs_baseline, stability_signal, failure_rate,
  compute_time_sec, topk_gap, recommendation

Optional fields:
  cost_estimate, resource_usage, cache_hit_rate, extra

Usage:
    from utils.stage_signals import StageSignal, write_stage_signal

    sig = StageSignal(
        stage_name="baseline_aggregate",
        stage_id="S05z",
        task_type="classification",
        config_name="config_classification_telecom_churn_azureml.yml",
        candidate_count_in=2,
        candidate_count_out=1,
        best_score=0.95,
        best_metric_name="Accuracy",
        recommendation="proceed",
        recommendation_reason="Baseline accuracy 0.95 — sufficient for Phase B",
    )
    write_stage_signal(sig, out_dir="outputs", filename="baseline_stage_signal.json")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageSignal:
    """Canonical stage signal emitted after each major pipeline stage."""

    # ---- Required identity ----
    stage_name: str                     # e.g. "baseline_aggregate"
    stage_id: str                       # e.g. "S05z"
    task_type: str                      # classification | regression | clustering
    config_name: str                    # config YAML filename

    # ---- Candidate flow ----
    candidate_count_in: int = 0         # candidates entering this stage
    candidate_count_out: int = 0        # candidates surviving this stage

    # ---- Score ----
    best_score: Optional[float] = None
    best_metric_name: str = ""          # e.g. "Accuracy", "R2", "silhouette_score"

    # ---- Deltas ----
    delta_vs_baseline: Optional[float] = None   # improvement over baseline (null if N/A)

    # ---- Stability ----
    stability_signal: Optional[float] = None    # std/var across folds/trials
    stability_reason: str = ""                  # why null, or description

    # ---- Failure rate ----
    failure_rate: Optional[float] = None        # failed / total (0.0 – 1.0)
    failure_count: int = 0
    total_count: int = 0

    # ---- Compute ----
    compute_time_sec: Optional[float] = None
    start_timestamp: str = ""
    end_timestamp: str = ""

    # ---- Tournament quality ----
    topk_gap: Optional[float] = None            # best − 2nd best
    bounded_tournament: bool = True             # always True for V3

    # ---- Recommendation ----
    recommendation: str = "proceed"             # proceed | stop
    recommendation_reason: str = ""

    # ---- Optional extras ----
    cost_estimate: Optional[float] = None
    resource_usage: Optional[str] = None
    cache_hit_rate: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- Auto-populated ----
    timestamp: str = ""
    pipeline_version: str = "v3"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_stage_signal(
    signal: StageSignal,
    out_dir: str = "outputs",
    filename: str = "stage_signal.json",
) -> Path:
    """Write stage signal as JSON into *out_dir*/*filename*. Never throws.

    Returns the Path written (or intended) even on error.
    """
    out_path = Path(out_dir) / filename
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temp file
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(signal), f, indent=2, default=str)
        os.replace(str(tmp), str(out_path))
        print(f"📡 Stage signal written: {out_path}")
    except Exception as exc:
        print(f"⚠️  write_stage_signal({out_path}): {exc}")
        # Clean up temp
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return out_path


def load_stage_signal(path: str) -> Optional[StageSignal]:
    """Load a StageSignal from a JSON file. Returns None on any error."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Remove unknown keys to be forward-compatible
        known = {f.name for f in StageSignal.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return StageSignal(**filtered)
    except Exception as exc:
        print(f"⚠️  load_stage_signal({path}): {exc}")
        return None


def summarize_signals(signal_paths: List[str]) -> "pd.DataFrame":
    """Load multiple stage signals and return a summary DataFrame.

    Columns: stage_id, stage_name, best_score, best_metric_name,
             candidate_in, candidate_out, delta_vs_baseline, failure_rate,
             compute_time_sec, recommendation, timestamp
    """
    import pandas as pd

    rows = []
    for p in signal_paths:
        sig = load_stage_signal(p)
        if sig is None:
            rows.append({"source_file": p, "stage_id": "?", "stage_name": "LOAD_FAILED"})
            continue
        rows.append({
            "source_file": p,
            "stage_id": sig.stage_id,
            "stage_name": sig.stage_name,
            "task_type": sig.task_type,
            "best_score": sig.best_score,
            "best_metric_name": sig.best_metric_name,
            "candidate_in": sig.candidate_count_in,
            "candidate_out": sig.candidate_count_out,
            "delta_vs_baseline": sig.delta_vs_baseline,
            "failure_rate": sig.failure_rate,
            "compute_time_sec": sig.compute_time_sec,
            "topk_gap": sig.topk_gap,
            "recommendation": sig.recommendation,
            "recommendation_reason": sig.recommendation_reason,
            "timestamp": sig.timestamp,
        })
    return pd.DataFrame(rows)
