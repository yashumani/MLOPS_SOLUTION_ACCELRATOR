"""
Candidate Ledger — Unified per-candidate tracking across all pipeline stages.

Every evaluated candidate (baseline model, Phase B recipe×engine, Phase C trial,
final evaluation comparison) gets ONE row in a CSV/Parquet ledger.  A user can
open ``outputs/all_candidates.csv`` after a pipeline run and see every candidate
with its parameters, metrics, rank, and provenance.

This module provides:

  - CandidateRow   — dict-based schema with validation
  - normalize_metrics()  — maps task-specific metrics to canonical column names
  - write_candidate_artifacts()  — writes inputs.json / metrics.json / status.json
  - append_rows_to_stage_table()  — batch-append to stage-level CSV (+ Parquet)
  - merge_ledgers()  — merge multiple stage CSVs into one consolidated file
  - sha256_file()  — hash a recipe YAML for provenance
  - infer_run_metadata()  — best-effort read of AzureML env vars

Design principles:
  - Never crash the pipeline — all writes are try/except guarded
  - Filesystem outputs are the source of truth (no MLflow dependency)
  - Works identically for classification / regression / clustering
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

IDENTITY_COLS = [
    "dataset_id", "task_type", "preset", "pipeline_version",
    "stage", "step_name", "engine", "candidate_id",
    "run_id", "timestamp_utc",
]

INPUT_COLS = [
    "recipe_name", "recipe_hash", "params_json", "pipeline_dims_json",
]

# Metric columns — all nullable.  Stages populate only the subset that applies.
CLASSIFICATION_METRIC_COLS = [
    "accuracy", "roc_auc", "f1", "precision", "recall", "logloss",
]
REGRESSION_METRIC_COLS = [
    "rmse", "mae", "r2", "mse",
]
CLUSTERING_METRIC_COLS = [
    "silhouette", "davies_bouldin", "calinski_harabasz",
]
OUTPUT_COLS = [
    "primary_metric_name", "primary_metric_value",
] + CLASSIFICATION_METRIC_COLS + REGRESSION_METRIC_COLS + CLUSTERING_METRIC_COLS

SIGNAL_COLS = [
    "candidate_rank", "delta_vs_baseline_best", "is_stage_best",
    "is_final_champion", "status", "failure_reason", "compute_time_sec",
]

PROVENANCE_COLS = [
    "source_path", "artifacts_json",
]

# Tournament columns — added post-hoc by AIM-Tournament in final_evaluation
TOURNAMENT_COLS = [
    "rank_accuracy", "rank_roc_auc", "rank_f1", "rank_precision", "rank_recall", "rank_logloss",
    "rank_r2", "rank_rmse", "rank_mae",
    "rank_silhouette", "rank_davies_bouldin", "rank_calinski_harabasz",
    "utility_score", "utility_rank", "pareto_optimal",
]

ALL_COLUMNS = IDENTITY_COLS + INPUT_COLS + OUTPUT_COLS + SIGNAL_COLS + PROVENANCE_COLS + TOURNAMENT_COLS

# Primary metric per task type
_PRIMARY_METRIC = {
    "classification": ("AUC", "roc_auc"),
    "regression": ("R2", "r2"),
    "clustering": ("silhouette_score", "silhouette"),
}


# ---------------------------------------------------------------------------
# CandidateRow helpers
# ---------------------------------------------------------------------------

def empty_row() -> Dict[str, Any]:
    """Return a dict with all canonical columns set to None."""
    return {c: None for c in ALL_COLUMNS}


def make_row(
    *,
    stage: str,
    step_name: str,
    engine: str,
    candidate_id: str,
    task_type: str = "classification",
    dataset_id: str = "",
    preset: str = "",
    status: str = "ok",
    failure_reason: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """Create a ledger row with required identity fields filled in.

    Any additional keyword args are merged (only known columns are kept).
    """
    row = empty_row()
    row.update({
        "stage": stage,
        "step_name": step_name,
        "engine": engine,
        "candidate_id": candidate_id,
        "task_type": task_type,
        "dataset_id": dataset_id,
        "preset": preset,
        "pipeline_version": "v3",
        "status": status,
        "failure_reason": failure_reason,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    # Merge run metadata from environment (best effort)
    meta = infer_run_metadata()
    if meta.get("run_id") and not row.get("run_id"):
        row["run_id"] = meta["run_id"]
    # Merge caller-supplied extras (only known cols)
    for k, v in kwargs.items():
        if k in row:
            row[k] = v
    return row


def normalize_metrics(task_type: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map a raw metrics dict (with mixed-case keys) to canonical ledger columns.

    Returns a dict whose keys are a subset of ALL_COLUMNS.
    Also sets primary_metric_name / primary_metric_value.
    """
    out: Dict[str, Any] = {}
    low = {k.lower().replace(" ", "_"): v for k, v in raw.items()}

    # Classification
    for col in CLASSIFICATION_METRIC_COLS:
        if col in low:
            out[col] = _safe_float(low[col])
    # Also accept "auc" for roc_auc
    if "auc" in low and "roc_auc" not in out:
        out["roc_auc"] = _safe_float(low["auc"])
    if "pr_auc" in low and "roc_auc" not in out:
        pass  # keep separate

    # Regression
    for col in REGRESSION_METRIC_COLS:
        if col in low:
            out[col] = _safe_float(low[col])

    # Clustering
    for col in CLUSTERING_METRIC_COLS:
        if col in low:
            out[col] = _safe_float(low[col])
    if "silhouette_score" in low:
        out["silhouette"] = _safe_float(low["silhouette_score"])

    # Primary metric
    pm_label, pm_col = _PRIMARY_METRIC.get(task_type, ("Accuracy", "accuracy"))
    out["primary_metric_name"] = pm_label
    out["primary_metric_value"] = out.get(pm_col)

    return out


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def write_candidate_artifacts(
    base_dir: Union[str, Path],
    candidate_row: Dict[str, Any],
    inputs_dict: Optional[Dict] = None,
    metrics_dict: Optional[Dict] = None,
    status_dict: Optional[Dict] = None,
) -> Optional[Path]:
    """Write per-candidate artifact files.

    Creates::
        <base_dir>/candidates/<stage>/<candidate_id>/inputs.json
        <base_dir>/candidates/<stage>/<candidate_id>/metrics.json
        <base_dir>/candidates/<stage>/<candidate_id>/status.json

    Returns the candidate folder Path, or None on error.
    """
    try:
        stage = candidate_row.get("stage", "unknown")
        cid = candidate_row.get("candidate_id", "unknown")
        # Sanitise candidate_id for filesystem
        safe_cid = str(cid).replace("/", "_").replace("\\", "_").replace(" ", "_")[:120]
        cand_dir = Path(base_dir) / "candidates" / stage / safe_cid
        cand_dir.mkdir(parents=True, exist_ok=True)

        if inputs_dict is not None:
            _safe_json_dump(inputs_dict, cand_dir / "inputs.json")
        if metrics_dict is not None:
            _safe_json_dump(metrics_dict, cand_dir / "metrics.json")

        # Build status from row if not provided
        if status_dict is None:
            status_dict = {
                "status": candidate_row.get("status", "ok"),
                "failure_reason": candidate_row.get("failure_reason", ""),
                "candidate_rank": candidate_row.get("candidate_rank"),
                "is_stage_best": candidate_row.get("is_stage_best", False),
                "is_final_champion": candidate_row.get("is_final_champion", False),
                "compute_time_sec": candidate_row.get("compute_time_sec"),
                "timestamp_utc": candidate_row.get("timestamp_utc"),
            }
        _safe_json_dump(status_dict, cand_dir / "status.json")

        return cand_dir
    except Exception as exc:
        print(f"⚠️  write_candidate_artifacts failed: {exc}")
        return None


def append_rows_to_stage_table(
    rows: List[Dict[str, Any]],
    csv_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
) -> None:
    """Append candidate rows to a stage-level CSV (and optionally Parquet).

    Creates the file with a header if it doesn't exist, appends otherwise.
    """
    if not rows:
        return
    csv_path = Path(csv_path)
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"📋 Ledger: appended {len(rows)} rows → {csv_path}")
    except Exception as exc:
        print(f"⚠️  append_rows_to_stage_table CSV failed: {exc}")

    # Parquet (optional)
    if parquet_path:
        _write_parquet(rows, parquet_path, append=True)


def write_stage_table(
    rows: List[Dict[str, Any]],
    csv_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write (overwrite) a stage-level ledger table."""
    if not rows:
        return
    csv_path = Path(csv_path)
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"📋 Ledger: wrote {len(rows)} rows → {csv_path}")
    except Exception as exc:
        print(f"⚠️  write_stage_table CSV failed: {exc}")

    if parquet_path:
        _write_parquet(rows, parquet_path, append=False)


def merge_ledgers(
    ledger_paths: List[Union[str, Path]],
    out_csv: Union[str, Path],
    out_parquet: Optional[Union[str, Path]] = None,
) -> int:
    """Merge multiple stage-level CSVs into one consolidated ledger.

    Returns total row count written.
    """
    import pandas as pd
    frames = []
    for p in ledger_paths:
        p = Path(p)
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p, dtype=str)
                frames.append(df)
                print(f"  📖 Loaded {len(df)} rows from {p.name}")
            except Exception as exc:
                print(f"  ⚠️  Failed to read {p}: {exc}")
    if not frames:
        print("  ⚠️  No ledger data to merge")
        return 0

    merged = pd.concat(frames, ignore_index=True)
    # Ensure all canonical columns present
    for col in ALL_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[ALL_COLUMNS]  # reorder

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"📋 Merged ledger: {len(merged)} rows → {out_csv}")

    if out_parquet:
        _write_parquet_df(merged, out_parquet)

    return len(merged)


def build_summary(
    csv_path: Union[str, Path],
) -> Dict[str, Any]:
    """Build a JSON-serialisable summary from the merged ledger CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str)
    df["primary_metric_value"] = pd.to_numeric(df["primary_metric_value"], errors="coerce")

    summary: Dict[str, Any] = {
        "total_candidates": len(df),
        "by_stage": {},
        "by_status": df["status"].value_counts().to_dict(),
        "champion": None,
    }
    for stage, grp in df.groupby("stage"):
        best_idx = grp["primary_metric_value"].idxmax()
        best_row = grp.loc[best_idx] if pd.notna(best_idx) else None
        summary["by_stage"][stage] = {
            "count": len(grp),
            "ok": int((grp["status"] == "ok").sum()),
            "failed": int((grp["status"] == "failed").sum()),
            "best_candidate_id": best_row["candidate_id"] if best_row is not None else None,
            "best_score": float(best_row["primary_metric_value"]) if best_row is not None and pd.notna(best_row["primary_metric_value"]) else None,
        }
    # Final champion
    champ = df[df["is_final_champion"] == "True"]
    if len(champ) > 0:
        c = champ.iloc[0]
        summary["champion"] = {
            "candidate_id": c.get("candidate_id"),
            "stage": c.get("stage"),
            "engine": c.get("engine"),
            "primary_metric_value": _safe_float(c.get("primary_metric_value")),
        }
    return summary


def build_readme_md(summary: Dict[str, Any]) -> str:
    """Generate a human-readable Markdown describing the ledger."""
    lines = [
        "# Candidate Ledger — All Evaluated Candidates",
        "",
        "This folder contains the consolidated candidate ledger from a V3 bounded-tournament pipeline run.",
        "",
        "## Files",
        "",
        "| File | Format | Description |",
        "|------|--------|-------------|",
        "| `all_candidates.csv` | CSV | Every evaluated candidate, one row per candidate |",
        "| `all_candidates.parquet` | Parquet | Same data in columnar format (if pyarrow available) |",
        "| `all_candidates_summary.json` | JSON | Counts by stage/status, best per stage, champion |",
        "| `all_candidates_README.md` | Markdown | This file |",
        "",
        "## Schema",
        "",
        "| Category | Columns |",
        "|----------|---------|",
        f"| Identity | {', '.join(IDENTITY_COLS)} |",
        f"| Inputs | {', '.join(INPUT_COLS)} |",
        f"| Outputs | {', '.join(OUTPUT_COLS)} |",
        f"| Signals | {', '.join(SIGNAL_COLS)} |",
        f"| Provenance | {', '.join(PROVENANCE_COLS)} |",
        "",
        "## Summary",
        "",
        f"- **Total candidates**: {summary.get('total_candidates', '?')}",
        f"- **By status**: {summary.get('by_status', {})}",
    ]
    for stage, info in summary.get("by_stage", {}).items():
        lines.append(f"- **{stage}**: {info.get('count', 0)} candidates, "
                      f"best={info.get('best_score', '?')}")
    champ = summary.get("champion")
    if champ:
        lines.extend([
            "",
            f"## 🏆 Champion: `{champ.get('candidate_id', '?')}`",
            f"- Stage: {champ.get('stage', '?')}",
            f"- Engine: {champ.get('engine', '?')}",
            f"- Score: {champ.get('primary_metric_value', '?')}",
        ])
    lines.extend([
        "",
        "## Usage",
        "",
        "```python",
        "import pandas as pd",
        "",
        "# CSV",
        "df = pd.read_csv('outputs/all_candidates.csv')",
        "",
        "# Parquet (faster for large ledgers)",
        "df = pd.read_parquet('outputs/all_candidates.parquet')",
        "",
        "# Best per stage",
        "best = df.loc[df.groupby('stage')['primary_metric_value'].idxmax()]",
        "print(best[['stage', 'candidate_id', 'engine', 'primary_metric_value']])",
        "```",
        "",
        "## No MLflow Dependency",
        "",
        "This ledger is written entirely to the filesystem. It does **not** depend on",
        "MLflow model registry or `azureml://` artifact repositories. It works even when",
        "MLflow logging fails or is disabled.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment / provenance helpers
# ---------------------------------------------------------------------------

def infer_run_metadata() -> Dict[str, str]:
    """Best-effort extraction of AzureML run metadata from env vars."""
    return {
        "run_id": os.environ.get("AZUREML_RUN_ID", ""),
        "experiment_name": os.environ.get("AZUREML_EXPERIMENT_NAME", ""),
        "workspace_name": os.environ.get("AZUREML_WORKSPACE_NAME", ""),
    }


def sha256_file(path: Union[str, Path]) -> str:
    """Return hex sha256 digest of a file, or '' on error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_json_dump(obj: Any, path: Path) -> None:
    """Write JSON with stable key ordering and utf-8."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str, ensure_ascii=False)
    except Exception as exc:
        print(f"⚠️  _safe_json_dump({path}): {exc}")


def _write_parquet(rows: List[Dict], path: Union[str, Path], append: bool = False) -> None:
    """Write rows to Parquet, falling back gracefully if pyarrow is missing."""
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        _write_parquet_df(df, path, append=append)
    except Exception as exc:
        print(f"⚠️  Parquet write skipped ({path}): {exc}")


def _write_parquet_df(df, path: Union[str, Path], append: bool = False) -> None:
    """Write a DataFrame to Parquet."""
    path = Path(path)
    try:
        import pyarrow  # noqa: F401
        path.parent.mkdir(parents=True, exist_ok=True)
        if append and path.exists():
            import pandas as pd
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        print(f"📋 Parquet: {len(df)} rows → {path}")
    except ImportError:
        print(f"⚠️  pyarrow not installed — Parquet write skipped for {path}")
    except Exception as exc:
        print(f"⚠️  Parquet write failed ({path}): {exc}")
