#!/usr/bin/env python3
"""Merge bounded Azure qualification waves and run complete-matrix acceptance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_qualification_evidence import verify_qualification_evidence  # noqa: E402


PATH_FIELDS = (
    "pipeline_evidence_dir",
    "registered_model_smoke_submission",
    "registered_model_smoke_evidence",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _single_manifest(root: Path) -> Path:
    matches = sorted(root.rglob("qualification-evidence-manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Wave input {root} must contain exactly one evidence manifest; "
            f"found {len(matches)}"
        )
    return matches[0]


def _resolve_contained(base: Path, value: object, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text or Path(text).is_absolute():
        raise RuntimeError(f"{field} must be a relative path")
    path = (base / text).resolve()
    if not path.is_relative_to(base.resolve()) or not path.exists():
        raise RuntimeError(f"{field} escapes or is missing from its wave")
    return path


def _asset_identity(payload: dict[str, Any]) -> str:
    selected = {
        field: payload.get(field)
        for field in (
            "schema_version",
            "asset_version",
            "asset_count",
            "scenario_count",
            "all_passed",
            "errors",
            "assets",
        )
    }
    return json.dumps(selected, sort_keys=True, separators=(",", ":"))


def merge_waves(wave_dirs: list[Path], output_dir: Path) -> tuple[Path, Path]:
    if not 1 <= len(wave_dirs) <= 15:
        raise ValueError("Complete acceptance requires between 1 and 15 wave inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    monitors: list[str] = []
    scenarios: list[dict[str, Any]] = []
    release_candidates: set[str] = set()
    archive_hashes: set[str] = set()
    asset_identities: set[str] = set()
    selected_asset_path: str | None = None

    for index, raw_wave in enumerate(wave_dirs, start=1):
        source_manifest = _single_manifest(raw_wave.resolve())
        partial = verify_qualification_evidence(source_manifest)
        if partial.get("state") != "passed":
            raise RuntimeError(f"Wave {index} failed partial evidence verification")
        source_root = source_manifest.parent
        target_root = output_dir / "waves" / f"{index:02d}"
        shutil.copytree(source_root, target_root)
        payload = _read_json(target_root / source_manifest.name)

        release_candidates.add(
            json.dumps(payload.get("release_candidate"), sort_keys=True)
        )
        archive_hashes.add(str(payload.get("source_archive_sha256") or ""))
        asset_path = _resolve_contained(
            target_root,
            payload.get("data_asset_audit"),
            field=f"wave[{index}].data_asset_audit",
        )
        asset_identities.add(_asset_identity(_read_json(asset_path)))
        if selected_asset_path is None:
            selected_asset_path = asset_path.relative_to(output_dir).as_posix()

        raw_monitors = payload.get("monitor_summaries")
        if not isinstance(raw_monitors, list) or not raw_monitors:
            raise RuntimeError(f"Wave {index} has no monitor summary")
        for monitor_index, value in enumerate(raw_monitors, start=1):
            monitor = _resolve_contained(
                target_root,
                value,
                field=f"wave[{index}].monitor_summaries[{monitor_index}]",
            )
            monitors.append(monitor.relative_to(output_dir).as_posix())

        raw_scenarios = payload.get("scenarios")
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise RuntimeError(f"Wave {index} has no scenarios")
        for scenario_index, raw_scenario in enumerate(raw_scenarios, start=1):
            if not isinstance(raw_scenario, dict):
                raise RuntimeError(f"Wave {index} has an invalid scenario entry")
            scenario = dict(raw_scenario)
            for field in PATH_FIELDS:
                source_path = _resolve_contained(
                    target_root,
                    scenario.get(field),
                    field=f"wave[{index}].scenarios[{scenario_index}].{field}",
                )
                scenario[field] = source_path.relative_to(output_dir).as_posix()
            scenarios.append(scenario)

    if len(release_candidates) != 1 or len(archive_hashes) != 1:
        raise RuntimeError("Qualification waves do not share one frozen source identity")
    if len(asset_identities) != 1:
        raise RuntimeError("Qualification waves do not share one data asset identity")
    if len(scenarios) != 15 or len(
        {str(item.get("scenario_id") or "") for item in scenarios}
    ) != 15:
        raise RuntimeError("Complete acceptance requires 15 unique scenarios")

    manifest = {
        "schema_version": "1.0",
        "monitor_summaries": monitors,
        "data_asset_audit": selected_asset_path,
        "release_candidate": json.loads(next(iter(release_candidates))),
        "source_archive_sha256": next(iter(archive_hashes)),
        "scenarios": scenarios,
    }
    manifest_path = output_dir / "qualification-final-manifest.json"
    _write_json_atomic(manifest_path, manifest)
    report = verify_qualification_evidence(
        manifest_path,
        require_complete_matrix=True,
    )
    report_path = output_dir / "qualification-final-report.json"
    _write_json_atomic(report_path, report)
    if report.get("state") != "passed" or report.get(
        "release_matrix_accepted"
    ) is not True:
        raise RuntimeError("Complete qualification matrix evidence was rejected")
    return manifest_path, report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = {
        "schema_version": "1.0",
        "azureml_run_id": os.environ.get("AZUREML_RUN_ID"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    try:
        if not os.environ.get("AZUREML_RUN_ID"):
            raise RuntimeError("Qualification merge must execute inside Azure ML")
        manifest, report = merge_waves(
            [path.resolve() for path in args.wave_dir],
            args.output_dir.resolve(),
        )
        summary.update(
            {
                "status": "passed",
                "manifest": str(manifest),
                "report": str(report),
                "wave_count": len(args.wave_dir),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve final remote failure
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"Qualification merge failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path = args.output_dir.resolve().parent / "qualification-merge-summary.json"
        _write_json_atomic(summary_path, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
