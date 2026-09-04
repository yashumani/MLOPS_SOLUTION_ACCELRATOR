from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_qualification_wave as wave


SCENARIO = "classification-healthcare-heart-disease"


def _argument(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_wave_runs_all_remote_acceptance_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "wave"
    calls: list[str] = []
    monkeypatch.setenv("AZUREML_RUN_ID", "qualification-wave")

    def audit(arguments: list[str]) -> int:
        calls.append("audit")
        path = Path(_argument(arguments, "--output-json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"all_passed": True}), encoding="utf-8")
        return 0

    def submit(arguments: list[str]) -> int:
        calls.append("submit")
        path = Path(_argument(arguments, "--result-json"))
        path.write_text(json.dumps({"complete": True}), encoding="utf-8")
        return 0

    def monitor(arguments: list[str]) -> int:
        calls.append("monitor")
        path = Path(_argument(arguments, "--output-dir")) / "monitor-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"state": "passed"}), encoding="utf-8")
        return 0

    def collect(_client, **kwargs) -> Path:
        calls.append("collect")
        path = kwargs["output_dir"] / "qualification-evidence-manifest.json"
        path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
        return path

    monkeypatch.setattr(wave, "audit_assets", audit)
    monkeypatch.setattr(wave, "submit_batch", submit)
    monkeypatch.setattr(wave, "monitor_batch", monitor)
    monkeypatch.setattr(wave, "collect_wave", collect)
    monkeypatch.setattr(
        wave,
        "load_azure_context",
        lambda: SimpleNamespace(
            subscription_id="subscription",
            resource_group="resource-group",
            workspace_name="workspace",
            compute="cluster",
        ),
    )
    client = SimpleNamespace(
        compute=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                name=name,
                max_instances=2,
            )
        )
    )
    monkeypatch.setattr(wave, "get_ml_client", lambda *_args: client)
    monkeypatch.setattr(
        wave,
        "verify_qualification_evidence",
        lambda _path: {
            "state": "passed",
            "accepted_scenario_count": 1,
        },
    )

    result = wave.main(
        [
            "--scenario",
            SCENARIO,
            "--datastore-canary-job",
            "datastore-canary",
            "--output-dir",
            str(output),
        ]
    )

    summary = json.loads(
        (output / "qualification-wave-summary.json").read_text(encoding="utf-8")
    )
    assert result == 0
    assert calls == ["audit", "submit", "monitor", "collect"]
    assert summary["status"] == "passed"
    assert summary["accepted_scenario_count"] == 1


def test_wave_rejects_more_than_two_scenarios_before_remote_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AZUREML_RUN_ID", "qualification-wave")
    monkeypatch.setattr(
        wave,
        "audit_assets",
        lambda _arguments: (_ for _ in ()).throw(
            AssertionError("remote audit must not run")
        ),
    )

    result = wave.main(
        [
            "--scenario",
            "classification-healthcare-heart-disease",
            "--scenario",
            "regression-education-final-grade",
            "--scenario",
            "clustering-retail-transaction-segments",
            "--datastore-canary-job",
            "datastore-canary",
            "--output-dir",
            str(tmp_path / "wave"),
        ]
    )

    assert result == 1
