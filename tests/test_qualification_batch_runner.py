from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "batch_submit_all.py"


def _load_module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("qualification_batch_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_catalog_has_exact_five_industries_per_task() -> None:
    module = _load_module()
    scenarios = module.load_execution_catalog()

    assert len(scenarios) == 15
    for task_type in module.TASK_TYPES:
        selected = [item for item in scenarios if item.task_type == task_type]
        assert len(selected) == 5
        assert len({item.industry for item in selected}) == 5


def test_selection_rejects_unknown_scenario() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="Unknown qualification scenarios"):
        module.select_scenarios(
            module.load_execution_catalog(),
            scenario_ids={"missing-scenario"},
        )


def test_submission_command_preserves_canonical_guards(tmp_path: Path) -> None:
    module = _load_module()
    scenario = module.load_execution_catalog()[0]
    context = SimpleNamespace(
        as_cli_args=lambda: [
            "--subscription_id",
            "sub",
            "--resource_group",
            "rg",
            "--workspace_name",
            "ws",
            "--compute",
            "cluster",
        ]
    )

    command = module.build_submission_command(
        scenario,
        result_path=tmp_path / "result.json",
        context=context,
    )

    assert str(module.SUBMITTER) in command
    assert "--force" not in command
    assert "--force_rerun" not in command
    assert "--result_json" in command
    assert "--tags_json" in command


def test_shell_entrypoints_delegate_without_force_bypass() -> None:
    for name in (
        "submit_all_15.sh",
        "submit_15_parallel.sh",
        "submit_all_dryrun.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "batch_submit_all.py" in source
        assert "--force" not in source
        assert "config_classification_cardiac_arrest" not in source

    dry_run = (ROOT / "scripts" / "submit_all_dryrun.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ "$argument" == "--execute" ]]' in dry_run
