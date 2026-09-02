from pathlib import Path

import yaml


def test_stage1_writes_eda_to_declared_component_output() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/steps/stage1_ingestion.py").read_text(
        encoding="utf-8"
    )
    component = yaml.safe_load(
        (root / "components/stage1_ingestion.yml").read_text(encoding="utf-8")
    )

    assert "job_outputs_dir = Path(args.eda_dir)" in source
    assert 'job_outputs_dir = Path("outputs")' not in source
    assert "--eda_dir ${{outputs.eda_report}}" in component["command"]
