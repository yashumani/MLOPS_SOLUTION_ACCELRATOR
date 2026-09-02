from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pipelines import submit_pipeline


def test_effective_round1_cap_honors_both_configured_limits() -> None:
    assert submit_pipeline._compiled_round1_cap(
        {
            "max_variants": 4,
            "planner": {"round1_max_variants": 40},
        }
    ) == 4
    assert submit_pipeline._compiled_round1_cap(
        {
            "max_variants": 40,
            "planner": {"round1_max_variants": 8},
        }
    ) == 8


def test_pipeline_job_settings_bind_working_output_datastore() -> None:
    job = SimpleNamespace(settings=SimpleNamespace())

    submit_pipeline._configure_pipeline_job_settings(
        job,
        default_compute="cluster",
        default_datastore="mlops_blob",
        force_rerun=True,
    )

    assert job.settings.default_compute == "cluster"
    assert job.settings.default_datastore == "mlops_blob"
    assert job.settings.force_rerun is True


def test_active_component_environment_identity_is_exact() -> None:
    identities = submit_pipeline._component_environment_identities()

    assert identities["variant_runner"].startswith("azureml:")
    assert identities["model_reg"].startswith("azureml:")
    assert identities["variant_runner"] != identities["model_reg"]
    assert submit_pipeline._normalize_azureml_environment(
        identities["variant_runner"]
    ) == "mlops-v3-unified:32"


def test_production_submission_requires_content_identity() -> None:
    assert submit_pipeline._production_data_identity_verified(
        {"preset": "diagnostic", "dataset": {}}
    )
    assert not submit_pipeline._production_data_identity_verified(
        {"preset": "production", "dataset": {"content_sha256": None}}
    )
    assert submit_pipeline._production_data_identity_verified(
        {
            "preset": "production",
            "dataset": {"content_sha256": "a" * 64},
        }
    )


def test_local_manifest_input_uses_native_absolute_path(tmp_path: Path) -> None:
    manifest = tmp_path / "execution-manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    resolved = submit_pipeline._sdk_local_input_path(manifest)

    assert Path(resolved) == manifest.resolve()
    assert not resolved.startswith("file:")


def test_local_manifest_input_uses_configured_datastore(tmp_path: Path) -> None:
    manifest = tmp_path / "execution-manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    sdk_input = submit_pipeline._sdk_local_file_input(
        manifest,
        datastore="mlops_blob",
    )

    assert sdk_input.path == str(manifest.resolve())
    assert sdk_input.datastore == "mlops_blob"


def test_legacy_operator_scripts_delegate_to_canonical_submitter() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/batch_submit_inline.py",
        "scripts/resubmit_6_failed.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "run_config_batch" in source
        assert "jobs.create_or_update" not in source
        assert "full_pipeline(" not in source

    helper = (root / "scripts/_canonical_batch_submit.py").read_text(
        encoding="utf-8"
    )
    assert "pipelines" in helper
    assert "submit_pipeline.py" in helper
