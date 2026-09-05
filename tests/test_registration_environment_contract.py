"""Static guards for the engine wrappers persisted in registration bundles."""

from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "config" / "s12_registration_environment"


def test_registration_runtime_pins_the_training_flaml_version():
    requirements = {
        requirement.name.lower(): requirement
        for line in (ENVIRONMENT / "requirements.in").read_text().splitlines()
        if line.strip() and not line.startswith("#")
        for requirement in [Requirement(line)]
    }
    assert str(requirements["flaml"].specifier) == "==2.2.0"
    assert not requirements["flaml"].extras
    lock = (ENVIRONMENT / "requirements.lock").read_text()
    assert "flaml==2.2.0 \\" in lock
    assert "--hash=sha256:eb7429801879f66901ec13892ea21a914e3a5a094151b621a924e554637ec4a4" in lock
    assert "flaml==2.2.0" in (ROOT / "config/mlops_v3_unified_environment/conda_v33.yml").read_text()


def test_registration_build_checks_persisted_wrapper_imports():
    dockerfile = (ENVIRONMENT / "Dockerfile").read_text()
    assert "--require-hashes" in dockerfile
    assert "python -m pip check" in dockerfile
    assert "from flaml.automl.model import" in dockerfile
    for wrapper in ("CatBoostEstimator", "LGBMEstimator", "LRL1Classifier", "XGBoostLimitDepthEstimator"):
        assert wrapper in dockerfile
