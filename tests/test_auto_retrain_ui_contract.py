"""Cross-language contract checks for auto-retrain controller plan payloads."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_react_auto_retrain_plan_sends_required_decision_path() -> None:
    page = (ROOT / "react-ui" / "src" / "pages" / "AutoRetrain.tsx").read_text(
        encoding="utf-8"
    )
    api_types = (ROOT / "react-ui" / "src" / "types" / "api.ts").read_text(
        encoding="utf-8"
    )

    assert "decision_path: decisionPath.trim()" in page
    assert "!decisionPath.trim()" in page
    assert "decision_path: string;" in api_types


def test_streamlit_auto_retrain_plan_validates_and_sends_decision_path() -> None:
    page = (ROOT / "ui" / "pages" / "4_Auto_Retrain.py").read_text(
        encoding="utf-8"
    )

    assert "if not decision_path.strip():" in page
    assert '"decision_path": decision_path.strip()' in page

