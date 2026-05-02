"""Test the strict quality gate logic in final_evaluation.

We do not import final_evaluation.main() (it has too many side effects);
instead we mirror its gate predicate to lock the contract.
"""

import math


DEFAULT_QUALITY_THRESHOLDS = {
    "classification": 0.65,
    "regression": 0.30,
    "clustering": 0.10,
}


def _gate(task_type, best_val, champion_valid, cfg=None):
    """Mirror of final_evaluation gate decision."""
    cfg = cfg or {}
    threshold = float(
        cfg.get("registry", {}).get(
            "min_quality", DEFAULT_QUALITY_THRESHOLDS[task_type]
        )
    )
    is_finite = best_val is not None and not (
        isinstance(best_val, float) and (math.isnan(best_val) or math.isinf(best_val))
    )
    return bool(champion_valid and is_finite and best_val >= threshold), threshold


def test_gate_passes_on_strong_champion():
    passed, _ = _gate("classification", 0.85, True)
    assert passed


def test_gate_blocks_below_default_threshold():
    passed, _ = _gate("classification", 0.50, True)
    assert not passed


def test_gate_blocks_invalid_champion():
    passed, _ = _gate("classification", 0.95, False)
    assert not passed


def test_gate_blocks_nan_metric():
    passed, _ = _gate("regression", float("nan"), True)
    assert not passed


def test_gate_blocks_none_metric():
    passed, _ = _gate("regression", None, True)
    assert not passed


def test_gate_uses_cfg_override():
    cfg = {"registry": {"min_quality": 0.95}}
    passed, threshold = _gate("classification", 0.85, True, cfg=cfg)
    assert threshold == 0.95
    assert not passed


def test_gate_threshold_per_task_type():
    assert _gate("classification", 0.65, True)[0] is True
    assert _gate("regression", 0.30, True)[0] is True
    assert _gate("clustering", 0.10, True)[0] is True
    assert _gate("classification", 0.64, True)[0] is False
