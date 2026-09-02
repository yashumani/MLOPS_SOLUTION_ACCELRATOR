from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest

from orchestration.config_compiler import (
    CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS,
    HPO_TIMEOUT_CAP_SECONDS,
    HPO_TRIALS_CAP,
    PHASE_B_TIMEOUT_CAP_SECONDS,
    ROUND1_MAX_VARIANTS_CAP,
    ROUND2_MAX_VARIANTS_CAP,
)
from steps.s06_phaseb_variant_runner import (
    HardDeadlineExceeded,
    require_phase_b_budget,
    rank_round2_proxy_survivors,
    resolve_recipe_paths,
    run_subprocess_with_hard_deadline,
    run_with_hard_timeout,
    select_feasible_round1_candidates,
)
from utils.variant_planner import VariantScore


def _blocking_operation():
    time.sleep(5)


def _raising_operation():
    raise ValueError("isolated failure detail")


def test_approved_budget_caps_are_locked() -> None:
    assert ROUND1_MAX_VARIANTS_CAP == 40
    assert ROUND2_MAX_VARIANTS_CAP == 8
    assert CANDIDATE_ENGINE_TIMEOUT_CAP_SECONDS == 600
    assert PHASE_B_TIMEOUT_CAP_SECONDS == 10800
    assert HPO_TRIALS_CAP == 50
    assert HPO_TIMEOUT_CAP_SECONDS == 3600


def test_round1_ranker_prunes_to_top_eight_deterministically() -> None:
    reports = [
        {
            "variant_id": f"v{index:02d}",
            "variant_path": f"classification/v{index:02d}.yml",
            "semantic_hash": f"{index:064x}",
            "status": "pass",
            "proxy_metric": 0.60 + index / 100,
        }
        for index in range(12)
    ]
    selected, rejected = rank_round2_proxy_survivors(
        reports,
        task_type="classification",
        configured_threshold=0.50,
        max_variants=8,
    )

    assert selected == [
        f"classification/v{index:02d}.yml" for index in range(11, 3, -1)
    ]
    assert len(rejected) == 4
    assert all(item["status"] == "pruned" for item in rejected)


def test_round1_ranker_fail_closed_on_error_or_threshold() -> None:
    selected, rejected = rank_round2_proxy_survivors(
        [
            {
                "variant_id": "warning",
                "variant_path": "warning.yml",
                "status": "warning",
                "proxy_metric": 0.90,
            },
            {
                "variant_id": "low",
                "variant_path": "low.yml",
                "status": "pass",
                "proxy_metric": 0.49,
            },
        ],
        task_type="classification",
        configured_threshold=0.50,
        max_variants=8,
    )
    assert selected == []
    assert {item["variant_id"] for item in rejected} == {"warning", "low"}


@pytest.mark.parametrize(
    ("task_type", "threshold", "score"),
    (
        ("regression", 0.25, 0.20),
        ("clustering", 0.40, 0.35),
    ),
)
def test_round1_ranker_honors_configured_threshold_for_every_task(
    task_type: str,
    threshold: float,
    score: float,
) -> None:
    selected, rejected = rank_round2_proxy_survivors(
        [
            {
                "variant_id": "below-configured-threshold",
                "variant_path": "candidate.yml",
                "status": "pass",
                "proxy_metric": score,
            }
        ],
        task_type=task_type,
        configured_threshold=threshold,
        max_variants=1,
    )

    assert selected == []
    assert rejected[0]["reason"] == f"proxy_metric below {threshold}"


def test_round1_cap_is_applied_after_feasibility_screening() -> None:
    scores = [
        VariantScore(
            variant_id=variant_id,
            variant_path=f"classification/{variant_id}.yml",
            relevance_score=relevance_score,
            reasoning=[],
            preprocessing_hash=variant_id,
            imputation="median",
            encoding="onehot",
            scaling="standard",
            imbalance="none",
            feature_selection="none",
        )
        for variant_id, relevance_score in (
            ("infeasible-high", 100.0),
            ("feasible-medium", 80.0),
            ("feasible-low", 60.0),
        )
    ]
    reports = [
        {
            "variant_path": "classification/infeasible-high.yml",
            "status": "fail",
        },
        {
            "variant_path": "classification/feasible-medium.yml",
            "status": "pass",
        },
        {
            "variant_path": "classification/feasible-low.yml",
            "status": "pass",
        },
    ]

    selected = select_feasible_round1_candidates(
        scores,
        reports,
        max_variants=2,
        diversity_min_hamming=0,
    )

    assert [item.variant_id for item in selected] == [
        "feasible-medium",
        "feasible-low",
    ]


def test_canonical_funnel_screens_catalog_before_shortlist() -> None:
    source = Path("src/steps/s06_phaseb_variant_runner.py").read_text(
        encoding="utf-8"
    )

    screening = source.index("for catalog_score in catalog_scores:")
    shortlist = source.index("round1_selected = select_feasible_round1_candidates(")

    assert screening < shortlist
    assert "variant_paths = [v.variant_path for v in round1_candidates]" not in source


def test_s06_has_no_budget_expansion_or_mlflow_endpoint_rewrite() -> None:
    source = Path("src/steps/s06_phaseb_variant_runner.py").read_text(
        encoding="utf-8"
    )

    assert "effective_budget +=" not in source
    assert "effective_budget = max" not in source
    assert "mlflow.set_tracking_uri" not in source
    assert "mlflow.set_registry_uri" not in source
    assert "candidate_budget = min(" in source
    assert "phase_b_deadline" in source


def test_recipe_paths_cannot_escape_canonical_root(tmp_path) -> None:
    recipe_root = tmp_path / "configs" / "recipes" / "classification"
    recipe_root.mkdir(parents=True)
    valid = recipe_root / "valid.yml"
    valid.write_text("recipe_name: valid\n", encoding="utf-8")
    external = tmp_path / "configs" / "external.yml"
    external.write_text("recipe_name: external\n", encoding="utf-8")

    assert resolve_recipe_paths(
        ["classification/valid.yml"],
        project_root=tmp_path,
    ) == [str(valid.resolve())]
    with pytest.raises(ValueError, match="escapes allowed root"):
        resolve_recipe_paths([str(external)], project_root=tmp_path)
    with pytest.raises(ValueError, match="escapes allowed root"):
        resolve_recipe_paths(["../external.yml"], project_root=tmp_path)


def test_hard_timeout_terminates_blocking_operation() -> None:
    started = time.monotonic()
    with pytest.raises(HardDeadlineExceeded, match="terminated"):
        run_with_hard_timeout(
            _blocking_operation,
            timeout_seconds=0.2,
        )
    assert time.monotonic() - started < 3


def test_isolated_operation_preserves_child_traceback() -> None:
    with pytest.raises(RuntimeError) as error:
        run_with_hard_timeout(
            _raising_operation,
            timeout_seconds=45,
        )

    message = str(error.value)
    assert "ValueError: isolated failure detail" in message
    assert "_raising_operation" in message


def test_expired_phase_b_budget_fails_closed() -> None:
    with pytest.raises(
        HardDeadlineExceeded,
        match="wall-clock budget exhausted",
    ):
        require_phase_b_budget(time.time() - 0.01, "regression-test")


def test_outer_watchdog_kills_entire_component_before_post_deadline_write(
    tmp_path,
) -> None:
    marker = tmp_path / "should-not-exist.txt"
    program = (
        "import pathlib,time;"
        "time.sleep(5);"
        f"pathlib.Path({str(marker)!r}).write_text('late')"
    )
    started = time.monotonic()

    with pytest.raises(HardDeadlineExceeded, match="end-to-end"):
        run_subprocess_with_hard_deadline(
            [sys.executable, "-c", program],
            timeout_seconds=0.2,
        )

    assert time.monotonic() - started < 2.0
    time.sleep(0.4)
    assert not marker.exists()


def test_round1_proxy_and_output_writes_are_deadline_guarded() -> None:
    source = Path("src/steps/s06_phaseb_variant_runner.py").read_text(
        encoding="utf-8"
    )

    assert "run_with_hard_timeout(\n                run_round1_proxy," in source
    assert "before_recipe_funnel_write" in source
    assert "creating minimal leaderboard" not in source.lower()
    assert "creating minimal manifest" not in source.lower()
