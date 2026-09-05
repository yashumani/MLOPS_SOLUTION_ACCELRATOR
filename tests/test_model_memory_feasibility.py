from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from utils.model_universe import (
    build_coverage_report,
    get_model_list,
    pycaret_memory_plan,
)


def test_large_regression_excludes_only_infeasible_kernel_without_mutating_catalog():
    catalog = get_model_list("regression", "pycaret")
    plan = pycaret_memory_plan("regression", 80000, memory_budget_bytes=2 * 1024**3)
    assert plan["included_models"] == [model for model in catalog if model != "kr"]
    assert plan["excluded_models"] == [{
        "model_id": "kr", "status": "skipped_memory_infeasible",
        "reason": "dense_kernel_estimate_exceeds_worker_budget",
        "estimated_peak_bytes": 153600000000,
    }]
    assert get_model_list("regression", "pycaret") == catalog
    assert plan["n_jobs"] == 1


@pytest.mark.parametrize("rows", [20, 1000])
def test_feasible_kernel_remains_in_search_including_exact_boundary(rows):
    plan = pycaret_memory_plan("regression", rows, memory_budget_bytes=rows * rows * 24)
    assert plan["included_models"] == get_model_list("regression", "pycaret")
    assert plan["excluded_models"] == []
    assert "kr" not in pycaret_memory_plan(
        "regression", rows, memory_budget_bytes=rows * rows * 24 - 1,
    )["included_models"]


def test_classification_retains_catalog_and_records_memory_budget():
    plan = pycaret_memory_plan("classification", 80000, memory_budget_bytes=1024**3)
    assert plan["included_models"] == get_model_list("classification", "pycaret")
    assert plan["excluded_models"] == []
    assert plan["memory_budget_bytes"] == 1024**3


@pytest.mark.parametrize("rows", [0, -1, True, 1.5, float("nan")])
def test_invalid_rows_fail_closed(rows):
    with pytest.raises(ValueError, match="positive training row count"):
        pycaret_memory_plan("regression", rows, memory_budget_bytes=1024)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, float("nan")])
def test_invalid_explicit_budget_fails_closed(budget):
    with pytest.raises(ValueError, match="positive memory budget"):
        pycaret_memory_plan("regression", 10, memory_budget_bytes=budget)


def test_unsupported_task_cannot_fall_back_to_unrestricted_search():
    with pytest.raises(ValueError, match="supervised PyCaret"):
        pycaret_memory_plan("unknown", 10, memory_budget_bytes=1024)


@pytest.mark.parametrize("files, expected_budget", [
    ({}, 8 * 1024**3),
    ({"/sys/fs/cgroup/memory.max": str(4 * 1024**3),
      "/sys/fs/cgroup/memory.current": str(2 * 1024**3)}, 1024**3),
    ({"/sys/fs/cgroup/memory.max": "max",
      "/sys/fs/cgroup/memory.current": "0"}, 8 * 1024**3),
    ({"/sys/fs/cgroup/memory/memory.limit_in_bytes": str(8 * 1024**3),
      "/sys/fs/cgroup/memory/memory.usage_in_bytes": str(2 * 1024**3)}, 3 * 1024**3),
    ({"/sys/fs/cgroup/memory.max": str(32 * 1024**3),
      "/sys/fs/cgroup/memory.current": "0"}, 8 * 1024**3),
])
def test_budget_respects_container_remaining_and_host_available_memory(
    monkeypatch, files, expected_budget,
):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=16 * 1024**3))
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) in files)
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    plan = pycaret_memory_plan("regression", 80000)
    assert plan["memory_budget_bytes"] == expected_budget
    assert plan["excluded_models"][0]["model_id"] == "kr"


def test_exhausted_container_fails_before_discovery(monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1024))
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    monkeypatch.setattr(Path, "read_text", lambda path: "2048")
    with pytest.raises(ValueError, match="positive memory budget"):
        pycaret_memory_plan("regression", 10)


def test_coverage_records_exclusion_without_counting_it_available():
    plan = pycaret_memory_plan("regression", 80000, memory_budget_bytes=1024**3)
    report = build_coverage_report("regression", memory_plan=plan)
    pycaret = report["engines"]["pycaret"]
    assert pycaret["skipped"] == 1
    assert pycaret["available"] == len(plan["included_models"])
    assert pycaret["models"]["kr"]["status"] == "skipped_memory_infeasible"
    assert report["engines"]["flaml"]["skipped"] == 0
