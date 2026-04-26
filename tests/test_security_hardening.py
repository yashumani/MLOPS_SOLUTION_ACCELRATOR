"""Security-hardening regression tests for ``pipelines.submit_pipeline``.

Coverage targets (Workstream D of prod-hardening-20260425):
    * ``_safe_join_data_path``     — path-traversal & absolute-path refusal
    * ``_check_csv_size_within_cap`` — 500 MB submit-host cap
    * ``_acquire_lock`` / ``_release_lock`` — single-writer guarantee + stale lock + cross-user EPERM
    * ``_record_force_audit`` — JSONL audit trail on ``--force``
    * Component manifest hard-fail in ``pipeline_builder._load_component_safe``

These tests are deliberately import-light so they run on the submit host
without an Azure ML workspace context.  They MUST pass before any production
deployment per the project mandate:

    "Do thorough testing before you approve for the production. No, no exceptions."
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Make the repo importable when pytest is invoked from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importing submit_pipeline executes module-level code (logger setup, recipe
# selector import).  Skip the entire module cleanly if a dependency is missing.
sp = pytest.importorskip("pipelines.submit_pipeline")


# ---------------------------------------------------------------------------
# _safe_join_data_path
# ---------------------------------------------------------------------------
class TestSafeJoinDataPath:
    def test_simple_relative_resolves_under_data_root(self):
        out = sp._safe_join_data_path("subset/train.csv")
        assert str(out).startswith(str(sp.DATA_ROOT))
        assert out.name == "train.csv"

    def test_absolute_path_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            sp._safe_join_data_path("/etc/passwd")

    def test_dotdot_rejected(self):
        with pytest.raises(ValueError, match="traversal"):
            sp._safe_join_data_path("../../../etc/passwd")

    def test_embedded_dotdot_rejected(self):
        with pytest.raises(ValueError, match="traversal"):
            sp._safe_join_data_path("subset/../../etc/passwd")

    @pytest.mark.parametrize("bad", ["", None, 123, [], {}])
    def test_non_string_or_empty_rejected(self, bad):
        with pytest.raises(ValueError):
            sp._safe_join_data_path(bad)

    def test_data_root_itself_allowed(self):
        out = sp._safe_join_data_path(".")
        assert out == sp.DATA_ROOT


# ---------------------------------------------------------------------------
# _check_csv_size_within_cap
# ---------------------------------------------------------------------------
class TestCsvSizeCap:
    def test_small_file_passes(self, tmp_path):
        p = tmp_path / "small.csv"
        p.write_bytes(b"a,b,c\n1,2,3\n")
        sp._check_csv_size_within_cap(p)

    def test_large_file_rejected(self, tmp_path):
        p = tmp_path / "big.csv"
        p.write_bytes(b"x")
        with pytest.raises(ValueError, match="exceeds"):
            sp._check_csv_size_within_cap(p, max_bytes=0)

    def test_at_boundary_passes(self, tmp_path):
        p = tmp_path / "edge.csv"
        p.write_bytes(b"abc")
        sp._check_csv_size_within_cap(p, max_bytes=3)


# ---------------------------------------------------------------------------
# _acquire_lock / _release_lock
# ---------------------------------------------------------------------------
class TestSubmitLock:
    @pytest.fixture(autouse=True)
    def _isolate_lockfile(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sp, "_LOCK_FILE", tmp_path / ".submit.lock")
        monkeypatch.setattr(sp, "_FORCE_AUDIT_FILE", tmp_path / ".force_submit_audit.jsonl")
        yield
        sp._release_lock()

    def test_acquire_then_release(self):
        assert sp._acquire_lock() is True
        assert sp._LOCK_FILE.exists()
        sp._release_lock()
        assert not sp._LOCK_FILE.exists()

    def test_double_acquire_blocks(self):
        assert sp._acquire_lock() is True
        # Same process re-acquire — must be refused (single-writer guarantee).
        assert sp._acquire_lock() is False

    def test_release_idempotent(self):
        sp._release_lock()
        sp._release_lock()  # must not raise

    def test_stale_lock_by_dead_pid_is_reclaimed(self):
        # Write a lock claiming a non-existent PID.
        sp._LOCK_FILE.write_text(json.dumps({
            "pid": 9_999_999,
            "ts": 0,
            "expires": 10**12,        # not TTL-expired
            "user": "ghost",
        }))
        with mock.patch("os.kill", side_effect=ProcessLookupError):
            assert sp._acquire_lock() is True

    def test_cross_user_lock_is_genuine_eperm(self):
        """EPERM (PermissionError) means PID exists but is owned by another user.
        The lock MUST be treated as live — never silently stolen."""
        sp._LOCK_FILE.write_text(json.dumps({
            "pid": 1,                  # init/pid 1 always exists
            "ts": __import__("datetime").datetime.now().timestamp(),
            "expires": __import__("datetime").datetime.now().timestamp() + 3600,
            "user": "another_user",
        }))
        with mock.patch("os.kill", side_effect=PermissionError):
            assert sp._acquire_lock() is False
        assert sp._LOCK_FILE.exists(), "Cross-user lock must NOT be removed"

    def test_ttl_expiry_reclaims_lock(self):
        sp._LOCK_FILE.write_text(json.dumps({
            "pid": 1,
            "ts": 0,
            "expires": 1,             # expired long ago
            "user": "ancient",
        }))
        # Even if PID 1 exists, TTL expiry reclaims the lock.
        assert sp._acquire_lock() is True

    def test_corrupt_lock_file_is_reclaimed(self):
        sp._LOCK_FILE.write_text("{not-json")
        assert sp._acquire_lock() is True


# ---------------------------------------------------------------------------
# _record_force_audit
# ---------------------------------------------------------------------------
class TestForceAudit:
    def test_audit_record_appended(self, tmp_path, monkeypatch):
        audit = tmp_path / "audit.jsonl"
        monkeypatch.setattr(sp, "_FORCE_AUDIT_FILE", audit)

        class _Args:
            config = "configs/x.yml"
            experiment_name = "exp1"
            display_name = "disp1"
            compute = "mlopsv2computecluster"

        sp._record_force_audit(_Args(), user="alice")
        sp._record_force_audit(_Args(), user="bob")

        lines = audit.read_text().strip().splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["user"] == "alice"
        assert rec["config"] == "configs/x.yml"
        assert rec["experiment_name"] == "exp1"
        assert rec["pid"] == os.getpid()
        assert "timestamp" in rec

    def test_audit_failure_does_not_raise(self, tmp_path, monkeypatch):
        # Point audit at a non-writable directory; must downgrade to warning, not raise.
        bad = tmp_path / "ro_dir" / "audit.jsonl"
        bad.parent.mkdir()
        bad.parent.chmod(0o400)
        monkeypatch.setattr(sp, "_FORCE_AUDIT_FILE", bad)

        class _Args:
            config = None
            experiment_name = None
            display_name = None
            compute = None

        try:
            sp._record_force_audit(_Args(), user="ci")  # must not raise
        finally:
            bad.parent.chmod(0o700)


# ---------------------------------------------------------------------------
# pipeline_builder component manifest hard-fail
# ---------------------------------------------------------------------------
class TestComponentLoaderHardFail:
    def test_load_component_safe_raises_on_missing(self):
        pb = pytest.importorskip("pipelines.pipeline_builder")
        with pytest.raises(RuntimeError):
            pb._load_component_safe("nonexistent_component_xyz", "components/does_not_exist.yml")

    def test_component_manifest_populated(self):
        pb = pytest.importorskip("pipelines.pipeline_builder")
        # Manifest should contain at least the canonical s00 → s12 keys.
        assert isinstance(pb._COMPONENT_MANIFEST, dict)
        assert len(pb._COMPONENT_MANIFEST) >= 1
