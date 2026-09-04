from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from orchestration import operational_state as state
from scripts import run_auto_retrain_daemon as daemon


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "_configured_path", "")
    binding = {
        "tenant_id": "3bc05bc3-19d1-4d30-89c5-134f4b278b11",
        "subscription_id": "93044a08-5661-4f1b-b424-5eafe066a9d1",
        "resource_group": "test-rg", "workspace_name": "test-workspace",
    }
    for name, value in binding.items():
        monkeypatch.setenv("AZURE_" + name.upper(), value)
    monkeypatch.setenv("AZURE_COMPUTE", "test-cluster")
    monkeypatch.setenv("MLOPS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("MLOPS_OPERATIONAL_STATE_DB", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER", str(tmp_path / "ledger.jsonl"))
    return binding


def test_bootstrap_is_explicit_idempotent_and_does_not_seed_users(configured):
    with pytest.raises(state.OperationalStateError, match="explicitly"):
        state.bind_workspace(configured)
    assert state.bind_workspace(configured, initialize=True) is True
    assert state.bind_workspace(configured, initialize=True) is False
    assert state.bind_workspace(configured) is False
    with state.transaction() as connection:
        assert state.get_document(connection, "configuration", "workspace") == configured
        assert state.get_document(connection, "access_control", "users") is None
        assert len(state.load_events(connection, "controller_audit")) == 1


def test_bootstrap_cannot_replace_existing_binding_or_ledger(configured):
    state.bind_workspace(configured, initialize=True)
    with state.transaction() as connection:
        state.append_event(connection, "retrain_ledger:test", {"decision_id": "keep"})
    with pytest.raises(state.OperationalStateError, match="another"):
        state.bind_workspace({**configured, "workspace_name": "other"}, initialize=True)
    with state.transaction() as connection:
        assert state.get_document(connection, "configuration", "workspace") == configured
        assert state.load_events(connection, "retrain_ledger:test") == [{"decision_id": "keep"}]


@pytest.mark.parametrize("kind", ["document", "event"])
def test_nonempty_unbound_database_cannot_be_adopted(configured, kind):
    with state.transaction() as connection:
        if kind == "document":
            state.put_document(connection, "legacy", "one", {"owner": "unknown"})
        else:
            state.append_event(connection, "legacy", {"owner": "unknown"})
    with pytest.raises(state.OperationalStateError, match="nonempty"):
        state.bind_workspace(configured, initialize=True)
    with state.transaction() as connection:
        assert state.get_document(connection, "configuration", "workspace") is None


def test_competing_initializers_only_write_one_binding_event(configured):
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _: state.bind_workspace(configured, initialize=True), range(4))) == 1
    with state.transaction() as connection:
        assert len(state.load_events(connection, "controller_audit")) == 1


@pytest.mark.parametrize("change", [{"tenant_id": "not-uuid"}, {"workspace_name": " "}, {"extra": "field"}])
def test_invalid_binding_is_rejected(configured, change):
    with pytest.raises(state.OperationalStateError):
        state.bind_workspace({**configured, **change}, initialize=True)


def test_initialization_checks_managed_identity_access_and_never_submits(configured, monkeypatch):
    calls = []
    def make_client(subscription, group, workspace, *, credential_mode):
        assert (subscription, group, workspace) == tuple(configured[key] for key in ("subscription_id", "resource_group", "workspace_name"))
        assert credential_mode == "managed_identity"
        return SimpleNamespace(workspaces=SimpleNamespace(get=lambda name: calls.append(name)))
    monkeypatch.setattr(daemon, "get_ml_client", make_client)
    monkeypatch.setattr(daemon, "process_source_job", lambda *args, **kwargs: pytest.fail("initialization submitted"))
    assert daemon.main(["--initialize-state"]) == 0
    assert calls == ["test-workspace"]
    assert state.bind_workspace(configured) is False


def test_failed_azure_access_does_not_initialize_state(configured, monkeypatch):
    def deny(name):
        raise RuntimeError("access denied")
    monkeypatch.setattr(
        daemon,
        "get_ml_client",
        lambda *_args, **_kwargs: SimpleNamespace(
            workspaces=SimpleNamespace(get=deny)
        ),
    )
    assert daemon.main(["--initialize-state"]) == 2
    with state.transaction() as connection:
        assert state.get_document(connection, "configuration", "workspace") is None


def test_obo_controller_mode_is_explicit_and_propagated(configured, monkeypatch):
    observed = {}
    monkeypatch.setenv("MLOPS_CONTROLLER_CREDENTIAL_MODE", "azureml_obo")
    monkeypatch.delenv("MLOPS_AZURE_CREDENTIAL_MODE", raising=False)

    def make_client(*_args, credential_mode):
        observed["credential_mode"] = credential_mode
        return SimpleNamespace(
            workspaces=SimpleNamespace(get=lambda _name: None)
        )

    monkeypatch.setattr(daemon, "get_ml_client", make_client)

    assert daemon.main(["--initialize-state"]) == 0
    assert observed["credential_mode"] == "azureml_obo"
    assert "MLOPS_AZURE_CREDENTIAL_MODE" not in daemon.os.environ


@pytest.mark.parametrize("flags", [["--initialize-state", "--execute"], ["--initialize-state", "--once"], ["--initialize-state", "--manifest", "unreviewed.yml"], ["--once"]])
def test_invalid_modes_do_not_touch_state(configured, flags):
    assert daemon.main(flags) == 2
    assert not state.database_path().exists()


def test_state_paths_must_stay_within_owner_root(configured, monkeypatch, tmp_path):
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER", str(tmp_path.parent / "outside.jsonl"))
    with pytest.raises(ValueError, match="contained"):
        daemon.validate_state_paths()
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER", str(state.database_path()))
    with pytest.raises(ValueError, match="distinct"):
        daemon.validate_state_paths()


def test_state_symlink_escape_is_rejected(configured, monkeypatch, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are not available on this host")
    monkeypatch.setenv("MLOPS_AUTO_RETRAIN_LEDGER", str(link / "ledger.jsonl"))
    with pytest.raises(ValueError, match="contained"):
        daemon.validate_state_paths()
