from concurrent.futures import ThreadPoolExecutor
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.core.entra_auth import Principal
from api.services import user_access_service as service
from orchestration import operational_state as state
from test_entra_auth import config, signing_key, _token, TENANT, USER, OTHER


def _save(actor, object_id=OTHER, role="viewer", enabled=True, revision=0, create=True):
    return service.save_user(actor, object_id=object_id, display_name="Test user", role=role, enabled=enabled, expected_revision=revision, create=create, audit_id="audit-test")


def test_only_initial_admin_exists_and_restart_does_not_reseed(config):
    actor = Principal(TENANT, USER, ("admin",))
    assert service.list_users(actor)["users"] == [{"object_id": USER, "display_name": "Initial administrator", "role": "admin", "enabled": True}]
    _save(actor, role="admin")
    service.initialize_users(config)
    assert len(service.list_users(actor)["users"]) == 2


@pytest.mark.parametrize("role,enabled", [("viewer", True), ("operator", True), ("admin", False)])
def test_last_admin_cannot_be_removed_or_demoted(config, role, enabled):
    actor = Principal(TENANT, USER, ("admin",))
    with pytest.raises(HTTPException) as error:
        _save(actor, object_id=USER, role=role, enabled=enabled, create=False)
    assert error.value.status_code == 409
    assert service.authorized_roles(TENANT, USER) == ("admin",)
    assert service.list_users(actor)["revision"] == 0


def test_revoked_admin_cannot_use_stale_principal(config):
    owner = Principal(TENANT, USER, ("admin",))
    _save(owner, role="admin")
    stale = Principal(TENANT, OTHER, ("admin",))
    _save(owner, role="viewer", revision=1, create=False)
    with pytest.raises(HTTPException) as error:
        _save(stale, revision=2)
    assert error.value.status_code == 403


def test_conflicting_edits_are_rejected_and_audited_atomically(config):
    owner = Principal(TENANT, USER, ("admin",))
    _save(owner)
    def update(role):
        try:
            _save(owner, role=role, revision=1, create=False)
            return 200
        except HTTPException as error:
            return error.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(update, ["operator", "admin"])) == [200, 409]
    with state.transaction() as connection:
        events = state.load_events(connection, "user_access_audit")
    assert len(events) == 3
    assert events[-1]["actor"]["object_id"] == USER
    assert events[-1]["api_request_audit_id"] == "audit-test"


@pytest.fixture
def client(config, monkeypatch):
    from api.core.config import Settings, settings
    from api.main import app
    for field in Settings.model_fields:
        monkeypatch.setattr(settings, field, getattr(config, field))
    return TestClient(app)


def test_admin_http_add_edit_revoke_and_denial(client, signing_key, config):
    headers = {"Authorization": "Bearer " + _token(signing_key)}
    assert client.get("/api/v1/users", headers=headers).json()["revision"] == 0
    body = {"object_id": OTHER, "display_name": "New admin", "role": "admin", "enabled": True, "expected_revision": 0}
    assert client.post("/api/v1/users", headers=headers, json=body).status_code == 201
    other_headers = {"Authorization": "Bearer " + _token(signing_key, oid=OTHER)}
    assert client.get("/api/v1/users", headers=other_headers).status_code == 200
    body.pop("object_id")
    body.update(expected_revision=1, enabled=False)
    assert client.put(f"/api/v1/users/{OTHER}", headers=headers, json=body).status_code == 200
    assert client.get("/api/v1/users", headers=other_headers).status_code == 403
    assert client.get("/api/v1/users", headers={"X-API-Key": "legacy"}).status_code == 401
    assert service.authorized_roles(TENANT, OTHER) is None


@pytest.mark.parametrize("role", ["viewer", "operator"])
def test_non_admin_cannot_manage_users(client, signing_key, config, role):
    _save(Principal(TENANT, USER, ("admin",)), role=role)
    headers = {"Authorization": "Bearer " + _token(signing_key, oid=OTHER)}
    assert client.get("/api/v1/users", headers=headers).status_code == 403
    assert client.post("/api/v1/users", headers=headers, json={}).status_code == 403


def test_client_cannot_override_tenant_or_actor(client, signing_key, config):
    headers = {"Authorization": "Bearer " + _token(signing_key)}
    body = {"object_id": OTHER, "display_name": "test", "role": "admin", "enabled": True, "expected_revision": 0, "actor": USER}
    assert client.post("/api/v1/users", headers=headers, json=body).status_code == 422


def test_seed_file_contains_only_verified_owner():
    from pathlib import Path
    seed = json.loads((Path(__file__).resolve().parents[1] / "configs/access/initial_admin.json").read_text())
    assert seed["tenant_id"] == TENANT
    assert seed["users"] == [{"object_id": "b03e4295-9fce-4b3b-b6ba-e7e750e639ef", "roles": ["admin"]}]
