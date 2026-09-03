from __future__ import annotations

import json
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.core import entra_auth
from api.core.config import Settings


TENANT = "3bc05bc3-19d1-4d30-89c5-134f4b278b11"
API = "10000000-0000-4000-8000-000000000001"
SPA = "20000000-0000-4000-8000-000000000002"
USER = "30000000-0000-4000-8000-000000000003"
OTHER = "40000000-0000-4000-8000-000000000004"


@pytest.fixture
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def config(tmp_path, monkeypatch, signing_key):
    allowlist = tmp_path / "users.json"
    allowlist.write_text(json.dumps({"schema_version": "1.0", "tenant_id": TENANT, "users": [{"object_id": USER, "roles": ["admin"]}]}))
    settings = Settings(
        _env_file=None,
        api_deployment_profile="multi_user",
        api_config_mutation_enabled=False,
        api_entra_tenant_id=TENANT,
        api_entra_api_client_id=API,
        api_entra_spa_client_id=SPA,
        api_entra_allowed_client_ids=SPA,
        api_entra_redirect_uri="https://mlops.example.test/auth/callback",
        api_user_allowlist_path=str(allowlist),
        cors_allow_origins="https://mlops.example.test",
        mlops_state_dir=str(tmp_path / "submitter"),
        mlops_submission_request_root=str(tmp_path / "requests"),
        mlops_auto_retrain_ledger_root=str(tmp_path / "ledgers"),
        notification_report_dir=str(tmp_path / "notifications"),
        mlops_operational_state_db=str(tmp_path / "state.sqlite3"),
        azure_subscription_id="test-subscription", azure_resource_group="test-rg", azure_workspace_name="test-workspace",
    )
    client = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=signing_key.public_key()))
    monkeypatch.setattr(entra_auth, "_key_client", lambda tenant: client)
    monkeypatch.setenv("MLOPS_OPERATIONAL_STATE_DB", settings.mlops_operational_state_db)
    from api.services.user_access_service import initialize_users
    initialize_users(settings)
    return settings


def _token(key, **changes):
    now = int(time.time())
    claims = {"exp": now + 300, "iat": now, "nbf": now - 1, "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0", "aud": API, "tid": TENANT, "oid": USER, "sub": "subject", "scp": "access_as_user", "azp": SPA, "ver": "2.0"}
    claims.update(changes)
    claims = {name: value for name, value in claims.items() if value is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})


def test_valid_delegated_user_and_hardened_runtime(config, signing_key):
    config.validate_runtime_security()
    user = entra_auth.authenticate_token(_token(signing_key), config)
    assert user.object_id == USER
    assert user.roles == ("admin",)
    assert user.submission_tags()["actor_tenant_id"] == TENANT


@pytest.mark.parametrize("changes", [
    {"aud": OTHER}, {"tid": OTHER}, {"iss": "https://attacker.example"},
    {"azp": OTHER}, {"ver": "1.0"}, {"idtyp": "app"},
    {"scp": None}, {"scp": "unrelated"}, {"oid": None},
    {"exp": int(time.time()) - 300}, {"nbf": int(time.time()) + 300},
    {"iat": int(time.time()) + 300},
])
def test_invalid_identity_claims_are_rejected(config, signing_key, changes):
    with pytest.raises(HTTPException) as exc:
        entra_auth.authenticate_token(_token(signing_key, **changes), config)
    assert exc.value.status_code == 401


def test_invalid_signature_and_symmetric_algorithm_are_rejected(config, signing_key):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    for token in (_token(other_key), jwt.encode({"sub": USER}, "x" * 32, algorithm="HS256", headers={"kid": "test-key"})):
        with pytest.raises(HTTPException) as exc:
            entra_auth.authenticate_token(token, config)
        assert exc.value.status_code == 401


def test_email_cannot_grant_access_and_removal_takes_effect_immediately(config, signing_key):
    with pytest.raises(HTTPException) as exc:
        entra_auth.authenticate_token(_token(signing_key, oid=OTHER, email="owner@example.test"), config)
    assert exc.value.status_code == 403
    from orchestration import operational_state as state
    with state.transaction() as connection:
        directory = state.get_document(connection, "access_control", "users")
        directory["users"][USER]["enabled"] = False
        state.put_document(connection, "access_control", "users", directory)
    with pytest.raises(HTTPException) as exc:
        entra_auth.authenticate_token(_token(signing_key), config)
    assert exc.value.status_code == 403


@pytest.mark.parametrize("payload", [
    {}, {"schema_version": "1.0", "tenant_id": OTHER, "users": []},
    {"schema_version": "1.0", "tenant_id": TENANT, "users": [{"object_id": USER, "roles": ["superuser"]}]},
    {"schema_version": "1.0", "tenant_id": TENANT, "users": [{"object_id": USER, "roles": ["viewer"]}, {"object_id": USER, "roles": ["operator"]}]},
])
def test_invalid_bootstrap_allowlist_fails_closed(config, signing_key, payload):
    from pathlib import Path
    Path(config.api_user_allowlist_path).write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        entra_auth.load_allowlist(config.api_user_allowlist_path, TENANT)


def test_bearer_only_api_and_transactional_actor_audit(config, signing_key, monkeypatch):
    from api.core.config import settings
    from api.main import app
    from orchestration import operational_state

    for field in Settings.model_fields:
        monkeypatch.setattr(settings, field, getattr(config, field))
    monkeypatch.setenv("MLOPS_OPERATIONAL_STATE_DB", config.mlops_operational_state_db)
    monkeypatch.setattr(settings, "api_key", "legacy-key")
    client = TestClient(app)
    assert client.get("/api/v1/auth/config").json()["mode"] == "entra"
    assert client.get("/api/v1/auth/me", headers={"X-API-Key": "legacy-key"}).status_code == 401
    token = _token(signing_key)
    response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200
    assert response.json()["object_id"] == USER
    with operational_state.transaction() as connection:
        rows = connection.execute("SELECT payload FROM documents WHERE namespace='request_audit'").fetchall()
    assert len(rows) == 1
    record = json.loads(rows[0][0])
    assert record["actor"]["object_id"] == USER
    assert record["status_code"] == 200
    assert token not in rows[0][0]


def test_viewer_cannot_submit(config, signing_key, monkeypatch):
    from pathlib import Path
    from api.core.config import settings
    from api.main import app

    from orchestration import operational_state as state
    with state.transaction() as connection:
        directory = state.get_document(connection, "access_control", "users")
        directory["users"][USER]["role"] = "viewer"
        state.put_document(connection, "access_control", "users", directory)
    for field in Settings.model_fields:
        monkeypatch.setattr(settings, field, getattr(config, field))
    monkeypatch.setenv("MLOPS_OPERATIONAL_STATE_DB", config.mlops_operational_state_db)
    client = TestClient(app)
    response = client.post("/api/v1/pipelines/submit", json={"config_name": "config_classification"}, headers={"Authorization": "Bearer " + _token(signing_key)})
    assert response.status_code == 403


def test_missing_state_database_prevents_multi_user_startup(config):
    config.mlops_operational_state_db = ""
    with pytest.raises(RuntimeError, match="MLOPS_OPERATIONAL_STATE_DB"):
        config.validate_runtime_security()
