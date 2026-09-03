"""Single-tenant delegated-token authentication with an operator-owned allowlist."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import jwt
from fastapi import HTTPException


MAX_ALLOWLIST_BYTES = 64 * 1024
MAX_TOKEN_BYTES = 24 * 1024
ALLOWED_ROLES = frozenset({"viewer", "operator", "admin"})


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    object_id: str
    roles: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"tenant_id": self.tenant_id, "object_id": self.object_id, "roles": list(self.roles)}

    def submission_tags(self) -> dict[str, str]:
        return {
            "actor_tenant_id": self.tenant_id,
            "actor_object_id": self.object_id,
            "actor_roles": ",".join(self.roles),
        }


def canonical_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Identity must be a UUID string")
    return str(UUID(value))


def load_allowlist(path: str, tenant_id: str) -> dict[str, tuple[str, ...]]:
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise ValueError("API_USER_ALLOWLIST_PATH must be absolute")
    with source.open("rb") as handle:
        raw = handle.read(MAX_ALLOWLIST_BYTES + 1)
    if len(raw) > MAX_ALLOWLIST_BYTES:
        raise ValueError("User allowlist exceeds 64 KiB")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("User allowlist must declare schema_version 1.0")
    if canonical_id(payload.get("tenant_id")) != canonical_id(tenant_id):
        raise ValueError("User allowlist tenant does not match the API tenant")
    users = payload.get("users")
    if not isinstance(users, list) or not users or len(users) > 1000:
        raise ValueError("User allowlist must contain 1-1000 users")
    result: dict[str, tuple[str, ...]] = {}
    for user in users:
        if not isinstance(user, dict):
            raise ValueError("Allowlist user must be an object")
        object_id = canonical_id(user.get("object_id"))
        roles = user.get("roles")
        if not isinstance(roles, list) or not roles or any(role not in ALLOWED_ROLES for role in roles):
            raise ValueError("User roles must be viewer, operator, or admin")
        if object_id in result:
            raise ValueError("Duplicate user object ID in allowlist")
        result[object_id] = tuple(sorted(set(roles)))
    return result


@lru_cache(maxsize=4)
def _key_client(tenant_id: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(
        f"https://login.microsoftonline.com/{canonical_id(tenant_id)}/discovery/v2.0/keys",
        cache_keys=False,
        lifespan=300,
        timeout=5,
    )


def authenticate_token(token: str, settings) -> Principal:
    unauthorized = HTTPException(401, "Invalid or missing access token", headers={"WWW-Authenticate": "Bearer"})
    if not token or len(token.encode("utf-8")) > MAX_TOKEN_BYTES:
        raise unauthorized
    try:
        tenant_id = canonical_id(settings.api_entra_tenant_id)
        audience = canonical_id(settings.api_entra_api_client_id)
        allowed_clients = {canonical_id(value.strip()) for value in settings.api_entra_allowed_client_ids.split(",") if value.strip()}
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
            raise unauthorized
        key = _key_client(tenant_id).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            issuer=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            audience=audience,
            leeway=30,
            options={"require": ["exp", "iat", "nbf", "iss", "aud", "tid", "oid", "sub", "scp", "azp", "ver"]},
        )
        if claims["ver"] != "2.0" or canonical_id(claims["tid"]) != tenant_id:
            raise unauthorized
        if claims.get("idtyp") == "app" or canonical_id(claims["azp"]) not in allowed_clients:
            raise unauthorized
        if not isinstance(claims["scp"], str) or settings.api_entra_required_scope not in claims["scp"].split():
            raise unauthorized
        object_id = canonical_id(claims["oid"])
    except jwt.PyJWKClientConnectionError as exc:
        raise HTTPException(503, "Identity signing keys are temporarily unavailable") from exc
    except (jwt.PyJWTError, ValueError, TypeError, KeyError) as exc:
        raise unauthorized from exc
    try:
        from api.services.user_access_service import authorized_roles
        from orchestration.operational_state import OperationalStateError
        roles = authorized_roles(tenant_id, object_id)
    except (OSError, ValueError, TypeError, sqlite3.Error, OperationalStateError) as exc:
        raise HTTPException(503, "User authorization configuration is unavailable") from exc
    if roles is None:
        raise HTTPException(403, "User is not authorized for this workspace")
    return Principal(tenant_id, object_id, roles)
