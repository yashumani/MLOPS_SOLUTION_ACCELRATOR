"""Transactional application access; never grants Azure RBAC or directory roles."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException

from api.core.entra_auth import ALLOWED_ROLES, Principal, canonical_id, load_allowlist
from orchestration import operational_state as state


def initialize_users(settings) -> None:
    users = load_allowlist(settings.api_user_allowlist_path, settings.api_entra_tenant_id)
    if len(users) != 1 or next(iter(users.values())) != ("admin",):
        raise ValueError("Bootstrap allowlist must contain exactly one admin")
    owner = next(iter(users))
    tenant = canonical_id(settings.api_entra_tenant_id)
    with state.transaction() as connection:
        current = state.get_document(connection, "access_control", "users")
        if current is not None:
            if current.get("tenant_id") != tenant or current.get("bootstrap_admin_id") != owner:
                raise ValueError("Access database belongs to another tenant or bootstrap admin")
            return
        record = {"tenant_id": tenant, "bootstrap_admin_id": owner, "revision": 0, "users": {
            owner: {"object_id": owner, "display_name": "Initial administrator", "role": "admin", "enabled": True},
        }}
        state.put_document(connection, "access_control", "users", record)
        state.append_event(connection, "user_access_audit", {"action": "bootstrap", "tenant_id": tenant, "object_id": owner, "timestamp_utc": datetime.now(timezone.utc).isoformat()})


def _directory(connection, tenant_id: str) -> dict:
    value = state.get_document(connection, "access_control", "users")
    if value is None or value.get("tenant_id") != canonical_id(tenant_id):
        raise state.OperationalStateError("Workspace user authorization is not initialized")
    return value


def authorized_roles(tenant_id: str, object_id: str) -> tuple[str, ...] | None:
    with state.transaction() as connection:
        user = _directory(connection, tenant_id)["users"].get(canonical_id(object_id))
        if user is None or user.get("enabled") is not True:
            return None
        if user.get("role") not in ALLOWED_ROLES:
            raise state.OperationalStateError("Stored user role is invalid")
        return (user["role"],)


def _require_current_admin(directory: dict, actor: Principal) -> None:
    user = directory["users"].get(actor.object_id)
    if not user or user.get("enabled") is not True or user.get("role") != "admin":
        raise HTTPException(403, "Administrator access is required")


def _public_directory(directory: dict) -> dict:
    return {"tenant_id": directory["tenant_id"], "revision": directory["revision"], "users": sorted(directory["users"].values(), key=lambda user: (user["display_name"].casefold(), user["object_id"]))}


def list_users(actor: Principal) -> dict:
    with state.transaction() as connection:
        directory = _directory(connection, actor.tenant_id)
        _require_current_admin(directory, actor)
        return _public_directory(directory)


def save_user(actor: Principal, *, object_id: str, display_name: str, role: str, enabled: bool, expected_revision: int, create: bool, audit_id: str) -> dict:
    object_id = canonical_id(object_id)
    if role not in ALLOWED_ROLES or type(enabled) is not bool or type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Invalid user role, enabled state, or revision")
    display_name = display_name.strip()
    if not display_name or len(display_name) > 160 or any(ord(char) < 32 for char in display_name):
        raise ValueError("Display name must contain 1-160 printable characters")
    with state.transaction() as connection:
        directory = _directory(connection, actor.tenant_id)
        _require_current_admin(directory, actor)
        if directory["revision"] != expected_revision:
            raise HTTPException(409, "User list changed; refresh before saving")
        before = directory["users"].get(object_id)
        if create and before is not None:
            raise HTTPException(409, "User already exists; edit their existing access")
        if not create and before is None:
            raise HTTPException(404, "User not found")
        if create and len(directory["users"]) >= 1000:
            raise HTTPException(409, "Workspace user limit reached")
        after = {"object_id": object_id, "display_name": display_name, "role": role, "enabled": enabled}
        directory["users"][object_id] = after
        if not any(user["enabled"] and user["role"] == "admin" for user in directory["users"].values()):
            raise HTTPException(409, "The last active administrator cannot be removed or demoted")
        directory["revision"] += 1
        state.append_event(connection, "user_access_audit", {
            "action": "create" if create else "update", "actor": actor.as_dict(),
            "api_request_audit_id": audit_id, "before": before, "after": after,
            "revision": directory["revision"], "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        state.put_document(connection, "access_control", "users", directory)
        return _public_directory(directory)
