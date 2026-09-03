"""Deployment-specific authentication and server-owned authorization."""

import secrets
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from api.core.config import settings
from api.core.entra_auth import Principal, authenticate_token
from orchestration import operational_state

_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


def _write_audit(audit: dict) -> None:
    with operational_state.transaction() as connection:
        operational_state.put_document(connection, "request_audit", audit["audit_id"], audit)


def complete_audit(audit_id: str, status_code: int) -> None:
    with operational_state.transaction() as connection:
        record = operational_state.get_document(connection, "request_audit", audit_id)
        if record is None:
            raise RuntimeError("Authenticated request audit record is missing")
        record.update({
            "status": "completed", "status_code": status_code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        operational_state.put_document(connection, "request_audit", audit_id, record)


def actor_tags(request: Request | None) -> dict[str, str]:
    principal = getattr(getattr(request, "state", None), "principal", None)
    if not isinstance(principal, Principal):
        return {}
    return {**principal.submission_tags(), "api_request_audit_id": request.state.audit_id}


async def verify_api_key(
    api_key: str = Depends(_header),
    request: Request = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str | Principal:
    if settings.api_deployment_profile.strip().lower() == "multi_user":
        if not isinstance(credentials, HTTPAuthorizationCredentials) or credentials.scheme.lower() != "bearer":
            raise HTTPException(401, "Bearer access token required", headers={"WWW-Authenticate": "Bearer"})
        principal = await run_in_threadpool(authenticate_token, credentials.credentials, settings)
        if request is None:
            raise HTTPException(503, "Authenticated request context is unavailable")
        audit_id = str(uuid4())
        audit = {
            "audit_id": audit_id, "actor": principal.as_dict(),
            "method": request.method, "path": request.url.path,
            "received_at": datetime.now(timezone.utc).isoformat(), "status": "received",
            "workspace": settings.azure_workspace_name,
        }
        try:
            await run_in_threadpool(_write_audit, audit)
        except Exception as exc:
            raise HTTPException(503, "Durable request audit is unavailable") from exc
        request.state.principal = principal
        request.state.audit_id = audit_id
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not {"operator", "admin"}.intersection(principal.roles):
            raise HTTPException(403, "Operator access is required")
        return principal
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server",
        )
    if not api_key or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


async def require_admin(principal=Depends(verify_api_key)) -> Principal:
    if not isinstance(principal, Principal) or "admin" not in principal.roles:
        raise HTTPException(403, "Administrator access is required")
    return principal


async def require_config_mutation_enabled() -> None:
    """Refuse checkout mutation unless the deployment explicitly enables it."""
    if not settings.api_config_mutation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Config mutation is disabled for this deployment; use the "
                "reviewed Git configuration workflow"
            ),
        )
