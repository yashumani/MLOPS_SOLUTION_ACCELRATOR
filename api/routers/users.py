"""Admin-only workspace user management."""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.core.entra_auth import Principal
from api.core.security import require_admin
from api.services import user_access_service


router = APIRouter(prefix="/api/v1/users", tags=["users"])


class UserChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=160, pattern=r"^[^\x00-\x1f]+$")
    role: Literal["admin", "operator", "viewer"]
    enabled: bool = Field(strict=True)
    expected_revision: int = Field(ge=0, strict=True)

    @field_validator("display_name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Display name cannot be blank")
        return value.strip()


class UserCreate(UserChange):
    object_id: UUID


@router.get("")
def list_users(actor: Principal = Depends(require_admin)):
    return user_access_service.list_users(actor)


@router.post("", status_code=201)
def create_user(change: UserCreate, request: Request, actor: Principal = Depends(require_admin)):
    return user_access_service.save_user(actor, **change.model_dump(exclude={"object_id"}), object_id=str(change.object_id), create=True, audit_id=request.state.audit_id)


@router.put("/{object_id}")
def update_user(object_id: UUID, change: UserChange, request: Request, actor: Principal = Depends(require_admin)):
    return user_access_service.save_user(actor, **change.model_dump(), object_id=str(object_id), create=False, audit_id=request.state.audit_id)
