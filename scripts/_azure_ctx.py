"""Shared Azure context loader.

All scripts and pipeline submitters must source the Azure subscription /
resource group / workspace / compute from environment variables (or an
explicit ``.env`` file), never from hardcoded values. This module is the
single chokepoint.

Environment variables (required unless noted):
    AZURE_SUBSCRIPTION_ID    Azure subscription GUID
    AZURE_RESOURCE_GROUP     Resource group containing the AML workspace
    AZURE_WORKSPACE_NAME     Azure ML workspace name
    AZURE_COMPUTE            Default compute target name
    MLOPS_STATE_DIR          (optional) Override for ~/.mlops state directory

Phase 1 of the production-hardening plan. Fails closed: missing required
variables raise ``MissingAzureContextError`` rather than silently falling
back to baked-in defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_ENV_VARS: tuple[str, ...] = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_WORKSPACE_NAME",
    "AZURE_COMPUTE",
)


class MissingAzureContextError(RuntimeError):
    """Raised when one or more required Azure context env vars are missing."""


@dataclass(frozen=True)
class AzureContext:
    """Resolved Azure context. Always populated; never holds ``None`` fields."""

    subscription_id: str
    resource_group: str
    workspace_name: str
    compute: str

    def as_cli_args(self) -> list[str]:
        """Render as ``--subscription_id ... --compute ...`` argument list."""
        return [
            "--subscription_id", self.subscription_id,
            "--resource_group", self.resource_group,
            "--workspace_name", self.workspace_name,
            "--compute", self.compute,
        ]


def _missing(env: Iterable[str]) -> list[str]:
    return [name for name in env if not os.environ.get(name)]


def load_azure_context() -> AzureContext:
    """Load Azure context from environment.

    Raises:
        MissingAzureContextError: if any of ``REQUIRED_ENV_VARS`` is unset
        or empty. The error message lists every missing variable so users
        can fix their environment in one shot.
    """
    missing = _missing(REQUIRED_ENV_VARS)
    if missing:
        raise MissingAzureContextError(
            "Missing required Azure context environment variables: "
            + ", ".join(missing)
            + ".\nSet them in your shell or a .env file. See .env.example "
              "in the repo root for the full list."
        )
    return AzureContext(
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
        compute=os.environ["AZURE_COMPUTE"],
    )


def get_state_dir() -> Path:
    """Return the user-state directory (``$MLOPS_STATE_DIR`` or ``~/.mlops``).

    Creates the directory on first call. Used for lock files, audit logs,
    and other operator state that must NOT live inside the repo (where
    ``git clean -fdx`` could destroy audit history).
    """
    override = os.environ.get("MLOPS_STATE_DIR")
    base = Path(override) if override else Path.home() / ".mlops"
    base.mkdir(parents=True, exist_ok=True)
    return base


__all__ = [
    "AzureContext",
    "MissingAzureContextError",
    "REQUIRED_ENV_VARS",
    "load_azure_context",
    "get_state_dir",
]
