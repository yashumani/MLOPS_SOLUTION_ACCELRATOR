"""Azure ML client factory.

Uses ``ChainedTokenCredential(ManagedIdentityCredential, AzureCliCredential)``
instead of ``DefaultAzureCredential``. Rationale:

* Production submitters (compute instance / container) authenticate via the
  workspace-attached managed identity. Tried first.
* Local developers authenticate via ``az login``. Tried second.
* All other ``DefaultAzureCredential`` legs (Visual Studio, environment
  variables holding a static client secret, Azure PowerShell, etc.) are
  intentionally excluded — they are common sources of credential leakage and
  silent fall-throughs in CI.
"""

from typing import Optional

from azure.ai.ml import MLClient
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
)


def _build_credential() -> ChainedTokenCredential:
    return ChainedTokenCredential(
        ManagedIdentityCredential(),
        AzureCliCredential(),
    )


def get_ml_client(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    tenant_id: Optional[str] = None,  # kept for API compatibility; ignored
) -> MLClient:
    """Create an MLClient using the unified credential chain."""
    return MLClient(
        credential=_build_credential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
