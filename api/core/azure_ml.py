"""Azure ML client singleton."""

from azure.ai.ml import MLClient
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
)

from api.core.config import settings

_ml_client: MLClient | None = None


def get_ml_client() -> MLClient:
    """Return a cached MLClient instance (created on first call)."""
    global _ml_client
    if _ml_client is None:
        _ml_client = MLClient(
            credential=ChainedTokenCredential(
                ManagedIdentityCredential(),
                AzureCliCredential(),
            ),
            subscription_id=settings.azure_subscription_id,
            resource_group_name=settings.azure_resource_group,
            workspace_name=settings.azure_workspace_name,
        )
    return _ml_client
