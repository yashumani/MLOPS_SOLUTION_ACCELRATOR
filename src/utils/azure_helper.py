"""Azure ML client factory with explicit runtime credential modes."""

import os
from typing import Optional

from azure.ai.ml import MLClient
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
)


CREDENTIAL_MODE_ENV = "MLOPS_AZURE_CREDENTIAL_MODE"
CREDENTIAL_MODES = {"operator", "managed_identity", "azureml_obo"}


class _AzureMLOBOCredentialAdapter:
    """Discard Azure Core token options unsupported by older AML OBO clients."""

    def __init__(self, credential: object) -> None:
        self._credential = credential

    def get_token(self, *scopes: str, **_kwargs: object):
        return self._credential.get_token(*scopes)

    def close(self) -> None:
        close = getattr(self._credential, "close", None)
        if callable(close):
            close()


def resolve_credential_mode(mode: str | None = None) -> str:
    normalized = str(
        mode or os.environ.get(CREDENTIAL_MODE_ENV) or "operator"
    ).strip().lower()
    if normalized not in CREDENTIAL_MODES:
        raise ValueError(
            f"{CREDENTIAL_MODE_ENV} must be one of: "
            + ", ".join(sorted(CREDENTIAL_MODES))
        )
    return normalized


def build_credential(mode: str | None = None):
    """Build only the credential class selected for this execution context."""

    selected = resolve_credential_mode(mode)
    if selected == "azureml_obo":
        if not os.environ.get("OBO_ENDPOINT"):
            raise RuntimeError(
                "azureml_obo requires an Azure ML user-identity job with OBO_ENDPOINT"
            )
        from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

        return _AzureMLOBOCredentialAdapter(AzureMLOnBehalfOfCredential())
    if selected == "managed_identity":
        return ManagedIdentityCredential(
            client_id=os.environ.get("AZURE_CLIENT_ID") or None
        )
    return ChainedTokenCredential(
        ManagedIdentityCredential(),
        AzureCliCredential(process_timeout=60),
    )


def get_ml_client(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    tenant_id: Optional[str] = None,  # kept for API compatibility; ignored
    credential_mode: str | None = None,
) -> MLClient:
    """Create an MLClient using the selected explicit credential mode."""
    return MLClient(
        credential=build_credential(credential_mode),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
