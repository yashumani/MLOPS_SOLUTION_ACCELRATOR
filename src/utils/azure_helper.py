from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str, tenant_id: Optional[str] = None) -> MLClient:
    """Create an MLClient using DefaultAzureCredential."""
    credential = DefaultAzureCredential()
    return MLClient(credential=credential, subscription_id=subscription_id, resource_group_name=resource_group, workspace_name=workspace_name)
