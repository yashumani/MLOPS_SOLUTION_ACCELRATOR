"""Azure ML Studio URL builder utility.

Centralises the logic for constructing portal links so that every
response model that references a job can include a clickable Studio URL.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from azure.ai.ml import MLClient

logger = logging.getLogger(__name__)


def build_studio_url(ml_client: "MLClient", job_name: str) -> str:
    """Return a portal deep-link to the given job in Azure ML Studio.

    URL format:
        https://ml.azure.com/runs/{job_name}
            ?wsid=/subscriptions/{sub}/resourceGroups/{rg}
            /providers/Microsoft.MachineLearningServices/workspaces/{ws}

    Parameters
    ----------
    ml_client : azure.ai.ml.MLClient
        An authenticated MLClient – subscription / resource-group / workspace
        fields are read from the client rather than from env vars.
    job_name : str
        The Azure ML job name (``submitted.name``).
    """
    sub = ml_client.subscription_id
    rg = ml_client.resource_group_name
    ws = ml_client.workspace_name

    return (
        f"https://ml.azure.com/runs/{job_name}"
        f"?wsid=/subscriptions/{sub}"
        f"/resourceGroups/{rg}"
        f"/providers/Microsoft.MachineLearningServices"
        f"/workspaces/{ws}"
    )
