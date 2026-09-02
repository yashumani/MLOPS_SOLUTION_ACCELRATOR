#!/usr/bin/env python3
"""Refresh the two shared Azure ML artifact datastore account keys."""

from __future__ import annotations

import argparse
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AccountKeyConfiguration, AzureBlobDatastore
from azure.identity import AzureCliCredential


SUBSCRIPTION_ID = "93044a08-5661-4f1b-b424-5eafe066a9d1"
RESOURCE_GROUP = "mvpv1"
WORKSPACE_NAME = "mlops-accelerator"
ACCOUNT_NAME = "mlopsaccelerat7263606092"
EXPECTED_DATASTORES = {
    "workspaceblobstore": "azureml-blobstore-12b6bc01-e563-4a39-9006-b7bb2efe45b7",
    "workspaceartifactstore": "azureml",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-shared-workspace-change", action="store_true")
    args = parser.parse_args()

    client = MLClient(
        AzureCliCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )
    datastores = []
    for name, expected_container in EXPECTED_DATASTORES.items():
        current = client.datastores.get(name)
        if not isinstance(current, AzureBlobDatastore):
            raise RuntimeError(f"{name} is not an AzureBlobDatastore")
        if current.account_name != ACCOUNT_NAME:
            raise RuntimeError(
                f"{name} account changed: {current.account_name!r} != {ACCOUNT_NAME!r}"
            )
        if current.container_name != expected_container:
            raise RuntimeError(
                f"{name} container changed: "
                f"{current.container_name!r} != {expected_container!r}"
            )
        datastores.append(current)
        print(
            f"Validated {name}: account={current.account_name}, "
            f"container={current.container_name}"
        )

    if not args.apply:
        print("Dry run only; no datastore credentials changed.")
        return 0
    if not args.confirm_shared_workspace_change:
        raise SystemExit(
            "Refusing shared workspace change without "
            "--confirm-shared-workspace-change"
        )

    account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "").strip()
    if not account_key:
        raise SystemExit("AZURE_STORAGE_ACCOUNT_KEY is required for --apply")

    for current in datastores:
        updated = AzureBlobDatastore(
            name=current.name,
            account_name=current.account_name,
            container_name=current.container_name,
            description=current.description,
            tags=current.tags,
            endpoint=current.endpoint,
            protocol=current.protocol,
            credentials=AccountKeyConfiguration(account_key=account_key),
        )
        client.datastores.create_or_update(updated)
        print(f"Refreshed stored account key for {current.name}")
    print("Credential refresh complete; run the bounded datastore canary next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
