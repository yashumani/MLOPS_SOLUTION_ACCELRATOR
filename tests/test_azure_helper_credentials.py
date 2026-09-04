from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils import azure_helper


def test_azureml_obo_is_explicit_and_strips_unsupported_token_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class OboCredential:
        def get_token(self, *scopes: str, **kwargs: object) -> str:
            calls.append((scopes, kwargs))
            return "token"

    monkeypatch.setenv("OBO_ENDPOINT", "https://identity.test")
    monkeypatch.setattr(
        "azure.ai.ml.identity.AzureMLOnBehalfOfCredential",
        OboCredential,
    )

    credential = azure_helper.build_credential("azureml_obo")

    assert (
        credential.get_token(
            "https://management.azure.com/.default",
            tenant_id="ignored",
        )
        == "token"
    )
    assert calls == [(('https://management.azure.com/.default',), {})]


def test_azureml_obo_refuses_non_job_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OBO_ENDPOINT", raising=False)

    with pytest.raises(RuntimeError, match="OBO_ENDPOINT"):
        azure_helper.build_credential("azureml_obo")


def test_managed_identity_mode_never_builds_cli_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()
    captured: dict[str, object] = {}
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-id")
    monkeypatch.setattr(
        azure_helper,
        "ManagedIdentityCredential",
        lambda **kwargs: captured.setdefault("managed_kwargs", kwargs) and expected,
    )
    monkeypatch.setattr(
        azure_helper,
        "AzureCliCredential",
        lambda **_kwargs: pytest.fail("managed mode created an Azure CLI credential"),
    )

    assert azure_helper.build_credential("managed_identity") is expected
    assert captured["managed_kwargs"] == {"client_id": "client-id"}


def test_operator_mode_keeps_managed_identity_then_cli_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = object()
    cli = object()
    chained = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        azure_helper,
        "ManagedIdentityCredential",
        lambda **_kwargs: managed,
    )
    monkeypatch.setattr(
        azure_helper,
        "AzureCliCredential",
        lambda **kwargs: captured.setdefault("cli_kwargs", kwargs) and cli,
    )
    monkeypatch.setattr(
        azure_helper,
        "ChainedTokenCredential",
        lambda *credentials: captured.setdefault("credentials", credentials) and chained,
    )

    assert azure_helper.build_credential("operator") is chained
    assert captured["credentials"] == (managed, cli)
    assert captured["cli_kwargs"] == {"process_timeout": 60}


def test_ml_client_receives_selected_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = object()
    expected = object()
    captured = SimpleNamespace()
    monkeypatch.setattr(
        azure_helper,
        "build_credential",
        lambda mode: credential if mode == "azureml_obo" else None,
    )

    def make_client(**kwargs: object):
        captured.kwargs = kwargs
        return expected

    monkeypatch.setattr(azure_helper, "MLClient", make_client)

    assert (
        azure_helper.get_ml_client(
            "subscription",
            "resource-group",
            "workspace",
            credential_mode="azureml_obo",
        )
        is expected
    )
    assert captured.kwargs == {
        "credential": credential,
        "subscription_id": "subscription",
        "resource_group_name": "resource-group",
        "workspace_name": "workspace",
    }


def test_unknown_credential_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="MLOPS_AZURE_CREDENTIAL_MODE"):
        azure_helper.build_credential("implicit-fallback")
