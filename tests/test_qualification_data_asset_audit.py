from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from scripts import audit_qualification_data_assets as audit
from scripts.register_qualification_data_assets import build_asset_plan


def _plan():
    candidates = audit._read_yaml(audit.DEFAULT_CANDIDATES)
    catalog = audit._read_yaml(audit.DEFAULT_CATALOG)
    return build_asset_plan(
        candidates,
        audit.build_profile_report(candidates, catalog),
        datastore="mlops_blob",
        version=audit.DEFAULT_VERSION,
        profile_run_id=catalog["profile_run_id"],
    )


class FakeData:
    def __init__(self, plan, *, bad_hash: bool = False) -> None:
        self.assets = {}
        for item in plan:
            tags = deepcopy(item["tags"])
            if bad_hash and not self.assets:
                tags["content_sha256"] = "0" * 64
            path = item["path"].replace(
                "azureml://datastores/",
                "azureml://subscriptions/sub/resourcegroups/rg/"
                "workspaces/ws/datastores/",
            )
            self.assets[(item["name"], item["version"])] = SimpleNamespace(
                version=item["version"],
                type="uri_file",
                path=path,
                tags=tags,
            )

    def get(self, *, name, version):
        return self.assets[(name, version)]


def test_catalog_builds_eleven_assets_covering_fifteen_scenarios() -> None:
    plan = _plan()

    report = audit.audit_assets(SimpleNamespace(data=FakeData(plan)), plan)

    assert len(plan) == 11
    assert report["asset_count"] == 11
    assert report["scenario_count"] == 15
    assert report["all_passed"] is True
    assert report["errors"] == []


def test_asset_hash_tag_mismatch_fails_audit() -> None:
    plan = _plan()

    report = audit.audit_assets(
        SimpleNamespace(data=FakeData(plan, bad_hash=True)),
        plan,
    )

    assert report["all_passed"] is False
    assert any("hashes" in error for error in report["errors"])
