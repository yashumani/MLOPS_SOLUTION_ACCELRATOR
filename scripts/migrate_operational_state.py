#!/usr/bin/env python3
"""Import stopped JSON request/ledger writers into transactional operational state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.services.submission_request_store import _validate_record  # noqa: E402
from orchestration import operational_state  # noqa: E402
from orchestration.auto_retrain_decision_ledger import validate_decision_record  # noqa: E402


def migrate_state(database: Path, request_root: Path | None, ledgers: list[Path], *, apply: bool) -> dict:
    batches: dict[str, list[dict]] = {}
    if request_root is not None:
        batches["submission_requests"] = [
            _validate_record(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(request_root.glob("req-*.json"))
        ]
    for path in ledgers:
        namespace = "retrain_ledger:" + str(path.resolve())
        batches[namespace] = [
            validate_decision_record(json.loads(line), source=str(path))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    report = {"applied": False, "counts": {key: len(rows) for key, rows in batches.items()}}
    if not apply:
        return report
    with operational_state.transaction(path=database) as connection:
        for namespace, rows in batches.items():
            fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            prior = operational_state.get_document(connection, "migrations", namespace)
            if prior is not None:
                if prior.get("fingerprint") != fingerprint:
                    raise operational_state.OperationalStateError("Legacy state changed after migration; refusing overwrite")
                continue
            if namespace == "submission_requests":
                for row in rows:
                    existing = operational_state.get_document(connection, namespace, row["request_id"])
                    if existing is not None and existing != row:
                        raise operational_state.OperationalStateError("Request migration conflicts with existing state")
                    operational_state.put_document(connection, namespace, row["request_id"], row)
            else:
                if operational_state.load_events(connection, namespace):
                    raise operational_state.OperationalStateError("Ledger migration target must be empty")
                for row in rows:
                    operational_state.append_event(connection, namespace, row)
            operational_state.put_document(connection, "migrations", namespace, {"fingerprint": fingerprint, "count": len(rows)})
    report["applied"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", required=True, type=Path)
    parser.add_argument("--request-root", type=Path)
    parser.add_argument("--ledger", action="append", type=Path, default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-writers-stopped", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.confirm_writers_stopped:
        parser.error("--apply requires --confirm-writers-stopped")
    if not args.state_db.is_absolute():
        parser.error("--state-db must be an absolute local-disk path")
    print(json.dumps(migrate_state(args.state_db, args.request_root, args.ledger, apply=args.apply), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
