"""Transactional state shared by API workers and the controller on one host."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID


_configured_path: str = ""


class OperationalStateError(RuntimeError):
    """Operational state is unavailable or requires an explicit migration."""


def _enable_wal(connection: sqlite3.Connection) -> None:
    # Changing journal mode can return SQLITE_BUSY without invoking busy_timeout
    # when multiple processes open a new database. Retry only that initialization.
    deadline = time.monotonic() + 10
    while True:
        try:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            if mode != "wal":
                mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode != "wal":
                raise OperationalStateError("Operational state requires WAL journaling")
            return
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", None)
            busy = (code & 255 in {5, 6}) if code is not None else str(exc) in {
                "database is locked", "database table is locked",
            }
            if not busy or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def configure_database(path: str) -> None:
    global _configured_path
    _configured_path = path.strip()
    database_path()


def database_path() -> Path | None:
    raw = os.environ.get("MLOPS_OPERATIONAL_STATE_DB", "").strip() or _configured_path
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
        raise OperationalStateError("MLOPS_OPERATIONAL_STATE_DB must be an absolute local-disk path")
    return path.resolve()


@contextmanager
def transaction(*, path: Path | None = None) -> Iterator[sqlite3.Connection]:
    target = path or database_path()
    if target is None:
        raise OperationalStateError("Transactional operational state is not configured")
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=10, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        _enable_wal(connection)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "namespace TEXT NOT NULL, record_id TEXT NOT NULL, payload TEXT NOT NULL, "
            "PRIMARY KEY(namespace, record_id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "namespace TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS events_namespace ON events(namespace, sequence)"
        )
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_document(connection: sqlite3.Connection, namespace: str, record_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT payload FROM documents WHERE namespace=? AND record_id=?",
        (namespace, record_id),
    ).fetchone()
    if row is None:
        return None
    value = json.loads(row[0])
    if not isinstance(value, dict):
        raise OperationalStateError("Operational state document must be an object")
    return value


def put_document(connection: sqlite3.Connection, namespace: str, record_id: str, payload: dict[str, Any]) -> None:
    connection.execute(
        "INSERT INTO documents(namespace, record_id, payload) VALUES(?,?,?) "
        "ON CONFLICT(namespace, record_id) DO UPDATE SET payload=excluded.payload",
        (namespace, record_id, json.dumps(payload, sort_keys=True, allow_nan=False)),
    )


def load_events(connection: sqlite3.Connection, namespace: str) -> list[dict[str, Any]]:
    return [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT payload FROM events WHERE namespace=? ORDER BY sequence", (namespace,)
        )
    ]


def append_event(connection: sqlite3.Connection, namespace: str, payload: dict[str, Any]) -> None:
    connection.execute(
        "INSERT INTO events(namespace, payload) VALUES(?,?)",
        (namespace, json.dumps(payload, sort_keys=True, allow_nan=False)),
    )


def bind_workspace(binding: dict[str, str], *, initialize: bool = False) -> bool:
    """Verify the workspace, or explicitly bind a new empty controller database."""
    required = {"tenant_id", "subscription_id", "resource_group", "workspace_name"}
    if set(binding) != required or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in binding.values()
    ):
        raise OperationalStateError("A complete, unambiguous workspace binding is required")
    try:
        UUID(binding["tenant_id"])
        UUID(binding["subscription_id"])
    except ValueError as exc:
        raise OperationalStateError("Tenant and subscription must be UUIDs") from exc
    with transaction() as connection:
        existing = get_document(connection, "configuration", "workspace")
        if existing is not None:
            if existing != binding:
                raise OperationalStateError("Operational state belongs to another tenant or workspace")
            return False
        if not initialize:
            raise OperationalStateError("Initialize controller state explicitly before starting the controller")
        if connection.execute("SELECT 1 FROM documents LIMIT 1").fetchone() or connection.execute(
            "SELECT 1 FROM events LIMIT 1"
        ).fetchone():
            raise OperationalStateError("Cannot bind nonempty state without an existing workspace identity")
        put_document(connection, "configuration", "workspace", binding)
        append_event(connection, "controller_audit", {
            "event": "workspace_initialized", "workspace": binding,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        return True


def require_legacy_import(connection: sqlite3.Connection, namespace: str, legacy_exists: bool) -> None:
    if legacy_exists and get_document(connection, "migrations", namespace) is None:
        raise OperationalStateError(
            "Legacy JSON state exists; import it with migrate_operational_state.py before enabling SQLite"
        )
