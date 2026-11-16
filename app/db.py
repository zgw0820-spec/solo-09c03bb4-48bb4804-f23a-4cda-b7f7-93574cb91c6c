"""SQLite connection management.

Two API instances share one SQLite file (e.g. via a Docker volume). Every
request opens its own connection; WAL mode + a busy timeout let multiple
readers and one writer cooperate across processes.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

DB_PATH = os.environ.get("RAIN_DB_PATH", "/data/rainfall.db")
BUSY_TIMEOUT_MS = int(os.environ.get("RAIN_BUSY_TIMEOUT_MS", "10000"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    group_id        TEXT PRIMARY KEY,
    forecast_time   TEXT NOT NULL,
    rows            INTEGER NOT NULL,
    cols            INTEGER NOT NULL,
    thresholds      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS members (
    group_id    TEXT NOT NULL REFERENCES groups(group_id),
    member_idx  INTEGER NOT NULL,
    member_id   TEXT NOT NULL,
    PRIMARY KEY (group_id, member_idx),
    UNIQUE (group_id, member_id)
);

CREATE TABLE IF NOT EXISTS member_revisions (
    group_id     TEXT NOT NULL,
    member_idx   INTEGER NOT NULL,
    revision     INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    rain_blob    BLOB NOT NULL,
    miss_blob    BLOB NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (group_id, member_idx, revision)
);

CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id     TEXT NOT NULL,
    revision_map TEXT NOT NULL,
    is_current   INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (group_id, revision_map)
);

CREATE TABLE IF NOT EXISTS product_thresholds (
    product_id        INTEGER NOT NULL REFERENCES products(product_id),
    threshold         REAL NOT NULL,
    probability_blob  BLOB NOT NULL,
    denom_blob        BLOB NOT NULL,
    PRIMARY KEY (product_id, threshold)
);

CREATE INDEX IF NOT EXISTS idx_products_group_current
    ON products(group_id, is_current);
"""


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(
        DB_PATH,
        timeout=BUSY_TIMEOUT_MS / 1000.0,
        isolation_level=None,  # explicit BEGIN/COMMIT; do not let sqlite3 autotransact
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create schema and switch the file into WAL mode (idempotent)."""
    conn = connect()
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(_SCHEMA)
    finally:
        conn.close()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()
