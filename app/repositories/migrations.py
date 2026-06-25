from __future__ import annotations

import json
import sqlite3

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS collections (
  name TEXT PRIMARY KEY,
  description TEXT,
  embedding_model TEXT NOT NULL,
  embedding_dimension INTEGER NOT NULL,
  chunk_strategy TEXT NOT NULL,
  metadata_schema_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  id TEXT NOT NULL,
  collection TEXT NOT NULL,
  source TEXT NOT NULL,
  doc_type TEXT NOT NULL,
  title TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  indexed_at TEXT,
  PRIMARY KEY (collection, id)
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT NOT NULL,
  collection TEXT NOT NULL,
  document_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  token_count INTEGER,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (collection, id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS keyword_index USING fts5(
  collection,
  document_id,
  chunk_id,
  title,
  text,
  metadata_json,
  tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS index_jobs (
  job_id TEXT PRIMARY KEY,
  collection TEXT NOT NULL,
  action TEXT NOT NULL,
  document_id TEXT,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 5,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS search_logs (
  id TEXT PRIMARY KEY,
  collection TEXT,
  query TEXT NOT NULL,
  mode TEXT NOT NULL,
  top_k INTEGER NOT NULL,
  filters_json TEXT,
  result_count INTEGER NOT NULL,
  latency_ms INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
"""


def run_migrations(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0 and _table_exists(conn, "collections"):
        _set_schema_version(conn, 1)
        version = 1
    if version < 1:
        conn.executescript(SCHEMA_V1)
        _set_schema_version(conn, 1)
        version = 1
    if version < 2:
        _migrate_to_v2(conn)
        _set_schema_version(conn, 2)
        version = 2
    if version < 3:
        _migrate_to_v3(conn)
        _set_schema_version(conn, 3)


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    columns = _column_names(conn, "collections")
    if "embedding_normalized" not in columns:
        conn.execute(
            "ALTER TABLE collections ADD COLUMN embedding_normalized INTEGER NOT NULL DEFAULT 1"
        )
    if "embedding_distance" not in columns:
        conn.execute(
            "ALTER TABLE collections ADD COLUMN embedding_distance TEXT NOT NULL DEFAULT 'Cosine'"
        )
    if "embedding_contract_version" not in columns:
        conn.execute(
            "ALTER TABLE collections ADD COLUMN embedding_contract_version "
            "TEXT NOT NULL DEFAULT 'embedding-api.v1'"
        )
    if "embedding_metadata_json" not in columns:
        conn.execute("ALTER TABLE collections ADD COLUMN embedding_metadata_json TEXT")
    rows = conn.execute(
        """
        SELECT name, embedding_model, embedding_dimension, created_at
        FROM collections
        """
    ).fetchall()
    for row in rows:
        metadata = {
            "provider": "embedding-api",
            "model": row["embedding_model"],
            "dimension": row["embedding_dimension"],
            "normalized": True,
            "distance": "Cosine",
            "contract_version": "embedding-api.v1",
            "created_at": row["created_at"],
            "resolved_from": "legacy",
            "validated": False,
        }
        conn.execute(
            """
            UPDATE collections
            SET embedding_metadata_json = ?
            WHERE name = ? AND embedding_metadata_json IS NULL
            """,
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True), row["name"]),
        )


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    columns = _column_names(conn, "collections")
    if "index_status" not in columns:
        conn.execute(
            "ALTER TABLE collections ADD COLUMN index_status TEXT NOT NULL DEFAULT 'ready'"
        )
    if "last_indexed_at" not in columns:
        conn.execute("ALTER TABLE collections ADD COLUMN last_indexed_at TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {version}")
