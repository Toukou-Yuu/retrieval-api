from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now().astimezone().isoformat()


class SQLiteRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
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
            )

    def health_check(self) -> bool:
        with self.connect() as conn:
            conn.execute("SELECT 1").fetchone()
        return True

    def create_collection(self, data: dict[str, Any]) -> bool:
        now = utc_now()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT * FROM collections WHERE name = ?", (data["name"],)
            ).fetchone()
            if existing:
                return False
            conn.execute(
                """
                INSERT INTO collections (
                  name, description, embedding_model, embedding_dimension,
                  chunk_strategy, metadata_schema_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["name"],
                    data.get("description"),
                    data["embedding_model"],
                    data["embedding_dimension"],
                    data["chunk_strategy"],
                    json.dumps(data.get("metadata_schema"), ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return True

    def get_collection(self, name: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def list_collections(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM collections ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def delete_collection(self, name: str) -> dict[str, int]:
        with self.connect() as conn:
            documents = conn.execute(
                "SELECT COUNT(*) AS count FROM documents WHERE collection = ?", (name,)
            ).fetchone()["count"]
            chunks = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE collection = ?", (name,)
            ).fetchone()["count"]
            conn.execute("DELETE FROM keyword_index WHERE collection = ?", (name,))
            conn.execute("DELETE FROM chunks WHERE collection = ?", (name,))
            conn.execute("DELETE FROM documents WHERE collection = ?", (name,))
            conn.execute("DELETE FROM collections WHERE name = ?", (name,))
        return {"documents": documents, "chunks": chunks}


def decode_collection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row["name"],
        "description": row["description"],
        "embedding_model": row["embedding_model"],
        "embedding_dimension": row["embedding_dimension"],
        "chunk_strategy": row["chunk_strategy"],
        "metadata_schema": json.loads(row["metadata_schema_json"] or "null"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
