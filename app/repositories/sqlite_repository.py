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

    def get_document(self, collection: str, document_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection = ? AND id = ?",
                (collection, document_id),
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self, collection: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE collection = ? ORDER BY updated_at DESC",
                (collection,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_document(self, document: dict[str, Any]) -> None:
        now = utc_now()
        existing = self.get_document(document["collection"], document["id"])
        created_at = existing["created_at"] if existing else now
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                  id, collection, source, doc_type, title, text_hash, metadata_json,
                  status, created_at, updated_at, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection, id) DO UPDATE SET
                  source = excluded.source,
                  doc_type = excluded.doc_type,
                  title = excluded.title,
                  text_hash = excluded.text_hash,
                  metadata_json = excluded.metadata_json,
                  status = excluded.status,
                  updated_at = excluded.updated_at,
                  indexed_at = excluded.indexed_at
                """,
                (
                    document["id"],
                    document["collection"],
                    document["source"],
                    document["doc_type"],
                    document["title"],
                    document["text_hash"],
                    json.dumps(document["metadata"], ensure_ascii=False, sort_keys=True),
                    document["status"],
                    created_at,
                    document["updated_at"],
                    now,
                ),
            )

    def replace_chunks(
        self,
        collection: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            old_count = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE collection = ? AND document_id = ?",
                (collection, document_id),
            ).fetchone()["count"]
            conn.execute(
                "DELETE FROM keyword_index WHERE collection = ? AND document_id = ?",
                (collection, document_id),
            )
            conn.execute(
                "DELETE FROM chunks WHERE collection = ? AND document_id = ?",
                (collection, document_id),
            )
            for chunk in chunks:
                metadata_json = json.dumps(
                    chunk["metadata"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                conn.execute(
                    """
                    INSERT INTO chunks (
                      id, collection, document_id, chunk_index, text, text_hash,
                      token_count, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        collection,
                        document_id,
                        chunk["chunk_index"],
                        chunk["text"],
                        chunk["text_hash"],
                        chunk["token_count"],
                        metadata_json,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO keyword_index (
                      collection, document_id, chunk_id, title, text, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        collection,
                        document_id,
                        chunk["id"],
                        chunk["metadata"].get("title", ""),
                        chunk["text"],
                        metadata_json,
                    ),
                )
        return old_count

    def get_chunks(self, collection: str, document_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM chunks WHERE collection = ?"
        params: list[Any] = [collection]
        if document_id:
            sql += " AND document_id = ?"
            params.append(document_id)
        sql += " ORDER BY document_id, chunk_index"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, collection: str, document_id: str) -> int:
        with self.connect() as conn:
            deleted_chunks = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE collection = ? AND document_id = ?",
                (collection, document_id),
            ).fetchone()["count"]
            conn.execute(
                "DELETE FROM keyword_index WHERE collection = ? AND document_id = ?",
                (collection, document_id),
            )
            conn.execute(
                "DELETE FROM chunks WHERE collection = ? AND document_id = ?",
                (collection, document_id),
            )
            conn.execute(
                "DELETE FROM documents WHERE collection = ? AND id = ?",
                (collection, document_id),
            )
        return deleted_chunks

    def update_document_status(self, collection: str, document_id: str, status: str) -> None:
        document = self.get_document(collection, document_id)
        if not document:
            return
        metadata = json.loads(document["metadata_json"])
        metadata["status"] = status
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, metadata_json = ?, updated_at = ?
                WHERE collection = ? AND id = ?
                """,
                (status, metadata_json, now, collection, document_id),
            )
            rows = conn.execute(
                "SELECT * FROM chunks WHERE collection = ? AND document_id = ?",
                (collection, document_id),
            ).fetchall()
            conn.execute(
                "DELETE FROM keyword_index WHERE collection = ? AND document_id = ?",
                (collection, document_id),
            )
            for row in rows:
                chunk_metadata = json.loads(row["metadata_json"])
                chunk_metadata["status"] = status
                chunk_metadata_json = json.dumps(
                    chunk_metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                conn.execute(
                    """
                    UPDATE chunks
                    SET metadata_json = ?, updated_at = ?
                    WHERE collection = ? AND id = ?
                    """,
                    (chunk_metadata_json, now, collection, row["id"]),
                )
                conn.execute(
                    """
                    INSERT INTO keyword_index (
                      collection, document_id, chunk_id, title, text, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        collection,
                        document_id,
                        row["id"],
                        chunk_metadata.get("title", ""),
                        row["text"],
                        chunk_metadata_json,
                    ),
                )

    def insert_job(self, job: dict[str, Any]) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO index_jobs (
                  job_id, collection, action, document_id, payload_json, status,
                  retry_count, max_retries, created_at, updated_at, started_at, finished_at,
                  last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["job_id"],
                    job["collection"],
                    job["action"],
                    job.get("document_id"),
                    json.dumps(job.get("payload", {}), ensure_ascii=False),
                    job["status"],
                    job.get("retry_count", 0),
                    job.get("max_retries", 5),
                    now,
                    now,
                    job.get("started_at"),
                    job.get("finished_at"),
                    job.get("last_error"),
                ),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [job_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE index_jobs SET {assignments} WHERE job_id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM index_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(
        self,
        collection: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM index_jobs WHERE 1=1"
        params: list[Any] = []
        if collection:
            sql += " AND collection = ?"
            params.append(collection)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM index_jobs WHERE job_id = ?", (job_id,))
        return cur.rowcount > 0


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
