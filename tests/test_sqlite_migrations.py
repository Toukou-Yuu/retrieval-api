import sqlite3

from app.repositories.migrations import SCHEMA_V1
from app.repositories.sqlite_repository import SQLiteRepository


def test_fresh_database_runs_all_migrations(tmp_path):
    repo = SQLiteRepository(tmp_path / "retrieval.sqlite3")

    with repo.connect() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(collections)").fetchall()
        }

    assert version == 3
    assert {
        "embedding_normalized",
        "embedding_distance",
        "embedding_contract_version",
        "embedding_metadata_json",
        "index_status",
        "last_indexed_at",
    } <= columns


def test_unversioned_v1_database_migrates_existing_collection(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_V1)
        conn.execute(
            """
            INSERT INTO collections (
              name, description, embedding_model, embedding_dimension,
              chunk_strategy, metadata_schema_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy",
                "legacy collection",
                "legacy-model",
                768,
                "plain_text",
                None,
                "2026-06-01T00:00:00+00:00",
                "2026-06-01T00:00:00+00:00",
            ),
        )

    repo = SQLiteRepository(db_path)
    migrated = repo.get_collection("legacy")

    assert migrated is not None
    assert migrated["embedding_normalized"] == 1
    assert migrated["embedding_distance"] == "Cosine"
    assert migrated["embedding_contract_version"] == "embedding-api.v1"
    assert migrated["index_status"] == "ready"

    with repo.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3

    reinitialized = SQLiteRepository(db_path)
    assert reinitialized.get_collection("legacy") == migrated
