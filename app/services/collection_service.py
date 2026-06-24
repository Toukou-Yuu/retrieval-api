from __future__ import annotations

from typing import Any

from app.config import Settings
from app.errors import api_error
from app.integrations.qdrant_client import QdrantClient
from app.repositories.sqlite_repository import SQLiteRepository, decode_collection
from app.schemas import CollectionCreate


class CollectionService:
    def __init__(self, repo: SQLiteRepository, qdrant: QdrantClient, settings: Settings) -> None:
        self.repo = repo
        self.qdrant = qdrant
        self.settings = settings

    def create(self, payload: CollectionCreate) -> dict[str, Any]:
        data = payload.model_dump()
        data["embedding_model"] = data["embedding_model"] or self.settings.default_embedding_model
        data["embedding_dimension"] = (
            data["embedding_dimension"] or self.settings.default_embedding_dimension
        )
        existing = self.repo.get_collection(payload.name)
        if existing:
            if existing["embedding_model"] != data["embedding_model"]:
                raise api_error(
                    409,
                    "EMBEDDING_MODEL_MISMATCH",
                    f"Collection already exists with model {existing['embedding_model']}",
                )
            if int(existing["embedding_dimension"]) != int(data["embedding_dimension"]):
                raise api_error(
                    409,
                    "DIMENSION_MISMATCH",
                    f"Collection already exists with dimension {existing['embedding_dimension']}",
                )
            return {"name": payload.name, "created": False}
        self.qdrant.create_collection(payload.name, int(data["embedding_dimension"]))
        created = self.repo.create_collection(data)
        return {"name": payload.name, "created": created}

    def list(self) -> list[dict[str, Any]]:
        return [decode_collection(row) for row in self.repo.list_collections()]

    def get(self, name: str) -> dict[str, Any]:
        row = self.repo.get_collection(name)
        if not row:
            raise api_error(404, "COLLECTION_NOT_FOUND", f"Collection not found: {name}")
        return decode_collection(row)

    def delete(self, name: str) -> dict[str, Any]:
        if not self.repo.get_collection(name):
            raise api_error(404, "COLLECTION_NOT_FOUND", f"Collection not found: {name}")
        counts = self.repo.delete_collection(name)
        self.qdrant.delete_collection(name)
        return {
            "name": name,
            "deleted": True,
            "deleted_documents": counts["documents"],
            "deleted_chunks": counts["chunks"],
        }
