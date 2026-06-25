from __future__ import annotations

from typing import Any

from app.config import Settings
from app.errors import ApiError, api_error
from app.integrations.embedding_http_client import EmbeddingHTTPClient
from app.integrations.qdrant_client import QdrantClient
from app.repositories.sqlite_repository import SQLiteRepository, decode_collection, utc_now
from app.schemas import CollectionCreate


class CollectionService:
    def __init__(
        self,
        repo: SQLiteRepository,
        qdrant: QdrantClient,
        embedding: EmbeddingHTTPClient,
        settings: Settings,
    ) -> None:
        self.repo = repo
        self.qdrant = qdrant
        self.embedding = embedding
        self.settings = settings

    def create(self, payload: CollectionCreate) -> dict[str, Any]:
        existing_row = self.repo.get_collection(payload.name)
        if existing_row:
            existing = decode_collection(existing_row)
            self._validate_existing_contract(existing, payload)
            self._ensure_vector_store_contract(
                payload.name,
                existing["embedding"],
            )
            return self._create_response(payload.name, False, existing["embedding"])

        embedding = self._resolve_embedding_contract(payload)
        if self.qdrant.collection_exists(payload.name):
            raise api_error(
                409,
                "COLLECTION_ALREADY_EXISTS",
                f"Qdrant collection already exists: {payload.name}",
            )
        data = payload.model_dump()
        data.update(
            {
                "embedding_model": embedding["model"],
                "embedding_dimension": embedding["dimension"],
                "embedding_normalized": embedding["normalized"],
                "embedding_distance": embedding["distance"],
                "embedding_contract_version": embedding["contract_version"],
                "embedding": embedding,
            }
        )
        created = self.repo.create_collection(data)
        try:
            self.qdrant.create_collection(
                payload.name,
                int(embedding["dimension"]),
                str(embedding["distance"]),
            )
        except ApiError:
            self.repo.delete_collection(payload.name)
            raise
        return self._create_response(payload.name, created, embedding)

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

    def _resolve_embedding_contract(self, payload: CollectionCreate) -> dict[str, Any]:
        model = payload.embedding_model or self.settings.default_embedding_model
        contract_version = (
            payload.embedding_contract_version or self.settings.embedding_api_contract_version
        )
        if contract_version != self.settings.embedding_api_contract_version:
            raise api_error(
                400,
                "COLLECTION_EMBEDDING_CONTRACT_INVALID",
                f"Unsupported embedding contract version: {contract_version}",
            )
        if self.settings.validate_embedding_contract_on_collection_create:
            model_info = self.embedding.get_model(model)
            if model_info.contract_version != contract_version:
                raise api_error(
                    502,
                    "COLLECTION_EMBEDDING_CONTRACT_INVALID",
                    "Embedding API returned an unexpected contract version",
                )
            if (
                payload.embedding_dimension is not None
                and payload.embedding_dimension != model_info.dimension
            ):
                raise api_error(
                    400,
                    "EMBEDDING_DIMENSION_MISMATCH",
                    "Requested embedding dimension does not match the model",
                    {
                        "requested": payload.embedding_dimension,
                        "model_dimension": model_info.dimension,
                    },
                )
            dimension = model_info.dimension
            normalized = (
                payload.embedding_normalized
                if payload.embedding_normalized is not None
                else model_info.normalized
            )
            distance = payload.embedding_distance or model_info.recommended_distance
            validated = True
            resolved_from = "embedding-api:/v1/models"
        else:
            if payload.embedding_dimension is None:
                raise api_error(
                    400,
                    "COLLECTION_EMBEDDING_CONTRACT_INVALID",
                    "embedding_dimension is required when contract validation is disabled",
                )
            dimension = payload.embedding_dimension
            normalized = (
                payload.embedding_normalized
                if payload.embedding_normalized is not None
                else self.settings.default_embedding_normalize
            )
            distance = payload.embedding_distance or self.settings.default_embedding_distance
            validated = False
            resolved_from = "explicit"
        return {
            "provider": "embedding-api",
            "model": model,
            "dimension": dimension,
            "normalized": normalized,
            "distance": distance,
            "contract_version": contract_version,
            "created_at": utc_now(),
            "resolved_from": resolved_from,
            "validated": validated,
        }

    def _validate_existing_contract(
        self,
        existing: dict[str, Any],
        payload: CollectionCreate,
    ) -> None:
        expected_model = payload.embedding_model or self.settings.default_embedding_model
        if existing["embedding_model"] != expected_model:
            raise api_error(
                409,
                "EMBEDDING_MODEL_MISMATCH",
                f"Collection already exists with model {existing['embedding_model']}",
            )
        if (
            payload.embedding_dimension is not None
            and int(existing["embedding_dimension"]) != payload.embedding_dimension
        ):
            raise api_error(
                409,
                "EMBEDDING_DIMENSION_MISMATCH",
                f"Collection already exists with dimension {existing['embedding_dimension']}",
            )
        if (
            payload.embedding_normalized is not None
            and existing["embedding_normalized"] != payload.embedding_normalized
        ):
            raise api_error(
                409,
                "EMBEDDING_NORMALIZE_MISMATCH",
                "Collection already exists with a different normalize setting",
            )
        if (
            payload.embedding_distance is not None
            and existing["embedding_distance"] != payload.embedding_distance
        ):
            raise api_error(
                409,
                "COLLECTION_EMBEDDING_CONTRACT_INVALID",
                "Collection already exists with a different distance setting",
            )
        if (
            payload.embedding_contract_version is not None
            and existing["embedding_contract_version"] != payload.embedding_contract_version
        ):
            raise api_error(
                409,
                "COLLECTION_EMBEDDING_CONTRACT_INVALID",
                "Collection already exists with a different contract version",
            )

    def _ensure_vector_store_contract(
        self,
        name: str,
        embedding: dict[str, Any],
    ) -> None:
        if not self.qdrant.collection_exists(name):
            raise api_error(
                409,
                "VECTOR_STORE_CONTRACT_MISMATCH",
                f"Qdrant collection is missing: {name}",
            )
        dimension, distance = self.qdrant.get_collection_vector_config(name)
        if dimension != int(embedding["dimension"]) or distance != embedding["distance"]:
            raise api_error(
                409,
                "VECTOR_STORE_CONTRACT_MISMATCH",
                "Qdrant collection configuration does not match SQLite metadata",
                {
                    "sqlite": {
                        "dimension": embedding["dimension"],
                        "distance": embedding["distance"],
                    },
                    "qdrant": {"dimension": dimension, "distance": distance},
                },
            )

    def _create_response(
        self,
        name: str,
        created: bool,
        embedding: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "name": name,
            "created": created,
            "embedding": embedding,
            "vector_store": {"type": "qdrant", "collection": name},
        }
