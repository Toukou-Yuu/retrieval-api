from pathlib import Path

import pytest

from app.config import Settings
from app.errors import ApiError, api_error
from app.integrations.embedding_contract import EmbeddingModelInfo
from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas import CollectionCreate
from app.services.collection_service import CollectionService


def test_collection_creation_resolves_model_contract(client):
    response = client.post(
        "/v1/collections",
        json={"name": "resolved", "chunk_strategy": "plain_text"},
    )

    assert response.status_code == 200
    assert response.json()["embedding"] == {
        "provider": "embedding-api",
        "model": "test-model",
        "dimension": 3,
        "normalized": True,
        "distance": "Cosine",
        "contract_version": "embedding-api.v1",
        "created_at": response.json()["embedding"]["created_at"],
        "resolved_from": "embedding-api:/v1/models",
        "validated": True,
    }


def test_existing_qdrant_contract_mismatch_is_rejected(client):
    client.post("/v1/collections", json={"name": "mismatch"})
    client.fake_qdrant.collections["mismatch"]["dimension"] = 4

    response = client.post("/v1/collections", json={"name": "mismatch"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "VECTOR_STORE_CONTRACT_MISMATCH"


class OfflineEmbedding:
    def get_model(self, name: str | None = None) -> EmbeddingModelInfo:
        raise AssertionError("offline creation must not query embedding-api")


class MemoryQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, tuple[int, str]] = {}

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, dimension: int, distance: str = "Cosine") -> None:
        self.collections[name] = (dimension, distance)

    def get_collection_vector_config(self, name: str) -> tuple[int, str]:
        return self.collections[name]


def test_offline_collection_creation_requires_explicit_dimension(tmp_path: Path):
    settings = Settings(
        retrieval_data_dir=tmp_path,
        validate_embedding_contract_on_collection_create=False,
    )
    service = CollectionService(
        SQLiteRepository(settings.sqlite_path),
        MemoryQdrant(),
        OfflineEmbedding(),
        settings,
    )

    with pytest.raises(ApiError) as exc_info:
        service.create(CollectionCreate(name="offline"))

    assert exc_info.value.code == "COLLECTION_EMBEDDING_CONTRACT_INVALID"

    created = service.create(CollectionCreate(name="offline", embedding_dimension=384))

    assert created["embedding"]["dimension"] == 384
    assert created["embedding"]["validated"] is False


class FailingQdrant(MemoryQdrant):
    def create_collection(self, name: str, dimension: int, distance: str = "Cosine") -> None:
        raise api_error(502, "QDRANT_UNAVAILABLE", "Qdrant is unavailable")


def test_collection_creation_rolls_back_metadata_when_qdrant_creation_fails(tmp_path: Path):
    settings = Settings(
        retrieval_data_dir=tmp_path,
        validate_embedding_contract_on_collection_create=False,
    )
    repo = SQLiteRepository(settings.sqlite_path)
    service = CollectionService(repo, FailingQdrant(), OfflineEmbedding(), settings)

    with pytest.raises(ApiError, match="Qdrant is unavailable"):
        service.create(CollectionCreate(name="failed", embedding_dimension=384))

    assert repo.get_collection("failed") is None
