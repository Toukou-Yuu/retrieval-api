from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import Settings
from app.main import app
from app.repositories.sqlite_repository import SQLiteRepository
from app.services.chunk_service import ChunkService
from app.services.collection_service import CollectionService
from app.services.document_service import DocumentService
from app.services.search_service import SearchService


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, int] = {}
        self.points: dict[str, dict[str, dict]] = {}

    def health_check(self) -> bool:
        return True

    def create_collection(self, name: str, dimension: int) -> None:
        self.collections[name] = dimension
        self.points.setdefault(name, {})

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)
        self.points.pop(name, None)

    def upsert_points(self, collection: str, points: list[dict]) -> None:
        self.points.setdefault(collection, {})
        for point in points:
            self.points[collection][point["id"]] = point

    def delete_document_points(self, collection: str, document_id: str) -> None:
        existing = self.points.setdefault(collection, {})
        for point_id, point in list(existing.items()):
            if point["payload"]["document_id"] == document_id:
                del existing[point_id]

    def set_document_payload(self, collection: str, document_id: str, payload: dict) -> None:
        for point in self.points.setdefault(collection, {}).values():
            if point["payload"]["document_id"] == document_id:
                point["payload"].update(payload)

    def search(self, collection: str, vector: list[float], limit: int, filters: dict | None = None):
        results = []
        for point in self.points.get(collection, {}).values():
            payload = point["payload"]
            if filters and filters.get("status") and payload.get("status") != filters["status"]:
                continue
            score = sum(a * b for a, b in zip(point["vector"], vector, strict=True))
            results.append({"id": point["id"], "score": score, "payload": payload})
        return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


class FakeEmbedding:
    model = "test-model"

    def health_check(self) -> bool:
        return True

    def embed(self, texts: list[str], normalize: bool = True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "subscription" in lowered else 0.0,
                    1.0 if "airport" in lowered else 0.0,
                    1.0,
                ]
            )
        return self.model, 3, vectors


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    settings = Settings(
        retrieval_data_dir=tmp_path,
        default_embedding_model="test-model",
        default_embedding_dimension=3,
    )
    repo = SQLiteRepository(settings.sqlite_path)
    qdrant = FakeQdrant()
    embedding = FakeEmbedding()
    app.dependency_overrides[deps.get_app_settings] = lambda: settings
    app.dependency_overrides[deps.get_repo] = lambda: repo
    app.dependency_overrides[deps.get_qdrant] = lambda: qdrant
    app.dependency_overrides[deps.get_embedding_client] = lambda: embedding
    app.dependency_overrides[deps.get_collection_service] = lambda: CollectionService(
        repo,
        qdrant,
        settings,
    )
    app.dependency_overrides[deps.get_document_service] = lambda: DocumentService(
        repo,
        qdrant,
        embedding,
        ChunkService(),
    )
    app.dependency_overrides[deps.get_search_service] = lambda: SearchService(
        repo,
        qdrant,
        embedding,
        settings,
    )
    with TestClient(app) as test_client:
        test_client.fake_qdrant = qdrant
        test_client.fake_embedding = embedding
        yield test_client
    app.dependency_overrides.clear()
