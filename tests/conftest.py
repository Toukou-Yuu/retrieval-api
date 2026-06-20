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
        return []


class FakeEmbedding:
    def health_check(self) -> bool:
        return True

    def embed(self, texts: list[str], normalize: bool = True):
        return "test-model", 3, [[1.0, 0.0, 0.0] for _ in texts]


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
    with TestClient(app) as test_client:
        test_client.fake_qdrant = qdrant
        yield test_client
    app.dependency_overrides.clear()
