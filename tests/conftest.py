from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import Settings
from app.main import app
from app.repositories.sqlite_repository import SQLiteRepository
from app.services.collection_service import CollectionService


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, int] = {}

    def health_check(self) -> bool:
        return True

    def create_collection(self, name: str, dimension: int) -> None:
        self.collections[name] = dimension

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    def upsert_points(self, collection: str, points: list[dict]) -> None:
        pass

    def delete_document_points(self, collection: str, document_id: str) -> None:
        pass

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
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
