from typing import Annotated

from fastapi import APIRouter, Depends

from app import __version__
from app.config import Settings
from app.integrations.embedding_http_client import EmbeddingHTTPClient
from app.integrations.qdrant_client import QdrantClient
from app.repositories.sqlite_repository import SQLiteRepository

from .deps import get_app_settings, get_embedding_client, get_qdrant, get_repo

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "retrieval-api", "version": __version__}


@router.get("/ready")
def ready(
    repo: Annotated[SQLiteRepository, Depends(get_repo)],
    qdrant: Annotated[QdrantClient, Depends(get_qdrant)],
    embedding: Annotated[EmbeddingHTTPClient, Depends(get_embedding_client)],
) -> dict[str, object]:
    dependencies = {
        "sqlite": "ok" if repo.health_check() else "error",
        "qdrant": "ok" if qdrant.health_check() else "error",
        "embedding_api": "ok" if embedding.health_check() else "error",
    }
    status = "ready" if all(value == "ok" for value in dependencies.values()) else "degraded"
    return {"status": status, "dependencies": dependencies}


@router.get("/info")
def info(settings: Annotated[Settings, Depends(get_app_settings)]) -> dict[str, object]:
    return {
        "service": "retrieval-api",
        "version": __version__,
        "embedding_model": settings.default_embedding_model,
        "embedding_dimension": settings.default_embedding_dimension,
        "vector_db": "qdrant",
        "keyword_index": "sqlite_fts5",
        "rerank_enabled": settings.rerank_enabled,
    }
