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
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> dict[str, object]:
    dependencies = {
        "sqlite": {
            "status": "ok" if repo.health_check() else "error",
            "path": str(settings.sqlite_path),
        },
        "qdrant": {
            "status": "ok" if qdrant.health_check() else "error",
            "url": settings.qdrant_url,
        },
        "embedding_api": {
            "status": "ok" if embedding.ready_check() else "error",
            "url": settings.embedding_api_url,
            "contract_version": settings.embedding_api_contract_version,
            "default_model": settings.default_embedding_model,
        },
    }
    status = (
        "ready"
        if all(dependency["status"] == "ok" for dependency in dependencies.values())
        else "degraded"
    )
    return {"status": status, "dependencies": dependencies}


@router.get("/info")
def info(settings: Annotated[Settings, Depends(get_app_settings)]) -> dict[str, object]:
    return {
        "service": "retrieval-api",
        "version": __version__,
        "contract": {"embedding_api": settings.embedding_api_contract_version},
        "defaults": {
            "embedding_model": settings.default_embedding_model,
            "embedding_normalized": settings.default_embedding_normalize,
            "embedding_distance": settings.default_embedding_distance,
        },
        "stores": {"vector_db": "qdrant", "keyword_index": "sqlite_fts5"},
        "limits": {
            "max_top_k": settings.max_top_k,
            "max_candidate_k": settings.max_candidate_k,
        },
        "embedding_model": settings.default_embedding_model,
        "embedding_dimension": settings.default_embedding_dimension,
        "vector_db": "qdrant",
        "keyword_index": "sqlite_fts5",
        "rerank_enabled": settings.rerank_enabled,
    }
