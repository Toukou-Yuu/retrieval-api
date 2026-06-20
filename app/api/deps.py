from functools import lru_cache

from app.config import Settings, get_settings
from app.integrations.embedding_http_client import EmbeddingHTTPClient
from app.integrations.qdrant_client import QdrantClient
from app.repositories.sqlite_repository import SQLiteRepository
from app.services.chunk_service import ChunkService
from app.services.collection_service import CollectionService
from app.services.document_service import DocumentService


@lru_cache
def get_repo() -> SQLiteRepository:
    settings = get_settings()
    return SQLiteRepository(settings.sqlite_path)


@lru_cache
def get_qdrant() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(settings.qdrant_url)


@lru_cache
def get_embedding_client() -> EmbeddingHTTPClient:
    settings = get_settings()
    return EmbeddingHTTPClient(settings.embedding_api_url, settings.embedding_timeout_seconds)


def get_collection_service() -> CollectionService:
    return CollectionService(get_repo(), get_qdrant(), get_settings())


def get_app_settings() -> Settings:
    return get_settings()


def get_chunk_service() -> ChunkService:
    return ChunkService()


def get_document_service() -> DocumentService:
    return DocumentService(get_repo(), get_qdrant(), get_embedding_client(), get_chunk_service())
