from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    retrieval_data_dir: Path = Path("/data")
    qdrant_url: str = "http://qdrant:6333"
    embedding_api_url: str = "http://embedding-api:8100"
    embedding_api_contract_version: str = "embedding-api.v1"
    embedding_timeout_seconds: float = 30.0
    default_embedding_model: str = "BAAI/bge-m3"
    default_embedding_dimension: int | None = None
    default_embedding_normalize: bool = True
    default_embedding_distance: str = "Cosine"
    validate_embedding_contract_on_startup: bool = False
    validate_embedding_contract_on_collection_create: bool = True
    validate_embedding_contract_on_search: bool = True
    allow_multi_model_search: bool = False
    rerank_enabled: bool = False
    rrf_k: int = 60
    max_top_k: int = 50
    max_candidate_k: int = 200

    @property
    def sqlite_path(self) -> Path:
        return self.retrieval_data_dir / "retrieval.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
