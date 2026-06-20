from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChunkStrategy(StrEnum):
    PLAIN_TEXT = "plain_text"
    MARKDOWN_SEMANTIC = "markdown_semantic"
    JSON_FIELDS = "json_fields"


class SearchMode(StrEnum):
    DENSE = "dense"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class JobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, description="Collection name.")
    description: str | None = Field(
        default=None,
        description="Human-readable collection description.",
    )
    embedding_model: str | None = Field(default=None, description="Embedding model identifier.")
    embedding_dimension: int | None = Field(
        default=None,
        gt=0,
        description="Embedding vector size.",
    )
    chunk_strategy: ChunkStrategy = Field(default=ChunkStrategy.PLAIN_TEXT)
    metadata_schema: dict[str, Any] | None = Field(default=None)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name is required")
        return value


class DocumentInput(BaseModel):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    doc_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None


class IndexingOptions(BaseModel):
    mode: str = "sync"


class DocumentsUpsertRequest(BaseModel):
    collection: str
    documents: list[DocumentInput]
    indexing: IndexingOptions = Field(default_factory=IndexingOptions)


class Filters(BaseModel):
    status: str | None = "published"
    source: str | None = None
    doc_type: str | None = None
    tags: list[str] | None = None
    date_gte: str | None = None
    date_lte: str | None = None


class SearchRequest(BaseModel):
    collections: list[str]
    query: str = Field(min_length=1)
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=5, gt=0)
    candidate_k: int = Field(default=30, gt=0)
    filters: Filters | None = None
    rerank: bool = True
    include_score_breakdown: bool = True
