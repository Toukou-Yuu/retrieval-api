from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class EmbeddingInputType(StrEnum):
    QUERY = "query"
    DOCUMENT = "document"


class EmbeddingModelInfo(BaseModel):
    name: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    normalized: bool
    recommended_distance: str = Field(min_length=1)
    max_input_tokens: int | None = Field(default=None, gt=0)
    supports_query_document_prefix: bool = False
    contract_version: str = "embedding-api.v1"


class EmbeddingModelsResponse(BaseModel):
    data: list[EmbeddingModelInfo]


class EmbeddingVector(BaseModel):
    index: int = Field(ge=0)
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    normalized: bool
    input_type: EmbeddingInputType
    data: list[EmbeddingVector]
    usage: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_vectors(self) -> EmbeddingResponse:
        indexes = [item.index for item in self.data]
        if sorted(indexes) != list(range(len(self.data))):
            raise ValueError("embedding data indexes must be contiguous and start at zero")
        if any(len(item.embedding) != self.dimension for item in self.data):
            raise ValueError("embedding vector length must match dimension")
        return self
