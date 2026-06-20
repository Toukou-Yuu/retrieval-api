from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.config import Settings
from app.errors import api_error
from app.integrations.embedding_http_client import EmbeddingHTTPClient
from app.integrations.qdrant_client import QdrantClient
from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas import Filters, SearchMode, SearchRequest


class SearchService:
    def __init__(
        self,
        repo: SQLiteRepository,
        qdrant: QdrantClient,
        embedding: EmbeddingHTTPClient,
        settings: Settings,
    ) -> None:
        self.repo = repo
        self.qdrant = qdrant
        self.embedding = embedding
        self.settings = settings

    def search(self, request: SearchRequest) -> dict[str, Any]:
        started = time.perf_counter()
        self._validate_request(request)
        filters = request.filters or Filters()
        all_items: list[dict[str, Any]] = []
        for collection in request.collections:
            if not self.repo.get_collection(collection):
                raise api_error(
                    404,
                    "COLLECTION_NOT_FOUND",
                    f"Collection not found: {collection}",
                )
            all_items.extend(self._search_collection(collection, request, filters))
        all_items = sorted(all_items, key=lambda item: item["score"], reverse=True)
        if request.rerank and self.settings.rerank_enabled:
            all_items = self._rerank(request.query, all_items)
        items = all_items[: request.top_k]
        latency_ms = int((time.perf_counter() - started) * 1000)
        self.repo.log_search(
            log_id=f"search_{uuid.uuid4().hex}",
            collection=",".join(request.collections),
            query=request.query,
            mode=request.mode.value,
            top_k=request.top_k,
            filters=filters.model_dump(exclude_none=True),
            result_count=len(items),
            latency_ms=latency_ms,
        )
        return {
            "query": request.query,
            "mode": request.mode.value,
            "items": [
                self._serialize_item(item, request.include_score_breakdown)
                for item in items
            ],
            "latency_ms": latency_ms,
        }

    def _search_collection(
        self,
        collection: str,
        request: SearchRequest,
        filters: Filters,
    ) -> list[dict[str, Any]]:
        dense_items: list[dict[str, Any]] = []
        keyword_items: list[dict[str, Any]] = []
        if request.mode in {SearchMode.DENSE, SearchMode.HYBRID}:
            dense_items = self._dense_search(
                collection,
                request.query,
                request.candidate_k,
                filters,
            )
        if request.mode in {SearchMode.KEYWORD, SearchMode.HYBRID}:
            keyword_items = self._keyword_search(
                collection,
                request.query,
                request.candidate_k,
                filters,
            )
        if request.mode == SearchMode.DENSE:
            return dense_items
        if request.mode == SearchMode.KEYWORD:
            return keyword_items
        return self._rrf_fusion(dense_items, keyword_items)

    def _dense_search(
        self,
        collection: str,
        query: str,
        limit: int,
        filters: Filters,
    ) -> list[dict[str, Any]]:
        _, _, vectors = self.embedding.embed([query])
        results = self.qdrant.search(
            collection,
            vectors[0],
            limit,
            filters.model_dump(exclude_none=True),
        )
        items: list[dict[str, Any]] = []
        for result in results:
            payload = result.get("payload", {})
            item = {
                "collection": collection,
                "document_id": payload.get("document_id"),
                "chunk_id": payload.get("chunk_id"),
                "title": payload.get("title", ""),
                "text": payload.get("text", ""),
                "metadata": payload,
                "score": float(result.get("score", 0.0)),
                "score_breakdown": {"dense": float(result.get("score", 0.0))},
            }
            if metadata_matches(payload, filters):
                items.append(item)
        return items

    def _keyword_search(
        self,
        collection: str,
        query: str,
        limit: int,
        filters: Filters,
    ) -> list[dict[str, Any]]:
        rows = self.repo.keyword_search(collection, query, limit)
        items: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            metadata = json.loads(row["metadata_json"])
            if not metadata_matches(metadata, filters):
                continue
            score = 1.0 / rank
            items.append(
                {
                    "collection": collection,
                    "document_id": row["document_id"],
                    "chunk_id": row["chunk_id"],
                    "title": row["title"],
                    "text": row["text"],
                    "metadata": metadata,
                    "score": score,
                    "score_breakdown": {"keyword": score},
                }
            )
        return items

    def _rrf_fusion(
        self,
        dense_items: list[dict[str, Any]],
        keyword_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for source_name, items in (("dense", dense_items), ("keyword", keyword_items)):
            for rank, item in enumerate(items, start=1):
                key = (item["collection"], item["chunk_id"])
                target = merged.setdefault(key, {**item, "score": 0.0, "score_breakdown": {}})
                rrf_score = 1.0 / (self.settings.rrf_k + rank)
                target["score"] += rrf_score
                target["score_breakdown"][source_name] = item["score_breakdown"].get(
                    source_name,
                    item["score"],
                )
                target["score_breakdown"]["fusion"] = target["score"]
        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)

    def _rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        terms = {term.lower() for term in query.split() if term.strip()}
        for item in items:
            text = f"{item['title']} {item['text']}".lower()
            overlap = sum(1 for term in terms if term in text)
            rerank_score = item["score"] + overlap * 0.1
            item["score"] = rerank_score
            item["score_breakdown"]["rerank"] = rerank_score
        return sorted(items, key=lambda item: item["score"], reverse=True)

    def _validate_request(self, request: SearchRequest) -> None:
        if not request.collections:
            raise api_error(400, "INVALID_REQUEST", "collections must not be empty")
        if request.top_k > self.settings.max_top_k:
            raise api_error(400, "INVALID_REQUEST", "top_k exceeds maximum")
        if request.candidate_k > self.settings.max_candidate_k:
            raise api_error(400, "INVALID_REQUEST", "candidate_k exceeds maximum")

    def _serialize_item(self, item: dict[str, Any], include_breakdown: bool) -> dict[str, Any]:
        payload = {
            "collection": item["collection"],
            "document_id": item["document_id"],
            "chunk_id": item["chunk_id"],
            "title": item["title"],
            "text": item["text"],
            "score": item["score"],
            "metadata": item["metadata"],
        }
        if include_breakdown:
            payload["score_breakdown"] = item["score_breakdown"]
        return payload


def metadata_matches(metadata: dict[str, Any], filters: Filters) -> bool:
    if filters.status and metadata.get("status") != filters.status:
        return False
    if filters.source and metadata.get("source") != filters.source:
        return False
    if filters.doc_type and metadata.get("doc_type") != filters.doc_type:
        return False
    if filters.tags:
        tags = metadata.get("tags", [])
        if not any(tag in tags for tag in filters.tags):
            return False
    date = metadata.get("date")
    if filters.date_gte and (not date or date < filters.date_gte):
        return False
    if filters.date_lte and (not date or date > filters.date_lte):
        return False
    return True
