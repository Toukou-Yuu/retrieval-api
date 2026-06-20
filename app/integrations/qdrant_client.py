from __future__ import annotations

from typing import Any

import httpx

from app.errors import api_error


class QdrantClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/", timeout=self.timeout)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def create_collection(self, name: str, dimension: int) -> None:
        response = httpx.put(
            f"{self.base_url}/collections/{name}",
            json={"vectors": {"size": dimension, "distance": "Cosine"}},
            timeout=self.timeout,
        )
        if response.status_code not in {200, 201}:
            raise api_error(502, "QDRANT_ERROR", "Failed to create Qdrant collection")

    def delete_collection(self, name: str) -> None:
        response = httpx.delete(f"{self.base_url}/collections/{name}", timeout=self.timeout)
        if response.status_code not in {200, 404}:
            raise api_error(502, "QDRANT_ERROR", "Failed to delete Qdrant collection")

    def upsert_points(self, collection: str, points: list[dict[str, Any]]) -> None:
        response = httpx.put(
            f"{self.base_url}/collections/{collection}/points",
            params={"wait": "true"},
            json={"points": points},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise api_error(502, "QDRANT_ERROR", "Failed to upsert Qdrant points")

    def delete_document_points(self, collection: str, document_id: str) -> None:
        payload = {
            "filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]},
            "wait": True,
        }
        response = httpx.post(
            f"{self.base_url}/collections/{collection}/points/delete",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise api_error(502, "QDRANT_ERROR", "Failed to delete Qdrant points")

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"vector": vector, "limit": limit, "with_payload": True}
        qdrant_filter = build_qdrant_filter(filters)
        if qdrant_filter:
            payload["filter"] = qdrant_filter
        response = httpx.post(
            f"{self.base_url}/collections/{collection}/points/search",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise api_error(502, "QDRANT_ERROR", "Failed to search Qdrant")
        return response.json().get("result", [])


def build_qdrant_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    must: list[dict[str, Any]] = []
    for key in ("status", "source", "doc_type"):
        if filters.get(key):
            must.append({"key": key, "match": {"value": filters[key]}})
    if filters.get("tags"):
        should = [{"key": "tags", "match": {"value": tag}} for tag in filters["tags"]]
        must.append({"should": should})
    if filters.get("date_gte") or filters.get("date_lte"):
        range_filter: dict[str, Any] = {}
        if filters.get("date_gte"):
            range_filter["gte"] = filters["date_gte"]
        if filters.get("date_lte"):
            range_filter["lte"] = filters["date_lte"]
        must.append({"key": "date", "range": range_filter})
    return {"must": must} if must else None
