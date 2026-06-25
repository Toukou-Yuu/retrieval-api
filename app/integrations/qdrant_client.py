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

    def collection_exists(self, name: str) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/collections/{name}", timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise api_error(502, "QDRANT_UNAVAILABLE", "Qdrant is unavailable") from exc
        if response.status_code == 404:
            return False
        if response.status_code >= 400:
            raise api_error(
                502,
                "QDRANT_ERROR",
                "Failed to read Qdrant collection",
                {"upstream_status_code": response.status_code},
            )
        return True

    def get_collection_vector_config(self, name: str) -> tuple[int, str]:
        try:
            response = httpx.get(f"{self.base_url}/collections/{name}", timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise api_error(502, "QDRANT_UNAVAILABLE", "Qdrant is unavailable") from exc
        if response.status_code == 404:
            raise api_error(404, "COLLECTION_NOT_FOUND", f"Qdrant collection not found: {name}")
        if response.status_code >= 400:
            raise api_error(
                502,
                "QDRANT_ERROR",
                "Failed to read Qdrant collection",
                {"upstream_status_code": response.status_code},
            )
        try:
            vectors = response.json()["result"]["config"]["params"]["vectors"]
            return int(vectors["size"]), str(vectors["distance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise api_error(
                502,
                "QDRANT_ERROR",
                "Qdrant returned an invalid collection configuration",
            ) from exc

    def create_collection(self, name: str, dimension: int, distance: str = "Cosine") -> None:
        self._request(
            "PUT",
            f"/collections/{name}",
            json={"vectors": {"size": dimension, "distance": distance}},
        )

    def delete_collection(self, name: str) -> None:
        self._request("DELETE", f"/collections/{name}", allowed_statuses={404})

    def upsert_points(self, collection: str, points: list[dict[str, Any]]) -> None:
        self._request(
            "PUT",
            f"/collections/{collection}/points",
            params={"wait": "true"},
            json={"points": points},
        )

    def delete_document_points(self, collection: str, document_id: str) -> None:
        payload = {
            "filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]},
            "wait": True,
        }
        self._request(
            "POST",
            f"/collections/{collection}/points/delete",
            json=payload,
        )

    def set_document_payload(
        self,
        collection: str,
        document_id: str,
        payload: dict[str, Any],
    ) -> None:
        request = {
            "payload": payload,
            "filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]},
        }
        self._request(
            "POST",
            f"/collections/{collection}/points/payload",
            params={"wait": "true"},
            json=request,
        )

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
        response = self._request(
            "POST",
            f"/collections/{collection}/points/search",
            json=payload,
        )
        return response.json().get("result", [])

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allowed_statuses: set[int] | None = None,
    ) -> httpx.Response:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                json=json,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise api_error(502, "QDRANT_UNAVAILABLE", "Qdrant is unavailable") from exc
        if response.status_code >= 400 and response.status_code not in (allowed_statuses or set()):
            raise api_error(
                502,
                "QDRANT_ERROR",
                "Qdrant returned an error",
                {"upstream_status_code": response.status_code},
            )
        return response


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
