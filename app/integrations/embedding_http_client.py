from __future__ import annotations

import httpx

from app.errors import api_error


class EmbeddingHTTPClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/v1/health", timeout=5.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def embed(self, texts: list[str], normalize: bool = True) -> tuple[str, int, list[list[float]]]:
        try:
            response = httpx.post(
                f"{self.base_url}/v1/embeddings",
                json={"texts": texts, "normalize": normalize},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise api_error(502, "EMBEDDING_API_ERROR", str(exc)) from exc
        if response.status_code >= 400:
            raise api_error(
                502,
                "EMBEDDING_API_ERROR",
                "Embedding API returned an error",
                {"status_code": response.status_code},
            )
        payload = response.json()
        return payload["model"], int(payload["dimension"]), payload["vectors"]
