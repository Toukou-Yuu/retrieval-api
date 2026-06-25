from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import ValidationError

from app.errors import api_error
from app.integrations.embedding_contract import (
    EmbeddingInputType,
    EmbeddingModelInfo,
    EmbeddingModelsResponse,
    EmbeddingResponse,
)


class EmbeddingHTTPClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        default_model: str = "BAAI/bge-m3",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_model = default_model
        self.transport = transport

    def health_check(self) -> bool:
        return self._check("/v1/health")

    def ready_check(self) -> bool:
        return self._check("/v1/ready")

    def get_info(self) -> dict[str, Any]:
        payload = self._json_response("GET", "/v1/info")
        if not isinstance(payload, dict):
            raise api_error(
                502,
                "EMBEDDING_CONTRACT_INVALID",
                "Embedding API info must be an object",
            )
        return payload

    def list_models(self) -> list[EmbeddingModelInfo]:
        payload = self._json_response("GET", "/v1/models")
        try:
            return EmbeddingModelsResponse.model_validate(payload).data
        except ValidationError as exc:
            raise api_error(
                502,
                "EMBEDDING_CONTRACT_INVALID",
                "Embedding API returned an invalid model list",
            ) from exc

    def get_model(self, name: str | None = None) -> EmbeddingModelInfo:
        selected_name = name
        if selected_name is None:
            info = self.get_info()
            default_model = info.get("default_model")
            if isinstance(default_model, str):
                selected_name = default_model
        models = self.list_models()
        if selected_name is None and len(models) == 1:
            return models[0]
        for model in models:
            if model.name == selected_name:
                return model
        raise api_error(
            400,
            "EMBEDDING_MODEL_NOT_FOUND",
            f"Embedding model not found: {selected_name}",
        )

    def embed_texts(
        self,
        texts: list[str],
        model: str,
        input_type: Literal["query", "document"] | EmbeddingInputType,
        normalize: bool,
    ) -> EmbeddingResponse:
        expected_input_type = EmbeddingInputType(input_type)
        payload = self._json_response(
            "POST",
            "/v1/embeddings",
            json={
                "model": model,
                "input": texts,
                "input_type": expected_input_type.value,
                "normalize": normalize,
                "truncate": True,
            },
            model_request=True,
        )
        response = self._parse_embedding_response(payload)
        if response.input_type != expected_input_type:
            raise api_error(
                502,
                "EMBEDDING_CONTRACT_INVALID",
                "Embedding API returned an unexpected input_type",
            )
        return response

    def embed(self, texts: list[str], normalize: bool = True) -> tuple[str, int, list[list[float]]]:
        response = self.embed_texts(
            texts,
            model=self.default_model,
            input_type=EmbeddingInputType.DOCUMENT,
            normalize=normalize,
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        return response.model, response.dimension, vectors

    def _check(self, path: str) -> bool:
        try:
            with self._client() as client:
                response = client.get(path)
            return response.status_code < 400
        except httpx.HTTPError:
            return False

    def _json_response(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        model_request: bool = False,
    ) -> Any:
        try:
            with self._client() as client:
                response = client.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise api_error(
                502,
                "EMBEDDING_API_UNAVAILABLE",
                "Embedding API is unavailable",
            ) from exc
        if response.status_code >= 400:
            code = "EMBEDDING_API_ERROR"
            if model_request and response.status_code == 404:
                code = "EMBEDDING_MODEL_NOT_FOUND"
            raise api_error(
                502,
                code,
                "Embedding API returned an error",
                {"upstream_status_code": response.status_code},
            )
        try:
            return response.json()
        except ValueError as exc:
            raise api_error(
                502,
                "EMBEDDING_CONTRACT_INVALID",
                "Embedding API returned invalid JSON",
            ) from exc

    def _parse_embedding_response(
        self,
        payload: Any,
    ) -> EmbeddingResponse:
        if not isinstance(payload, dict):
            raise api_error(
                502,
                "EMBEDDING_CONTRACT_INVALID",
                "Embedding response must be an object",
            )
        try:
            return EmbeddingResponse.model_validate(payload)
        except (TypeError, ValidationError) as exc:
            raise api_error(
                502,
                "EMBEDDING_CONTRACT_INVALID",
                "Embedding API returned an invalid embedding response",
            ) from exc

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        )
