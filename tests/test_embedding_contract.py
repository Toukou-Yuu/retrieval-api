import json

import httpx
import pytest

from app.errors import ApiError
from app.integrations.embedding_contract import EmbeddingInputType
from app.integrations.embedding_http_client import EmbeddingHTTPClient


def test_embed_texts_uses_v1_request_and_response_contract():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/embeddings"
        assert json.loads(request.content) == {
            "model": "test-model",
            "input": ["first", "second"],
            "input_type": "document",
            "normalize": True,
            "truncate": True,
        }
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "dimension": 3,
                "normalized": True,
                "input_type": "document",
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                ],
            },
        )

    client = EmbeddingHTTPClient(
        "http://embedding-api:8100",
        transport=httpx.MockTransport(handler),
    )

    response = client.embed_texts(
        ["first", "second"],
        model="test-model",
        input_type=EmbeddingInputType.DOCUMENT,
        normalize=True,
    )

    assert response.model == "test-model"
    assert response.dimension == 3
    assert response.data[1].embedding == [0.0, 1.0, 0.0]


def test_list_models_and_get_model_use_v1_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "name": "test-model",
                        "dimension": 3,
                        "normalized": True,
                        "recommended_distance": "Cosine",
                        "supports_query_document_prefix": True,
                    }
                ]
            },
        )

    client = EmbeddingHTTPClient(
        "http://embedding-api:8100",
        transport=httpx.MockTransport(handler),
    )

    model = client.get_model("test-model")

    assert model.dimension == 3
    assert model.contract_version == "embedding-api.v1"


def test_invalid_embedding_response_raises_contract_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "dimension": 3,
                "normalized": True,
                "input_type": "query",
                "data": [{"index": 0, "embedding": [1.0, 0.0]}],
            },
        )

    client = EmbeddingHTTPClient(
        "http://embedding-api:8100",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ApiError, match="invalid embedding response") as exc_info:
        client.embed_texts(["query"], "test-model", "query", normalize=True)

    assert exc_info.value.code == "EMBEDDING_CONTRACT_INVALID"


def test_embedding_api_error_keeps_upstream_status_code():
    client = EmbeddingHTTPClient(
        "http://embedding-api:8100",
        transport=httpx.MockTransport(lambda _: httpx.Response(404)),
    )

    with pytest.raises(ApiError) as exc_info:
        client.embed_texts(["query"], "missing-model", "query", normalize=True)

    assert exc_info.value.code == "EMBEDDING_MODEL_NOT_FOUND"
    assert exc_info.value.details == {"upstream_status_code": 404}


def test_unavailable_embedding_api_does_not_expose_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = EmbeddingHTTPClient(
        "http://embedding-api:8100",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ApiError) as exc_info:
        client.embed_texts(["query"], "test-model", "query", normalize=True)

    assert exc_info.value.code == "EMBEDDING_API_UNAVAILABLE"
    assert exc_info.value.message == "Embedding API is unavailable"
