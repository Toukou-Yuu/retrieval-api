def test_health_ready_and_info(client):
    assert client.get("/v1/health").json()["status"] == "ok"
    ready = client.get("/v1/ready").json()
    assert ready["status"] == "ready"
    assert ready["dependencies"]["embedding_api"] == {
        "status": "ok",
        "url": "http://embedding-api:8100",
        "contract_version": "embedding-api.v1",
        "default_model": "test-model",
    }
    info = client.get("/v1/info").json()
    assert info["service"] == "retrieval-api"
    assert info["keyword_index"] == "sqlite_fts5"
    assert info["contract"] == {"embedding_api": "embedding-api.v1"}
    assert info["defaults"]["embedding_normalized"] is True


def test_ready_is_degraded_when_embedding_api_is_not_ready(client):
    client.fake_embedding.ready = False

    response = client.get("/v1/ready")

    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"]["embedding_api"]["status"] == "error"


def test_collection_create_idempotent_and_delete(client):
    payload = {
        "name": "200iq_cases",
        "description": "case collection",
        "embedding_model": "test-model",
        "embedding_dimension": 3,
        "chunk_strategy": "plain_text",
    }

    created = client.post("/v1/collections", json=payload)
    repeated = client.post("/v1/collections", json=payload)
    listed = client.get("/v1/collections")
    fetched = client.get("/v1/collections/200iq_cases")
    deleted = client.delete("/v1/collections/200iq_cases")

    assert created.status_code == 200
    assert created.json()["name"] == "200iq_cases"
    assert created.json()["created"] is True
    assert created.json()["embedding"]["model"] == "test-model"
    assert created.json()["embedding"]["validated"] is True
    assert created.json()["vector_store"] == {"type": "qdrant", "collection": "200iq_cases"}
    assert repeated.json()["created"] is False
    assert listed.json()["items"][0]["name"] == "200iq_cases"
    assert fetched.json()["embedding_dimension"] == 3
    assert deleted.json()["deleted"] is True


def test_collection_dimension_mismatch(client):
    client.post(
        "/v1/collections",
        json={"name": "docs", "embedding_dimension": 3, "chunk_strategy": "plain_text"},
    )
    response = client.post(
        "/v1/collections",
        json={"name": "docs", "embedding_dimension": 4, "chunk_strategy": "plain_text"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EMBEDDING_DIMENSION_MISMATCH"


def test_collection_embedding_model_mismatch(client):
    client.post(
        "/v1/collections",
        json={"name": "docs", "embedding_model": "model-a", "embedding_dimension": 3},
    )
    response = client.post(
        "/v1/collections",
        json={"name": "docs", "embedding_model": "model-b", "embedding_dimension": 3},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EMBEDDING_MODEL_MISMATCH"
