def test_health_ready_and_info(client):
    assert client.get("/v1/health").json()["status"] == "ok"
    assert client.get("/v1/ready").json()["status"] == "ready"
    info = client.get("/v1/info").json()
    assert info["service"] == "retrieval-api"
    assert info["keyword_index"] == "sqlite_fts5"


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
    assert created.json() == {"name": "200iq_cases", "created": True}
    assert repeated.json() == {"name": "200iq_cases", "created": False}
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
    assert response.json()["error"]["code"] == "DIMENSION_MISMATCH"
