def create_collection(client, name="200iq_cases", strategy="plain_text", dimension=3):
    response = client.post(
        "/v1/collections",
        json={
            "name": name,
            "embedding_model": "test-model",
            "embedding_dimension": dimension,
            "chunk_strategy": strategy,
        },
    )
    assert response.status_code == 200


def document_payload(text="subscription lesson", document_id="200iq:case:001"):
    return {
        "id": document_id,
        "source": "200iq-moments",
        "doc_type": "case",
        "title": "Airport subscription",
        "text": text,
        "metadata": {
            "case_id": "001",
            "status": "published",
            "tags": ["subscription"],
            "date": "2025-06-20",
        },
    }


def test_document_upsert_idempotent_get_and_delete(client):
    create_collection(client)
    request = {"collection": "200iq_cases", "documents": [document_payload()]}

    first = client.post("/v1/documents/upsert", json=request)
    repeated = client.post("/v1/documents/upsert", json=request)
    fetched = client.get("/v1/documents/200iq_cases/200iq:case:001")
    deleted = client.delete("/v1/documents/200iq_cases/200iq:case:001")

    assert first.status_code == 200
    assert first.json()["upserted"] == 1
    assert repeated.json()["skipped"] == 1
    assert fetched.json()["chunks"][0]["chunk_index"] == 0
    assert deleted.json()["deleted_chunks"] == 1
    assert deleted.json()["deleted"] is True


def test_delete_document_is_idempotent(client):
    create_collection(client)

    response = client.delete("/v1/documents/200iq_cases/200iq:case:missing")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 0
    assert response.json()["deleted"] is True


def test_document_update_removes_old_chunks(client):
    create_collection(client)
    client.post(
        "/v1/documents/upsert",
        json={"collection": "200iq_cases", "documents": [document_payload("old text")]},
    )
    client.post(
        "/v1/documents/upsert",
        json={"collection": "200iq_cases", "documents": [document_payload("new text")]},
    )

    fetched = client.get("/v1/documents/200iq_cases/200iq:case:001").json()
    assert len(fetched["chunks"]) == 1
    assert len(client.fake_qdrant.points["200iq_cases"]) == 1


def test_archive_and_restore_update_status(client):
    create_collection(client)
    client.post(
        "/v1/documents/upsert",
        json={"collection": "200iq_cases", "documents": [document_payload()]},
    )

    archived = client.post("/v1/documents/200iq_cases/200iq:case:001/archive")
    restored = client.post("/v1/documents/200iq_cases/200iq:case:001/restore")

    assert archived.json()["status"] == "archived"
    assert restored.json()["status"] == "published"
    point = next(iter(client.fake_qdrant.points["200iq_cases"].values()))
    assert point["payload"]["status"] == "published"


def test_rebuild_collection_replace(client):
    create_collection(client)
    client.post(
        "/v1/documents/upsert",
        json={"collection": "200iq_cases", "documents": [document_payload(document_id="old")]},
    )
    response = client.post(
        "/v1/index/rebuild-collection",
        json={
            "collection": "200iq_cases",
            "mode": "replace",
            "documents": [document_payload(document_id="new")],
        },
    )

    assert response.status_code == 200
    assert client.get("/v1/documents/200iq_cases/new").status_code == 200
    assert client.get("/v1/documents/200iq_cases/old").status_code == 404


def test_async_upsert_creates_pending_job_and_retry_indexes(client):
    create_collection(client)
    response = client.post(
        "/v1/documents/upsert",
        json={
            "collection": "200iq_cases",
            "documents": [document_payload()],
            "indexing": {"mode": "async"},
        },
    )
    job_id = response.json()["jobs"][0]["job_id"]
    retried = client.post(f"/v1/index/jobs/{job_id}/retry")

    assert response.json()["jobs"][0]["status"] == "pending"
    assert retried.json()["status"] == "done"
    assert client.get("/v1/documents/200iq_cases/200iq:case:001").status_code == 200


def test_async_upsert_background_processes_job(client):
    create_collection(client)
    response = client.post(
        "/v1/documents/upsert",
        json={
            "collection": "200iq_cases",
            "documents": [document_payload()],
            "indexing": {"mode": "async"},
        },
    )
    job_id = response.json()["jobs"][0]["job_id"]
    job = client.get(f"/v1/index/jobs/{job_id}").json()

    assert job["status"] == "done"
    assert client.get("/v1/documents/200iq_cases/200iq:case:001").status_code == 200


def test_dimension_mismatch_marks_job_failed(client):
    create_collection(client, dimension=4)
    response = client.post(
        "/v1/documents/upsert",
        json={"collection": "200iq_cases", "documents": [document_payload()]},
    )
    jobs = client.get("/v1/index/jobs").json()["items"]

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DIMENSION_MISMATCH"
    assert jobs[0]["status"] == "failed"


def test_embedding_model_mismatch_marks_job_failed_before_writing_index(client):
    create_collection(client)
    client.fake_embedding.model = "other-model"

    response = client.post(
        "/v1/documents/upsert",
        json={"collection": "200iq_cases", "documents": [document_payload()]},
    )
    jobs = client.get("/v1/index/jobs").json()["items"]

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EMBEDDING_MODEL_MISMATCH"
    assert jobs[0]["status"] == "failed"
    assert client.get("/v1/documents/200iq_cases/200iq:case:001").status_code == 404
    assert client.fake_qdrant.points["200iq_cases"] == {}
