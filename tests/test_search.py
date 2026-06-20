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


def seed_documents(client):
    create_collection(client)
    client.post(
        "/v1/documents/upsert",
        json={
            "collection": "200iq_cases",
            "documents": [
                document_payload(
                    "airport subscription expires after one month",
                    "200iq:case:001",
                ),
                {
                    **document_payload("irrelevant notebook text", "200iq:case:002"),
                    "metadata": {
                        "case_id": "002",
                        "status": "archived",
                        "tags": ["notes"],
                        "date": "2025-07-01",
                    },
                },
            ],
        },
    )


def test_keyword_search_filters_published_by_default(client):
    seed_documents(client)
    response = client.post(
        "/v1/search/keyword",
        json={
            "collections": ["200iq_cases"],
            "query": "subscription",
            "top_k": 5,
            "candidate_k": 10,
        },
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["document_id"] == "200iq:case:001"


def test_dense_and_hybrid_search(client):
    seed_documents(client)

    dense = client.post(
        "/v1/search/dense",
        json={"collections": ["200iq_cases"], "query": "airport subscription"},
    )
    hybrid = client.post(
        "/v1/search",
        json={
            "collections": ["200iq_cases"],
            "query": "airport subscription",
            "mode": "hybrid",
            "include_score_breakdown": True,
        },
    )

    assert dense.status_code == 200
    assert dense.json()["items"][0]["document_id"] == "200iq:case:001"
    assert hybrid.json()["items"][0]["score_breakdown"]["fusion"] > 0


def test_metadata_filters(client):
    seed_documents(client)
    response = client.post(
        "/v1/search/keyword",
        json={
            "collections": ["200iq_cases"],
            "query": "subscription",
            "filters": {"status": "published", "tags": ["subscription"]},
        },
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["metadata"]["tags"] == ["subscription"]


def test_search_requires_collection(client):
    response = client.post(
        "/v1/search",
        json={"collections": [], "query": "anything"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
