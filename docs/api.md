# retrieval-api HTTP API

Base URL:

```text
http://127.0.0.1:8300/v1
```

## Embedding contract

retrieval-api expects `embedding-api.v1`. The contract is intentionally limited to model discovery and text-to-vector conversion:

```text
GET  /v1/health
GET  /v1/ready
GET  /v1/info
GET  /v1/models
POST /v1/embeddings
```

`GET /v1/models` returns the supported models:

```json
{
  "data": [
    {
      "name": "BAAI/bge-m3",
      "dimension": 1024,
      "normalized": true,
      "recommended_distance": "Cosine",
      "max_input_tokens": 8192,
      "supports_query_document_prefix": true,
      "contract_version": "embedding-api.v1"
    }
  ]
}
```

`POST /v1/embeddings` uses the collection-bound model and intent:

```json
{
  "model": "BAAI/bge-m3",
  "input": ["text 1", "text 2"],
  "input_type": "document",
  "normalize": true,
  "truncate": true
}
```

The v1 response is:

```json
{
  "model": "BAAI/bge-m3",
  "dimension": 1024,
  "normalized": true,
  "input_type": "document",
  "data": [
    {"index": 0, "embedding": [0.1, 0.2]},
    {"index": 1, "embedding": [0.3, 0.4]}
  ]
}
```

Each vector must have `dimension` values and indexes must be contiguous from zero. retrieval-api only accepts the v1 `data[].embedding` response format.

## System

### GET `/health`

Returns process liveness.

### GET `/ready`

Checks SQLite, Qdrant, and `embedding-api /v1/ready` without loading a model.

```json
{
  "status": "ready",
  "dependencies": {
    "sqlite": {"status": "ok", "path": "/data/retrieval.sqlite3"},
    "qdrant": {"status": "ok", "url": "http://qdrant:6333"},
    "embedding_api": {
      "status": "ok",
      "url": "http://embedding-api:8100",
      "contract_version": "embedding-api.v1",
      "default_model": "BAAI/bge-m3"
    }
  }
}
```

### GET `/info`

Returns the supported embedding contract, configured defaults, stores, and search limits. Existing top-level `embedding_model`, `embedding_dimension`, `vector_db`, `keyword_index`, and `rerank_enabled` fields remain available.

## Collections

### POST `/collections`

Creates SQLite metadata and a matching Qdrant collection.

```json
{
  "name": "200iq_cases",
  "description": "Case collection",
  "embedding_model": "BAAI/bge-m3",
  "embedding_dimension": 1024,
  "embedding_normalized": true,
  "embedding_distance": "Cosine",
  "embedding_contract_version": "embedding-api.v1",
  "chunk_strategy": "markdown_semantic",
  "metadata_schema": {"status": "string", "tags": "string[]"}
}
```

All embedding fields are optional when contract validation is enabled. retrieval-api resolves the model from `/v1/models`, verifies a supplied dimension, and stores the resolved contract. If `VALIDATE_EMBEDDING_CONTRACT_ON_COLLECTION_CREATE=false`, `embedding_dimension` is mandatory and the collection is stored with `validated=false`.

```json
{
  "name": "200iq_cases",
  "created": true,
  "embedding": {
    "provider": "embedding-api",
    "model": "BAAI/bge-m3",
    "dimension": 1024,
    "normalized": true,
    "distance": "Cosine",
    "contract_version": "embedding-api.v1",
    "created_at": "2026-06-24T00:00:00+00:00",
    "resolved_from": "embedding-api:/v1/models",
    "validated": true
  },
  "vector_store": {"type": "qdrant", "collection": "200iq_cases"}
}
```

Repeated creation verifies the stored SQLite contract and Qdrant vector dimension/distance. A mismatch returns `VECTOR_STORE_CONTRACT_MISMATCH` instead of silently reusing the collection.

### GET `/collections`, GET `/collections/{collection}`, DELETE `/collections/{collection}`

List, retrieve, or delete collection metadata and its indexes. Collection responses include both existing flat embedding fields and the full `embedding` contract object.

## Documents and jobs

### POST `/documents/upsert`

Indexes one or more documents. `indexing.mode` is `sync` or `async`.

```json
{
  "collection": "200iq_cases",
  "documents": [
    {
      "id": "200iq:case:001",
      "source": "200iq-moments",
      "doc_type": "case",
      "title": "Airport subscription",
      "text": "Complete searchable document text.",
      "metadata": {"tags": ["subscription"], "status": "published"}
    }
  ],
  "indexing": {"mode": "sync"}
}
```

Each chunk is embedded with the collection model in `document` mode. retrieval-api validates returned model, dimension, normalize state, and vector count before writing Qdrant. Reindexing a document deletes its prior points by `document_id` before writing replacement points.

Other document endpoints are unchanged:

```text
GET    /documents/{collection}/{document_id}
DELETE /documents/{collection}/{document_id}
POST   /documents/{collection}/{document_id}/archive
POST   /documents/{collection}/{document_id}/restore
POST   /index/reindex
POST   /index/rebuild-collection
GET    /index/jobs
GET    /index/jobs/{job_id}
POST   /index/jobs/{job_id}/retry
DELETE /index/jobs/{job_id}
```

## Search

### POST `/search`

Runs `dense`, `keyword`, or `hybrid` retrieval. `/search/dense`, `/search/keyword`, and `/search/hybrid` set the mode directly.

```json
{
  "collections": ["200iq_cases"],
  "query": "subscription lesson",
  "mode": "hybrid",
  "top_k": 5,
  "candidate_k": 30,
  "filters": {"status": "published", "tags": ["subscription"]}
}
```

Dense search sends the query in `query` mode using each collection's contract. For multi-collection dense or hybrid requests, all collection contracts must have the same model, dimension, normalize state, distance, and contract version. Otherwise the default response is `400 MULTI_MODEL_SEARCH_NOT_ALLOWED`. Keyword-only searches do not request embeddings and are not restricted by that rule.

Set `ALLOW_MULTI_MODEL_SEARCH=true` only when intentionally accepting independent embeddings per collection and merging their results.

## Errors

All errors use one shape:

```json
{
  "error": {
    "code": "EMBEDDING_DIMENSION_MISMATCH",
    "message": "Expected 1024, got 768",
    "details": {}
  }
}
```

Relevant codes:

```text
COLLECTION_NOT_FOUND
COLLECTION_ALREADY_EXISTS
COLLECTION_EMBEDDING_CONTRACT_INVALID
VECTOR_STORE_CONTRACT_MISMATCH
DOCUMENT_NOT_FOUND
DOCUMENT_INDEX_FAILED
EMBEDDING_API_UNAVAILABLE
EMBEDDING_API_ERROR
EMBEDDING_MODEL_NOT_FOUND
EMBEDDING_MODEL_MISMATCH
EMBEDDING_DIMENSION_MISMATCH
EMBEDDING_NORMALIZE_MISMATCH
EMBEDDING_CONTRACT_INVALID
MULTI_MODEL_SEARCH_NOT_ALLOWED
QDRANT_UNAVAILABLE
QDRANT_ERROR
SQLITE_ERROR
INVALID_REQUEST
```
