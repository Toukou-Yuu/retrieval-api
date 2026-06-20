# retrieval-api HTTP API

Base URL:

```text
http://127.0.0.1:8300/v1
```

## System

### GET `/health`

Returns service health.

```json
{
  "status": "ok",
  "service": "retrieval-api",
  "version": "1.0.0"
}
```

### GET `/ready`

Checks SQLite, Qdrant, and embedding API connectivity.

### GET `/info`

Returns service configuration such as embedding model, vector database, keyword index, and rerank status.

## Collections

### POST `/collections`

Creates a collection and the matching Qdrant collection.

```json
{
  "name": "200iq_cases",
  "description": "Case collection",
  "embedding_model": "BAAI/bge-m3",
  "embedding_dimension": 1024,
  "chunk_strategy": "markdown_semantic",
  "metadata_schema": {
    "status": "string",
    "tags": "string[]",
    "date": "string"
  }
}
```

Response:

```json
{
  "name": "200iq_cases",
  "created": true
}
```

### GET `/collections`

Lists collections.

### GET `/collections/{collection}`

Returns collection metadata.

### DELETE `/collections/{collection}`

Deletes collection metadata, SQLite index rows, and Qdrant collection data.

## Documents

### POST `/documents/upsert`

Indexes one or more documents.

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
      "metadata": {
        "case_id": "001",
        "status": "published",
        "tags": ["subscription"],
        "date": "2025-06-20"
      }
    }
  ],
  "indexing": {
    "mode": "sync"
  }
}
```

`indexing.mode` accepts `sync` or `async`.

### GET `/documents/{collection}/{document_id}`

Returns document index status and chunk summaries. It does not return the full upstream document body.

### DELETE `/documents/{collection}/{document_id}`

Deletes the document index from SQLite and Qdrant.

### POST `/documents/{collection}/{document_id}/archive`

Marks the indexed document as archived. Default search excludes archived documents.

### POST `/documents/{collection}/{document_id}/restore`

Restores the indexed document to `published`.

## Index Jobs

### POST `/index/reindex`

Reindexes a document batch.

```json
{
  "collection": "200iq_cases",
  "documents": [],
  "delete_missing": false
}
```

### POST `/index/rebuild-collection`

Rebuilds a collection.

```json
{
  "collection": "200iq_cases",
  "documents": [],
  "mode": "replace"
}
```

`mode` accepts `replace` or `merge`.

### GET `/index/jobs`

Lists index jobs.

Query parameters:

```text
collection=200iq_cases
status=pending|processing|done|failed
limit=50
```

### GET `/index/jobs/{job_id}`

Returns a single job.

### POST `/index/jobs/{job_id}/retry`

Retries a stored job.

### DELETE `/index/jobs/{job_id}`

Deletes a job record.

## Search

### POST `/search`

Runs `dense`, `keyword`, or `hybrid` search.

```json
{
  "collections": ["200iq_cases"],
  "query": "subscription lesson",
  "mode": "hybrid",
  "top_k": 5,
  "candidate_k": 30,
  "filters": {
    "status": "published",
    "tags": ["subscription"]
  },
  "rerank": true,
  "include_score_breakdown": true
}
```

Response:

```json
{
  "query": "subscription lesson",
  "mode": "hybrid",
  "items": [
    {
      "collection": "200iq_cases",
      "document_id": "200iq:case:001",
      "chunk_id": "200iq:case:001:chunk:0000:abcdef123456",
      "title": "Airport subscription",
      "text": "Matched evidence text.",
      "score": 0.91,
      "score_breakdown": {
        "dense": 0.82,
        "keyword": 0.5,
        "fusion": 0.03
      },
      "metadata": {
        "source": "200iq-moments",
        "doc_type": "case",
        "status": "published",
        "tags": ["subscription"]
      }
    }
  ],
  "latency_ms": 12
}
```

### POST `/search/dense`

Runs dense vector search.

### POST `/search/keyword`

Runs SQLite FTS5 keyword search.

### POST `/search/hybrid`

Runs hybrid search using reciprocal rank fusion.

## Error Format

```json
{
  "error": {
    "code": "COLLECTION_NOT_FOUND",
    "message": "Collection not found: 200iq_cases",
    "details": {}
  }
}
```
