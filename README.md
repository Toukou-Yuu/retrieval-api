# retrieval-api

`retrieval-api` owns collections, document chunking, indexing jobs, keyword search, and vector-search orchestration. It does not load embedding models. Embeddings are requested from a separate `embedding-api`; vectors are stored in Qdrant and metadata/FTS5 data in SQLite.

## Local development without Docker

Core tests run without Docker, Qdrant, or a model download.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export RETRIEVAL_DATA_DIR="$(pwd)/data/dev"
export EMBEDDING_API_URL="http://127.0.0.1:8100"
export QDRANT_URL="http://127.0.0.1:6333"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8300
```

Run the default validation suite:

```bash
pytest
ruff check .
python -m compileall app
```

Optional integration tests that use real dependencies must be explicitly enabled:

```bash
RUN_INTEGRATION_TESTS=1 pytest
```

Deleting `data/dev` resets only local development state. The application migrates existing SQLite metadata automatically, but never deletes Qdrant collections as part of a schema migration.

## Docker deployment

`docker-compose.yml` is the full deployment stack: retrieval-api, embedding-api, and Qdrant. It is not required for local unit testing.

```bash
docker network create agent-services
docker compose up -d
```

The services bind their public ports to `127.0.0.1`. The compose stack expects `toukouyuu/embedding-api:latest` to implement the v1 contract documented in [docs/api.md](docs/api.md). Copy `.env.example` to `.env` to override service addresses or contract settings.

The default endpoint is `http://127.0.0.1:8300`; OpenAPI is available at `/docs`.


## MCP tools

`retrieval-api` provides first-class MCP tools for Alice/Hermes in addition to the REST API.

```bash
uv run retrieval-api mcp
uv run retrieval-api mcp-http --host 127.0.0.1 --port 8301 --path /mcp
```

Production deployment uses the same image as the REST API with a separate `retrieval-api-mcp` sidecar. See [docs/mcp.md](docs/mcp.md) for Hermes configuration, tool inventory, permissions, side effects, and tests.

Tool names:

```text
retrieval_list_collections
retrieval_create_collection
retrieval_search
retrieval_get_document
retrieval_upsert_documents
retrieval_rebuild_collection
retrieval_index_status
retrieval_retry_job
```
