# retrieval-api MCP Tools

`retrieval-api` exposes native MCP tools for Alice/Hermes alongside the REST API. The MCP adapter shares the same service layer as FastAPI; it does not call the local REST endpoints.

## Start commands

Local stdio MCP:

```bash
uv run retrieval-api mcp
```

Local HTTP/Streamable HTTP MCP:

```bash
uv run retrieval-api mcp-http --host 127.0.0.1 --port 8301 --path /mcp
```

REST API:

```bash
uv run retrieval-api serve --host 127.0.0.1 --port 8300
```

Production Docker uses the same image with a separate MCP sidecar process:

```text
retrieval-api       -> REST API on 127.0.0.1:8300/v1
retrieval-api-mcp   -> MCP HTTP on 127.0.0.1:8301/mcp
```

## Hermes configuration

```yaml
mcp_servers:
  retrieval:
    url: http://127.0.0.1:8301/mcp
    timeout: 120
    connect_timeout: 60
```

Verify after deployment:

```bash
hermes mcp list
hermes mcp test retrieval
```

Current Hermes sessions may need `/reload-mcp`, `/new`, or a process restart before new tools appear.

## Tool inventory

| Tool | Permission | Side effects |
|---|---:|---|
| `retrieval_list_collections` | read | None |
| `retrieval_create_collection` | admin | Creates SQLite collection metadata and Qdrant collection unless `dry_run=true`. |
| `retrieval_search` | read | Logs search metadata to SQLite. Does not mutate documents. |
| `retrieval_get_document` | read | None |
| `retrieval_upsert_documents` | write | Writes documents/chunks to SQLite/FTS and vectors to Qdrant unless `dry_run=true`. |
| `retrieval_rebuild_collection` | admin | In `replace` mode deletes existing docs before rebuilding unless `dry_run=true` (default). |
| `retrieval_index_status` | read | None |
| `retrieval_retry_job` | write | Re-runs a stored indexing job unless `dry_run=true`. |

## Response envelope

Success:

```json
{
  "ok": true,
  "summary": "Found 3 retrieval matches for query: subscription lesson",
  "data": {},
  "warnings": []
}
```

Error:

```json
{
  "ok": false,
  "error": {
    "code": "COLLECTION_NOT_FOUND",
    "message": "Collection not found: 200iq_cases",
    "retryable": false,
    "suggested_action": "Call retrieval_list_collections first, then retry with an existing collection."
  }
}
```

`retrieval_search` returns Agent-oriented items with `title`, `text_preview`, `source`, `document_id`, `chunk_id`, `score`, and metadata. Long text is truncated with `text_truncated=true`; call `retrieval_get_document` for document details.

## Common errors

- `COLLECTION_NOT_FOUND`: call `retrieval_list_collections` and create/choose a collection.
- `DOCUMENT_NOT_FOUND`: search first, then retry with an existing `collection` and `document_id`.
- `EMBEDDING_*`: check `embedding-api` readiness and the collection embedding contract.
- `QDRANT_*`: check Qdrant health and collection vector configuration.
- `INVALID_REQUEST`: fix tool arguments to match the schema.

## Tests

```bash
uv run --extra dev pytest tests/mcp -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
docker compose config --quiet
```
