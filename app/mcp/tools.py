from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import ValidationError

from app.api import deps
from app.errors import ApiError
from app.schemas import (
    CollectionCreate,
    DocumentInput,
    DocumentsUpsertRequest,
    Filters,
    IndexingOptions,
    SearchRequest,
)

Permission = Literal["read", "write", "admin"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    permission: Permission
    description: str


ToolResult = dict[str, Any]
RawTool = Callable[..., Any]


def success(
    summary: str,
    data: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> ToolResult:
    return {
        "ok": True,
        "summary": summary,
        "data": data or {},
        "warnings": warnings or [],
    }


def error(
    *,
    code: str,
    message: str,
    retryable: bool = False,
    suggested_action: str,
) -> ToolResult:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "suggested_action": suggested_action,
        },
    }


def retrieval_list_collections(limit: int = 100, offset: int = 0) -> ToolResult:
    limit = _clamp_limit(limit, maximum=100)
    offset = max(offset, 0)
    collections = deps.get_collection_service().list()
    items = collections[offset : offset + limit]
    return success(
        f"Found {len(items)} retrieval collections.",
        {
            "collections": items,
            "count": len(items),
            "total": len(collections),
            "limit": limit,
            "offset": offset,
        },
    )


def retrieval_create_collection(
    *,
    name: str,
    description: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    embedding_normalized: bool | None = None,
    embedding_distance: str | None = None,
    embedding_contract_version: str | None = None,
    chunk_strategy: str = "plain_text",
    metadata_schema: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> ToolResult:
    payload = CollectionCreate(
        name=name,
        description=description,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_normalized=embedding_normalized,
        embedding_distance=embedding_distance,
        embedding_contract_version=embedding_contract_version,
        chunk_strategy=chunk_strategy,
        metadata_schema=metadata_schema,
    )
    if dry_run:
        return success(
            f"Validated retrieval collection create request for {payload.name}.",
            {"collection": payload.model_dump(mode="json"), "dry_run": True},
        )
    created = deps.get_collection_service().create(payload)
    verb = "Created" if created.get("created") else "Verified existing"
    return success(f"{verb} retrieval collection {created['name']}.", {"collection": created})


def retrieval_search(
    *,
    collections: list[str],
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    candidate_k: int = 30,
    filters: dict[str, Any] | None = None,
    rerank: bool = True,
    include_score_breakdown: bool = True,
    preview_chars: int = 700,
) -> ToolResult:
    request = SearchRequest(
        collections=collections,
        query=query,
        mode=mode,
        top_k=top_k,
        candidate_k=candidate_k,
        filters=Filters.model_validate(filters) if filters is not None else None,
        rerank=rerank,
        include_score_breakdown=include_score_breakdown,
    )
    raw = deps.get_search_service().search(request)
    items = [_agent_search_item(item, preview_chars) for item in raw["items"]]
    return success(
        f"Found {len(items)} retrieval matches for query: {query}",
        {
            "query": raw["query"],
            "mode": raw["mode"],
            "items": items,
            "count": len(items),
            "latency_ms": raw["latency_ms"],
        },
    )


def retrieval_get_document(
    *,
    collection: str,
    document_id: str,
    include_chunks: bool = True,
) -> ToolResult:
    document = deps.get_document_service().get(collection, document_id)
    if not include_chunks:
        document = {key: value for key, value in document.items() if key != "chunks"}
    return success(
        f"Read retrieval document {document_id} from {collection}.",
        {"document": document},
    )


def retrieval_upsert_documents(
    *,
    collection: str,
    documents: list[dict[str, Any]],
    indexing_mode: str = "sync",
    dry_run: bool = False,
) -> ToolResult:
    validated = [DocumentInput.model_validate(document) for document in documents]
    if dry_run:
        return success(
            f"Validated {len(validated)} documents for retrieval upsert into {collection}.",
            {
                "collection": collection,
                "documents": [document.model_dump(mode="json") for document in validated],
                "indexing_mode": indexing_mode,
                "dry_run": True,
            },
        )
    result = deps.get_document_service().upsert(
        DocumentsUpsertRequest(
            collection=collection,
            documents=validated,
            indexing=IndexingOptions(mode=indexing_mode),
        )
    )
    return success(
        f"Upserted {result['upserted']} retrieval documents into {collection}.",
        {"result": result},
    )


def retrieval_rebuild_collection(
    *,
    collection: str,
    documents: list[dict[str, Any]],
    mode: str = "replace",
    dry_run: bool = True,
) -> ToolResult:
    validated = [DocumentInput.model_validate(document) for document in documents]
    if dry_run:
        return success(
            "Validated rebuild of retrieval collection "
            f"{collection} with {len(validated)} documents.",
            {
                "collection": collection,
                "mode": mode,
                "document_count": len(validated),
                "dry_run": True,
            },
        )
    result = deps.get_document_service().rebuild_collection(collection, validated, mode)
    return success(
        f"Rebuilt retrieval collection {collection} with {result['accepted']} documents.",
        {"result": result},
    )


def retrieval_index_status(
    collection: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> ToolResult:
    limit = _clamp_limit(limit, maximum=100)
    collections = deps.get_collection_service().list()
    jobs = deps.get_document_service().list_jobs(collection=collection, status=status, limit=limit)
    data = {
        "collections": collections,
        "collection_count": len(collections),
        "jobs": jobs,
        "job_count": len(jobs),
        "filters": {"collection": collection, "status": status, "limit": limit},
    }
    return success(
        f"Read retrieval index status: {len(collections)} collections, {len(jobs)} jobs.",
        data,
    )


def retrieval_retry_job(job_id: str, dry_run: bool = False) -> ToolResult:
    service = deps.get_document_service()
    if dry_run:
        job = service.get_job(job_id)
        return success(
            f"Validated retry request for retrieval job {job_id}.",
            {"job": job, "dry_run": True},
        )
    job = service.retry_job(job_id)
    return success(f"Retried retrieval job {job_id}.", {"job": job})


TOOL_SPECS: dict[str, ToolSpec] = {
    "retrieval_list_collections": ToolSpec(
        "retrieval_list_collections",
        "read",
        "List retrieval collections and embedding contracts. Read-only.",
    ),
    "retrieval_create_collection": ToolSpec(
        "retrieval_create_collection",
        "admin",
        "Create or verify a retrieval collection. Side effect: writes SQLite metadata "
        "and creates a Qdrant collection unless dry_run=true.",
    ),
    "retrieval_search": ToolSpec(
        "retrieval_search",
        "read",
        "Search one or more retrieval collections with agent-friendly previews. Read-only.",
    ),
    "retrieval_get_document": ToolSpec(
        "retrieval_get_document",
        "read",
        "Read indexed document metadata and chunk references by collection and document_id.",
    ),
    "retrieval_upsert_documents": ToolSpec(
        "retrieval_upsert_documents",
        "write",
        "Upsert documents into a retrieval collection. Side effect: writes SQLite/FTS "
        "metadata and Qdrant vectors unless dry_run=true.",
    ),
    "retrieval_rebuild_collection": ToolSpec(
        "retrieval_rebuild_collection",
        "admin",
        "Rebuild a retrieval collection from supplied documents. Destructive in replace "
        "mode unless dry_run=true.",
    ),
    "retrieval_index_status": ToolSpec(
        "retrieval_index_status",
        "read",
        "Read retrieval collection metadata and recent indexing jobs. Read-only.",
    ),
    "retrieval_retry_job": ToolSpec(
        "retrieval_retry_job",
        "write",
        "Retry a failed retrieval indexing job. Side effect: reruns the stored job "
        "unless dry_run=true.",
    ),
}

TOOLS: dict[str, RawTool] = {
    "retrieval_list_collections": retrieval_list_collections,
    "retrieval_create_collection": retrieval_create_collection,
    "retrieval_search": retrieval_search,
    "retrieval_get_document": retrieval_get_document,
    "retrieval_upsert_documents": retrieval_upsert_documents,
    "retrieval_rebuild_collection": retrieval_rebuild_collection,
    "retrieval_index_status": retrieval_index_status,
    "retrieval_retry_job": retrieval_retry_job,
}


def call_retrieval_tool(tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    if tool_name not in TOOLS:
        return error(
            code="TOOL_NOT_FOUND",
            message=f"Unknown retrieval MCP tool: {tool_name}",
            suggested_action=(
                "Call MCP list_tools and retry with a supported retrieval_* tool name."
            ),
        )
    try:
        result = TOOLS[tool_name](**arguments)
    except ApiError as exc:
        return error(
            code=exc.code,
            message=exc.message,
            retryable=exc.status_code >= 500,
            suggested_action=_suggested_action(exc.code),
        )
    except ValidationError as exc:
        return error(
            code="INVALID_REQUEST",
            message=str(exc),
            suggested_action="Fix the tool arguments to match the documented schema, then retry.",
        )
    except Exception as exc:
        return error(
            code="TOOL_CALL_FAILED",
            message=str(exc),
            retryable=True,
            suggested_action="Inspect retrieval-api settings, dependencies, and logs, then retry.",
        )
    if isinstance(result, dict) and "ok" in result:
        return cast(ToolResult, result)
    return success(f"{tool_name} completed.", {"result": result})


def _agent_search_item(item: dict[str, Any], preview_chars: int) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    text = str(item.get("text") or "")
    preview, truncated = _preview(text, preview_chars)
    payload = {
        "collection": item.get("collection"),
        "document_id": item.get("document_id"),
        "chunk_id": item.get("chunk_id"),
        "title": item.get("title") or "",
        "text_preview": preview,
        "text_truncated": truncated,
        "source": metadata.get("source"),
        "doc_type": metadata.get("doc_type"),
        "score": item.get("score"),
        "metadata": metadata,
    }
    if "score_breakdown" in item:
        payload["score_breakdown"] = item["score_breakdown"]
    return payload


def _preview(text: str, max_chars: int) -> tuple[str, bool]:
    max_chars = _clamp_limit(max_chars, maximum=4000)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "…", True


def _clamp_limit(value: int, *, maximum: int) -> int:
    return max(1, min(int(value), maximum))


def _suggested_action(code: str) -> str:
    suggestions = {
        "COLLECTION_NOT_FOUND": (
            "Call retrieval_list_collections first, then retry with an existing collection."
        ),
        "DOCUMENT_NOT_FOUND": (
            "Check collection/document_id with retrieval_search or retrieval_index_status, "
            "then retry."
        ),
        "INDEX_JOB_FAILED": (
            "Call retrieval_index_status first, then retry with an existing job_id."
        ),
        "INVALID_REQUEST": "Fix the tool arguments to match the documented schema, then retry.",
    }
    return suggestions.get(code, "Inspect retrieval-api logs and tool arguments, then retry.")
