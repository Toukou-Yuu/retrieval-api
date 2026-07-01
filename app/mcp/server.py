from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.types import ASGIApp

from app.mcp.tools import TOOL_SPECS, call_retrieval_tool


def create_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8301,
    path: str = "/mcp",
) -> FastMCP:
    server = FastMCP(
        "retrieval-api",
        instructions=(
            "Agent-facing tools for hikari's retrieval-api. Use these tools to manage "
            "retrieval collections, search indexed knowledge, inspect documents, and "
            "operate indexing jobs."
        ),
        host=host,
        port=port,
        streamable_http_path=path,
        stateless_http=True,
        json_response=True,
    )
    _register_tools(server)
    return server


def create_mcp_http_app(
    *,
    host: str = "127.0.0.1",
    port: int = 8301,
    path: str = "/mcp",
) -> ASGIApp:
    return create_mcp_server(host=host, port=port, path=path).streamable_http_app()


def run_stdio_server() -> None:
    create_mcp_server().run("stdio")


def run_streamable_http_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8301,
    path: str = "/mcp",
) -> None:
    create_mcp_server(host=host, port=port, path=path).run("streamable-http")


def _register_tools(server: FastMCP) -> None:
    server.tool(
        name="retrieval_list_collections",
        description=TOOL_SPECS["retrieval_list_collections"].description,
    )(retrieval_list_collections_tool)
    server.tool(
        name="retrieval_create_collection",
        description=TOOL_SPECS["retrieval_create_collection"].description,
    )(retrieval_create_collection_tool)
    server.tool(
        name="retrieval_search",
        description=TOOL_SPECS["retrieval_search"].description,
    )(retrieval_search_tool)
    server.tool(
        name="retrieval_get_document",
        description=TOOL_SPECS["retrieval_get_document"].description,
    )(retrieval_get_document_tool)
    server.tool(
        name="retrieval_upsert_documents",
        description=TOOL_SPECS["retrieval_upsert_documents"].description,
    )(retrieval_upsert_documents_tool)
    server.tool(
        name="retrieval_rebuild_collection",
        description=TOOL_SPECS["retrieval_rebuild_collection"].description,
    )(retrieval_rebuild_collection_tool)
    server.tool(
        name="retrieval_index_status",
        description=TOOL_SPECS["retrieval_index_status"].description,
    )(retrieval_index_status_tool)
    server.tool(
        name="retrieval_retry_job",
        description=TOOL_SPECS["retrieval_retry_job"].description,
    )(retrieval_retry_job_tool)


def retrieval_list_collections_tool(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    return call_retrieval_tool("retrieval_list_collections", {"limit": limit, "offset": offset})


def retrieval_create_collection_tool(
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
) -> dict[str, Any]:
    return call_retrieval_tool(
        "retrieval_create_collection",
        {
            "name": name,
            "description": description,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "embedding_normalized": embedding_normalized,
            "embedding_distance": embedding_distance,
            "embedding_contract_version": embedding_contract_version,
            "chunk_strategy": chunk_strategy,
            "metadata_schema": metadata_schema,
            "dry_run": dry_run,
        },
    )


def retrieval_search_tool(
    collections: list[str],
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    candidate_k: int = 30,
    filters: dict[str, Any] | None = None,
    rerank: bool = True,
    include_score_breakdown: bool = True,
    preview_chars: int = 700,
) -> dict[str, Any]:
    return call_retrieval_tool(
        "retrieval_search",
        {
            "collections": collections,
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "filters": filters,
            "rerank": rerank,
            "include_score_breakdown": include_score_breakdown,
            "preview_chars": preview_chars,
        },
    )


def retrieval_get_document_tool(
    collection: str,
    document_id: str,
    include_chunks: bool = True,
) -> dict[str, Any]:
    return call_retrieval_tool(
        "retrieval_get_document",
        {"collection": collection, "document_id": document_id, "include_chunks": include_chunks},
    )


def retrieval_upsert_documents_tool(
    collection: str,
    documents: list[dict[str, Any]],
    indexing_mode: str = "sync",
    dry_run: bool = False,
) -> dict[str, Any]:
    return call_retrieval_tool(
        "retrieval_upsert_documents",
        {
            "collection": collection,
            "documents": documents,
            "indexing_mode": indexing_mode,
            "dry_run": dry_run,
        },
    )


def retrieval_rebuild_collection_tool(
    collection: str,
    documents: list[dict[str, Any]],
    mode: str = "replace",
    dry_run: bool = True,
) -> dict[str, Any]:
    return call_retrieval_tool(
        "retrieval_rebuild_collection",
        {"collection": collection, "documents": documents, "mode": mode, "dry_run": dry_run},
    )


def retrieval_index_status_tool(
    collection: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    return call_retrieval_tool(
        "retrieval_index_status",
        {"collection": collection, "status": status, "limit": limit},
    )


def retrieval_retry_job_tool(job_id: str, dry_run: bool = False) -> dict[str, Any]:
    return call_retrieval_tool("retrieval_retry_job", {"job_id": job_id, "dry_run": dry_run})
