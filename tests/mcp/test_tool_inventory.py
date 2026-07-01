import asyncio

from app.mcp.server import create_mcp_server
from app.mcp.tools import TOOL_SPECS, call_retrieval_tool

EXPECTED_TOOLS = {
    "retrieval_list_collections",
    "retrieval_create_collection",
    "retrieval_search",
    "retrieval_get_document",
    "retrieval_upsert_documents",
    "retrieval_rebuild_collection",
    "retrieval_index_status",
    "retrieval_retry_job",
}


def test_mcp_tool_inventory_matches_architecture_standard() -> None:
    server = create_mcp_server()

    tools = asyncio.run(server.list_tools())
    tool_names = {tool.name for tool in tools}

    assert tool_names == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description
        assert tool.name in TOOL_SPECS
        assert TOOL_SPECS[tool.name].permission in {"read", "write", "admin"}


def test_tool_calls_return_standard_success_envelope(tmp_path, monkeypatch) -> None:
    from app.api import deps
    from app.config import Settings
    from app.repositories.sqlite_repository import SQLiteRepository
    from app.services.collection_service import CollectionService

    class MemoryQdrant:
        def collection_exists(self, name: str) -> bool:
            return False

        def create_collection(self, name: str, dimension: int, distance: str = "Cosine") -> None:
            return None

        def get_collection_vector_config(self, name: str) -> tuple[int, str]:
            return 3, "Cosine"

    class OfflineEmbedding:
        pass

    settings = Settings(
        retrieval_data_dir=tmp_path,
        validate_embedding_contract_on_collection_create=False,
        default_embedding_dimension=3,
    )
    repo = SQLiteRepository(settings.sqlite_path)
    service = CollectionService(repo, MemoryQdrant(), OfflineEmbedding(), settings)
    monkeypatch.setattr(deps, "get_collection_service", lambda: service)

    result = call_retrieval_tool("retrieval_list_collections", {})

    assert result == {
        "ok": True,
        "summary": "Found 0 retrieval collections.",
        "data": {"collections": [], "count": 0, "total": 0, "limit": 100, "offset": 0},
        "warnings": [],
    }


def test_tool_calls_return_standard_error_envelope(tmp_path, monkeypatch) -> None:
    from app.api import deps
    from app.config import Settings
    from app.repositories.sqlite_repository import SQLiteRepository
    from app.services.chunk_service import ChunkService
    from app.services.document_service import DocumentService

    class MemoryQdrant:
        def delete_document_points(self, collection: str, document_id: str) -> None:
            return None

    class OfflineEmbedding:
        pass

    settings = Settings(retrieval_data_dir=tmp_path)
    repo = SQLiteRepository(settings.sqlite_path)
    service = DocumentService(repo, MemoryQdrant(), OfflineEmbedding(), ChunkService())
    monkeypatch.setattr(deps, "get_document_service", lambda: service)

    result = call_retrieval_tool(
        "retrieval_get_document",
        {"collection": "missing", "document_id": "doc-1"},
    )

    assert result["ok"] is False
    assert result["error"] == {
        "code": "DOCUMENT_NOT_FOUND",
        "message": "Document not found: doc-1",
        "retryable": False,
        "suggested_action": (
            "Check collection/document_id with retrieval_search or retrieval_index_status, "
            "then retry."
        ),
    }
