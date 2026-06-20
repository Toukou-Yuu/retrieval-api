from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import DocumentsUpsertRequest
from app.services.document_service import DocumentService

from .deps import get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upsert")
def upsert_documents(
    payload: DocumentsUpsertRequest,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.upsert(payload)


@router.get("/{collection}/{document_id:path}")
def get_document(
    collection: str,
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.get(collection, document_id)


@router.delete("/{collection}/{document_id:path}")
def delete_document(
    collection: str,
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.delete(collection, document_id)


@router.post("/{collection}/{document_id:path}/archive")
def archive_document(
    collection: str,
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.archive(collection, document_id)


@router.post("/{collection}/{document_id:path}/restore")
def restore_document(
    collection: str,
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.restore(collection, document_id)
