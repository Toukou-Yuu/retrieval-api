from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.schemas import DocumentInput
from app.services.document_service import DocumentService

from .deps import get_document_service

router = APIRouter(prefix="/index", tags=["index"])


class ReindexRequest(BaseModel):
    collection: str
    documents: list[DocumentInput]
    delete_missing: bool = False


class RebuildCollectionRequest(BaseModel):
    collection: str
    documents: list[DocumentInput]
    mode: str = "replace"


@router.post("/reindex")
def reindex(
    payload: ReindexRequest,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.reindex(payload.collection, payload.documents, payload.delete_missing)


@router.post("/rebuild-collection")
def rebuild_collection(
    payload: RebuildCollectionRequest,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.rebuild_collection(payload.collection, payload.documents, payload.mode)


@router.get("/jobs")
def list_jobs(
    service: Annotated[DocumentService, Depends(get_document_service)],
    collection: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, object]:
    return {"items": service.list_jobs(collection, status, limit)}


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.get_job(job_id)


@router.post("/jobs/{job_id}/retry")
def retry_job(
    job_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.retry_job(job_id)


@router.delete("/jobs/{job_id}")
def delete_job(
    job_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> dict[str, object]:
    return service.delete_job(job_id)
