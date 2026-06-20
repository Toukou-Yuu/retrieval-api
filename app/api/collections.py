from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import CollectionCreate
from app.services.collection_service import CollectionService

from .deps import get_collection_service

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("")
def create_collection(
    payload: CollectionCreate,
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> dict[str, object]:
    return service.create(payload)


@router.get("")
def list_collections(
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> dict[str, object]:
    return {"items": service.list()}


@router.get("/{collection}")
def get_collection(
    collection: str,
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> dict[str, object]:
    return service.get(collection)


@router.delete("/{collection}")
def delete_collection(
    collection: str,
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> dict[str, object]:
    return service.delete(collection)
