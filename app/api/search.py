from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import SearchMode, SearchRequest
from app.services.search_service import SearchService

from .deps import get_search_service

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
def search(
    payload: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> dict[str, object]:
    return service.search(payload)


@router.post("/dense")
def dense_search(
    payload: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> dict[str, object]:
    return service.search(payload.model_copy(update={"mode": SearchMode.DENSE}))


@router.post("/keyword")
def keyword_search(
    payload: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> dict[str, object]:
    return service.search(payload.model_copy(update={"mode": SearchMode.KEYWORD}))


@router.post("/hybrid")
def hybrid_search(
    payload: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> dict[str, object]:
    return service.search(payload.model_copy(update={"mode": SearchMode.HYBRID}))
