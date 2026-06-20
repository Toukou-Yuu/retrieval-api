from fastapi import APIRouter

from app.api import collections, documents, index, search, system

router = APIRouter(prefix="/v1")
router.include_router(system.router)
router.include_router(collections.router)
router.include_router(documents.router)
router.include_router(index.router)
router.include_router(search.router)
