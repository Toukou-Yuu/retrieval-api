from fastapi import APIRouter

from app.api import collections, system

router = APIRouter(prefix="/v1")
router.include_router(system.router)
router.include_router(collections.router)
