from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError

from app import __version__
from app.api import deps, router
from app.config import get_settings
from app.errors import ApiError, error_response, http_exception_response


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.validate_embedding_contract_on_startup:
        model = deps.get_embedding_client().get_model(settings.default_embedding_model)
        if model.contract_version != settings.embedding_api_contract_version:
            raise RuntimeError(
                "Embedding API contract version does not match retrieval-api settings"
            )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="retrieval-api",
        description="Local unified retrieval API for agent systems",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError):
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException):
    return http_exception_response(exc)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return error_response(
        422,
        "INVALID_REQUEST",
        "Request validation failed",
        {"errors": exc.errors()},
    )
