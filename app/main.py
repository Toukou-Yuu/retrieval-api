from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError

from app import __version__
from app.api import router
from app.errors import ApiError, error_response, http_exception_response


def create_app() -> FastAPI:
    app = FastAPI(
        title="retrieval-api",
        description="Local unified retrieval API for agent systems",
        version=__version__,
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
