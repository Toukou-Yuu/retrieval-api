from fastapi import HTTPException
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def api_error(status_code: int, code: str, message: str, details: dict | None = None) -> ApiError:
    return ApiError(status_code, code, message, details)


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or {}}},
    )


def http_exception_response(exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    code = detail.get("code", "INVALID_REQUEST")
    message = detail.get("message", str(exc.detail))
    details = detail.get("details", {})
    return error_response(exc.status_code, code, message, details)
