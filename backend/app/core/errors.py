"""Errors raised by the service layer; ``app.main`` turns them into JSON responses."""

from __future__ import annotations


class AppError(Exception):
    """``code`` is machine readable; ``message`` is shown to the user."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def not_found(doc_id: str) -> AppError:
    return AppError(404, "not_found", f"Document {doc_id!r} not found")
