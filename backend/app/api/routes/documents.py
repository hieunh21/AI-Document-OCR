from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ai.schemas import DocumentStatus
from app.schemas.documents import DocumentDetail, DocumentList
from app.services import mock_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentList)
def list_documents(status: DocumentStatus | None = None) -> DocumentList:
    items = mock_store.list_documents(status)
    return DocumentList(items=items, total=len(items))


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str) -> DocumentDetail:
    doc = mock_store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id!r} not found")
    return doc
