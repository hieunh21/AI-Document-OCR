from fastapi.testclient import TestClient

from ai.schemas import ReceiptExtraction
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.headers["X-Request-ID"]


def test_request_id_is_echoed():
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_list_and_filter_documents():
    body = client.get("/api/v1/documents").json()
    assert body["total"] == len(body["items"]) >= 3
    failed = client.get("/api/v1/documents", params={"status": "failed"}).json()
    assert failed["total"] >= 1 and all(d["status"] == "failed" for d in failed["items"])
    assert client.get("/api/v1/documents", params={"status": "nope"}).status_code == 422


def test_document_detail_matches_schema():
    doc_id = client.get("/api/v1/documents").json()["items"][-1]["id"]
    detail = client.get(f"/api/v1/documents/{doc_id}").json()
    if detail["extraction"] is not None:
        ReceiptExtraction.model_validate(detail["extraction"])
        span_ids = {s["id"] for s in detail["ocr"]["spans"]}
        for ev in detail["extraction"]["evidence"].values():
            assert set(ev["span_ids"]) <= span_ids


def test_unknown_document_is_404():
    assert client.get("/api/v1/documents/does-not-exist").status_code == 404
