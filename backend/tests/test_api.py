import csv
import io

from sqlalchemy.orm import sessionmaker

from ai.schemas import ReceiptExtraction
from app.db import make_engine
from app.models import Document
from app.services.documents import recover_interrupted

API = "/api/v1/documents"


# ----------------------------------------------------------------------------- basics


def test_health_reports_database_and_pipeline(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["pipeline"] == {"name": "default", "version": "pipeline-test", "loaded": False}


def test_request_id_is_echoed_and_errors_share_one_shape(client):
    r = client.get(f"{API}/nope", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 404 and r.headers["X-Request-ID"] == "abc123"
    assert r.json() == {
        "detail": "Document 'nope' not found",
        "code": "not_found",
        "request_id": "abc123",
    }


# ----------------------------------------------------------------------------- upload


def test_upload_stores_image_and_returns_document(image, client, upload):
    data = image()
    r = upload(data, name="../../evil/receipt.jpg")
    assert r.status_code == 201
    doc = r.json()
    assert doc["status"] == "uploaded" and doc["original_name"] == "receipt.jpg"
    assert (doc["image_width"], doc["image_height"]) == (600, 900)
    assert doc["extraction"] is None and doc["events"][0]["action"] == "uploaded"
    img = client.get(doc["image_url"])
    assert img.status_code == 200 and img.content == data
    assert img.headers["content-type"] == "image/jpeg"


def test_same_bytes_return_the_existing_document(image, upload):
    data = image()
    first, second = upload(data), upload(data, name="again.jpg")
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["id"] == first.json()["id"]


def test_upload_rejects_bad_files(image, client, upload, settings):
    assert upload(b"").json()["code"] == "empty_file"
    r = upload(b"%PDF-1.4 not an image", name="x.pdf")
    assert (r.status_code, r.json()["code"]) == (415, "unsupported_file")
    r = upload(image(size=(120, 120)))
    assert (r.status_code, r.json()["code"]) == (422, "image_too_small")
    truncated = image()[:2000]
    assert upload(truncated).status_code in (415, 422)
    settings.max_upload_mb = 0  # every non-empty file is now too large
    r = upload(image())
    assert (r.status_code, r.json()["code"]) == (413, "file_too_large")
    assert client.post(API).status_code == 422  # no file part


def test_png_upload(image, upload):
    r = upload(image("PNG"), name="r.png")
    assert r.status_code == 201 and r.json()["content_type"] == "image/png"


def test_upload_rate_limit(image, upload, settings):
    settings.uploads_per_minute = 2
    codes = [upload(image(text=f"r{i}")).status_code for i in range(3)]
    assert codes[:2] == [201, 201] and codes[2] == 429


def test_api_key_when_configured(client, upload, settings):
    settings.api_key = "s3cret"
    assert client.get(API).status_code == 401
    assert client.get(API, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get(API, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays public


# ----------------------------------------------------------------------------- process


def test_process_runs_pipeline_and_keeps_evidence(client, upload):
    doc_id = upload().json()["id"]
    r = client.post(f"{API}/{doc_id}/process")
    assert r.status_code == 200
    doc = r.json()
    # The mock KIE flags the amount, so the document needs review.
    assert doc["status"] == "needs_review"
    assert doc["fields"]["total_cost"] == 150000 and doc["total_cost"] == 150000
    assert doc["field_sources"] == dict.fromkeys(
        ["seller", "seller_address", "timestamp", "total_cost"], "ai"
    )
    assert doc["pipeline_version"] == "pipeline-test" and doc["processing_time_ms"] >= 0
    ex = ReceiptExtraction.model_validate(doc["extraction"])
    span_ids = {s["id"] for s in doc["ocr"]["spans"]}
    for ev in ex.evidence.values():
        assert set(ev.span_ids) <= span_ids
    assert [w["code"] for w in doc["warnings"]] == ["low_confidence"]
    assert [e["action"] for e in doc["events"]] == ["uploaded", "processed"]


def test_upload_and_process_in_one_call(upload):
    doc = upload(process=True).json()
    assert doc["status"] == "needs_review" and doc["extraction_id"] is not None


def test_inference_failure_is_recorded_and_retryable(client, upload, pipeline_factory):
    doc_id = upload().json()["id"]
    pipeline_factory.fail = True
    r = client.post(f"{API}/{doc_id}/process")
    assert (r.status_code, r.json()["code"]) == (500, "inference_failed")
    doc = client.get(f"{API}/{doc_id}").json()
    assert doc["status"] == "failed" and "RuntimeError" in doc["error"]
    pipeline_factory.fail = False
    doc = client.post(f"{API}/{doc_id}/process").json()
    assert doc["status"] == "needs_review" and doc["error"] is None
    assert [e["action"] for e in doc["events"]] == ["uploaded", "failed", "processed"]


def test_missing_file_fails_cleanly(client, upload, settings):
    doc = upload().json()
    (settings.storage_path / "uploads" / doc["id"][:2] / f"{doc['id']}.jpg").unlink()
    r = client.post(f"{API}/{doc['id']}/process")
    assert (r.status_code, r.json()["code"]) == (500, "file_missing")
    assert client.get(f"{API}/{doc['id']}").json()["status"] == "failed"


def test_reprocess_keeps_history_and_resets_current_values(client, upload):
    doc_id = upload(process=True).json()["id"]
    client.patch(f"{API}/{doc_id}/fields", json={"total_cost": 99000})
    doc = client.post(f"{API}/{doc_id}/process").json()
    assert doc["fields"]["total_cost"] == 150000  # back to the new AI result
    assert doc["field_sources"]["total_cost"] == "ai"
    assert len(doc["corrections"]) == 1  # the old correction stays in history


# ----------------------------------------------------------------------------- review


def test_corrections_are_normalized_and_logged(client, upload):
    doc_id = upload(process=True).json()["id"]
    r = client.patch(
        f"{API}/{doc_id}/fields",
        json={"total_cost": "200.000", "timestamp": "09/10/2026 10:15", "reviewer": "an"},
    )
    assert r.status_code == 200
    doc = r.json()
    assert doc["fields"]["total_cost"] == 200000
    assert doc["fields"]["timestamp"] == "2026-10-09T10:15"
    assert doc["field_sources"]["total_cost"] == "human"
    assert doc["field_sources"]["seller"] == "ai"
    assert doc["extraction"]["fields"]["total_cost"] == 150000  # AI output is untouched
    old_new = {c["field_name"]: (c["old_value"], c["new_value"]) for c in doc["corrections"]}
    assert old_new == {
        "total_cost": (150000, 200000),
        "timestamp": ("2026-10-08T09:26", "2026-10-09T10:15"),
    }
    assert all(c["reviewer"] == "an" for c in doc["corrections"])
    # The AI warning was on the corrected amount: nothing left to review.
    assert doc["warnings"] == [] and doc["status"] == "ready"


def test_correction_to_null_and_validation(client, upload):
    doc_id = upload(process=True).json()["id"]
    doc = client.patch(f"{API}/{doc_id}/fields", json={"seller_address": None}).json()
    assert doc["fields"]["seller_address"] is None
    assert not any(w["field"] == "seller_address" for w in doc["warnings"])  # confirmed absent
    bad = client.patch(f"{API}/{doc_id}/fields", json={"timestamp": "hôm qua"})
    assert (bad.status_code, bad.json()["code"]) == (422, "invalid_timestamp")
    bad = client.patch(f"{API}/{doc_id}/fields", json={"total_cost": "abc"})
    assert bad.json()["code"] == "invalid_total_cost"
    assert client.patch(f"{API}/{doc_id}/fields", json={}).json()["code"] == "no_fields"
    assert client.patch(f"{API}/{doc_id}/fields", json={"colour": "red"}).status_code == 422
    future = client.patch(f"{API}/{doc_id}/fields", json={"timestamp": "2099-01-01"}).json()
    assert any(w["code"] == "timestamp_out_of_range" for w in future["warnings"])
    assert future["status"] == "needs_review"


def test_unchanged_value_adds_no_correction(client, upload):
    doc_id = upload(process=True).json()["id"]
    doc = client.patch(f"{API}/{doc_id}/fields", json={"total_cost": 150000}).json()
    assert doc["corrections"] == [] and doc["field_sources"]["total_cost"] == "ai"


def test_cannot_edit_or_approve_before_processing(client, upload):
    doc_id = upload().json()["id"]
    assert client.patch(f"{API}/{doc_id}/fields", json={"seller": "X"}).status_code == 409
    assert client.post(f"{API}/{doc_id}/approve").json()["code"] == "not_processed"


def test_approve_locks_the_document(client, upload):
    doc_id = upload(process=True).json()["id"]
    doc = client.post(f"{API}/{doc_id}/approve", json={"reviewer": "binh"}).json()
    assert doc["status"] == "approved" and doc["approved_by"] == "binh"
    assert doc["approved_at"] is not None
    approved = doc["events"][-1]
    assert approved["action"] == "approved" and approved["actor"] == "binh"
    assert approved["detail"]["open_warnings"] == ["low_confidence"]
    assert client.patch(f"{API}/{doc_id}/fields", json={"seller": "X"}).status_code == 409
    assert client.post(f"{API}/{doc_id}/process").json()["code"] == "approved_locked"
    assert client.post(f"{API}/{doc_id}/approve").json()["code"] == "already_approved"


def test_approve_without_body(client, upload):
    doc_id = upload(process=True).json()["id"]
    assert client.post(f"{API}/{doc_id}/approve").json()["approved_by"] is None


# ----------------------------------------------------------------------------- list / export


def _three_docs(client, upload, image):
    a = upload(image(text="a"), process=True).json()["id"]
    b = upload(image(text="b"), process=True).json()["id"]
    c = upload(image(text="c")).json()["id"]
    client.patch(f"{API}/{b}/fields", json={"seller": "Bách Hoá Xanh", "timestamp": "2020-08-15"})
    client.post(f"{API}/{b}/approve")
    return a, b, c


def test_list_filters_and_pagination(client, upload, image):
    a, b, c = _three_docs(client, upload, image)
    body = client.get(API).json()
    assert body["total"] == 3 and [d["id"] for d in body["items"]] == [c, b, a]
    assert [d["id"] for d in client.get(API, params={"status": "approved"}).json()["items"]] == [b]
    assert client.get(API, params={"seller": "hoá xanh"}).json()["total"] == 1
    r = client.get(API, params={"receipt_from": "2020-08-15", "receipt_to": "2020-08-15"})
    assert [d["id"] for d in r.json()["items"]] == [b]
    page2 = client.get(API, params={"page": 2, "page_size": 2}).json()
    assert page2["total"] == 3 and [d["id"] for d in page2["items"]] == [a]
    assert client.get(API, params={"status": "nope"}).json()["code"] == "invalid_request"
    assert client.get(API, params={"page_size": 1000}).status_code == 422


def test_export_single_document(client, upload):
    doc_id = upload(process=True).json()["id"]
    client.patch(f"{API}/{doc_id}/fields", json={"seller": "Cửa hàng Đức"})
    js = client.get(f"{API}/{doc_id}/export").json()
    assert js["seller"] == "Cửa hàng Đức" and js["field_sources"]["seller"] == "human"
    assert js["evidence"]["total_cost"]["span_ids"] and js["corrections"][0]["field"] == "seller"
    r = client.get(f"{API}/{doc_id}/export", params={"format": "csv"})
    assert r.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0]["seller"] == "Cửa hàng Đức" and rows[0]["total_cost"] == "150000"


def test_bulk_export_of_approved_documents(client, upload, image):
    _, b, _ = _three_docs(client, upload, image)
    r = client.get(f"{API}/export", params={"status": "approved", "format": "csv"})
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert [row["id"] for row in rows] == [b] and rows[0]["seller"] == "Bách Hoá Xanh"
    js = client.get(f"{API}/export", params={"format": "json"}).json()
    assert len(js) == 3


# ----------------------------------------------------------------------------- persistence


def test_results_survive_a_restart(client, upload, settings):
    doc_id = upload(process=True).json()["id"]
    client.patch(f"{API}/{doc_id}/fields", json={"total_cost": 175000})
    fresh = make_engine(settings.database_url)  # a new process would open a new engine
    with sessionmaker(bind=fresh)() as db:
        doc = db.get(Document, doc_id)
        assert doc.total_cost == 175000 and doc.status == "ready"
        assert len(doc.extractions) == 1 and doc.corrections[0].new_value == 175000
    fresh.dispose()


def test_interrupted_processing_is_marked_failed(client, upload, engine):
    doc_id = upload().json()["id"]
    with sessionmaker(bind=engine)() as db:
        db.get(Document, doc_id).status = "processing"
        db.commit()
        assert recover_interrupted(db) == 1
    doc = client.get(f"{API}/{doc_id}").json()
    assert doc["status"] == "failed" and "interrupted" in doc["error"]


# ----------------------------------------------------------------------------- phase 6 helpers


def test_stats_counts_statuses_and_approved_total(client, upload, image):
    a = upload(image(text="a"), process=True).json()["id"]
    upload(image(text="b"), process=True)
    upload(image(text="c"))
    client.post(f"{API}/{a}/approve")
    s = client.get(f"{API}/stats").json()
    assert s["total"] == 3 and s["approved_total_vnd"] == 150000
    assert s["by_status"]["approved"] == 1 and s["by_status"]["needs_review"] == 1
    assert s["by_status"]["uploaded"] == 1 and s["by_status"]["failed"] == 0


def test_image_with_exif_rotation_is_served_upright(client, upload):
    from PIL import Image

    im = Image.new("RGB", (900, 600), "white")  # stored landscape, EXIF says "rotate 90"
    exif = im.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    doc = upload(buf.getvalue()).json()
    assert (doc["image_width"], doc["image_height"]) == (600, 900)  # same frame as OCR boxes
    served = Image.open(io.BytesIO(client.get(doc["image_url"]).content))
    assert served.size == (600, 900)


def test_health_reports_upload_limit(client):
    assert client.get("/health").json()["limits"] == {"max_upload_mb": 10}
