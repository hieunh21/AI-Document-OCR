import pytest
from pydantic import ValidationError

from ai.mock import MockKIE, MockOCR
from ai.pipeline import ReceiptPipeline
from ai.schemas import (
    DocumentStatus,
    Evidence,
    FieldEvidence,
    OCRSpan,
    ReceiptExtraction,
    ReceiptFields,
)


def test_all_fields_may_be_null():
    ex = ReceiptExtraction(model_version="t")
    assert ex.fields.model_dump() == {
        "seller": None,
        "seller_address": None,
        "timestamp": None,
        "total_cost": None,
    }
    assert ex.missing_fields() == ["seller", "seller_address", "timestamp", "total_cost"]
    assert ex.suggested_status() is DocumentStatus.NEEDS_REVIEW


def test_json_schema_allows_null_for_every_field():
    schema = ReceiptExtraction.model_json_schema()
    fields = schema["$defs"]["ReceiptFields"]["properties"]
    for name in ("seller", "seller_address", "timestamp", "total_cost"):
        types = {t.get("type") for t in fields[name]["anyOf"]}
        assert "null" in types, name


@pytest.mark.parametrize("ts", ["2020-08-11", "2020-08-11T08:06", "2020-08-11T08:06:59"])
def test_timestamp_accepts_iso(ts):
    assert ReceiptFields(timestamp=ts).timestamp == ts


@pytest.mark.parametrize("ts", ["11/08/2020", "2020-02-30", "2020-08-11T25:00", "2020-08-11 08:06"])
def test_timestamp_rejects_non_iso_or_impossible(ts):
    with pytest.raises(ValidationError):
        ReceiptFields(timestamp=ts)


def test_total_cost_must_be_non_negative_int():
    assert ReceiptFields(total_cost=150000).total_cost == 150000
    with pytest.raises(ValidationError):
        ReceiptFields(total_cost=-1)


def test_blank_strings_become_null():
    f = ReceiptFields(seller="   ", seller_address=" 12  Nguyễn  Trãi ")
    assert f.seller is None
    assert f.seller_address == "12 Nguyễn Trãi"


def test_bbox_must_be_xyxy():
    with pytest.raises(ValidationError):
        OCRSpan(id="s", text="x", bbox=[10, 10, 5, 20])
    with pytest.raises(ValidationError):
        OCRSpan(id="s", text="x", bbox=[1, 2, 3])


def test_field_evidence_fills_union_bbox_and_span_ids():
    ev = FieldEvidence(
        regions=[
            Evidence(bbox=[10, 10, 50, 20], source_id="a"),
            Evidence(bbox=[60, 5, 90, 25], source_id="b"),
        ]
    )
    assert ev.bbox == [10, 5, 90, 25]
    assert ev.span_ids == ["a", "b"]


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        ReceiptExtraction(model_version="t", fields={"vendor": "x"})


def test_mock_pipeline_round_trip():
    result = ReceiptPipeline(MockOCR(), MockKIE()).run(_blank_image())
    ex = result.extraction
    assert ex.fields.total_cost == 150000
    span_ids = {s.id for s in result.ocr.spans}
    for ev in ex.evidence.values():  # evidence must point at real OCR spans
        assert set(ev.span_ids) <= span_ids
    assert ReceiptExtraction.model_validate_json(ex.model_dump_json()) == ex
    assert set(result.timings_ms) == {"preprocess", "ocr", "kie", "quality", "total"}


def _blank_image():
    from PIL import Image

    return Image.new("RGB", (768, 1024), "white")
