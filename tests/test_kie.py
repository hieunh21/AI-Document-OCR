"""KIE tests on hand-written OCR spans (no dataset, no network)."""

import pytest

from ai.kie.llm import LLMKIE, LLMBackend, LLMUsage
from ai.kie.rule_based import RuleBasedKIE
from ai.schemas import OCRSpan


def S(i, text, x0, y0, x1, y1):
    return OCRSpan(id=f"s{i:03d}", text=text, bbox=[x0, y0, x1, y1])


RECEIPT = [
    S(0, "MINIMART ANAN", 230, 80, 470, 108),
    S(1, "Chợ Sủi Phú Thị Gia Lâm", 200, 120, 498, 146),
    S(2, "Tel: 0912345678", 50, 150, 300, 172),
    S(3, "HÓA ĐƠN BÁN HÀNG", 190, 200, 515, 240),
    S(4, "Sữa TH 110ml", 50, 280, 300, 305),
    S(5, "6,750", 420, 280, 480, 305),
    S(6, "54,000", 550, 282, 640, 307),
    S(7, "Tổng cộng :", 35, 540, 200, 565),
    S(8, "74,000", 560, 536, 650, 560),
    S(9, "Tổng tiền:", 34, 582, 170, 613),
    S(10, "74,000", 575, 572, 655, 602),  # slightly higher than its keyword
    S(11, "Tiền khách trả:", 35, 620, 250, 645),
    S(12, "100,000", 560, 620, 655, 645),
    S(13, "Tiền thừa", 35, 660, 200, 685),
    S(14, "26,000", 560, 660, 655, 685),
    S(15, "Số GD: 000AC2212008001407 Ngày: 11/08/2020-08:06", 70, 973, 660, 1013),
]


def test_rules_extract_all_fields_with_evidence():
    ex = RuleBasedKIE().extract(None, RECEIPT)
    f = ex.fields
    assert f.seller == "MINIMART ANAN"
    assert f.seller_address == "Chợ Sủi Phú Thị Gia Lâm"
    assert f.timestamp == "2020-08-11T08:06"
    assert f.total_cost == 74000
    assert ex.evidence["total_cost"].span_ids == ["s009", "s010"]
    assert ex.evidence["timestamp"].span_ids == ["s015"]
    assert not ex.warnings


def test_rules_ignore_cash_given_and_change():
    spans = [s for s in RECEIPT if s.id not in ("s007", "s008", "s009", "s010")]
    assert RuleBasedKIE().extract(None, spans).fields.total_cost is None


def test_rules_never_crash_on_empty_input():
    ex = RuleBasedKIE().extract(None, [])
    assert ex.missing_fields() == ["seller", "seller_address", "timestamp", "total_cost"]


class FakeBackend(LLMBackend):
    name, model = "fake", "fake-1"

    def __init__(self, response):
        self.response, self.calls, self.last_user = response, 0, None

    def complete_json(self, system, user, schema):
        self.calls += 1
        self.last_user = user
        return self.response, LLMUsage(input_tokens=100, output_tokens=20, latency_ms=5)


def _llm(response, tmp_path):
    return LLMKIE(FakeBackend(response), cache_dir=tmp_path)


def test_llm_prompt_tags_every_span(tmp_path):
    kie = _llm({}, tmp_path)
    kie.extract(None, RECEIPT)
    assert "[s009] Tổng tiền: [s010] 74,000" in kie.backend.last_user


def test_llm_output_is_normalized_and_evidence_is_real(tmp_path):
    response = {
        "seller": {"text": "MINIMART ANAN", "span_ids": ["s000"]},
        "seller_address": {"text": "Chợ Sủi Phú Thị Gia Lâm", "span_ids": ["s001", "s999"]},
        "timestamp": {"text": "Ngày: 11/08/2020-08:06", "span_ids": ["s015"]},
        "total_cost": {"text": "74,000", "span_ids": ["s010"]},
    }
    ex = _llm(response, tmp_path).extract(None, RECEIPT)
    assert ex.fields.timestamp == "2020-08-11T08:06"
    assert ex.fields.total_cost == 74000
    assert ex.evidence["seller_address"].span_ids == ["s001"]  # invented id dropped
    assert [w.code for w in ex.warnings] == ["llm_unknown_span"]


def test_llm_empty_text_means_absent(tmp_path):
    response = {
        f: {"text": "", "span_ids": []}
        for f in ("seller", "seller_address", "timestamp", "total_cost")
    }
    response["total_cost"] = {"text": "không rõ", "span_ids": []}
    ex = _llm(response, tmp_path).extract(None, RECEIPT)
    assert ex.missing_fields() == ["seller", "seller_address", "timestamp", "total_cost"]
    assert any(w.code == "unparsable_value" for w in ex.warnings)


def test_llm_responses_are_cached(tmp_path):
    kie = _llm({"seller": {"text": "X", "span_ids": []}}, tmp_path)
    _, u1 = kie.extract_with_usage(RECEIPT)
    _, u2 = kie.extract_with_usage(RECEIPT)
    assert kie.backend.calls == 1
    assert not u1.cached and u2.cached and u2.input_tokens == 100


def test_dotenv_strips_inline_comments(tmp_path, monkeypatch):
    from ai.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text(
        "A_URL=https://x.example   # comment\n"
        "A_EMPTY=                  # empty = default\n"
        'A_QUOTED="va#lue"  # c\n'
        "# full comment\n",
        encoding="utf-8",
    )
    for k in ("A_URL", "A_EMPTY", "A_QUOTED"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(env)
    import os

    assert os.environ["A_URL"] == "https://x.example"
    assert "A_EMPTY" not in os.environ
    assert os.environ["A_QUOTED"] == "va#lue"


def test_parse_json_object_tolerates_fences_and_noise():
    import pytest

    from ai.kie.llm import LLMResponseError, parse_json_object

    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(LLMResponseError):
        parse_json_object("no json here")
    with pytest.raises(LLMResponseError):
        parse_json_object("[1, 2]")


def test_llm_tolerates_flattened_fields(tmp_path):
    ex = _llm(
        {"seller": "MINIMART ANAN", "total_cost": {"text": "74,000", "span_ids": "s010"}}, tmp_path
    ).extract(None, RECEIPT)
    assert ex.fields.seller == "MINIMART ANAN" and ex.fields.total_cost == 74000


def test_validation_warns_but_keeps_values():
    from datetime import date

    from ai.kie.validate import validate_extraction
    from ai.schemas import ReceiptExtraction, ReceiptFields

    ex = ReceiptExtraction(
        model_version="t", fields=ReceiptFields(timestamp="2031-01-01", total_cost=1_500_000_001)
    )
    validate_extraction(ex, today=date(2026, 10, 18))
    codes = {w.code for w in ex.warnings}
    assert codes == {"timestamp_out_of_range", "total_cost_out_of_range"}
    assert ex.fields.total_cost == 1_500_000_001  # never dropped
    validate_extraction(ex, today=date(2026, 10, 18))
    assert len(ex.warnings) == 2  # idempotent


@pytest.mark.parametrize("photo_turn", ["cw", "ccw"])
def test_kie_on_rotated_ocr_uses_upright_layout_and_original_boxes(photo_turn):
    from ai.kie.base import extract_from_ocr
    from ai.schemas import OCRResult

    W, H = 768, 1024  # upright page; the photo is landscape (H x W)

    def photo_box(b):
        x0, y0, x1, y1 = b
        if photo_turn == "cw":  # page turned 90° clockwise: (x, y) -> (H - y, x)
            return [H - y1, x0, H - y0, x1]
        return [y0, W - x1, y1, W - x0]  # counter-clockwise: (x, y) -> (y, W - x)

    # rotation = CCW turn that makes the page upright (orientation classifier convention)
    rotation = 90 if photo_turn == "cw" else 270
    spans = [s.model_copy(update={"bbox": photo_box(s.bbox)}) for s in RECEIPT]
    ocr = OCRResult(image_width=H, image_height=W, spans=spans, rotation=rotation)
    ex, _ = extract_from_ocr(RuleBasedKIE(), ocr)
    assert ex.fields.total_cost == 74000 and ex.fields.seller == "MINIMART ANAN"
    assert ex.fields.timestamp == "2020-08-11T08:06"
    ev = ex.evidence["total_cost"].regions
    assert ev[1].bbox == photo_box(RECEIPT[10].bbox)  # evidence in photo coordinates


def test_rules_attach_time_printed_on_the_line_above():
    spans = [
        S(0, "VinCommerce", 97, 387, 300, 427),
        S(1, "Thời gian:10:43", 318, 433, 520, 461),  # other column, OCR line above
        S(2, "Ngày:12/08/2020", 100, 449, 300, 478),
        S(3, "NVBH:09021212", 337, 473, 520, 497),
    ]
    ex = RuleBasedKIE().extract(None, spans)
    assert ex.fields.timestamp == "2020-08-12T10:43"
    assert ex.evidence["timestamp"].span_ids == ["s001", "s002"]


def test_rules_find_address_by_place_names_and_repair_accents():
    from ai.normalization.address import AdminUnits

    admin = AdminUnits.load()
    if admin is None:
        pytest.skip("configs/vn_admin_units.json not built")
    spans = [
        S(0, "THỨC COFFEE", 167, 65, 560, 162),
        S(1, "22 Quang Trung, P10, Gò Vãp", 234, 175, 520, 214),  # no address keyword
        S(2, "Tel:028 3588 3591", 307, 212, 500, 242),
        S(3, "Cảm ơn quý khách", 200, 900, 500, 930),  # bottom of the receipt
    ]
    assert RuleBasedKIE(seller_fallback=False).extract(None, spans).fields.seller_address is None
    kie = RuleBasedKIE(seller_fallback=False, admin=admin)
    assert kie.extract(None, spans).fields.seller_address == "22 Quang Trung, P10, Gò Vấp"
