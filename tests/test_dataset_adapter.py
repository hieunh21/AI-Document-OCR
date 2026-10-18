"""Adapter tests.

The synthetic rows below are written by hand in the MC-OCR CSV format and do not
contain dataset content. Tests marked ``requires_dataset`` run only where the data
is present locally.
"""

import json

import pytest

from evaluation.dataset_adapter import build_ground_truth, parse_regions
from tests.conftest import PRIVATE_CHECKS, requires_dataset


def _poly(cat, x, y, w, h, W=768, H=1024):
    seg = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    return {"category_id": cat, "segmentation": seg, "bbox": [x, y, w, h], "width": W, "height": H}


def _row(items, quality=0.7):
    """items: list of (category_id, label, text, (x, y, w, h))."""
    return {
        "img_id": "synthetic.jpg",
        "anno_polygons": repr([_poly(c, *box) for c, _, _, box in items]),
        "anno_texts": "|||".join(t for _, _, t, _ in items),
        "anno_labels": "|||".join(lab for _, lab, _, _ in items),
        "anno_num": str(len(items)),
        "anno_image_quality": str(quality),
    }


def _gt(items):
    regions, size, issues = parse_regions(_row(items))
    gt, gt_issues = build_ground_truth(regions)
    return gt, regions, size, issues + gt_issues


def test_full_receipt():
    gt, regions, size, issues = _gt(
        [
            (15, "SELLER", "CỬA HÀNG MẪU", (230, 80, 240, 28)),
            (16, "ADDRESS", "P. Láng Hạ, Q. Đống Đa", (200, 150, 300, 26)),  # 2nd line first in CSV
            (16, "ADDRESS", "Số 1 Đường Mẫu", (200, 120, 300, 26)),
            (17, "TIMESTAMP", "Ngày : 11/08/2020 08:06", (400, 973, 259, 40)),
            (18, "TOTAL_COST", "Tổng tiền:", (34, 582, 136, 31)),
            (18, "TOTAL_COST", "74,000", (575, 572, 80, 30)),
        ]
    )
    assert size == (768, 1024)
    assert issues == []
    f = gt.fields
    assert f.seller == "CỬA HÀNG MẪU"
    assert f.seller_address == "Số 1 Đường Mẫu P. Láng Hạ, Q. Đống Đa"  # reading order
    assert f.timestamp == "2020-08-11T08:06"
    assert f.total_cost == 74000
    assert gt.raw_fields.total_cost == "74,000"
    # evidence = annotated regions, xyxy, keyword included
    ev = gt.evidence["total_cost"]
    assert len(ev.regions) == 2 and ev.bbox == [34.0, 572.0, 655.0, 613.0]
    assert regions[0].bbox == [230.0, 80.0, 470.0, 108.0]


def test_missing_fields_are_null_and_flagged():
    gt, _, _, issues = _gt([(17, "TIMESTAMP", "Ngày: 10-08-2020", (10, 10, 100, 20))])
    assert gt.fields.seller is None and gt.fields.total_cost is None
    assert gt.fields.timestamp == "2020-08-10"
    assert {
        "missing_seller",
        "missing_seller_address",
        "missing_total_cost",
        "timestamp_no_time",
    } <= set(issues)


def test_category_id_wins_over_label_typo():
    gt, regions, _, issues = _gt(
        [
            (18, "TOTAL_TOTAL_COST", "TỔNG TIỀN PHẢI T.TOÁN", (10, 500, 200, 20)),
            (18, "TOTAL_COST", "30.900", (400, 500, 80, 20)),
        ]
    )
    assert regions[0].field == "total_cost"
    assert gt.fields.total_cost == 30900
    assert "label_category_mismatch:TOTAL_TOTAL_COST->18" in issues


def test_keyword_only_total_is_null_and_flagged():
    gt, _, _, issues = _gt([(18, "TOTAL_COST", "Tổng thanh toán", (10, 500, 200, 20))])
    assert gt.fields.total_cost is None
    assert "total_cost_no_amount" in issues


def test_item_count_is_not_the_total():
    gt, _, _, issues = _gt(
        [
            (18, "TOTAL_COST", "TẠI QUẦY", (10, 500, 100, 20)),
            (18, "TOTAL_COST", "03 03", (150, 500, 50, 20)),
            (18, "TOTAL_COST", "540.000 Đ", (400, 500, 120, 20)),
        ]
    )
    assert gt.fields.total_cost == 540000
    assert "total_cost_ambiguous" not in issues


def test_label_swap_is_flagged():
    _, _, _, issues = _gt([(17, "TIMESTAMP", "TỔNG TIỀN PHẢI T.TOÁN 31.700", (10, 10, 300, 20))])
    assert "suspect_timestamp_contains_total" in issues
    assert "timestamp_unparsed" in issues


def test_empty_row():
    regions, size, issues = parse_regions(_row([]))
    assert regions == [] and size is None and issues == []


# ----------------------------------------------------------------- real data


@requires_dataset
def test_loads_all_labelled_images(mcocr_records):
    assert len(mcocr_records) == 1155
    for r in mcocr_records.values():
        assert r.image_path.exists()
        for reg in r.regions:
            x0, y0, x1, y1 = reg.bbox
            assert 0 <= x0 <= x1 <= r.width + 1 and 0 <= y0 <= y1 <= r.height + 1, r.img_id


@requires_dataset
def test_category_label_mapping_is_consistent(mcocr_records):
    mismatches = [
        i
        for r in mcocr_records.values()
        for i in r.issues
        if i.startswith("label_category_mismatch")
    ]
    assert mismatches == ["label_category_mismatch:TOTAL_TOTAL_COST->18"]


@requires_dataset
@pytest.mark.skipif(not PRIVATE_CHECKS.exists(), reason="manual checks file not present")
def test_manual_annotation_checks(mcocr_records):
    """Regression on hand-verified records (see docs/annotation_manual_check.md)."""
    checks = json.loads(PRIVATE_CHECKS.read_text(encoding="utf-8"))["records"]
    assert len(checks) >= 10
    for c in checks:
        got = mcocr_records[c["img_id"]].gt.fields.model_dump()
        assert got == c["adapter_expected"], c["img_id"]
