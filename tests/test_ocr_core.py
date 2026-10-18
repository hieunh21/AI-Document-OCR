import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ai.preprocessing.image import ImageError, Transform, load_image, prepare_for_ocr
from evaluation.ocr_eval import (
    aggregate,
    overlap_ratio,
    strip_accents,
    substring_edit_distance,
)


def _marker_image(w=300, h=500, box=(40, 60, 90, 80)):
    im = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(im).rectangle(box, fill="black")
    return im


def _black_bbox(im: Image.Image) -> list[float]:
    a = np.asarray(im.convert("L")) < 128
    ys, xs = np.nonzero(a)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("max_side", [None, 250])
def test_transform_maps_back_to_original(rotation, max_side):
    box = (40, 60, 90, 80)
    im = _marker_image(box=box)
    work, tf = prepare_for_ocr(im, max_side=max_side, rotation=rotation)
    if max_side:
        assert max(work.size) <= max_side
    wb = _black_bbox(work)  # where the marker is in the working image
    corners = np.array([[wb[0], wb[1]], [wb[2], wb[1]], [wb[2], wb[3]], [wb[0], wb[3]]])
    back = tf.bbox_to_original(corners)
    expected = [box[0], box[1], box[2] + 1, box[3] + 1]
    tol = 2 / (tf.scale)  # resampling blur
    assert np.allclose(back, expected, atol=tol), (back, expected)


def test_transform_clips_to_image():
    tf = Transform(orig_w=100, orig_h=200)
    assert tf.bbox_to_original(np.array([[-5, -5], [120, 250]])) == [0, 0, 100, 200]


def test_load_image_validates():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, "PNG")
    with pytest.raises(ImageError) as e:
        load_image(buf.getvalue())
    assert e.value.code == "image_too_small"
    with pytest.raises(ImageError) as e:
        load_image(b"not an image")
    assert e.value.code == "unreadable_image"
    buf = io.BytesIO()
    Image.new("L", (400, 600)).save(buf, "JPEG")
    assert load_image(buf.getvalue()).mode == "RGB"


def test_load_image_applies_exif_orientation():
    im = _marker_image(w=300, h=500)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    assert load_image(buf.getvalue()).size == (500, 300)


@pytest.mark.parametrize(
    "gt, pred, dist",
    [
        ("abc", "abc", 0),
        ("abc", "xxabcxx", 0),  # substring match is free
        ("abc", "xxabxx", 1),
        ("74,000", "Tổng tiền: 74,000", 0),
        ("abc", "", 3),
        ("", "abc", 0),
        ("kitten", "sitting", 2),  # 'sittin' substring: k->s, e->i
    ],
)
def test_substring_edit_distance(gt, pred, dist):
    assert substring_edit_distance(gt, pred) == dist


def test_strip_accents_vietnamese():
    assert strip_accents("chợ sủi phú thị gia lâm đường") == "cho sui phu thi gia lam duong"


def test_overlap_ratio():
    assert overlap_ratio([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert overlap_ratio([0, 0, 10, 10], [2, 2, 4, 4]) == 1.0  # small box inside big one
    assert overlap_ratio([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_aggregate_counts_missed_regions_as_full_error():
    rows = [
        {
            "field": "seller",
            "detected": True,
            "cer_len": 10,
            "cer_err": 1,
            "cer_folded_len": 10,
            "cer_folded_err": 1,
            "cer_noacc_len": 10,
            "cer_noacc_err": 0,
        },
        {
            "field": "seller",
            "detected": False,
            "cer_len": 10,
            "cer_err": 10,
            "cer_folded_len": 10,
            "cer_folded_err": 10,
            "cer_noacc_len": 10,
            "cer_noacc_err": 10,
        },
    ]
    m = aggregate(rows)
    assert m["detected"] == 0.5
    assert m["cer"] == pytest.approx(11 / 20)
    assert m["line_exact"] == 0.0


def test_crop_line_rectifies_rotated_quad():
    from ai.ocr.hybrid import crop_line, order_quad

    rgb = np.full((200, 300, 3), 255, np.uint8)
    quad = np.array([[50, 100], [250, 60], [256, 90], [56, 130]], np.float32)  # tilted line
    q = order_quad(quad)
    assert q[0][0] < q[1][0] and q[0][1] < q[3][1]  # TL, TR, BR, BL
    crop = crop_line(rgb, quad, pad_ratio=0.2)
    assert crop.shape[1] > crop.shape[0]  # stays horizontal
    assert crop.shape[0] >= 30 * 1.4  # height + padding on both sides


def test_quality_model_fit_predict_roundtrip():
    from ai.quality.assess import LinearQualityModel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.clip(0.6 + 0.1 * X[:, 0] - 0.05 * X[:, 2] + rng.normal(0, 0.01, 200), 0, 1)
    m = LinearQualityModel.fit(X, y, ["a", "b", "c"])
    pred = m.predict(X)
    assert np.abs(pred - y).mean() < 0.02
    m2 = LinearQualityModel.model_validate_json(m.model_dump_json())
    assert np.allclose(m2.predict(X), pred)


def test_quality_features_on_blank_and_sharp_images():
    from ai.quality.assess import QualityAssessor, image_features
    from ai.schemas import OCRResult

    blank = Image.new("RGB", (600, 800), "white")
    sharp = _marker_image(600, 800, box=(100, 100, 500, 120))
    assert image_features(sharp)["log_lap_var"] > image_features(blank)["log_lap_var"]
    report = QualityAssessor().assess(blank, OCRResult(image_width=600, image_height=800))
    assert "no_text_detected" in report.warnings


def test_factory_parses_presets_without_loading_models():
    from ai.ocr.factory import available_presets, build_ocr

    p = build_ocr("v6_tiny+orient", device="cpu")
    assert p.auto_orient and p.preset_name == "v6_tiny+orient" and p._ocr is None
    assert "v6med_det+vietocr_seq2seq" in available_presets()
    with pytest.raises(ValueError):
        build_ocr("does_not_exist")


def test_value_level_ocr_checks():
    from ai.schemas import OCRResult, OCRSpan
    from evaluation.dataset_adapter import build_ground_truth, parse_regions
    from evaluation.ocr_eval import score_image
    from tests.test_dataset_adapter import _row

    regions, _, _ = parse_regions(
        _row(
            [
                (17, "TIMESTAMP", "Ngày bán: 17/08/2020 06:28", (10, 10, 300, 20)),
                (18, "TOTAL_COST", "74,000", (400, 500, 80, 20)),
            ]
        )
    )

    class Rec:
        img_id = "x.jpg"

    rec = Rec()
    rec.regions = regions
    gt, _ = build_ground_truth(regions)
    ocr = OCRResult(
        image_width=768,
        image_height=1024,
        spans=[
            OCRSpan(id="a", text="Ngày bán:17/08/2020 06:28", bbox=[8, 8, 310, 32]),
            OCRSpan(id="b", text="74.000", bbox=[398, 498, 482, 522]),
        ],
    )
    rows = {r["field"]: r for r in score_image(rec, ocr)}
    assert rows["timestamp"]["ts_ok"] is True
    assert rows["total_cost"]["amount_ok"] is True
