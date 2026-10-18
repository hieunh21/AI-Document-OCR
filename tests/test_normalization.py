import pytest

from ai.layout import reading_order
from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.money import find_amounts, parse_vnd
from ai.normalization.text import fold_for_match, normalize_text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("150.000", 150000),
        ("150,000", 150000),
        ("150 000", 150000),
        ("150000", 150000),
        ("150.000đ", 150000),
        ("VND 150,000", 150000),
        ("VNd359,732", 359732),
        ("880,400.00", 880400),
        ("43.500,00", 43500),
        ("43.500.00", 43500),  # same separator for thousands and decimals (seen in labels)
        ("41,520.40", 41520),  # rounded half-up to whole dong
        ("0.357.075", 357075),
        ("47 . 500", 47500),
        ("61 ,000", 61000),
        ("VMD1, 528, 279", 1528279),
        ("10.000 10.000", 10000),  # repeated identical amount is fine
        ("Tổng tiền:", None),
        ("", None),
        (None, None),
        ("7.000 9.000", None),  # two different amounts -> ambiguous
    ],
)
def test_parse_vnd(text, expected):
    assert parse_vnd(text) == expected


def test_dates_and_times_are_not_amounts():
    assert find_amounts("Ngày: 10/09/2020 - 17:39") == []
    assert [int(a) for a in find_amounts("Ngày : 15/08/2020 09:05 Tổng 209,000")] == [209000]


@pytest.mark.parametrize(
    "text, iso",
    [
        ("Ngày bán: 15/08/2020 11:41", "2020-08-15T11:41"),
        ("Ngày : 11/08/2020 08:06", "2020-08-11T08:06"),
        ("Thời gian: 16:40:27 - 14/08/2020", "2020-08-14T16:40:27"),
        ("Thời gian: 10:09:06-15/08/2020", "2020-08-15T10:09:06"),
        ("Ngày: 10/08/2020 (08:57 CH - 08:57 CH)", "2020-08-10T20:57"),
        ("Ngày: 08/08/2020 (08.41 CH 08.41 CH)", "2020-08-08T20:41"),
        ("Thời gian : 01.10.2020 08.58", "2020-10-01T08:58"),
        ("Ngày: 12/08/2020 - 13:17", "2020-08-12T13:17"),
        ("NGÀY KINH DOANH: 14/08/2020", "2020-08-14"),
        ("Ngày: 10-08-2020", "2020-08-10"),
        ("Ngày 14 tháng 08 năm 2020", "2020-08-14"),
        ("2020-08-18 20:54", "2020-08-18T20:54"),
        ("13:06:2020 17:28:03", "2020-06-13T17:28:03"),
        ("Thời gian: 02,10.2020 14.19", "2020-10-02T14:19"),
        ("15 Aug 20 19:12:07", "2020-08-15T19:12:07"),
        ("Ngày 22/01/2020 Giờ: 16:49", "2020-01-22T16:49"),
        ("Ngày: 23/09/2019 9:10:04", "2019-09-23T09:10:04"),
        ("Giờ vào: 14-08-2020 20:53 Giờ ra: 14-08-2020 21:31", "2020-08-14T20:53"),
        # OCR output often drops the space after the colon (regression, found in Phase 2)
        ("Ngày bán:17/08/2020 06:28", "2020-08-17T06:28"),
        ("Ngay:10-08-2020", "2020-08-10"),
    ],
)
def test_parse_timestamp(text, iso):
    assert parse_timestamp(text).iso == iso


@pytest.mark.parametrize(
    "text",
    [
        "Ngày:",
        "TỔNG TIỀN PHẢI T.TOÁN 31.700",
        "Ngày: 09.10.2C20",  # OCR/annotation typo: never repaired
        "Ngày: 19/09/20199",
        "Ngày: 04/20/19 11:25",  # month 20 does not exist: no US-format guess
        "NGÀY 13/88/2020",
    ],
)
def test_parse_timestamp_without_valid_date(text):
    parsed = parse_timestamp(text)
    assert parsed is None or parsed.iso is None


def test_time_only_is_not_a_timestamp():
    parsed = parse_timestamp("18:11")
    assert parsed.time is not None and parsed.iso is None


def test_multiple_times_flagged():
    parsed = parse_timestamp("Ngày: 13/08/2020 (08:09 CH - 08:35 CH)")
    assert parsed.iso == "2020-08-13T20:09"
    assert "multiple_times" in parsed.flags


def test_text_normalization():
    decomposed = "Hòa"  # "Hòa" written with a combining grave accent
    assert normalize_text(f"  {decomposed}   Bình ") == "Hòa Bình"  # precomposed ò
    assert normalize_text("   ") is None
    assert fold_for_match("ĐC: 212 Đường Trần Phú-Cẩm Phả") == "đc 212 đường trần phú cẩm phả"


def test_reading_order_groups_same_line():
    boxes = {
        "amount": [500, 578, 650, 602],  # slightly higher than the keyword, same line
        "keyword": [30, 582, 170, 612],
        "header": [200, 80, 470, 110],
        "below": [30, 650, 200, 680],
    }
    order = reading_order(list(boxes), lambda k: boxes[k])
    assert order == ["header", "keyword", "amount", "below"]
