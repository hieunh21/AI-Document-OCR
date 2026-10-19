from ai.normalization.text import canonical_tones
from ai.schemas import ReceiptExtraction, ReceiptFields
from evaluation.dataset_adapter import McocrRecord
from evaluation.metrics import field_match, paired_bootstrap, score, text_key, ts_match


def test_canonical_tones_merges_old_and_new_placement():
    assert canonical_tones("thuỵ") == canonical_tones("thụy")
    assert canonical_tones("hoà bình") == canonical_tones("hòa bình")
    assert canonical_tones("khoẻ") == canonical_tones("khỏe")
    # a different tone or vowel quality is still different
    assert canonical_tones("thủy") != canonical_tones("thụy")
    assert canonical_tones("hoà") != canonical_tones("hóa")
    assert canonical_tones("tô") != canonical_tones("tơ")
    assert canonical_tones("Bách Hoá Xanh") == canonical_tones("Bách Hóa Xanh")


def test_field_match_rules():
    assert field_match("seller", "VINMART+ Thuỵ Khuê", "vinmart thụy khuê")
    assert not field_match("seller", "Thủy Khuê", "Thụy Khuê")
    assert field_match("total_cost", None, None)
    assert not field_match("total_cost", 1000, None)
    assert ts_match("2020-08-13T20:09:15", "2020-08-13T20:09")  # GT precision
    assert ts_match("2020-08-13T20:09", "2020-08-13")  # date-only GT
    assert not ts_match("2020-08-13", "2020-08-13T20:09")
    assert text_key("Hoà, Bình!") == text_key("hòa bình")


def _rec(img_id: str, **fields) -> McocrRecord:
    gt = ReceiptExtraction(model_version="gt", fields=ReceiptFields(**fields))
    return McocrRecord(img_id, None, 1, 1, 1.0, [], gt)  # type: ignore[arg-type]


def _pred(**fields) -> ReceiptExtraction:
    return ReceiptExtraction(model_version="test", fields=ReceiptFields(**fields))


def test_score_and_bootstrap():
    recs = [_rec(f"i{k}", seller="A", total_cost=1000) for k in range(10)]
    good = {r.img_id: _pred(seller="a", total_cost=1000) for r in recs}
    half = {r.img_id: _pred(seller="a") for r in recs}  # total missing
    m = score(recs, good)["metrics"]
    assert m["overall"]["em"] == 1.0 and m["seller"]["cer"] == 0.0
    m = score(recs, half)["metrics"]
    assert m["total_cost"]["recall"] == 0.0 and m["overall"]["em"] == 0.75  # 4 fields x 10
    b = paired_bootstrap(recs, half, good, n=200)
    assert b["diff"] == 0.25 and b["low"] == b["high"] == 0.25 and b["p_b_not_better"] == 0.0
