"""KIE field metrics shared by ``kie_eval`` and ``run_benchmark``.

Definitions: ``evaluation/data_contract.md`` §5. Per field, over images where the
field is *scorable*:

* **EM** — normalized prediction equals ground truth (``None == None`` counts);
  text fields compared on ``text_key`` (``fold_for_match`` + canonical tone placement);
  ``timestamp`` at the precision of the ground truth (date-only GT -> date part, GT
  without seconds -> ``HH:MM``);
* **P / R / F1** over non-null values (TP = non-null pred that matches);
* **CER** (text fields) of ``text_key(pred)`` vs ``text_key(gt)``, on images with a GT
  value (a missing prediction counts as 100 % error);
* ``timestamp`` also gets a **date-only EM**.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ai.normalization.text import canonical_tones, fold_for_match
from ai.schemas import FIELD_NAMES, ReceiptExtraction
from evaluation.dataset_adapter import McocrRecord

# v1 (Phase 3): text compared on fold_for_match.
# v2 (Phase 4): + canonical tone placement ("Thuỵ" == "Thụy"); GT unchanged.
METRICS_VERSION = "kie-metrics-v2"
TEXT_FIELDS = ("seller", "seller_address")


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_key(s: str | None) -> str:
    return canonical_tones(fold_for_match(s))


def ts_match(pred: str | None, gt: str | None, date_only: bool = False) -> bool:
    if gt is None or pred is None:
        return gt is None and pred is None
    if date_only or "T" not in gt:
        return pred[:10] == gt[:10]
    return pred[: len(gt)] == gt if len(gt) > 16 else pred[:16] == gt


def field_match(field: str, pred, gt) -> bool:
    if field == "timestamp":
        return ts_match(pred, gt)
    if pred is None or gt is None:
        return pred is None and gt is None
    if field in TEXT_FIELDS:
        return text_key(pred) == text_key(gt)
    return pred == gt


def image_outcomes(rec: McocrRecord, pred: ReceiptExtraction) -> dict[str, bool]:
    """Field -> exact match, for the scorable fields of one image."""
    skip = rec.unscorable_fields()
    return {
        f: field_match(f, getattr(pred.fields, f), getattr(rec.gt.fields, f))
        for f in FIELD_NAMES
        if f not in skip
    }


def score(records: list[McocrRecord], preds: dict[str, ReceiptExtraction]) -> dict:
    stats: dict[str, dict] = {f: defaultdict(float) for f in FIELD_NAMES}
    rows = []
    for rec in records:
        pred = preds[rec.img_id]
        for f, ok in image_outcomes(rec, pred).items():
            p, g = getattr(pred.fields, f), getattr(rec.gt.fields, f)
            s = stats[f]
            s["n"] += 1
            s["em"] += ok
            s["pred_nonnull"] += p is not None
            s["gt_nonnull"] += g is not None
            s["tp"] += ok and p is not None
            if f in TEXT_FIELDS and g is not None:
                gk, pk = text_key(g), text_key(p or "")
                s["cer_err"] += levenshtein(gk, pk) if p else len(gk)
                s["cer_len"] += len(gk)
            if f == "timestamp":
                s["date_em"] += ts_match(p, g, date_only=True)
            if not ok:
                rows.append(
                    {
                        "img_id": rec.img_id,
                        "field": f,
                        "gt": g,
                        "pred": p,
                        "raw": getattr(pred.raw_fields, f),
                    }
                )
    out = {}
    tot = defaultdict(float)
    for f, s in stats.items():
        prec = s["tp"] / s["pred_nonnull"] if s["pred_nonnull"] else float("nan")
        rec_ = s["tp"] / s["gt_nonnull"] if s["gt_nonnull"] else float("nan")
        out[f] = {
            "n": int(s["n"]),
            "em": s["em"] / s["n"] if s["n"] else float("nan"),
            "precision": prec,
            "recall": rec_,
            "f1": 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0,
        }
        if s["cer_len"]:
            out[f]["cer"] = s["cer_err"] / s["cer_len"]
        if f == "timestamp":
            out[f]["date_em"] = s["date_em"] / s["n"] if s["n"] else float("nan")
        for k in ("tp", "pred_nonnull", "gt_nonnull", "em", "n"):
            tot[k] += s[k]
    p = tot["tp"] / tot["pred_nonnull"] if tot["pred_nonnull"] else float("nan")
    r = tot["tp"] / tot["gt_nonnull"] if tot["gt_nonnull"] else float("nan")
    out["overall"] = {
        "n": int(tot["n"]),
        "em": tot["em"] / tot["n"] if tot["n"] else float("nan"),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }
    return {"metrics": out, "errors": rows}


def paired_bootstrap(
    records: list[McocrRecord],
    preds_a: dict[str, ReceiptExtraction],
    preds_b: dict[str, ReceiptExtraction],
    n: int = 2000,
    seed: int = 2026,
) -> dict:
    """Overall-EM difference B - A with a 95 % percentile interval, resampling images."""
    per_image = []
    for rec in records:
        a, b = image_outcomes(rec, preds_a[rec.img_id]), image_outcomes(rec, preds_b[rec.img_id])
        per_image.append((sum(a.values()), sum(b.values()), len(a)))

    def diff(sample) -> float:
        k = sum(x[2] for x in sample)
        return (sum(x[1] for x in sample) - sum(x[0] for x in sample)) / k if k else 0.0

    rng = random.Random(seed)
    boots = sorted(diff(rng.choices(per_image, k=len(per_image))) for _ in range(n))
    return {
        "diff": diff(per_image),
        "low": boots[int(0.025 * n)],
        "high": boots[int(0.975 * n) - 1],
        "p_b_not_better": sum(d <= 0 for d in boots) / n,
    }
