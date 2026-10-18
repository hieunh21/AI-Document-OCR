"""OCR benchmark on MC-OCR annotated regions.

MC-OCR has no full-page transcription: only the KIE regions (seller, address,
timestamp, total) carry text. We therefore measure, per annotated region:

* **detected** — at least one OCR span overlaps it (intersection >= 50 % of the
  smaller box);
* **CER** — Levenshtein distance between the region text and the *best matching
  substring* of the overlapping spans' text (semi-global alignment), divided by
  the region length. The substring match avoids penalising an OCR line that also
  contains neighbouring text (e.g. "Số GD: … Ngày: 11/08/2020").
  Undetected regions count as 100 % error.

Three text views separate error sources:

* ``cer``         NFC, whitespace collapsed, case-sensitive;
* ``cer_folded``  lower-case, punctuation removed (``fold_for_match``);
* ``cer_noacc``   folded and Vietnamese diacritics removed — the gap between
  ``cer_folded`` and ``cer_noacc`` is the error caused by wrong/missing accents.

CLI::

    python -m evaluation.ocr_eval run --split smoke --preset v6_medium [--device gpu:0]
    python -m evaluation.ocr_eval report --split smoke --presets v6_medium v6_small
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from ai.layout import reading_order
from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.money import find_amounts, to_vnd
from ai.normalization.text import fold_for_match, normalize_text
from ai.schemas import OCRResult, OCRSpan
from evaluation.dataset_adapter import McocrRecord, Region, load_split

OUT_ROOT = Path("outputs/ocr")
REPORTS = Path("evaluation/reports")
MIN_OVERLAP = 0.5


# --------------------------------------------------------------------------- text


def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def views(s: str) -> dict[str, str]:
    base = normalize_text(s) or ""
    folded = fold_for_match(base)
    return {"cer": base, "cer_folded": folded, "cer_noacc": strip_accents(folded)}


def substring_edit_distance(gt: str, pred: str) -> int:
    """min over substrings p' of pred of Levenshtein(gt, p')."""
    if not gt:
        return 0
    if not pred:
        return len(gt)
    prev = [0] * (len(pred) + 1)  # free start anywhere in pred
    for i, g in enumerate(gt, 1):
        cur = [i] + [0] * len(pred)
        for j, p in enumerate(pred, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (g != p))
        prev = cur
    return min(prev)  # free end anywhere in pred


# --------------------------------------------------------------------------- matching


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap_ratio(a, b) -> float:
    """Intersection over the smaller box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min(_area(a), _area(b))
    return (ix * iy) / small if small > 0 else 0.0


def spans_for_region(region: Region, spans: list[OCRSpan]) -> list[OCRSpan]:
    hit = [s for s in spans if overlap_ratio(region.bbox, s.bbox) >= MIN_OVERLAP]
    return reading_order(hit, lambda s: s.bbox)


def score_image(rec: McocrRecord, ocr: OCRResult) -> list[dict]:
    rows = []
    for region in rec.regions:
        if not region.text:
            continue
        hit = spans_for_region(region, ocr.spans)
        pred = " ".join(s.text for s in hit)
        gt_v, pred_v = views(region.text), views(pred)
        row = {
            "img_id": rec.img_id,
            "field": region.field,
            "detected": bool(hit),
            "gt": region.text,
            "pred": pred,
        }
        for k in gt_v:
            row[f"{k}_len"] = len(gt_v[k])
            row[f"{k}_err"] = substring_edit_distance(gt_v[k], pred_v[k]) if hit else len(gt_v[k])
        # Value-level checks: what KIE actually needs from OCR.
        if region.field == "total_cost":
            gt_amounts = {v for v in map(to_vnd, find_amounts(region.text)) if v >= 100}
            if gt_amounts:
                row["amount_ok"] = gt_amounts <= set(map(to_vnd, find_amounts(pred)))
        if region.field == "timestamp":
            gt_ts = parse_timestamp(region.text)
            if gt_ts and gt_ts.iso:
                pred_ts = parse_timestamp(pred)
                row["ts_ok"] = bool(pred_ts) and pred_ts.iso == gt_ts.iso
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    def cer(rs, k):
        n = sum(r[f"{k}_len"] for r in rs)
        return sum(r[f"{k}_err"] for r in rs) / n if n else float("nan")

    out = {
        "regions": len(rows),
        "detected": sum(r["detected"] for r in rows) / len(rows) if rows else float("nan"),
    }
    for k in ("cer", "cer_folded", "cer_noacc"):
        out[k] = cer(rows, k)
    out["line_exact"] = sum(r["cer_err"] == 0 for r in rows) / len(rows) if rows else float("nan")
    for key in ("amount_ok", "ts_ok"):
        rs = [r[key] for r in rows if key in r]
        out[key] = sum(rs) / len(rs) if rs else float("nan")
        out[f"{key}_n"] = len(rs)
    out["by_field"] = {}
    by_field = defaultdict(list)
    for r in rows:
        by_field[r["field"]].append(r)
    for f, rs in sorted(by_field.items()):
        out["by_field"][f] = {
            "regions": len(rs),
            "cer": cer(rs, "cer"),
            "cer_noacc": cer(rs, "cer_noacc"),
        }
    return out


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))] if s else float("nan")


# --------------------------------------------------------------------------- running


def run(split: str, preset: str, device: str, limit: int | None = None, tag: str = "") -> Path:
    from ai.ocr.factory import build_ocr
    from ai.preprocessing.image import load_image

    out = OUT_ROOT / f"{preset}@{device.replace(':', '')}{tag}" / split
    out.mkdir(parents=True, exist_ok=True)
    records = load_split(split)[:limit] if limit else load_split(split)
    provider = build_ocr(preset, device=device)
    t0 = time.perf_counter()
    provider.warmup()
    load_s = time.perf_counter() - t0
    failures = []
    for k, rec in enumerate(records, 1):
        path = out / f"{rec.img_id}.json"
        if path.exists():
            continue
        try:
            result = provider.predict(load_image(rec.image_path))
        except Exception as e:  # keep the batch going, record the failure
            failures.append({"img_id": rec.img_id, "error": repr(e)})
            print(f"[{k}/{len(records)}] {rec.img_id} FAILED: {e!r}", flush=True)
            continue
        path.write_text(result.model_dump_json(), encoding="utf-8")
        print(
            f"[{k}/{len(records)}] {rec.img_id} {len(result.spans)} spans "
            f"{result.latency_ms:.0f} ms",
            flush=True,
        )
    meta = {
        "preset": preset,
        "device": device,
        "version": provider.version,
        "model_load_s": round(load_s, 1),
        "failures": failures,
    }
    (out / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_results(split: str, run_name: str) -> dict[str, OCRResult]:
    d = OUT_ROOT / run_name / split
    return {
        p.name[: -len(".json")]: OCRResult.model_validate_json(p.read_text(encoding="utf-8"))
        for p in d.glob("*.jpg.json")
    }


def evaluate(split: str, run_name: str) -> dict:
    records = load_split(split)
    results = load_results(split, run_name)
    missing = [r.img_id for r in records if r.img_id not in results]
    rows = [
        row for r in records if r.img_id in results for row in score_image(r, results[r.img_id])
    ]
    lat = [results[r.img_id].latency_ms for r in records if r.img_id in results]
    meta_path = OUT_ROOT / run_name / split / "_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "run": run_name,
        "split": split,
        "images": len(results),
        "missing_images": missing,
        "version": meta.get("version"),
        "metrics": aggregate(rows),
        "latency_ms": {
            "p50": statistics.median(lat) if lat else None,
            "p95": pct(lat, 0.95),
            "mean": statistics.mean(lat) if lat else None,
        },
        "spans_per_image": statistics.mean(len(results[i].spans) for i in results)
        if results
        else 0,
        "rows": rows,
    }


def export_errors(e: dict) -> Path:
    """Per-region errors, worst first, for error analysis (contains label text:
    stays under outputs/, which is git-ignored)."""
    import csv

    path = OUT_ROOT / e["run"] / e["split"] / "_errors.csv"
    rows = sorted(e["rows"], key=lambda r: -r["cer_folded_err"] / max(1, r["cer_folded_len"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img_id", "field", "detected", "cer_folded", "cer_noacc", "gt", "pred"])
        for r in rows:
            w.writerow(
                [
                    r["img_id"],
                    r["field"],
                    r["detected"],
                    round(r["cer_folded_err"] / max(1, r["cer_folded_len"]), 3),
                    round(r["cer_noacc_err"] / max(1, r["cer_noacc_len"]), 3),
                    r["gt"],
                    r["pred"],
                ]
            )
    return path


def latency_table(split: str, runs: list[str]) -> list[str]:
    lines = [
        "",
        "Latency per image, measured in separate runs with nothing else on the machine "
        "(model already loaded; includes orientation, detection, recognition):",
        "",
        "| Run | Images | P50 ms | P95 ms | Mean ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in runs:
        lat = [x.latency_ms for x in load_results(split, r).values()]
        lines.append(
            f"| `{r}` | {len(lat)} | {statistics.median(lat):.0f} | {pct(lat, 0.95):.0f} | "
            f"{statistics.mean(lat):.0f} |"
        )
    return lines


def write_report(
    split: str, runs: list[str], hardware: str, latency_runs: list[str] | None = None
) -> Path:
    evals = [evaluate(split, r) for r in runs]
    for e in evals:
        if e["images"]:
            export_errors(e)
    lines = [
        f"# OCR benchmark — split `{split}`\n",
        "> Generated by `python -m evaluation.ocr_eval report`. Metric definitions: module "
        "docstring of `evaluation/ocr_eval.py`. Regions = annotated KIE regions only.\n",
        f"Hardware: {hardware}\n",
        "| Run | Images | Regions | Detected | CER | CER folded | CER no-accent | "
        "Line exact | Amount read | Date-time read | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in evals:
        m, lat = e["metrics"], e["latency_ms"]
        if not e["images"]:
            lines.append(f"| `{e['run']}` | 0 | – | – | – | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| `{e['run']}` | {e['images']} | {m['regions']} | {m['detected']:.1%} | "
            f"{m['cer']:.1%} | {m['cer_folded']:.1%} | {m['cer_noacc']:.1%} | "
            f"{m['line_exact']:.1%} | {m['amount_ok']:.1%} | {m['ts_ok']:.1%} | "
            f"{lat['p50']:.0f} | {lat['p95']:.0f} |"
        )
    lines += [
        "",
        "CER by field (case-sensitive / no-accent):",
        "",
        "| Run | " + " | ".join(("seller", "seller_address", "timestamp", "total_cost")) + " |",
        "|---|---|---|---|---|",
    ]
    for e in evals:
        bf = e["metrics"]["by_field"]
        cells = [
            f"{bf[f]['cer']:.1%} / {bf[f]['cer_noacc']:.1%}" if f in bf else "-"
            for f in ("seller", "seller_address", "timestamp", "total_cost")
        ]
        lines.append(f"| `{e['run']}` | " + " | ".join(cells) + " |")
    if evals and evals[0]["images"]:
        m0 = evals[0]["metrics"]
        lines += [
            "",
            f"*Amount read*: share of TOTAL_COST regions holding an amount ({m0['amount_ok_n']}) "
            "whose amount appears in the OCR text. *Date-time read*: share of TIMESTAMP "
            f"regions with a parseable value ({m0['ts_ok_n']}) whose parsed ISO value equals "
            "the one parsed from the OCR text. Both use the Phase 1 parsers.",
        ]
    if latency_runs:
        lines += latency_table(split, latency_runs)
    lines += ["", "Versions:", ""] + [f"- `{e['run']}`: {e['version']}" for e in evals]
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"ocr_{split}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", default="smoke")
    r.add_argument("--preset", default="v6_medium")
    r.add_argument("--device", default="cpu")
    r.add_argument("--limit", type=int)
    r.add_argument("--tag", default="", help="suffix for the output folder, e.g. _latency")
    e = sub.add_parser("report")
    e.add_argument("--split", default="smoke")
    e.add_argument("--runs", nargs="+", required=True, help="e.g. v6_medium@gpu0")
    e.add_argument("--hardware", default="unspecified")
    e.add_argument("--latency-runs", nargs="*", help="runs to report latency from")
    args = ap.parse_args()
    if args.cmd == "run":
        print(run(args.split, args.preset, args.device, args.limit, args.tag))
    else:
        print(write_report(args.split, args.runs, args.hardware, args.latency_runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
