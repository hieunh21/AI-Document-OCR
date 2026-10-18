"""KIE evaluation on cached OCR output (metric definitions: data_contract.md §5).

Per field, over images where the field is *scorable*:

* **EM** — normalized prediction equals ground truth (``None == None`` counts);
  text fields compared with ``fold_for_match``; ``timestamp`` at the precision of the
  ground truth (date-only GT -> date part, GT without seconds -> ``HH:MM``);
* **P / R / F1** over non-null values (TP = non-null pred that matches);
* **CER** (text fields) of the folded prediction vs folded GT, on images with a GT
  value (a missing prediction counts as 100 % error);
* ``timestamp`` also gets a **date-only EM**.

CLI::

    python -m evaluation.kie_eval --split dev --ocr-run "v6_medium+orient@gpu0" --kie rules
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ai.kie.base import extract_from_ocr
from ai.normalization.text import fold_for_match
from ai.schemas import FIELD_NAMES, ReceiptExtraction
from evaluation.dataset_adapter import McocrRecord, load_split
from evaluation.ocr_eval import load_results

OUT_ROOT = Path("outputs/kie")
REPORTS = Path("evaluation/reports")
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
        return fold_for_match(pred) == fold_for_match(gt)
    return pred == gt


def score(records: list[McocrRecord], preds: dict[str, ReceiptExtraction]) -> dict:
    stats: dict[str, dict] = {f: defaultdict(float) for f in FIELD_NAMES}
    rows = []
    for rec in records:
        pred = preds[rec.img_id]
        skip = rec.unscorable_fields()
        for f in FIELD_NAMES:
            if f in skip:
                continue
            p, g = getattr(pred.fields, f), getattr(rec.gt.fields, f)
            ok = field_match(f, p, g)
            s = stats[f]
            s["n"] += 1
            s["em"] += ok
            s["pred_nonnull"] += p is not None
            s["gt_nonnull"] += g is not None
            s["tp"] += ok and p is not None
            if f in TEXT_FIELDS and g is not None:
                gf, pf = fold_for_match(g), fold_for_match(p or "")
                s["cer_err"] += levenshtein(gf, pf) if p else len(gf)
                s["cer_len"] += len(gf)
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
            out[f]["date_em"] = s["date_em"] / s["n"]
        for k in ("tp", "pred_nonnull", "gt_nonnull", "em", "n"):
            tot[k] += s[k]
    p = tot["tp"] / tot["pred_nonnull"] if tot["pred_nonnull"] else float("nan")
    r = tot["tp"] / tot["gt_nonnull"] if tot["gt_nonnull"] else float("nan")
    out["overall"] = {
        "em": tot["em"] / tot["n"],
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }
    return {"metrics": out, "errors": rows}


def run(split: str, ocr_run: str, kie_name: str, workers: int = 1) -> dict:
    from ai.kie.factory import build_kie

    kie = build_kie(kie_name)
    records = load_split(split)
    ocr = load_results(split, ocr_run)
    missing = [r.img_id for r in records if r.img_id not in ocr]
    if missing:
        raise SystemExit(f"{len(missing)} images of {split} have no OCR in {ocr_run}")

    def one(rec):
        t0 = time.perf_counter()
        pred, usage = extract_from_ocr(kie, ocr[rec.img_id])  # upright layout
        return rec.img_id, pred, usage, (time.perf_counter() - t0) * 1000

    preds, lat, usages, failures = {}, [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, rec): rec.img_id for rec in records}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                img_id, pred, usage, ms = fut.result()
            except Exception as e:  # count as an empty prediction, keep going
                img_id = futures[fut]
                failures.append({"img_id": img_id, "error": repr(e)[:300]})
                print(f"[{k}/{len(records)}] {img_id} FAILED: {e!r}"[:300], flush=True)
                preds[img_id] = ReceiptExtraction(model_version="failed")
                continue
            preds[img_id] = pred
            lat.append(ms)
            if usage is not None:
                usages.append(usage)
            if workers > 1 and k % 20 == 0:
                print(f"[{k}/{len(records)}]", flush=True)
    result = score(records, preds)
    result |= {
        "split": split,
        "ocr_run": ocr_run,
        "kie": f"{kie.name}-{kie.version}",
        "images": len(records),
        "failures": failures,
        "kie_ms_mean": sum(lat) / len(lat) if lat else float("nan"),
    }
    if usages:
        live = [u for u in usages if not u.cached]
        result["llm"] = {
            "calls": len(usages),
            "cached_calls": len(usages) - len(live),
            "input_tokens": sum(u.input_tokens for u in usages),
            "output_tokens": sum(u.output_tokens for u in usages),
            "latency_ms_p50": sorted(u.latency_ms for u in usages)[len(usages) // 2],
            "latency_ms_p95": sorted(u.latency_ms for u in usages)[
                min(len(usages) - 1, int(0.95 * len(usages)))
            ],
        }
    out = OUT_ROOT / f"{kie_name}@{ocr_run}".replace(":", "_")  # ":" is invalid on Windows
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
        for i, p in preds.items():
            fh.write(json.dumps({"img_id": i, **p.model_dump(mode="json")}, ensure_ascii=False))
            fh.write("\n")
    with open(out / f"{split}_errors.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["img_id", "field", "gt", "pred", "raw"])
        w.writeheader()
        w.writerows(result["errors"])
    return result


def fmt(x: float) -> str:
    return "–" if x != x else f"{x:.1%}"


def report(split: str, results: list[dict]) -> Path:
    lines = [
        f"# KIE — split `{split}`\n",
        "> Generated by `python -m evaluation.kie_eval`. Metrics: `evaluation/data_contract.md` "
        "§5 (EM over scorable images; P/R/F1 over non-null values).\n",
        "| KIE | OCR | Seller EM | Address EM | Timestamp EM (date) | Total EM | "
        "Overall EM | Overall F1 | Seller CER | Address CER |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| `{r['kie']}` | `{r['ocr_run']}` | {fmt(m['seller']['em'])} | "
            f"{fmt(m['seller_address']['em'])} | {fmt(m['timestamp']['em'])} "
            f"({fmt(m['timestamp']['date_em'])}) | {fmt(m['total_cost']['em'])} | "
            f"{fmt(m['overall']['em'])} | {fmt(m['overall']['f1'])} | "
            f"{fmt(m['seller'].get('cer', float('nan')))} | "
            f"{fmt(m['seller_address'].get('cer', float('nan')))} |"
        )
    lines += [
        "",
        "Per field P / R / F1 (non-null values):",
        "",
        "| KIE | OCR | " + " | ".join(FIELD_NAMES) + " |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["metrics"]
        cells = [
            f"{fmt(m[f]['precision'])} / {fmt(m[f]['recall'])} / {fmt(m[f]['f1'])}"
            for f in FIELD_NAMES
        ]
        lines.append(f"| `{r['kie']}` | `{r['ocr_run']}` | " + " | ".join(cells) + " |")
    n = results[0]["metrics"] if results else {}
    lines += [
        "",
        "Scorable images per field: " + ", ".join(f"{f} {n[f]['n']}" for f in FIELD_NAMES) + ".",
    ]
    llm = [r for r in results if r.get("llm")]
    if llm:
        lines += [
            "",
            "LLM calls (tokens as reported by the API; latency of the original call, "
            "including network and any proxy):",
            "",
            "| KIE | Calls | Input tokens / receipt | Output tokens / receipt | "
            "P50 latency ms | P95 latency ms | Failed |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in llm:
            u = r["llm"]
            lines.append(
                f"| `{r['kie']}` | {u['calls']} | {u['input_tokens'] / u['calls']:.0f} | "
                f"{u['output_tokens'] / u['calls']:.0f} | {u['latency_ms_p50']:.0f} | "
                f"{u['latency_ms_p95']:.0f} | {len(r['failures'])} |"
            )
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"kie_{split}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ocr-run", nargs="+", default=["v6_medium+orient@gpu0"])
    ap.add_argument("--kie", nargs="+", default=["rules"])
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="parallel calls (LLM)")
    args = ap.parse_args()
    results = [run(args.split, o, k, args.workers) for k in args.kie for o in args.ocr_run]
    for r in results:
        m = r["metrics"]
        print(
            f"{r['kie']} @ {r['ocr_run']}: "
            + "  ".join(f"{f}={m[f]['em']:.3f}" for f in (*FIELD_NAMES, "overall"))
        )
    if not args.no_report:
        print(report(args.split, results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
