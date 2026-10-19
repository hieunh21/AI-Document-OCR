"""One command: OCR -> KIE -> metrics -> report, for named pipelines on one split.

    python -m evaluation.run_benchmark --split dev --model baseline rules llm \
        [--latency-images 30] [--workers 8] [--report model_comparison]

* OCR output is cached per preset in ``outputs/ocr/<preset>@<device>/<split>`` and
  only missing images are processed (``evaluation.ocr_eval.run``).
* KIE runs through ``extract_from_ocr`` exactly as in the app; LLM responses are
  cached on disk (``outputs/llm_cache``), so a re-run makes no new paid calls.
* Accuracy is reported on all scorable images and on subsets (clean annotations,
  store seen in train or not, image quality, rotation, seller stratum), with a paired
  bootstrap interval against the first pipeline.
* ``--latency-images N`` additionally times the real ``ReceiptPipeline`` (image load
  -> OCR -> KIE -> quality) on the first N images, sequentially, models warm.
  A cached LLM response is charged its original call latency.
* ``--split test`` requires ``--final``: the test split is run once per frozen pipeline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from ai.normalization.text import fold_noacc
from ai.schemas import FIELD_NAMES
from evaluation.dataset_adapter import GT_VERSION, McocrRecord, load_split
from evaluation.metrics import METRICS_VERSION, paired_bootstrap, score

REPORTS = Path("evaluation/reports")
OUT = Path("outputs/benchmark")
OCR_ROOT = Path("outputs/ocr")
GAZETTEER = Path("configs/kie_gazetteer.json")
QUALITY_MODEL = Path("configs/quality_model.json")
HARDWARE = "NVIDIA RTX 3050 Laptop 4 GB (gpu0), laptop CPU, Windows 11"
# Google list price for Gemini 2.5 Flash (USD per 1M tokens, paid tier). The proxy we
# call bills differently; this only gives an order of magnitude.
LLM_PRICE_USD_PER_M = {"input": 0.30, "output": 2.50}
# Flags that do not make an annotation suspect: a receipt without a printed time,
# or a field that is simply absent (a valid null target).
BENIGN_FLAGS = ("timestamp_no_time", "missing_")


@dataclass(frozen=True)
class Pipeline:
    ocr: str
    kie: str
    note: str


PIPELINES: dict[str, Pipeline] = {
    "baseline": Pipeline("v6_medium+orient", "rules", "rules, no store list"),
    "rules": Pipeline("v6_medium+orient", "rules+gaz-nofb", "rules + gazetteer"),
    "rules-adm": Pipeline(
        "v6_medium+orient", "rules+gaz+adm-nofb", "rules + gazetteer + official place names"
    ),
    "rules-vietocr": Pipeline(
        "v6med_det+vietocr_transformer+orient", "rules+gaz-nofb", "rules + gazetteer, VietOCR"
    ),
    "llm-nogaz": Pipeline("v6_medium+orient", "llm:gemini", "Gemini, no store list"),
    "llm": Pipeline("v6_medium+orient", "llm:gemini+gaz", "Gemini + gazetteer"),
    "llm-rerun": Pipeline(
        "v6_medium+orient",
        "llm:gemini:rerun2+gaz",
        "Gemini + gazetteer, same requests re-sent later (proxy backend check)",
    ),
    "llm-vietocr": Pipeline(
        "v6med_det+vietocr_transformer+orient", "llm:gemini+gaz", "Gemini + gazetteer, VietOCR"
    ),
    # pipeline-v3: PP-OCRv6 recognizer fine-tuned for Vietnamese (docs/finetune_rec.md)
    "rules-vi3": Pipeline(
        "v6_medium_vi3+orient", "rules+gaz+adm-nofb", "rules + gazetteer + place names, vi rec"
    ),
    "llm-vi3": Pipeline("v6_medium_vi3+orient", "llm:gemini+gaz", "Gemini + gazetteer, vi rec"),
    # pipeline-v4: learned span classifier instead of line rules (ai/kie/learned.py)
    "lines-vi3": Pipeline(
        "v6_medium_vi3+orient", "lines+gaz+adm", "span classifier + gazetteer + place names"
    ),
}


# --------------------------------------------------------------------------- helpers


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))] if s else float("nan")


def sha256_short(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "missing"


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ocr_run_name(preset: str, device: str) -> str:
    return f"{preset}@{device.replace(':', '')}"


def ensure_ocr(split: str, preset: str, device: str, records: list[McocrRecord]) -> str:
    """Run OCR for the images of ``split`` that have no cached result yet."""
    from evaluation.ocr_eval import run as run_ocr

    name = ocr_run_name(preset, device)
    d = OCR_ROOT / name / split
    if any(not (d / f"{r.img_id}.json").exists() for r in records):
        print(f"OCR {name} on {split} (missing images only)", flush=True)
        run_ocr(split, preset, device)
        free_gpu()
    missing = [r.img_id for r in records if not (d / f"{r.img_id}.json").exists()]
    if missing:
        raise SystemExit(f"OCR failed on {len(missing)} images of {split}: {missing[:3]}")
    return name


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - CPU-only install
        pass


@lru_cache(maxsize=1)
def train_sellers() -> frozenset[str]:
    return frozenset(
        fold_noacc(r.gt.fields.seller) for r in load_split("train") if r.gt.fields.seller
    )


def subsets(records: list[McocrRecord], split: str, ocr) -> dict[str, list[McocrRecord]]:
    """Named image subsets for breakdowns (``ocr``: img_id -> OCRResult)."""
    manifest = json.loads(Path(f"data/splits/{split}.json").read_text(encoding="utf-8"))
    stratum = {it["img_id"]: it["stratum"] for it in manifest["items"]}

    def clean(r: McocrRecord) -> bool:
        return all(i.startswith(BENIGN_FLAGS) for i in r.issues)

    def seen(r: McocrRecord) -> bool | None:
        s = r.gt.fields.seller
        return None if not s else fold_noacc(s) in train_sellers()

    out = {
        "all": records,
        "clean annotations": [r for r in records if clean(r)],
        "store seen in train": [r for r in records if seen(r) is True],
        "store not seen in train": [r for r in records if seen(r) is False],
        "no seller labelled": [r for r in records if seen(r) is None],
        "quality < 0.5": [r for r in records if r.quality < 0.5],
        "quality 0.5–0.7": [r for r in records if 0.5 <= r.quality < 0.7],
        "quality ≥ 0.7": [r for r in records if r.quality >= 0.7],
        "rotated photo": [r for r in records if ocr[r.img_id].rotation],
        "upright photo": [r for r in records if not ocr[r.img_id].rotation],
    }
    for s in sorted({stratum[r.img_id] for r in records}):
        out[f"stratum {s}"] = [r for r in records if stratum[r.img_id] == s]
    return out


def time_pipelines(
    names: list[str], device: str, records: list[McocrRecord]
) -> dict[str, dict[str, list[float]]]:
    """Live per-stage timings of ``ReceiptPipeline`` for each named pipeline.

    All models stay loaded and the pipelines run *interleaved* on each image, so a
    laptop GPU changing clock speed mid-run affects every pipeline alike.
    """
    from ai.kie.factory import build_kie
    from ai.ocr.factory import build_ocr
    from ai.pipeline import ReceiptPipeline
    from ai.preprocessing.image import load_image
    from ai.quality.assess import QualityAssessor

    quality = QualityAssessor()
    ocrs = {p: build_ocr(p, device=device) for p in sorted({PIPELINES[n].ocr for n in names})}
    kies = {k: build_kie(k) for k in sorted({PIPELINES[n].kie for n in names})}
    pipes = {
        n: ReceiptPipeline(ocrs[PIPELINES[n].ocr], kies[PIPELINES[n].kie], quality) for n in names
    }
    warm = load_image(records[0].image_path)
    for pipe in pipes.values():
        pipe.run(warm)
    out: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    for k, rec in enumerate(records, 1):
        for n, pipe in pipes.items():
            t0 = time.perf_counter()
            image = load_image(rec.image_path)
            load_ms = (time.perf_counter() - t0) * 1000
            res = pipe.run(image)
            ms = dict(res.timings_ms, load=load_ms)
            if res.llm_usage is not None and res.llm_usage.cached:
                ms["kie"] += res.llm_usage.latency_ms
                ms["total"] += res.llm_usage.latency_ms
            ms["total"] += load_ms
            for key, v in ms.items():
                out[n].setdefault(key, []).append(v)
        print(f"latency [{k}/{len(records)}]", flush=True)
    del pipes, ocrs
    free_gpu()
    return out


# --------------------------------------------------------------------------- report


def fmt(x: float) -> str:
    return "–" if x != x else f"{x:.1%}"


def ms(x: float) -> str:
    return "–" if x != x else f"{x:,.0f}"


def write_report(path: Path, ctx: dict, results: dict[str, dict]) -> None:
    names = list(results)
    lines = [
        f"# Benchmark — split `{ctx['split']}`\n",
        f"> Generated by `{ctx['command']}` on {ctx['date']}. Do not edit by hand.\n",
        "| Item | Value |",
        "|---|---|",
        f"| Split | `{ctx['split']}` ({ctx['split_version']}), {ctx['images']} images |",
        f"| Ground truth | `{GT_VERSION}` (`evaluation/data_contract.md`) |",
        f"| Metrics | `{METRICS_VERSION}` (`evaluation/metrics.py`) |",
        f"| Code | git `{ctx['git']}` |",
        f"| Gazetteer | `{GAZETTEER.as_posix()}` sha256 `{sha256_short(GAZETTEER)}` "
        "(built from train labels) |",
        f"| Quality model | `{QUALITY_MODEL.as_posix()}` sha256 `{sha256_short(QUALITY_MODEL)}` |",
        f"| Hardware | {HARDWARE}; Python {platform.python_version()} |",
        "",
        "## Accuracy (all scorable images)",
        "",
        "| Pipeline | OCR | KIE | Seller EM | Address EM | Timestamp EM (date only) | "
        "Total EM | Overall EM | Overall F1 | Address CER | E2E P50 / P95 ms | USD / 1k receipts |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in names:
        r = results[n]
        m = r["subsets"]["all"]
        lat = r.get("latency", {}).get("total")
        lat_s = f"{ms(statistics.median(lat))} / {ms(pct(lat, 0.95))}" if lat else "–"
        cost = f"{r['llm']['usd_per_1k']:.2f}" if r.get("llm") else "0"
        lines.append(
            f"| **{n}** | `{r['ocr_version']}` | `{r['kie']}` | {fmt(m['seller']['em'])} | "
            f"{fmt(m['seller_address']['em'])} | {fmt(m['timestamp']['em'])} "
            f"({fmt(m['timestamp']['date_em'])}) | {fmt(m['total_cost']['em'])} | "
            f"**{fmt(m['overall']['em'])}** | {fmt(m['overall']['f1'])} | "
            f"{fmt(m['seller_address'].get('cer', float('nan')))} | {lat_s} | {cost} |"
        )
    m0 = results[names[0]]["subsets"]["all"]
    lines += [
        "",
        "Scorable images per field: "
        + ", ".join(f"{f} {m0[f]['n']}" for f in FIELD_NAMES)
        + ". EM = exact match of the normalized value (null = null counts); F1 over non-null "
        "values; CER on case/punctuation-folded text. A failed call counts as an empty "
        "prediction.",
        "",
        "Field precision / recall / F1 (non-null values):",
        "",
        "| Pipeline | " + " | ".join(FIELD_NAMES) + " |",
        "|---|---|---|---|---|",
    ]
    for n in names:
        m = results[n]["subsets"]["all"]
        cells = [
            f"{fmt(m[f]['precision'])} / {fmt(m[f]['recall'])} / {fmt(m[f]['f1'])}"
            for f in FIELD_NAMES
        ]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")

    sub_names = list(results[names[0]]["subsets"])
    lines += [
        "",
        "## Overall EM by subset",
        "",
        "Subsets: *clean annotations* = no adapter flag other than a missing field or a "
        "date-only timestamp; *store seen in train* = the labelled seller name (accent-free) "
        "occurs in the train labels; *quality* = official `anno_image_quality`; *rotated* = "
        "the orientation classifier turned the photo. Small subsets (n < 20) are noisy.",
        "",
        "| Subset | Images | " + " | ".join(names) + " |",
        "|---|---:|" + "---:|" * len(names),
    ]
    for s in sub_names:
        k = results[names[0]]["subset_sizes"][s]
        cells = [fmt(results[n]["subsets"][s]["overall"]["em"]) for n in names]
        lines.append(f"| {s} | {k} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Per-field EM, store seen vs not seen in train:",
        "",
        "| Pipeline | Subset | Seller | Address | Timestamp | Total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for n in names:
        for s in ("store seen in train", "store not seen in train"):
            m = results[n]["subsets"][s]
            lines.append(
                f"| {n} | {s} | " + " | ".join(fmt(m[f]["em"]) for f in FIELD_NAMES) + " |"
            )

    if len(names) > 1:
        lines += [
            "",
            f"## Paired comparison against `{names[0]}`",
            "",
            "Overall-EM difference on the same images; 95 % interval from 2,000 paired "
            "bootstrap resamples of images (seed 2026).",
            "",
            "| Pipeline | Δ overall EM | 95 % interval | P(Δ ≤ 0) |",
            "|---|---:|---:|---:|",
        ]
        for n in names[1:]:
            b = results[n]["vs_first"]
            lines.append(
                f"| {n} | {b['diff']:+.1%} | [{b['low']:+.1%}, {b['high']:+.1%}] | "
                f"{b['p_b_not_better']:.3f} |"
            )

    timed = [n for n in names if results[n].get("latency")]
    if timed:
        k = len(results[timed[0]]["latency"]["total"])
        lines += [
            "",
            f"## Latency (live `ReceiptPipeline`, first {k} images, models warm)",
            "",
            "P50 / P95 in ms per stage; pipelines run one after another on each image "
            "(interleaved, one request at a time). *load* = read + decode the file; *ocr* includes "
            "orientation, detection and recognition; *kie* for an LLM = the API call through "
            "the proxy (original call time when the response came from the cache); "
            "*quality* = advisory image-quality score.",
            "",
            "| Pipeline | load | ocr | kie | quality | **total** |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for n in timed:
            t = results[n]["latency"]
            cells = [
                f"{ms(statistics.median(t[s]))} / {ms(pct(t[s], 0.95))}"
                for s in ("load", "ocr", "kie", "quality", "total")
            ]
            lines.append(f"| {n} | " + " | ".join(cells) + " |")

    llm = [n for n in names if results[n].get("llm")]
    if llm:
        lines += [
            "",
            "## LLM usage",
            "",
            "Tokens as reported by the API (the proxy adds ~250 prompt tokens and reports no "
            "thinking tokens). Cost at Google's list price for Gemini 2.5 Flash "
            f"(USD {LLM_PRICE_USD_PER_M['input']} / {LLM_PRICE_USD_PER_M['output']} per 1M "
            "input / output tokens) — an order of magnitude, not our bill.",
            "",
            "| Pipeline | Calls | New calls this run | Input tok / receipt | "
            "Output tok / receipt | Call P50 / P95 ms | Failed | USD / 1k receipts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for n in llm:
            u = results[n]["llm"]
            lines.append(
                f"| {n} | {u['calls']} | {u['calls'] - u['cached_calls']} | "
                f"{u['input_tokens'] / u['calls']:.0f} | {u['output_tokens'] / u['calls']:.0f} | "
                f"{u['latency_ms_p50']:.0f} / {u['latency_ms_p95']:.0f} | "
                f"{results[n]['failures']} | {u['usd_per_1k']:.2f} |"
            )

    lines += ["", "## Versions", ""]
    for n in names:
        r = results[n]
        lines.append(
            f"- **{n}** ({PIPELINES[n].note}): OCR run `{r['ocr_run']}` = `{r['ocr_version']}`; "
            f"KIE `{r['kie']}` (factory name `{PIPELINES[n].kie}`)."
        )
    lines += [
        "",
        "Not comparable with the RIVF 2021 leaderboard (different split and protocol).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> int:
    from evaluation.kie_eval import run as run_kie
    from evaluation.ocr_eval import load_results

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["smoke", "dev", "test"])
    ap.add_argument("--model", nargs="+", default=["baseline"], choices=list(PIPELINES))
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    ap.add_argument("--latency-images", type=int, default=0)
    ap.add_argument("--report", help="report name (default benchmark_<split>)")
    ap.add_argument("--final", action="store_true", help="required for --split test")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("test is run once per frozen pipeline: pass --final deliberately")

    records = load_split(args.split)
    manifest = json.loads(Path(f"data/splits/{args.split}.json").read_text(encoding="utf-8"))
    results: dict[str, dict] = {}
    for name in args.model:
        p = PIPELINES[name]
        ocr_run = ensure_ocr(args.split, p.ocr, args.device, records)
        ocr = load_results(args.split, ocr_run)
        r = run_kie(args.split, ocr_run, p.kie, args.workers)
        groups = subsets(records, args.split, ocr)
        meta = json.loads((OCR_ROOT / ocr_run / args.split / "_meta.json").read_text())
        res = {
            "ocr_run": ocr_run,
            "ocr_version": meta.get("version", p.ocr),
            "kie": r["kie"],
            "failures": len(r["failures"]),
            "subset_sizes": {s: len(g) for s, g in groups.items()},
            "subsets": {s: score(g, r["preds"])["metrics"] for s, g in groups.items()},
            "ocr_ms": [ocr[x.img_id].latency_ms for x in records],
            "kie_ms": list(r["latency_ms"].values()),
            "_preds": r["preds"],
        }
        if r.get("llm"):
            u = r["llm"]
            usd = (
                u["input_tokens"] * LLM_PRICE_USD_PER_M["input"]
                + u["output_tokens"] * LLM_PRICE_USD_PER_M["output"]
            ) / 1e6
            res["llm"] = u | {"usd_per_1k": usd / u["calls"] * 1000}
        results[name] = res
        print(f"{name}: overall EM {res['subsets']['all']['overall']['em']:.3f}", flush=True)

    first = args.model[0]
    for name in args.model[1:]:
        results[name]["vs_first"] = paired_bootstrap(
            records, results[first]["_preds"], results[name]["_preds"]
        )

    if args.latency_images:
        timings = time_pipelines(args.model, args.device, records[: args.latency_images])
        for name in args.model:
            results[name]["latency"] = timings[name]

    ctx = {
        "split": args.split,
        "split_version": manifest["version"],
        "images": len(records),
        "git": git_revision(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "command": "python -m evaluation.run_benchmark " + " ".join(sys.argv[1:]),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = REPORTS / f"{args.report or 'benchmark_' + args.split}.md"
    write_report(report, ctx, results)
    dump = {n: {k: v for k, v in r.items() if k != "_preds"} for n, r in results.items()}
    (OUT / f"{report.stem}.json").write_text(
        json.dumps({"context": ctx, "results": dump}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
