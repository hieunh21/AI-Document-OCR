"""Train the learned span classifier for KIE (``ai.kie.learned``) on the **train** split.

    python -m evaluation.train_line_classifier [--ocr-runs RUN ...] [--folds 5]

1. **Data.** OCR spans of every train image, from cached OCR runs:
   - the fine-tuned recognizer (deployed);
   - the stock one, noisier text.

   The fine-tuned recognizer was trained on these same images, so its train text is
   cleaner than it will be on new receipts. The stock run keeps the model used to OCR
   noise. A span's class is the annotated region covering at least half of its box
   (``span_labels``); dev and test are never read.
2. **Out-of-fold gazetteer.** Images are split into folds. The gazetteer features of a
   fold come from a gazetteer built on the *other* folds, so a store seen in only one
   image looks unknown, as a new store will in production.
3. **Out-of-fold predictions.** LightGBM is trained on the other folds and predicts each
   fold. These predictions:
   - pick the per-field thresholds and the confidence level that sends a field to
     review;
   - give a train estimate against the rule extractor, which runs with the same
     out-of-fold gazetteer.
4. **Final model.** It is trained on all folds and saved to
   ``configs/kie_line_model.txt``, with features, thresholds and the out-of-fold report
   in ``configs/kie_line_model.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from ai.kie.base import upright_spans
from ai.kie.gazetteer import Gazetteer
from ai.kie.learned import META_PATH, MODEL_PATH, Thresholds, assemble
from ai.kie.line_features import CLASSES, FEATURE_NAMES, order_spans, span_features, span_labels
from ai.kie.rule_based import RuleBasedKIE
from ai.normalization.address import AdminUnits
from ai.schemas import FIELD_NAMES, OCRSpan
from evaluation.build_gazetteer import build as build_gazetteer
from evaluation.dataset_adapter import GT_VERSION, McocrRecord, load_split
from evaluation.metrics import image_outcomes
from evaluation.ocr_eval import load_results

VERSION = "lgbm-v1"
SEED = 2026
DEFAULT_RUNS = ["v6_medium_vi3+orient@gpu0", "v6_medium+orient@gpu0"]
PARAMS = {
    "objective": "multiclass",
    "num_class": len(CLASSES),
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "num_threads": 4,
}
GRID = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
CONFIDENT_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


@dataclass
class Item:
    rec: McocrRecord
    run: str
    fold: int
    spans: list[OCRSpan]
    x: np.ndarray
    y: np.ndarray
    proba: np.ndarray | None = None


def prepare(runs: list[str], n_folds: int) -> tuple[list[Item], list[Gazetteer], AdminUnits]:
    records = load_split("train")
    ids = sorted(r.img_id for r in records)
    random.Random(SEED).shuffle(ids)
    fold_of = {img: i % n_folds for i, img in enumerate(ids)}
    gazetteers = [
        Gazetteer.from_dict(build_gazetteer(records=[r for r in records if fold_of[r.img_id] != k]))
        for k in range(n_folds)
    ]
    admin = AdminUnits.load()
    items = []
    for run in runs:
        results = load_results("train", run)
        missing = [r.img_id for r in records if r.img_id not in results]
        if missing:
            raise SystemExit(f"{run}: {len(missing)} train images without OCR; run ocr_eval first")
        t0 = time.perf_counter()
        for rec in records:
            ocr = results[rec.img_id]
            spans = order_spans(upright_spans(ocr))
            original = {s.id: s for s in ocr.spans}  # labels: same frame as the regions
            y = span_labels([original[s.id] for s in spans], rec.regions)
            k = fold_of[rec.img_id]
            items.append(Item(rec, run, k, spans, span_features(spans, gazetteers[k], admin), y))
        print(f"{run}: {len(records)} images, features in {time.perf_counter() - t0:.0f} s")
    return items, gazetteers, admin


def stack(items: list[Item]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([it.x for it in items]), np.concatenate([it.y for it in items])


def fit(train: list[Item], valid: list[Item] | None, rounds: int):
    import lightgbm as lgb

    x, y = stack(train)
    data = lgb.Dataset(x, y, feature_name=FEATURE_NAMES, free_raw_data=False)
    if valid is None:
        return lgb.train(PARAMS, data, num_boost_round=rounds)
    vx, vy = stack(valid)
    return lgb.train(
        PARAMS,
        data,
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(vx, vy, reference=data)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def field_em(items: list[Item], th: Thresholds, gazetteers, admin) -> dict[str, float]:
    hits, n = Counter(), Counter()
    for it in items:
        ex = assemble(it.spans, it.proba, th, gazetteers[it.fold], admin, "oof")
        for f, ok in image_outcomes(it.rec, ex).items():
            hits[f] += ok
            n[f] += 1
    out = {f: hits[f] / n[f] for f in FIELD_NAMES}
    out["overall"] = sum(hits.values()) / sum(n.values())
    return out


def rules_em(items: list[Item], gazetteers, admin) -> dict[str, float]:
    hits, n = Counter(), Counter()
    rules = {
        k: RuleBasedKIE(gazetteer=g, seller_fallback=False, admin=admin, admin_hint=True)
        for k, g in enumerate(gazetteers)
    }
    for it in items:
        ex = rules[it.fold].extract(None, it.spans)
        for f, ok in image_outcomes(it.rec, ex).items():
            hits[f] += ok
            n[f] += 1
    out = {f: hits[f] / n[f] for f in FIELD_NAMES}
    out["overall"] = sum(hits.values()) / sum(n.values())
    return out


def ready_table(items: list[Item], th: Thresholds, gazetteers, admin) -> list[dict]:
    """Share of receipts without warnings and how many of those are fully correct."""
    rows = []
    for c in CONFIDENT_GRID:
        t = Thresholds(**{**asdict(th), "confident": c})
        ready = ready_ok = 0
        for it in items:
            ex = assemble(it.spans, it.proba, t, gazetteers[it.fold], admin, "oof")
            if not ex.warnings:
                ready += 1
                ready_ok += all(image_outcomes(it.rec, ex).values())
        rows.append(
            {
                "confident": c,
                "ready_share": ready / len(items),
                "ready_precision": ready_ok / ready if ready else None,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--target-precision", type=float, default=0.9)
    args = ap.parse_args()

    items, gazetteers, admin = prepare(args.ocr_runs, args.folds)
    x, y = stack(items)
    print(f"{len(items)} receipts x OCR runs, {len(y)} spans; classes", Counter(y.tolist()))

    best_iters = []
    for k in range(args.folds):
        train = [it for it in items if it.fold != k]
        valid = [it for it in items if it.fold == k]
        booster = fit(train, valid, rounds=2000)
        best_iters.append(booster.best_iteration)
        for it in valid:
            it.proba = booster.predict(it.x, num_iteration=booster.best_iteration)
        print(f"fold {k}: best iteration {booster.best_iteration}")

    # Per-field thresholds, one field at a time on the out-of-fold predictions.
    th = Thresholds()
    for f in FIELD_NAMES:
        scores = {
            t: field_em(items, Thresholds(**{**asdict(th), f: t}), gazetteers, admin)[f]
            for t in GRID
        }
        th = Thresholds(**{**asdict(th), f: max(scores, key=lambda t: (scores[t], -abs(t - 0.5)))})
        print(f"threshold {f}: " + " ".join(f"{t}:{v:.3f}" for t, v in scores.items()))

    # Confidence that keeps a field out of review: the lowest one whose "ready" receipts
    # reach the target precision (deployed OCR run only). When none does, the lowest:
    # a stricter level would send more receipts to review without making "ready" more
    # reliable (the class probability says where a field is, not whether its text is right).
    deployed = [it for it in items if it.run == args.ocr_runs[0]]
    table = ready_table(deployed, th, gazetteers, admin)
    ok = [
        r for r in table if r["ready_precision"] and r["ready_precision"] >= args.target_precision
    ]
    th = Thresholds(**{**asdict(th), "confident": ok[0]["confident"] if ok else CONFIDENT_GRID[0]})

    report = {}
    for run in args.ocr_runs:
        sub = [it for it in items if it.run == run]
        report[run] = {
            "lines": field_em(sub, th, gazetteers, admin),
            "rules": rules_em(sub, gazetteers, admin),
        }
    rounds = int(np.mean(best_iters))
    final = fit(items, None, rounds)
    final.save_model(str(MODEL_PATH))
    importance = sorted(
        zip(FEATURE_NAMES, final.feature_importance("gain").tolist(), strict=True),
        key=lambda kv: -kv[1],
    )
    meta = {
        "version": VERSION,
        "classes": list(CLASSES),
        "features": FEATURE_NAMES,
        "thresholds": asdict(th),
        "params": PARAMS,
        "num_boost_round": rounds,
        "trained_on": {
            "split": "train",
            "ground_truth": GT_VERSION,
            "images": len({it.rec.img_id for it in items}),
            "ocr_runs": args.ocr_runs,
            "spans": int(len(y)),
            "class_counts": {CLASSES[c]: int(n) for c, n in sorted(Counter(y.tolist()).items())},
        },
        "out_of_fold": {
            "folds": args.folds,
            "gazetteer": "built from the other folds",
            "field_em": report,
            "ready_vs_confident": table,
        },
        "top_features": [[n, round(g, 1)] for n, g in importance[:20]],
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("\nout-of-fold exact match (train):")
    for run, r in report.items():
        for name, em in r.items():
            cells = " ".join(f"{f}={em[f]:.3f}" for f in [*FIELD_NAMES, "overall"])
            print(f"  {run:28s} {name:6s} {cells}")
    print("ready (no warning) vs confidence:", json.dumps(table))
    print("thresholds:", asdict(th))
    print(f"saved {MODEL_PATH} and {META_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
