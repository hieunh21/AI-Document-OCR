# VietDocAI — Vietnamese Receipt Intelligence

Read a photo of a Vietnamese retail receipt and extract **seller, seller address,
timestamp and total cost**, with the evidence region for each value, a human review
step, and a reproducible offline benchmark on **MC-OCR 2021**.

> Status: **Phase 0–3 done** (data contracts, MC-OCR adapter and splits, OCR + orientation +
> image quality, rule-based and LLM key-information extraction with dev benchmarks).
> The app still serves mock data (Phase 5).
> Roadmap: `VietDocAI_MC_OCR_Phase_Roadmap.md`; progress: `docs/progress/`.

## Layout

```text
ai/            AI core: schemas (data contracts), OCR/KIE interfaces, normalization, pipeline, mocks
backend/       FastAPI app (GET /health, /api/v1/documents — mock store until Phase 5)
frontend/      React + Ant Design (Upload, History, Review, Dashboard)
evaluation/    MC-OCR adapter, dedup, splits, EDA, data contract, (Phase 4) metrics & reports
data/splits/   train/dev/test/smoke manifests (ids + SHA-256 only, no images/labels)
docs/          requirements, EDA, data usage rules, manual annotation check, schemas, progress
scripts/       environment checks and utilities
tests/         unit tests (+ dataset tests that run only where the data is present)
```

## Quick start

Requirements: Python 3.11–3.12, Node ≥ 20.19 (developed on Node 24 LTS), optional NVIDIA GPU.

```bash
# Python
uv venv .venv --python 3.12              # or: python -m venv .venv
uv pip install --python .venv -e ".[dev]"  # core + tests
# uv pip install --python .venv -e ".[ocr]"  # PaddleOCR on CPU; for GPU + VietOCR see docs/setup_gpu.md

# Backend  (run from the repository root)
.venv/Scripts/python -m uvicorn app.main:app --app-dir backend --reload   # Windows
# .venv/bin/python -m uvicorn app.main:app --app-dir backend --reload     # Linux/macOS
curl http://127.0.0.1:8000/health

# Frontend (proxies /api and /health to :8000)
cd frontend && npm install && npm run dev   # http://localhost:5173

# Tests & lint
.venv/Scripts/python -m pytest
.venv/Scripts/ruff check .
```

## Dataset (not included)

Download the public MC-OCR 2021 copy from Kaggle (it is not stored in this repo, see
`docs/data_usage.md`) and place it as:

```text
data/raw/mcocr/
├── mcocr_train_df.csv   # 1,155 labelled images
├── train_images/
└── val_images/          # 391 unlabelled images (not used for metrics)
```

or set `VIETDOCAI_MCOCR_DIR`. Then:

```bash
python -m evaluation.dataset_adapter --show 1   # inspect normalized ground truth + data-quality flags
python -m evaluation.splits --check             # verify the committed manifests are reproduced
python -m evaluation.eda                        # regenerate docs/dataset_eda.md
```

## Data contract in one paragraph

Ground truth per image is an `ai.schemas.ReceiptExtraction` (the type models return):
`seller`/`seller_address` text (NFC, regions joined in reading order), `timestamp` ISO 8601
(date-only when no time is printed), `total_cost` integer VND, `null` when absent.
Annotations that exist but cannot be normalized are *unscorable*, not `null`. Splits are
70/15/15 with duplicate groups (SHA-256, 1024-bit dHash, same transaction) kept together
and stratified by seller. Details: `evaluation/data_contract.md`; numbers: `docs/dataset_eda.md`.

## OCR (Phase 2)

```bash
# OCR a split with a preset (outputs cached under outputs/ocr/, git-ignored)
python -m evaluation.ocr_eval run --split dev --preset "v6_medium+orient" --device gpu:0
python -m evaluation.ocr_eval report --split dev --runs "v6_medium+orient@gpu0"
python -m evaluation.iqa_eval fit --ocr-run v6_medium@gpu0     # image-quality model
python scripts/render_ocr_overlay.py --run "v6_medium+orient@gpu0" --split dev --n 4
```

Presets (`ai/ocr/factory.py`): PP-OCRv6 `v6_tiny|v6_small|v6_medium`, PP-OCRv5 latin, and
PP-OCR detection + VietOCR recognition (`v6med_det+vietocr_seq2seq|transformer`); append
`+orient` for automatic page orientation. Dev results: `evaluation/reports/ocr_dev.md`,
image quality: `evaluation/reports/iqa_dev.md`.

## Key information extraction (Phase 3)

```bash
python -m evaluation.build_gazetteer                       # store list from the train split
python -m evaluation.kie_eval --split dev --ocr-run "v6_medium+orient@gpu0"     --kie rules+gaz-nofb llm:gemini+gaz --workers 4         # LLM needs GEMINI_* in .env
python scripts/extract_receipt.py path/to/receipt.jpg --kie rules+gaz-nofb
```

KIE options (`ai/kie/factory.py`): rule baseline (`rules`, `rules+gaz-nofb`) and Gemini over
OCR text (`llm:gemini`, `llm:gemini+gaz`). The LLM only sees OCR text with span ids and must
cite them; evidence boxes always come from OCR. Dev numbers: `evaluation/reports/kie_dev.md`.

## Benchmark

Dev results so far are in `evaluation/reports/` (OCR, image quality, KIE). The full benchmark
with per-stratum results and the single held-out test run comes in Phase 4.

## References

- MC-OCR 2021 challenge and dataset — <https://www.rivf2021-mc-ocr.vietnlp.com/>
- PaddleOCR — <https://www.paddleocr.ai/>
