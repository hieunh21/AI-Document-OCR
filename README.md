# VietDocAI — Vietnamese Receipt Intelligence

Read a photo of a Vietnamese retail receipt and extract **seller, seller address,
timestamp and total cost**, with the evidence region for each value, a human review
step, and a reproducible offline benchmark on **MC-OCR 2021**.

> Status: **Phase 0–1 done** (project skeleton, data contracts, MC-OCR adapter, EDA,
> leakage-controlled splits). OCR/KIE models come in Phase 2–3. The app currently
> serves mock data. Roadmap: `VietDocAI_MC_OCR_Phase_Roadmap.md`.

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
# uv pip install --python .venv -e ".[ocr]"  # PaddleOCR (large download), needed from Phase 2

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

## Benchmark

Not run yet (Phase 4). No accuracy numbers are claimed.

## References

- MC-OCR 2021 challenge and dataset — <https://www.rivf2021-mc-ocr.vietnlp.com/>
- PaddleOCR — <https://www.paddleocr.ai/>
