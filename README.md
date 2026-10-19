# VietDocAI — Vietnamese Receipt Intelligence

Read a photo of a Vietnamese retail receipt and extract **seller, seller address,
timestamp and total cost**, with the evidence region for each value, a human review
step, and a reproducible offline benchmark on **MC-OCR 2021**.

> Status: **Phase 0–6 done**.
> - Phases 0–4: data contracts, MC-OCR adapter and splits, OCR with orientation and image
>   quality, rule-based and LLM key-information extraction, benchmark with error analysis,
>   and a single held-out test run.
> - Phase 5: FastAPI + PostgreSQL service running the real pipeline.
> - Phase 6: React app covering upload, review with evidence boxes, corrections, approval,
>   history and export.
> Roadmap: `VietDocAI_MC_OCR_Phase_Roadmap.md`; progress: `docs/progress/`.

## Layout

```text
ai/            AI core: schemas (data contracts), OCR/KIE interfaces, normalization, pipeline, mocks
backend/       FastAPI + SQLAlchemy/PostgreSQL service, Alembic migrations, API tests
frontend/      React + Ant Design: Dashboard, Upload (+ demo receipts), Review (evidence, edit, approve), History (filter, export)
evaluation/    MC-OCR adapter, dedup, splits, EDA, data contract, metrics, benchmark, error analysis
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

# Database + backend  (run from the repository root; copy .env.example to .env first)
docker compose up -d db                                  # PostgreSQL 16 on :5432
.venv/Scripts/alembic -c backend/alembic.ini upgrade head  # create / migrate tables
.venv/Scripts/python -m uvicorn app.main:app --app-dir backend --reload   # Windows
# .venv/bin/python -m uvicorn app.main:app --app-dir backend --reload     # Linux/macOS
curl http://127.0.0.1:8000/health                        # API docs: http://127.0.0.1:8000/docs

# Frontend (proxies /api and /health to :8000)
cd frontend && npm install && npm run dev   # http://localhost:5173

# Tests & lint
.venv/Scripts/python -m pytest
.venv/Scripts/ruff check .
```

## Web app (Phase 6)

Open http://localhost:5173 with the database, the backend and `npm run dev` running.

1. **Upload:** drop a receipt photo, or pick one of the demo receipts
   (`scripts/make_demo_receipts.py`, invented content).
2. **Review:** the four fields sit next to the image.
   - Selecting a field highlights where its value was read.
   - Each value is tagged as coming from the AI or from a person.
   - Warnings show missing or implausible values.
   - **Save correction** stores the edits; **Approve** locks the record.
3. **History:** filter by status, seller or dates, and export the selection as CSV or
   JSON, e.g. approved records only.
4. **Dashboard:** counts per status. The money total covers approved receipts only.

## Backend API (Phase 5)

The backend runs the frozen pipeline from `configs/pipeline.json` (`VIETDOCAI_PIPELINE`,
default `default`). Results are stored in PostgreSQL; image files stay on disk under
`storage/`. Interactive docs are at `/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/documents[?process=true]` | Upload a JPG/PNG/WebP. Checks size, type, decodability and dimensions, with a per-client rate limit. The same bytes return the existing document. |
| `POST /api/v1/documents/{id}/process` | Run the pipeline synchronously (about 0.5 s on GPU once the models are loaded). |
| `GET /api/v1/documents/{id}` | Status, current fields with their source (`ai` / `human`), warnings, AI extraction, OCR spans and evidence boxes, corrections, events. |
| `GET /api/v1/documents/{id}/image` | The uploaded image. |
| `PATCH /api/v1/documents/{id}/fields` | Correct fields (`"510.760"` and `"30/04/2020 10:41"` are normalized); old and new values are logged. |
| `POST /api/v1/documents/{id}/approve` | Human approval; the document is then locked. |
| `GET /api/v1/documents` | List with paging and filters: status, seller, upload date, receipt date. |
| `GET /api/v1/documents/{id}/export`, `GET /api/v1/documents/export?status=approved` | JSON or CSV (UTF-8 with BOM for Excel). |

- **Tables:** `documents`, `extractions` (every pipeline run), `field_corrections`,
  `document_events` (audit trail).
- **Errors:** every error returns `{"detail", "code", "request_id"}`.
- **Tests:** `backend/tests` covers the main flow and the error paths. They run on SQLite
  with the mock pipeline, so they need neither PostgreSQL nor a GPU.

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

The stock PP-OCRv6 recognizer cannot write the 90 Vietnamese letters with a tone mark
(ả, ẩ, ộ, …): its dictionary lacks them.

`docs/finetune_rec.md` fine-tunes it with a 237-character Vietnamese dictionary:
- `scripts/export_rec_dataset.py` exports the train/dev line crops;
- `scripts/remap_rec_head.py` builds the new output layers from the stock ones;
- `notebooks/finetune_ppocrv6_rec_vi.ipynb` trains on Colab.

On dev, folded OCR CER drops from 11.0 % to 3.2 % (`evaluation/reports/ocr_vi3_dev.md`).
The model has to be trained at the stock 320 px width: the backbone pools training
features to a fixed 40 time steps.

The result is preset `v6_medium_vi3`, the OCR of `pipeline-v3`. Its weights are not in
git: place the exported model in `models/ppocrv6_medium_rec_vi3/`.

## Key information extraction (Phase 3)

```bash
python -m evaluation.build_gazetteer                       # store list from the train split
python scripts/build_admin_units.py                        # official place names (GSO directory)
python -m evaluation.kie_eval --split dev --ocr-run "v6_medium+orient@gpu0"     --kie rules+gaz-nofb llm:gemini+gaz --workers 4         # LLM needs GEMINI_* in .env
python scripts/extract_receipt.py path/to/receipt.jpg [--pipeline default|llm]
```

KIE options (`ai/kie/factory.py`): rule baseline (`rules`, `rules+gaz-nofb`) and Gemini over
OCR text (`llm:gemini`, `llm:gemini+gaz`). The LLM only sees OCR text with span ids and must
cite them; evidence boxes always come from OCR. Dev numbers: `evaluation/reports/kie_dev.md`.

## Benchmark

```bash
python -m evaluation.run_benchmark --split dev --model rules baseline llm     --latency-images 30 --report model_comparison   # OCR -> KIE -> metrics -> report
python -m evaluation.error_analysis --split dev --model rules llm
```

Held-out **test** split (173 images), run once after the pipeline was frozen
(`configs/pipeline.json`). EM = exact match of the normalized field value
(`evaluation/data_contract.md` §5). Latency is end to end (file → JSON) on an RTX 3050
Laptop GPU.

| Pipeline | Seller EM | Address EM | Timestamp EM | Total EM | Overall EM | P50 / P95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Rules, no store list (baseline) | 50.9 % | 4.0 % | 87.9 % | 89.4 % | 57.1 % | 0.5 / 0.7 s |
| **Rules + store gazetteer (pipeline-v1 default, offline)** | 72.8 % | 52.6 % | 87.9 % | 89.4 % | **75.3 %** | 0.5 / 0.7 s |
| Gemini 2.5 Flash + gazetteer (optional) | 74.0 % | 53.8 % | 88.5 % | 92.5 % | **76.8 %** | 4.4 / 8.7 s |

- **OCR:** PP-OCRv6 medium with automatic orientation, for all three pipelines.
- **Gemini vs rules:** on test the gap is +1.5 points with a 95 % paired-bootstrap interval
  of [−0.6, +3.6]. On dev it was +3.7 [+1.3, +6.2].
- **Proxy check:** the LLM proxy's backend changed during the day (about 2.5× more output
  tokens, about 2× slower). Re-sending the same 174 dev requests gave +3.3 [+1.0, +5.6]
  over rules (`llm_backend_check_dev.md`). Accuracy held; latency and cost did not.
- **Rejected after dev comparison:**
  - VietOCR recognition: better OCR CER, but no KIE gain and about 3 s slower per image.
  - A lower detection threshold: no KIE gain either.
- **Error analysis (`docs/error_analysis.md`):** in a manual review of 29 errors, about a
  third were label omissions or typos, so these numbers are lower bounds.
- **Weak spot:** seller and address for stores never seen in training.
- **Rules v2 (after the test run, dev only):** official Vietnamese place names (GSO
  directory) find address lines and repair their accents. Dev overall EM rises from 74.3 %
  to 75.6 %, +1.2 [+0.4, +2.2] over v1 (`evaluation/reports/rules_v2_dev.md`). It is now
  `pipeline-v2`. It has no test number, because the test split was used once, for v1.
- **Pipeline v3 (dev only), the current default:** the same pipelines on the fine-tuned
  Vietnamese recognizer (`configs/pipeline.json`).
  - Folded OCR CER: 11.0 % → 3.2 %.
  - Overall EM, rules: 75.6 → 76.2 %.
  - Overall EM, Gemini: 77.5 → 78.1 % on the same proxy backend.
  - Both KIE gains are within noise (`evaluation/reports/ocr_vi3_dev.md`).
  - What v3 adds is correct Vietnamese in the evidence and raw OCR text.
- **Pipeline v4 (dev only), the current default:** a LightGBM span classifier replaces
  the hand-written line rules (`ai/kie/learned.py`, trained on train with
  `python -m evaluation.train_line_classifier`).
  - Overall EM: 76.2 → **79.0 %** (+2.8 [+1.0, +4.7]).
  - Address: 52.3 → 60.9 %.
  - The free pipeline is now at least as accurate as Gemini on dev (78.1 %).
  - Report: `evaluation/reports/lines_dev.md`.

Full reports:

- `evaluation/reports/final_test.md` — test split.
- `evaluation/reports/model_comparison.md` — six pipelines on dev, plus breakdowns by
  subset and latency per stage.
- `evaluation/reports/baseline_dev.md` — the baseline alone.
- `evaluation/reports/llm_backend_check_dev.md` — dev LLM requests re-sent to the changed
  proxy backend.

The numbers are not comparable with the RIVF 2021 leaderboard, which used a different
split and protocol.

## References

- MC-OCR 2021 challenge and dataset — <https://www.rivf2021-mc-ocr.vietnlp.com/>
- PaddleOCR — <https://www.paddleocr.ai/>
