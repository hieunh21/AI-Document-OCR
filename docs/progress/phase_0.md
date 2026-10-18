# Phase 0 — Scope & Project Setup

- Start date: 2026-10-18
- End date: 2026-10-18
- Status: Done

## Objectives
- Khởi tạo cấu trúc repo, môi trường, schema dữ liệu, app skeleton chạy được với mock data.

## Work completed
- [x] Cây thư mục theo roadmap (`ai/`, `backend/`, `frontend/`, `evaluation/`, `data/`, `docs/`, `scripts/`, `tests/`, `configs/`). Repo root chính là project root (không lồng thêm `vietdocai/`).
- [x] Dataset chuyển từ root vào `data/raw/mcocr/` (git-ignored).
- [x] `pyproject.toml` (extras `ocr`, `dev`), `.venv` Python 3.12.5 qua `uv`, `.gitignore`, `.env.example`.
- [x] `ai/schemas.py`: `ReceiptExtraction`, `ReceiptFields` (cho phép `null`), `RawFields`, `Evidence`, `FieldEvidence`, `OCRSpan`, `OCRResult`, `ExtractionWarning`, `DocumentStatus` (6 trạng thái). JSON Schema xuất ra `docs/schemas/`.
- [x] Interface `OCRProvider.predict`, `KIEProvider.extract`, `ReceiptPipeline` (đo latency từng stage), mock providers.
- [x] FastAPI: `GET /health`, `GET /api/v1/documents[?status=]`, `GET /api/v1/documents/{id}` (mock store), middleware `X-Request-ID` + log latency, CORS.
- [x] React + Ant Design + Vite 8: Upload (placeholder), History (lọc status), Review (bảng field + click → highlight evidence trên SVG theo toạ độ ảnh gốc), Dashboard; badge trạng thái API.
- [x] `docs/requirements.md`: 3 user stories + bảng trạng thái.
- [x] Kiểm tra PaddleOCR trên máy (`scripts/check_paddleocr.py`).

## Technical decisions
- Decision: bbox nội bộ là `[x_min, y_min, x_max, y_max]` theo pixel ảnh gốc (sau EXIF transpose).
  - Alternatives: `[x, y, w, h]` như COCO/MC-OCR. Reason: dễ union/so sánh; adapter chuyển đổi một chỗ.
- Decision: `timestamp` chuẩn hoá là chuỗi ISO (date-only hợp lệ), `total_cost` là `int` VND.
  - Reason: receipt có thể chỉ in ngày; VND không có đơn vị lẻ.
- Decision: `ready` ≠ `approved` — `ready` chỉ nghĩa là AI không cảnh báo, chưa ai kiểm tra.
- Decision: frontend dùng Vite 8 + TypeScript 5.9 trên Node 24 LTS (ban đầu là Vite 4 / TS 5.4 vì máy có Node 16; đã nâng Node cùng ngày).

## Evidence
- `pytest`: 82 passed (gồm 5 API tests). `ruff check .`: sạch. `npm run build`: thành công.
- Chạy thật: `GET /health` → 200 `{"status":"ok","version":"0.1.0","env":"dev"}` kèm `X-Request-ID`; frontend dev server proxy `/api`, `/health` OK.
- PaddleOCR check (CPU, ảnh 768×1024):

  | Item | Value |
  |---|---|
  | paddlepaddle / paddleocr | 3.3.1 (CPU build) / 3.7.0 |
  | Models (lang=`vi`) | PP-OCRv6_medium_det + PP-OCRv6_medium_rec |
  | Lines detected | 45 |
  | Predict latency (warm, CPU, oneDNN off) | ~80 s |

## Known limitations
- **PaddleOCR + oneDNN lỗi trên Windows** (`ConvertPirAttribute2RuntimeAttribute not support`) → phải tắt `enable_mkldnn`, dẫn tới **~80 s/ảnh trên CPU**: không dùng được cho app. Phase 2 cần thử: bản GPU (`paddlepaddle-gpu`, RTX 3050 4 GB), model `mobile`, hoặc hạ phiên bản Paddle.
- **Mất dấu tiếng Việt** ở lần chạy đầu: "Chợ Sủi" → "Ch Sùi", "HÓA ĐƠN" → "HÓA DON". Cần so sánh recognizer khác (ví dụ VietOCR) ở Phase 2 — đo bằng CER trên dev, không đánh giá bằng mắt.
- Frontend chưa có ảnh thật trong Review (chỉ vẽ box trên nền trắng); bundle 1 MB (chưa code-split).

## DoD verification
- [x] `GET /health` trả 200; giao diện mở được.
- [x] JSON schema hợp lệ, cho phép mọi field là `null` (test `test_json_schema_allows_null_for_every_field`).
- [x] Repo không chứa token, mật khẩu, ảnh gốc hay dữ liệu có điều kiện cấp phép (`.gitignore` + `git status` đã kiểm tra).

## Next phase
- Phase 1 (làm cùng ngày, xem `phase_1.md`).
