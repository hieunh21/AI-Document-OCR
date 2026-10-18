# Phase 2 — OCR Baseline & Image Quality

- Start date: 2026-10-18
- End date: 2026-10-18
- Status: Done

## Objectives
- Nhận diện dòng chữ + toạ độ trên ảnh gốc; đo chất lượng OCR trên dev; đánh giá chất lượng ảnh so với điểm IQA chính thức.

## Work completed
- [x] `ai/preprocessing/image.py`: đọc ảnh (EXIF), kiểm tra ảnh lỗi / quá nhỏ / tỉ lệ bất thường (`ImageError` có `code`), xoay + thu nhỏ với `Transform` để **quy đổi toạ độ về ảnh gốc** (test cho cả 4 góc xoay và resize).
- [x] `ai/preprocessing/orientation.py`: phân loại hướng trang 0/90/180/270 (`PP-LCNet_x1_0_doc_ori`), ngưỡng 0.8.
- [x] `ai/ocr/paddle.py`: `PaddleOCRProvider` (PP-OCRv6 tiny/small/medium, PP-OCRv5 latin), span sắp theo reading order, polygon + bbox + confidence.
- [x] `ai/ocr/hybrid.py`: detection PaddleOCR + recognizer thay thế được (VietOCR seq2seq/transformer, hoặc PP-OCR rec); crop chỉnh phối cảnh; batch VietOCR tự viết (nhanh gấp 2 lần bản gốc).
- [x] `ai/ocr/factory.py`: preset theo tên, hậu tố `+orient`; `DEFAULT_PRESET`, `ACCURATE_PRESET`, `CPU_PRESET`.
- [x] `evaluation/ocr_eval.py`: batch inference có cache, log thời gian, ghi lỗi; metric trên vùng KIE (detected, CER ×3 dạng, line exact, **đọc đúng số tiền**, **đọc đúng ngày giờ**); xuất `_errors.csv` các vùng sai nhất để phân tích.
- [x] `ai/quality/assess.py` + `evaluation/iqa_eval.py`: dự đoán điểm IQA chính thức (ridge, fit trên train), so với baseline; cảnh báo `low_image_quality` / `no_text_detected` đưa vào `ReceiptPipeline` — **không chặn**.
- [x] `scripts/render_ocr_overlay.py`: vẽ box OCR + vùng nhãn lên ảnh gốc để kiểm tra bằng mắt.
- [x] `docs/setup_gpu.md`: cài Paddle-GPU + PyTorch + VietOCR chung một môi trường.

## Kết quả (dev, 174 ảnh, 975 vùng — `evaluation/reports/ocr_dev.md`)

| OCR | CER folded | CER no-accent | Line exact | Amount read | Date-time read | P50 latency |
|---|---:|---:|---:|---:|---:|---:|
| PP-OCRv6 medium | 12.8% | 6.4% | 27.6% | 95.5% | 85.6% | — |
| PP-OCRv6 medium + orient | 11.0% | 4.4% | 28.0% | 95.5% | **87.5%** | 749 ms |
| det v6 + VietOCR seq2seq + orient | 7.5% | 5.3% | **39.9%** | 95.5% | 81.2% | **678 ms** |
| det v6 + VietOCR transformer + orient | **5.2%** | **3.1%** | 35.3% | 94.9% | 85.6% | 2982 ms |
| PP-OCRv6 tiny + orient (CPU) | — | — | — | — | — | 3735 ms |

IQA (`evaluation/reports/iqa_dev.md`): RMSE 0.126 vs 0.140 (baseline đoán trung bình), Spearman 0.43; cảnh báo ảnh kém (GT < 0.5): precision 39 %, recall 52 %.

## Technical decisions
- Decision: **GPU** (paddlepaddle-gpu cu126). CPU `medium` ≈ 80 s/ảnh → GPU ≈ 0.4–0.75 s/ảnh, cùng độ chính xác (CPU và GPU cho CER giống hệt trên smoke).
- Decision: **bật nhận diện hướng trang**. 5/174 ảnh dev bị xoay 90/270°: CER folded của chúng 0.7 → 0.1; tổng CER 14.9 → 13.0 %. Ngưỡng 0.8 loại 1 ca xoay nhầm (ảnh dọc trên khăn trải bàn hoa văn, score 0.64; ảnh xoay thật ≈ 0.92).
- Decision: **chưa chốt OCR cho app**. VietOCR transformer đọc dấu tốt hơn hẳn nhưng chậm 4× và hay "bịa" chữ số; PP-OCR đọc ngày giờ tốt nhất. Chọn ở Phase 4 theo **metric KIE** (mục tiêu thật), không theo CER. Mặc định tạm: `v6_medium+orient`.
- Decision: metric OCR đo trên **vùng KIE có nhãn** với CER theo substring (MC-OCR không có transcript toàn trang); thêm "amount/date-time read" vì đó là thứ KIE cần.
- Decision: VietOCR cài `--no-deps` (package pin `pillow==10.2`, `imgaug` phá môi trường); cuDNN của Paddle ghim về 9.10.2 trùng torch; luôn import torch trước paddle (`ai/ocr/_runtime.py`).
- Experiment (smoke, `ocr_smoke.md`): batch VietOCR theo đúng bề rộng (bản gốc) vs pad lên bề rộng chung tỉ lệ ≤1.5/≤1.2 → nhanh 2×, chênh CER ≤0.5 điểm trên 87 vùng (trong nhiễu); giữ 1.5.

## Evidence
- `pytest`: 110 passed. `ruff check .`: sạch. `python -m evaluation.splits --check`: reproducible.
- Overlay kiểm tra bằng mắt: ảnh thẳng và ảnh xoay 270° — box OCR trùng vùng nhãn trên ảnh gốc.
- Batch dev/train `v6_medium`: 0 lỗi; mọi lần chạy ghi `_meta.json` (version, thời gian load, lỗi).

## Bugs found & fixed
- **Parser ngày (Phase 1)**: từ chối ngày đứng ngay sau `:` (`"Ngày bán:17/08/2020"`). OCR thường bỏ dấu cách nên "date-time read" của PP-OCR bị báo 54 % thay vì 86 %. Sửa + test hồi quy; GT phục hồi thêm 5 timestamp → `mcocr2021-gt-v2` (changelog trong `data_contract.md`), split không đổi.
- Batch VietOCR lỗi pad 4D (`F.pad` cần 4 giá trị) — lỗi bị nuốt vào `_meta.json`; nay batch in `FAILED` ra màn hình.

## Known limitations
- IQA chỉ đủ để **gợi ý chụp lại**, không đủ để chặn ảnh (precision 39 %). Nhãn IQA là cảm nhận của người gán nhãn.
- Logo cách điệu (VinCommerce, co.opmart) đôi khi không được detect hoặc đọc sai — Phase 3 nên có luật/danh sách chuỗi cửa hàng.
- OCR hay gộp "Số GD: … Ngày: 12/08/2020-13:00" thành một dòng → KIE phải tách giá trị bên trong span.
- Latency đo trên laptop RTX 3050 4 GB, 30 ảnh; VRAM 4 GB đủ cho Paddle + VietOCR cùng lúc nhưng không còn nhiều dư.
- Model quality được fit trên output của `v6_medium` (không orient); đổi OCR thì nên fit lại (`python -m evaluation.iqa_eval fit --ocr-run …`).

## DoD verification
- [x] Module OCR trả text + boxes nhất quán (schema `OCRResult`, reading order, id ổn định).
- [x] Toạ độ box hiển thị đúng trên ảnh gốc (test 4 góc xoay + overlay thật).
- [x] Batch inference trên dev, ghi thời gian xử lý (P50/P95, đo sạch riêng).
- [x] Chất lượng ảnh được đánh giá so với nhãn IQA chính thức; chỉ cảnh báo, không chặn.

## Next phase
- Phase 3: KIE rule-based trên OCR spans (có thể chạy trên cả output PP-OCR và VietOCR đã cache), parser tách ngày/tiền bên trong span, evidence trỏ về span id; candidate thứ hai (OCR + LLM hoặc kết hợp PP-OCR cho số + VietOCR cho chữ).
