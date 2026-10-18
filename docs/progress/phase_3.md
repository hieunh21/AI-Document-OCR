# Phase 3 — Key Information Extraction (KIE)

- Start date: 2026-10-18
- End date: 2026-10-18
- Status: Done

## Objectives
- Biến output OCR thành JSON 4 trường có evidence; có ≥ 2 pipeline KIE chạy trên cùng input để benchmark.

## Work completed
**3A — Rule baseline** (`ai/kie/rule_based.py`)
- [x] Timestamp: parse mọi dòng trực quan; ưu tiên dòng có từ khoá ("ngày", "thời gian", "giờ") + có giờ; ghép "Ngày: …" với "Giờ: …" dòng dưới; loại ngày hạn dùng / voucher.
- [x] Total: từ khoá theo độ ưu tiên ("… phải thanh toán" > "tổng thanh toán" > "tổng tiền" > "tổng cộng" > "cộng tiền hàng"), so khớp mờ cho từ khoá OCR làm hỏng ("Tőngcng"), loại "tiền khách đưa / tiền thừa / chiết khấu / VAT / số lượng"; số tiền lấy theo hình học (span bên phải có tâm dọc gần nhất, rồi dòng dưới); lọc số điện thoại, mã thẻ "XXXX8722", MST, mã chữ-số, số âm, "VND/UND" đứng trước.
- [x] Seller / address: khối header (dừng ở tiêu đề, dòng ngày, bảng hàng hoặc dòng giá); địa chỉ = các dòng có dấu hiệu địa chỉ, bỏ dòng ĐT/MST/web.
- [x] Gazetteer (`ai/kie/gazetteer.py`, `configs/kie_gazetteer.json`): 29 seller + 196 địa chỉ cửa hàng xây **chỉ từ nhãn train** (`python -m evaluation.build_gazetteer`) — như danh mục nhà cung cấp của doanh nghiệp; so khớp mờ trên chữ bỏ dấu.

**3B — LLM candidate** (`ai/kie/llm.py`)
- [x] Gemini 2.5 Flash qua `google-genai` (base URL tuỳ chỉnh: shopaikey). Chỉ gửi **text OCR** gắn id span; model trả text + `span_ids`; code bỏ id không tồn tại, chuẩn hoá ngày/tiền bằng parser chung → evidence luôn là box OCR thật.
- [x] Cache phản hồi trên đĩa theo hash request (chạy lại miễn phí, tái lập được); ghi token + latency thật; chịu được proxy bỏ qua JSON mode (bóc ```json```).
- [x] `llm:gemini+gaz`: chuẩn hoá seller/address của LLM qua cùng gazetteer.

**3C — Chuẩn hoá & evidence**
- [x] `ReceiptExtraction` raw + normalized + evidence (span ids + bbox) cho mọi trường tìm được; `null` khi không thấy.
- [x] `ai/kie/validate.py`: cảnh báo (không xoá) ngày ngoài khoảng, tổng tiền bất thường; `missing_field`, `llm_unknown_span`, `unparsable_value`.
- [x] `extract_from_ocr`: KIE suy luận trên layout **đã dựng thẳng**, evidence trả về toạ độ ảnh gốc (ảnh xoay).
- [x] `evaluation/kie_eval.py` (metric theo data contract §5), `scripts/extract_receipt.py` (chạy cả pipeline cho 1 ảnh).

## Kết quả (dev, 174 ảnh — `evaluation/reports/kie_dev.md`; OCR `v6_medium+orient`)

| KIE | Seller EM | Address EM | Timestamp EM | Total EM | Overall EM | Address CER | Latency KIE |
|---|---:|---:|---:|---:|---:|---:|---:|
| rules | 50.6% | 4.0% | 87.4% | 90.1% | 57.1% | 39% | < 10 ms |
| rules + gazetteer (no fallback) | 74.7% | 45.4% | 87.4% | 90.1% | 74.0% | 27% | < 15 ms |
| Gemini 2.5 Flash | 70.1% | 31.0% | 88.6% | 93.2% | 70.1% | 22% | P50 1.6 s |
| **Gemini + gazetteer** | **75.3%** | **53.4%** | **88.6%** | **93.2%** | **77.2%** | **18%** | P50 1.6 s |

Gemini: ~1,409 token input + ~220 output / hoá đơn (số proxy báo), 0 lỗi gọi trên 180 request.

## Technical decisions
- Decision: luật phát triển **chỉ trên train**; dev chỉ dùng để chọn biến thể (bỏ seller fallback vì nó làm giảm EM khi có gazetteer (train: 71.6 % có fallback, 74.6 % không có)).
- Decision: gazetteer là thành phần riêng, đo ablation. Nó dùng tri thức cửa hàng từ train — hợp lệ vì test cùng phân bố cửa hàng; Phase 4 phải báo cáo theo stratum để thấy phần "cửa hàng lạ".
- Decision: LLM **không** được trả toạ độ; chỉ trích dẫn span id, code kiểm tra lại.
- Decision: proxy shopaikey bỏ qua `response_json_schema`, cộng ~250 token/request và không báo token thinking → không xác minh được model thật sau proxy; ghi rõ trong báo cáo, kết quả LLM gắn với "Gemini 2.5 Flash qua shopaikey".

## Bugs found & fixed
- KIE chạy trên toạ độ ảnh xoay → hỏng layout (5/174 dev). Thêm `upright_spans`/`restore_evidence`; total rules 87.6 → 90.1 %.
- Parser giờ: "09:03. 13/07/2020" (dấu chấm sau phút) và năm dính giờ "11/06/202021:07:25" — sửa + test hồi quy; GT không đổi.
- Regex viết qua heredoc mất `\b` (thành ký tự backspace) → lọc số tiền sai; viết lại bằng regex biên dịch sẵn, quét toàn repo không còn ký tự điều khiển.
- `.env` có chú thích cuối dòng làm hỏng `GEMINI_BASE_URL` → loader hỗ trợ chú thích (test).

## Known limitations
- EM địa chỉ/seller bị chặn bởi nhiễu nhãn (cùng cửa hàng được gõ khác nhau; nhãn "PHẢ", "CỮA HÀNG"). CER phản ánh tốt hơn.
- Chưa chuẩn hoá vị trí dấu thanh kiểu cũ/mới ("Thuỵ" vs "Thụy") trong metric; LLM đôi khi "sửa" sai dấu ("Thuỵ" → "Thủy").
- Gemini mới chạy zero-shot với prompt `p1` trên OCR PP-OCRv6; chưa thử few-shot, VietOCR + LLM, hay tắt thinking (proxy không báo thinking).
- Rules không tìm được seller cho cửa hàng ngoài gazetteer khi tắt fallback (đánh đổi precision/recall).

## DoD verification
- [x] Cùng một ảnh luôn trả JSON hợp schema; field thiếu không gây lỗi (test rỗng, ảnh xoay, LLM trả sai định dạng).
- [x] Truy ngược được mọi field tìm thấy về OCR span (evidence span ids + bbox gốc).
- [x] 2 pipeline KIE (rules, OCR + LLM) chạy trên cùng input.
- [x] Unit tests cho parser ngày/tiền, Unicode, dấu phân tách, KIE, LLM (backend giả) — 126 tests pass.

## Next phase
- Phase 4: benchmark đầy đủ (theo stratum, clean subset, latency end-to-end), error analysis có phân nhóm, chốt pipeline cho app, chạy test **một lần**.
