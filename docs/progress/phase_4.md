# Phase 4 — Evaluation, Benchmark & Error Analysis

- Start date: 2026-10-18
- End date: 2026-10-18
- Status: Done

## Objectives
- Có bằng chứng kỹ thuật (accuracy **và** latency/cost) để chọn pipeline cho app, rồi chạy held-out test **một lần**.

## Work completed
- [x] `python -m evaluation.run_benchmark --split dev --model rules llm …` (`evaluation/run_benchmark.py`):
  - một lệnh chạy cả chuỗi OCR (chỉ ảnh chưa có cache) → KIE (`extract_from_ocr`, như app) → metric → report;
  - report ghi split + số ảnh, version GT/metric, git commit, sha256 gazetteer/model IQA, version OCR/KIE, phần cứng.
- [x] `evaluation/metrics.py`: metric dùng chung (EM, P/R/F1, CER, EM chỉ ngày) + **paired bootstrap** (khoảng tin cậy 95 % cho chênh lệch giữa hai pipeline trên cùng ảnh).
- [x] Metric `kie-metrics-v2`: so khớp text sau khi đưa dấu thanh về vị trí chuẩn (`Thuỵ` = `Thụy`); GT không đổi; đã ghi changelog trong data contract, **chốt trước khi chạy test**.
- [x] Báo cáo theo nhóm ảnh: clean annotations, cửa hàng đã/chưa gặp trong train, chất lượng ảnh, ảnh xoay, seller stratum.
- [x] Latency end-to-end bằng `ReceiptPipeline` thật (load → OCR → KIE → IQA):
  - các pipeline chạy **xen kẽ trên từng ảnh**, vì GPU laptop đổi xung nhịp giữa chừng (lần đo tuần tự đầu tiên cho VietOCR lúc 1.3 s, lúc 2.9 s);
  - LLM tính theo thời gian gọi gốc.
- [x] Chi phí LLM: số token / hoá đơn và USD theo giá niêm yết của Google (chỉ là ước lượng bậc độ lớn).
- [x] `evaluation/error_analysis.py`:
  - tự gán mỗi lỗi cho giai đoạn hỏng đầu tiên (thiếu nhãn / OCR / KIE);
  - xuất crop để xem tay;
  - đã xem tay 29 lỗi → `docs/error_analysis.md`.
- [x] Ablation OCR: hạ `text_det_box_thresh` 0.6 → 0.5 / 0.4 (sửa lỗi mất dòng in mảnh) → **không cải thiện KIE**, không dùng.
- [x] Chốt pipeline (`configs/pipeline.json`, `ai.pipeline.build_pipeline`); chạy test một lần → `evaluation/reports/final_test.md`.

## Kết quả

**Dev (174 ảnh, `model_comparison.md`)**

| Pipeline | Seller | Address | Timestamp | Total | **Overall EM** | Δ vs rules (95 % CI) | E2E P50 / P95 | USD / 1k |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| baseline (rules, no store list) | 50.6 % | 4.0 % | 87.4 % | 90.1 % | 57.1 % | −17.2 [−19.9, −14.6] | 0.8 / 1.1 s | 0 |
| **rules** (+ gazetteer) | 74.7 % | 46.6 % | 87.4 % | 90.1 % | **74.3 %** | — | 0.9 / 1.3 s | 0 |
| rules, VietOCR | 74.7 % | 48.3 % | 83.2 % | 89.4 % | 73.5 % | −0.7 [−2.4, +0.9] | 3.0 / 3.7 s | 0 |
| Gemini, no store list | 70.1 % | 32.8 % | 88.6 % | 93.2 % | 70.6 % | −3.7 [−6.4, −1.0] | 2.5 / 4.2 s | ~1.0 |
| **Gemini + gazetteer** | 75.3 % | 56.3 % | 88.6 % | 93.2 % | **78.0 %** | +3.7 [+1.3, +6.2] | 2.4 / 4.2 s | ~1.0 |
| Gemini + gaz, VietOCR | 74.7 % | 56.9 % | 84.4 % | 91.9 % | 76.6 % | +2.4 [+0.0, +4.8] | 5.4 / 6.8 s | ~1.6 |

**Test (173 ảnh, chạy một lần với cấu hình đã đóng băng — `final_test.md`)**

| Pipeline | Seller | Address | Timestamp | Total | **Overall EM** | Δ vs rules (95 % CI) | E2E P50 / P95 |
|---|---:|---:|---:|---:|---:|---|---:|
| baseline | 50.9 % | 4.0 % | 87.9 % | 89.4 % | 57.1 % | −18.2 [−21.0, −15.5] | 0.5 / 0.7 s |
| **rules** (app default) | 72.8 % | 52.6 % | 87.9 % | 89.4 % | **75.3 %** | — | 0.5 / 0.7 s |
| Gemini + gazetteer (option) | 74.0 % | 53.8 % | 88.5 % | 92.5 % | **76.8 %** | +1.5 [−0.6, +3.6] | 4.4 / 8.7 s |

Rules trên test (75.3 %) khớp dev (74.3 %). Lợi thế của Gemini nhỏ lại (+3.7 → +1.5 điểm) và không còn có ý nghĩa thống kê; xem *Technical decisions* về proxy.

## Technical decisions
- **Decision: app mặc định = `rules`** (PP-OCRv6 medium + orientation + luật + gazetteer).
  - Lý do: offline, không cần key, không gửi dữ liệu ra ngoài, P50 < 1 s trên GPU, chi phí 0.
  - Kém LLM ít nhất 3.7 điểm trên dev, chỉ 1.5 điểm (không có ý nghĩa thống kê) trên test.
- **Decision: Gemini + gazetteer là tuỳ chọn** (`--pipeline llm`) khi có `GEMINI_API_KEY`.
  - Tốt hơn chủ yếu ở total (+3 điểm) và address.
  - Đổi lại: chậm hơn 2.5–5 lần, khoảng 1–2 USD / 1k hoá đơn, và phụ thuộc proxy.
- **Decision: không dùng VietOCR.** CER OCR tốt hơn nhiều nhưng KIE không tăng (dev 73.5 / 76.6 % so với 74.3 / 78.0 %) và chậm hơn khoảng 3 s / ảnh.
- **Decision: giữ ngưỡng detection mặc định (0.6).** Ngưỡng 0.5 lấy lại 100 % vùng nhãn nhưng KIE không tăng.
- **Decision: metric `kie-metrics-v2` (chuẩn hoá vị trí dấu thanh).**
  - Đây là khác biệt chính tả, không phải lỗi đọc.
  - Đã quyết định trên dev và ghi vào data contract trước khi chạy test.
- **Phát hiện: backend phía sau proxy shopaikey đã thay đổi trong ngày.**
  - Cache cho thấy các request lúc 17:4x có khoảng 215 token output, P50 1.6 s, 1,321 token input.
  - Từ 19:4x trở đi: khoảng 500 token output, 3.2–4.2 s, khoảng 1,200 token input.
  - Prompt không đổi, JSON trả về vẫn khoảng 300 ký tự.
  - Hệ quả: số dev của Gemini (gọi lúc 17:4x) và số test (gọi lúc 20:1x) **không chắc đến từ cùng một model/cấu hình**. Đây là rủi ro thực tế của việc dùng reseller, đã ghi rõ trong báo cáo.
  - **Kiểm tra lại** (`evaluation/reports/llm_backend_check_dev.md`): gửi lại **đúng 174 request dev** (cùng prompt và OCR, cache riêng qua `llm:gemini:rerun2+gaz`) lúc 20:25–20:35.
    - Backend mới: khoảng 574 token output, P50 3.3 s; 21 request mất hơn 30 s; 0 lỗi.
    - Overall EM **77.5 %**, so với 78.0 % lúc chiều; chênh −0.4 điểm, CI [−1.6, +0.7].
    - 657/696 giá trị trường giống hệt nhau; 15 trường đổi đúng/sai (9 từ đúng sang sai, 6 từ sai sang đúng).
    - So với rules: **+3.3 điểm, CI [+1.0, +5.6]**, vẫn có ý nghĩa thống kê.
    - Kết luận: backend đổi làm **chậm hơn khoảng 2 lần và tốn khoảng 1.8 lần chi phí**, nhưng độ chính xác gần như không đổi. Mức +1.5 trên test so với +3.3 trên dev là chênh lệch giữa dev và test, không phải do proxy đổi model.

## Bugs found & fixed
- `kie_eval` / `error_analysis` lấy `img_id` ra sau khi validate JSON (schema cấm key thừa) → sửa thứ tự.
- Đo latency tuần tự bị nhiễu do GPU laptop đổi xung nhịp → chạy xen kẽ các pipeline trên từng ảnh.
- Phát hiện (không phải bug của ta): pipeline PaddleOCR bỏ box detection có điểm < 0.6. Đã tìm ra qua error analysis, đo ablation, và ghi lại.

## Known limitations
- Exact match bị chặn bởi nhiễu nhãn: trong mẫu xem tay khoảng 1/3 "lỗi" là do nhãn thiếu hoặc sai. Chênh lệch 1–2 điểm giữa dev và test nằm trong nhiễu này.
- Cửa hàng chưa gặp trong train (dev n = 13, test n = 20): seller EM gần 0 % với cả hai phương pháp. App phải đánh dấu cần review.
- Latency đo trên laptop RTX 3050 với 30 ảnh, cùng một model dao động khoảng ±15 % giữa các lần đo; latency LLM tuỳ vào proxy.
- Chưa thử few-shot, sửa prompt, hay VLM (đọc trực tiếp ảnh); các hướng này để sau. Không tinh chỉnh gì thêm dựa trên test.

## DoD verification
- [x] Một lệnh chạy ra report kèm config/model version và số mẫu (`run_benchmark`).
- [x] Ít nhất 2 phương pháp được so sánh công bằng (6 pipeline, cùng dev, paired bootstrap).
- [x] Không dùng held-out test để tinh chỉnh; test chạy một lần sau khi đóng băng `configs/pipeline.json`.
- [x] Quyết định có căn cứ accuracy **và** latency/cost/offline.
- [x] Phân nhóm lỗi (ảnh mờ, xoay, OCR, seller, total, date) + ví dụ → `docs/error_analysis.md`.
- [x] Tests: 129 pass (thêm test cho metric, dấu thanh và bootstrap).

## Follow-up sau khi chạy test: rules v2 (chỉ đo trên train + dev)

Mục đích: cải thiện phương án miễn phí theo `docs/error_analysis.md`. Mọi thay đổi đều đo trên **train (808 ảnh) và dev (174 ảnh)**, mỗi thay đổi một lần; **không chạy lại test**.

| Thay đổi | Train overall | Dev overall | Giữ? |
|---|---:|---:|---|
| Mốc `rules-v1+gaz-nofb` | 74.5 % | 74.3 % | — |
| Total: ưu tiên số tiền không thuộc dòng "tiền khách trả / tiền thừa" | 74.5 % (sửa 3 / hỏng 2) | 74.3 % (sửa 1 / hỏng 1) | **Không**: toạ độ OCR ở hoá đơn in sát dòng quá mập mờ |
| Timestamp: tìm giờ ở dòng **trên hoặc dưới** (tối đa 2 dòng), ưu tiên dòng có từ khoá | 74.5 % (TS 86.5 → 86.6) | 74.4 % (TS 87.4 → 88.0) | Có |
| Danh mục hành chính: chỉ sửa dấu | 74.5 % | 74.4 % (CER địa chỉ 26.2 → 25.9) | (gộp vào dòng dưới) |
| **Danh mục hành chính: sửa dấu + nhận dòng địa chỉ có tên hành chính** | **75.2 %** (địa chỉ 50.0 → 52.5) | **75.6 %** (địa chỉ 46.6 → 51.1, CER 26.2 → 21.3) | **Có → `pipeline-v2`** |

- **Danh mục hành chính** (`configs/vn_admin_units.json`, dựng bằng `scripts/build_admin_units.py`):
  - nguồn là danh mục của Tổng cục Thống kê (danhmuchanhchinh.gso.gov.vn), qua bản JSON công khai github.com/dvhcvn/data;
  - gồm 63 tỉnh, 696 quận/huyện, 10,047 phường/xã của giai đoạn trước 2025, cộng thêm tên cũ: tổng 13,259 mục.
- **Sửa dấu** (`ai/normalization/address.py`): chỉ đổi dấu, không đổi chữ cái. Có hai lớp an toàn:
  - **kiểm tra cấp hành chính:** xã/phường phải thuộc tỉnh/huyện đã nhắc trong địa chỉ, nên tên thôn "Phú Thụy" không bị đổi thành xã "Phú Thủy" ở Quảng Bình;
  - **gặp ca mơ hồ thì giữ nguyên.**
- **Paired bootstrap trên dev:** `rules-adm` hơn `rules` **+1.2 điểm [+0.4, +2.2]**. Khoảng cách tới Gemini + gazetteer còn 2.4 điểm (trước là 3.6).
- **Chi phí:** KIE vẫn khoảng 2 ms mỗi hoá đơn; nạp danh mục khoảng 0.5 s một lần lúc khởi động.
- **Với Gemini, sửa dấu không đổi gì** (78.0 % → 78.0 %), vì Gemini vốn viết dấu đúng. Vì vậy không thêm bước này vào pipeline `llm`.
- **Hạn chế:**
  - **"Cửa hàng chưa gặp" hiện được xác định bằng cách so khớp đúng chính tả tên seller**, nên nhãn gõ sai ("THỨC COFEE", "VinCommercce") cũng bị tính là cửa hàng lạ. Con số của nhóm này vì vậy lẫn nhiễu nhãn.
  - **OCR sai cả chữ cái** ("QUAMG MIMH", "Thunt Nen") thì việc sửa dấu không cứu được.
- **Trạng thái tập test:** `pipeline-v2` chưa có số test, vì test đã dùng một lần cho v1 (75.3 %). Chạy test lần hai thì phải ghi rõ là test đã được dùng lại.

## Theo sau Phase 4: fine-tune bộ nhận dạng chữ tiếng Việt → `pipeline-v3`

Chi tiết trong `docs/finetune_rec.md`; báo cáo ở `evaluation/reports/ocr_vi3_dev.md`. Chỉ đo trên dev; **không dùng test**.

- **Nguyên nhân gốc của lỗi dấu:** bộ ký tự của PP-OCRv6 rec thiếu toàn bộ 90 chữ có dấu thanh. Cách sửa:
  - thay bằng bộ ký tự tiếng Việt 237 ký tự;
  - lớp đầu ra chép từ model gốc (`scripts/remap_rec_head.py`);
  - fine-tune trên các dòng của tập train.
- **Bẫy độ rộng ảnh:** khi train, backbone PPLCNetV4 luôn pool đặc trưng về 40 bước thời gian.
  - Vòng 1–2 train ảnh rộng 640 px nên model lặp ký tự ("119.000" → "1119.000"), và KIE giảm còn 67.8 %.
  - Vòng 3 train ảnh rộng 320 px như config gốc, chỉ dữ liệu thật, trên máy local khoảng 25 phút.

| Dev | OCR CER (folded) | Rules EM | Gemini EM |
|---|---:|---:|---:|
| `pipeline-v2` (OCR gốc) | 11.0 % | 75.6 % | 77.5 % (cùng proxy backend) |
| **`pipeline-v3` (OCR vòng 3)** | **3.2 %** | **76.2 %** (+0.6 [−0.4, +1.6]) | **78.1 %** (+0.6 [−1.0, +2.1]) |

- **Vì sao vẫn dùng dù KIE tăng không có ý nghĩa thống kê:**
  - OCR tốt hơn rõ rệt;
  - KIE không giảm;
  - chữ tiếng Việt đúng ở evidence, ở text OCR thô và với cửa hàng chưa gặp.
- **Vì sao KIE tăng ít:** gazetteer và danh mục hành chính vốn đã tự sửa dấu cho seller và địa chỉ; Gemini thì tự viết dấu đúng.
- **Trọng số model không nằm trong git** (`models/ppocrv6_medium_rec_vi3/`, khoảng 62 MB). Cần phát hành riêng, ví dụ đính kèm vào GitHub Release, để người clone repo chạy được pipeline mặc định.

## Next phase
- Phase 5: FastAPI + PostgreSQL. Backend dùng `ai.pipeline.build_pipeline("default")`; tuỳ chọn `llm` khi có key. Lưu `model_version` và pipeline version cho mỗi extraction.
