# Phase 1 — MC-OCR Dataset & Evaluation Protocol

- Start date: 2026-10-18
- End date: 2026-10-18
- Status: Done

## Objectives
- Hiểu chính xác annotation, chuyển sang schema nội bộ, chốt cách chia dữ liệu và metric trước khi chọn model.

## Work completed
- [x] Đọc trang dataset/challenge chính thức (định nghĩa task, format nhãn) → `docs/data_usage.md`.
- [x] Xác định cấu trúc dữ liệu thật: 1.155 ảnh có nhãn, 391 ảnh val **không nhãn**; không có `label_dict.json` → suy ra và kiểm chứng mapping 15/16/17/18 trên toàn bộ 6.585 vùng.
- [x] `evaluation/dataset_adapter.py`: CSV → `ReceiptExtraction` (cùng kiểu với output model) + evidence + cờ chất lượng dữ liệu + quy tắc *unscorable*.
- [x] Bộ chuẩn hoá dùng chung (sẽ tái sử dụng ở Phase 3): `ai/normalization/{text,money,datetime_vi}.py`, `ai/layout.py` (reading order).
- [x] Quy ước field thiếu → `null` (target hợp lệ), khác với annotation có nhưng không chuẩn hoá được → *unscorable*.
- [x] EDA tự sinh: `python -m evaluation.eda` → `docs/dataset_eda.md`.
- [x] Phát hiện trùng/gần trùng: SHA-256 + dHash 1024-bit (đã hiệu chỉnh ngưỡng) + khoá giao dịch → `evaluation/dedup.py`.
- [x] Split 70/15/15 theo nhóm trùng, phân tầng theo seller, seed 2026 → `data/splits/{train,dev,test,smoke,summary}.json` (808/174/173, smoke 15 ảnh từ dev). `--check` xác nhận tái lập.
- [x] `evaluation/data_contract.md`: annotation, chuẩn hoá, split, quy tắc dùng split, định nghĩa metric cho Phase 4.
- [x] Kiểm tra thủ công 12 ảnh (chỉ từ train) → `docs/annotation_manual_check.md`; giá trị chi tiết ở `data/private/` (git-ignored), được assert bằng test.

## Technical decisions
- Decision: **dHash 64-bit bị loại**, dùng 1024-bit với ngưỡng 150.
  - Lý do (đo được): hai hoá đơn *khác nhau* cùng chuỗi chỉ cách nhau 1–2/64 bit; với 1024-bit, bản re-encode/screenshot cùng ảnh cách 78 và 110 bit, hoá đơn khác nhau gần nhất cách 223 bit.
- Decision: thêm **khoá giao dịch** (seller + phút + tổng tiền) — bắt được 4 cặp "chụp lại cùng hoá đơn từ góc khác" (≈500 bit dHash, không hash toàn cục nào bắt được); đã xem ảnh xác nhận.
- Decision: chia theo **seller đã biết** (stratified), không chia theo "store chưa thấy".
  - Alternatives: group-by-store. Lý do: MINIMART ANAN (267 ảnh, 1 cửa hàng) và VinCommerce (322) chiếm 51 % dữ liệu — group theo store làm split lệch nặng. Hạn chế này được ghi rõ; Phase 4 báo cáo theo stratum.
- Decision: `total_cost` bỏ các số < 100 (số lượng/quầy), lấy giá trị đa số; hoà → `null`.
- Decision: không sửa nhãn sai; chỉ gắn cờ và báo cáo (nhãn sai không phát hiện được tự động vẫn giữ nguyên).

## Evidence
- `pytest`: 82 passed (gồm test trên dữ liệu thật: load 1.155 ảnh, bbox trong ảnh, mapping category, manual checks, split disjoint/không cắt nhóm, SHA-256 khớp).
- `python -m evaluation.splits --check` → `reproducible`.
- Số liệu chính (từ `docs/dataset_eda.md`):
  - Field có mặt: seller 92,7 %, address 93,2 %, timestamp 94,2 %, total 93,7 %; đủ cả 4: 82,1 %.
  - Timestamp chuẩn hoá được 95,6 % (date+time 897, date-only 143); total 93,0 %.
  - 37 nhóm trùng (76 ảnh); 20 ảnh train cũng có trong `val_images/`.
  - Kiểm tra tay: 41/48 giá trị khớp bản in; 7 sai do nhãn (2 lỗi tiền ×10).

## Known limitations
- Nhiễu nhãn ~15 % trên mẫu 12 ảnh (mẫu nhỏ) → EM bị chặn trên; một phần nhiễu không phát hiện tự động được.
- Test split đo hiệu năng trên cửa hàng **đã thấy**, không đo khả năng tổng quát sang mẫu hoá đơn mới.
- Quy ước "nhiều giờ → lấy giờ đầu" (giờ vào/ra) là quy ước, không phải sự thật nghiệp vụ; Phase 4 báo cáo thêm date-only match.
- Chuỗi giờ dạng `20 05` (thiếu dấu `:`) không được parse → GT date-only.

## DoD verification
- [x] Adapter đọc được nhãn, ảnh và metadata của toàn bộ split được phép sử dụng.
- [x] Train/dev/test tái lập cùng seed, không trùng file, không cắt nhóm trùng.
- [x] ≥ 10 record kiểm thử annotation thủ công (12).
- [x] Quy định dùng dữ liệu ở `docs/data_usage.md`.

## Next phase
- Phase 2: giải quyết tốc độ PaddleOCR (GPU / mobile model), đo CER recognizer trên dev (crop theo vùng annotation), module `OCRProvider` thật, quy đổi toạ độ về ảnh gốc, image-quality baseline so với điểm IQA chính thức.
