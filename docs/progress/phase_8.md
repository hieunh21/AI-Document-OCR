# Phase 8 — Nâng cao, Option A: cải thiện độ chính xác có chứng cứ

- Start date: 2026-10-19
- Status: đang làm (đã xong bộ phân loại dòng; còn sửa nhãn dev)

## Objectives
Sau pipeline-v3, phân tích lỗi (`docs/error_analysis.md` §6) cho thấy nút thắt đã chuyển từ OCR sang **khâu chọn dòng** (KIE): 57 % lỗi thuộc KIE. Mục tiêu là thay luật viết tay bằng một model học từ dữ liệu, và giữ pipeline miễn phí, offline.

## Work completed
- [x] **Bộ phân loại dòng** (`ai/kie/line_features.py`, `ai/kie/learned.py`, `evaluation/train_line_classifier.py`).
  - **Model:** LightGBM, mỗi khung OCR được xếp vào một trong 5 lớp: seller / address / timestamp / total / other.
  - **Đặc trưng:**
    - vị trí, cỡ chữ, chỗ trong dòng;
    - tỉ lệ chữ số và chữ hoa;
    - các mẫu dùng lại từ luật (ngày, từ khoá tổng tiền và từ phủ định, tiêu đề, số điện thoại, gợi ý địa chỉ, số tiền lớn nhất);
    - độ khớp gazetteer và danh mục hành chính;
    - ngữ cảnh của khung trước, khung sau và cùng dòng.
  - **Nhãn train:** khung OCR nhận nhãn của vùng chú thích chồng lấp ≥ 50 % so với khung **nhỏ hơn**. Ban đầu đo theo khung OCR; cách đó làm mất nhãn ở các dòng kiểu "Số GD … Ngày: …", và timestamp out-of-fold chỉ đạt 62.7 %.
  - **Dữ liệu train:** 808 ảnh train × 2 bản OCR (bộ nhận dạng fine-tune và bản gốc), tổng 67k khung. Bản OCR gốc giúp model quen với chữ đọc sai.
  - **Cross-fitting gazetteer:** đặc trưng gazetteer của mỗi fold lấy từ gazetteer dựng bằng các fold khác, để cửa hàng chỉ xuất hiện một lần trông như cửa hàng lạ.
  - **Ghép trường:** seller và địa chỉ lấy các khung vượt ngưỡng, nối theo thứ tự đọc; ngày giờ thì parse chung các khung và tự gắn thêm khung giờ ở gần; tổng tiền lấy số trong khung có xác suất cao nhất, nếu khung đó chỉ có từ khoá thì lấy số bên cạnh. Sau đó chuẩn hoá như luật.
  - **Ngưỡng:** chọn trên dự đoán out-of-fold của train, không dùng dev.
- [x] **Đo trên dev một lần, rồi đưa vào `pipeline-v4`** (`configs/pipeline.json`). Factory có thêm tên `lines`, `lines+gaz+adm`; benchmark có thêm `lines-vi3`. Thêm 5 test (`tests/test_line_kie.py`).

## Kết quả (`evaluation/reports/lines_dev.md`)

| Dev | seller | address | timestamp | total | **Tổng** |
|---|---:|---:|---:|---:|---:|
| Rules (v3) | 75.3 % | 52.3 % | 87.4 % | 91.3 % | 76.2 % |
| **Bộ phân loại (v4)** | **78.2 %** | **60.9 %** | 87.4 % | 90.7 % | **79.0 %** |
| Gemini (tốn tiền) | 74.7 % | 57.5 % | 87.4 % | 94.4 % | 78.1 % |

- **So với rules:** +2.8 điểm, khoảng tin cậy [+1.0, +4.7], có ý nghĩa thống kê. Trên train out-of-fold là +3.4.
- **So với Gemini:** phương án miễn phí nay ngang hoặc nhỉnh hơn (−0.9 [−2.7, +0.9] cho Gemini).
- **Độ trễ KIE:** 29 ms (P50) so với 2 ms của luật, vẫn rất nhỏ so với OCR (khoảng 450 ms).

## Known limitations
- **Nhãn "ready" vẫn chưa đáng tin:** khoảng 44 % hoá đơn "ready" đúng cả 4 trường. Nâng ngưỡng xác suất lớp không cải thiện con số này, vì xác suất cho biết *vùng* đúng chứ không cho biết *chữ* đúng. Muốn "ready" đáng tin cần độ tin cậy ở mức giá trị (từ độ tin cậy OCR, độ khớp gazetteer, kết quả parse…).
- **Cửa hàng mới thật** (5 hoá đơn trên dev): tìm được 2/5 nhưng chưa khớp tuyệt đối với nhãn.
- **Tổng tiền:** thấp hơn luật 0.6 điểm (1 hoá đơn) và thấp hơn Gemini 3.7 điểm.

## Next
- Sửa nhãn dev (xem tay các giá trị mà ít nhất một pipeline lệch nhãn, cộng một mẫu ngẫu nhiên) để đo điểm thật.
- Độ tin cậy ở mức giá trị cho nhãn "ready".
