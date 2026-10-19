# Phase 6 — React Frontend & Human-in-the-Loop

- Start date: 2026-10-19
- End date: 2026-10-19 (chờ người dùng thử trên trình duyệt)
- Status: Done

## Objectives
Một ứng dụng end-to-end dùng được không cần dòng lệnh: **Upload → AI → Review → Edit → Approve → History → Export**.

## Work completed
- [x] **API client** (`frontend/src/api/client.ts`): mọi lời gọi của Phase 5, đọc lỗi dạng `{detail, code, request_id}`, báo riêng khi không kết nối được backend. Kiểu dữ liệu trong `types.ts` khớp schema của backend.
- [x] **Upload** (`pages/UploadPage.tsx`):
  - kéo thả hoặc chọn ảnh, xem trước;
  - kiểm tra định dạng và dung lượng ngay ở trình duyệt (giới hạn lấy từ `/health`);
  - các bước Upload → Extract → Review, có trạng thái đang xử lý;
  - ảnh trùng thì mở tài liệu cũ; xử lý lỗi thì vẫn mở trang Review để thử lại;
  - 3 hoá đơn demo tự vẽ (`scripts/make_demo_receipts.py`, nội dung tự đặt).
- [x] **Review** (`pages/ReviewPage.tsx`):
  - **Ảnh và evidence:** ảnh hoá đơn và khung evidence vẽ bằng SVG theo toạ độ ảnh gốc (`viewBox` = kích thước ảnh), nên khung bám đúng khi co giãn. Ảnh có EXIF xoay được backend trả về đã dựng thẳng, cùng hệ toạ độ với OCR.
  - **Chọn trường:** bấm hoặc focus vào một trường thì vùng chữ AI đọc được sáng lên. Có công tắc hiện mọi khung OCR; rê chuột trên danh sách chữ OCR thì khung tương ứng sáng lên.
  - **Nguồn giá trị:** mỗi trường có nhãn **AI**, **edited by a person** (kèm người sửa và thời điểm) hoặc **unsaved**. Ngoài giá trị hiện tại còn hiện chữ đọc được trên hoá đơn, giá trị AI gốc khi đã sửa, và cảnh báo của trường.
  - **Save correction và Approve là hai việc khác nhau:**
    - Save chỉ gửi các trường đã đổi, kèm tên reviewer (lưu trong trình duyệt).
    - Approve hỏi xác nhận, liệt kê các cảnh báo còn mở, tự lưu thay đổi chưa lưu trước khi duyệt; sau đó tài liệu bị khoá.
  - **Trạng thái:** chưa xử lý thì có nút Extract now; lỗi thì có nút Retry; đang xử lý thì trang tự làm mới; đã duyệt thì khoá. Có thêm Re-process (xác nhận trước) và Export JSON/CSV.
  - **Các tab:** chữ OCR kèm độ tin cậy; lịch sử sửa (trước/sau, ai sửa, lần chạy nào); nhật ký hoạt động; thông tin kỹ thuật (model, thời gian từng bước, sha256).
- [x] **History:**
  - lọc theo trạng thái (lưu trên URL), seller, ngày upload, ngày trên hoá đơn;
  - phân trang phía server;
  - **export** CSV/JSON theo đúng bộ lọc đang chọn, ví dụ chỉ các bản ghi đã duyệt;
  - bấm vào dòng để mở Review.
- [x] **Dashboard:** dùng endpoint mới `GET /documents/stats`:
  - số tài liệu, số cần review, số đã duyệt (kèm % trên số đã xử lý);
  - **tổng tiền chỉ tính bản ghi đã duyệt**;
  - danh sách 5 tài liệu chờ review; bấm vào thì chuyển sang History đã lọc sẵn.
- [x] **Header:** badge hiện pipeline version, trạng thái database, model đã nạp hay chưa.
- [x] **Backend bổ sung** (3 test mới, tổng 28 test backend):
  - `GET /documents/stats`;
  - ảnh EXIF trả về đã dựng thẳng;
  - `/health` báo giới hạn upload.

## Evidence
- `npm run build` (tsc + vite) thành công; ruff và toàn bộ pytest đều qua.
- Chạy thật với backend (PostgreSQL + pipeline-v3, model nạp lúc khởi động) và 3 ảnh demo:
  - **thời gian:** đọc đúng cả 3;
  - **tổng tiền:** đúng 2/3 (ảnh nhà thuốc lấy nhầm dòng "Tiền khách đưa");
  - **seller:** để trống ở cả 3, vì là cửa hàng chưa gặp và dòng tên bị gộp vào địa chỉ.

  Cả 3 đều ở trạng thái `needs_review`, là đúng chỗ để con người sửa và duyệt.

## Technical decisions
- **Giao diện tiếng Anh, `ConfigProvider` locale `vi_VN`** (ngày, số), giữ như các phase trước.
- **Một trang Review cho cả xem, sửa và duyệt**: trạng thái tài liệu quyết định những nút nào hiện ra.
- **Upload và process gọi riêng**: nếu bước xử lý lỗi, người dùng vẫn có tài liệu và nút Retry.

## Known limitations
- Chưa có test trình duyệt tự động (Playwright); kiểm tra bằng build và thử tay. Có thể thêm ở Phase 7.
- Bundle JS khoảng 1.3 MB (Ant Design), chưa tách chunk.
- Không có phóng to/thu nhỏ ảnh; ảnh co theo chiều rộng cột.

## Next phase
- Phase 7: test tích hợp, Docker cho toàn bộ app (db + backend + frontend), triển khai, tài liệu portfolio.
