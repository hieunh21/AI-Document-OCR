# Phase 5 — FastAPI Backend & PostgreSQL

- Start date: 2026-10-19
- End date: 2026-10-19
- Status: Done

## Objectives
Biến pipeline AI thành dịch vụ có trạng thái và lưu trữ: upload → xử lý → xem kết quả và evidence → sửa trường → duyệt → xuất file. Mọi thao tác có lịch sử và còn nguyên sau khi khởi động lại.

## Work completed
- [x] **PostgreSQL 16** (`docker-compose.yml`), SQLAlchemy 2 và **Alembic** (`backend/alembic.ini`, migration `0001`). `alembic check` không thấy khác biệt giữa migration và model, trên cả SQLite lẫn PostgreSQL.
- [x] **Bảng** (`backend/app/models`):

  | Bảng | Nội dung |
  |---|---|
  | `documents` | file, sha256, kích thước ảnh, trạng thái, lỗi, **giá trị hiện tại** của 4 trường (để lọc và xuất không cần đọc JSON), người và thời điểm duyệt |
  | `extractions` | mỗi lần chạy pipeline là một dòng: pipeline version, model version, `raw_json` (OCR spans, chất lượng ảnh, thời gian từng bước), `normalized_json` (`ReceiptExtraction` có evidence) |
  | `field_corrections` | trường, giá trị cũ và mới, người sửa, extraction được sửa |
  | `document_events` | nhật ký: uploaded, processed, failed, corrected, approved |

  Kết quả JSON lưu bằng `JSONB`; file ảnh nằm trên đĩa (`storage/`), không lưu trong database.
- [x] **API** (`backend/app/api/routes/documents.py`, logic ở `backend/app/services/documents.py`):

  | Endpoint | Chức năng |
  |---|---|
  | `POST /api/v1/documents` | kiểm tra file (rỗng, quá 10 MB, không phải JPG/PNG/WebP, file hỏng, ảnh quá nhỏ hoặc quá lớn); giới hạn 30 upload mỗi phút mỗi địa chỉ; cùng nội dung thì trả lại tài liệu cũ; `?process=true` để xử lý luôn |
  | `POST /{id}/process` | chạy `build_pipeline("default")` đồng bộ; model nạp lười một lần, có khoá để GPU xử lý từng ảnh một |
  | `GET /{id}` | giá trị hiện tại kèm nguồn `ai`/`human`, cảnh báo tính lại trên giá trị hiện tại, kết quả AI gốc, OCR và evidence, lịch sử sửa, nhật ký |
  | `GET /{id}/image` | ảnh gốc |
  | `PATCH /{id}/fields` | chuẩn hoá giá trị (`"510.760"` → 510760, `"30/04/2020 10:41"` → `2020-04-30T10:41`), ghi giá trị cũ và mới, kiểm tra lại (ngày tương lai, số tiền bất thường) và cập nhật trạng thái |
  | `POST /{id}/approve` | duyệt rồi khoá: không sửa, không xử lý lại |
  | `GET /documents` | phân trang; lọc theo trạng thái, seller, ngày upload, ngày trên hoá đơn |
  | `GET /{id}/export`, `GET /documents/export` | JSON hoặc CSV (UTF-8 có BOM để Excel đọc đúng tiếng Việt) |

- [x] **Trạng thái:** `uploaded → processing → ready | needs_review | failed`; sửa trường thì tính lại `ready`/`needs_review`; `approved` khoá tài liệu.
  - Xử lý lại thì giữ toàn bộ lịch sử, nhưng giá trị hiện tại bắt đầu lại từ kết quả AI mới.
  - Tài liệu bị kẹt ở `processing` (server tắt giữa chừng) được đánh dấu `failed` khi khởi động.
- [x] **Lỗi, logging, bảo mật:**
  - mọi lỗi trả `{"detail", "code", "request_id"}`; lỗi inference ghi vào tài liệu chứ không làm sập server;
  - log có `request_id` và latency;
  - `X-API-Key` tuỳ chọn (`VIETDOCAI_API_KEY`) khi đưa lên internet;
  - `/health` báo trạng thái database và pipeline.
- [x] **Test** (`backend/tests`, 25 test, chạy trên SQLite với pipeline mock nên không cần PostgreSQL hay GPU):
  - luồng chính, xử lý lại, chuẩn hoá giá trị khi sửa, khoá sau duyệt;
  - lọc, phân trang, xuất JSON/CSV;
  - dữ liệu còn sau khi mở engine mới;
  - lỗi: file rỗng, quá lớn, sai định dạng, hỏng, ảnh nhỏ, thiếu file, inference lỗi, vượt giới hạn upload, sai API key, body không hợp lệ;
  - migration khớp model.

## Kết quả chạy thật (PostgreSQL + pipeline-v3 trên RTX 3050)

| Kiểm tra | Kết quả |
|---|---|
| Upload + xử lý ảnh đầu tiên | 10 s (gồm nạp model) |
| Các ảnh sau | **0.58 s** mỗi request (OCR 0.46 s, KIE 1 ms, IQA 31 ms) |
| Sửa 2 trường → trạng thái | `needs_review` → `ready`, nguồn = `human`, có 2 dòng `field_corrections` |
| Duyệt → sửa lại | 409 `approved_locked` |
| Xuất CSV các tài liệu đã duyệt | tiếng Việt đúng, kèm pipeline version |
| Khởi động lại backend | 2 tài liệu, 4 sự kiện và 2 lần sửa vẫn còn nguyên |

## Technical decisions
- **Xử lý đồng bộ:** một ảnh mất khoảng 0.5 s nên chưa cần hàng đợi job. Bảng `processing_jobs` để lại cho khi cần xử lý nhiều ảnh.
- **Giá trị hiện tại lưu trên `documents`**, tính từ extraction mới nhất cộng các lần sửa sau nó. Lọc, xuất file và dashboard không phải đọc JSON. Kết quả AI gốc vẫn giữ nguyên trong `extractions`.
- **Trường người dùng đặt `null`** được coi là "xác nhận không có trên hoá đơn", nên không bị cảnh báo thiếu.
- **Duyệt khi còn cảnh báo vẫn được**, vì con người quyết định; các cảnh báo còn mở được ghi vào sự kiện `approved`.
- **Trùng file (sha256)** trả lại tài liệu cũ với mã 200, để upload lại không tạo bản sao.

## Known limitations
- Giới hạn upload nằm trong bộ nhớ của từng process; chạy nhiều worker thì cần Redis.
- Chưa có tài khoản người dùng; `reviewer` là chuỗi tự do và API key dùng chung.
- File ảnh nằm trên đĩa local; triển khai thật nên dùng object storage.
- Giao diện React vẫn chỉ đọc (danh sách, chi tiết); upload, sửa và duyệt trên UI là Phase 6.

## Next phase
- Phase 6: nối các trang Upload, Review (sửa trường, Save Correction, Approve), History và Export của React vào các API trên.
