# ADR-002: Tích hợp Hệ trọng số động Rolling EWMA ở Layer 5
**Ngày**: 2026-07-14
**Trạng thái**: APPROVED (Chấp thuận)

## Problem (Vấn đề)
Trọng số cố định (static weights) cho Meta Fusion không phản ánh đúng phong độ thực tế của các mô hình thành phần theo thời gian (ví dụ: mô hình Poisson hoặc EWMA có thể hoạt động tốt ở chu kỳ này nhưng mất lợi thế ở chu kỳ khác). Việc chỉ dựa vào ROI lịch sử để tính trọng số rất nhiễu và dễ dẫn đến overfit.

## Alternatives (Các giải pháp thay thế)
* **Lựa chọn A**: Tính trọng số dựa trên ROI tĩnh 180 ngày.
* **Lựa chọn B (Được chọn)**: Sử dụng Rolling Quality Score dựa trên Brier, LogLoss, Precision@10 và ROI chéo trên 3 cửa sổ trượt (30d, 60d, 120d).

## Decision (Quyết định chọn)
Chọn Lựa chọn B. Trọng số của mỗi mô hình tỷ lệ thuận với điểm chất lượng dự báo tổng hợp, tự động cập nhật định kỳ mỗi chu kỳ trượt.

## Consequences (Hệ quả)
* Điểm số phản ánh đúng năng lực dự báo khoa học (Brier, LogLoss) kết hợp năng lực chọn lọc thực tế (Precision, ROI).
* Trọng số của mô hình ML (LightGBM) và mô hình thống kê tự điều hòa chéo giúp hệ thống ổn định trước các biến động dữ liệu.
