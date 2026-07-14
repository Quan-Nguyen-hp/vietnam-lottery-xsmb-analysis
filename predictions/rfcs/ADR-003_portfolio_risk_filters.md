# ADR-003: Triển khai Bộ lọc Rủi ro Danh mục và Đa dạng hóa Kelly
**Ngày**: 2026-07-14
**Trạng thái**: APPROVED (Chấp thuận)

## Problem (Vấn đề)
Phân bổ Kelly độc lập từng số rất dễ dẫn đến tập trung vốn quá lớn khi mô hình đề xuất nhiều số có tương quan cao (ví dụ: các cặp lộn 12-21 hoặc các số cùng một đầu số). Điều này vi phạm nguyên tắc quản trị rủi ro danh mục trong tài chính định lượng, làm tăng đột biến mức drawdown (MDD).

## Alternatives (Các giải pháp thay thế)
* **Lựa chọn A**: Giới hạn vốn Kelly cứng trên mỗi số (Heuristic).
* **Lựa chọn B (Được chọn)**: Tích hợp bộ lọc Diversification Score, điều chỉnh giảm Kelly theo mức độ tương quan chéo thực nghiệm và áp dụng Exposure Limits (Đầu/Đuôi giới hạn ở mức tối đa 20% tổng vốn).

## Decision (Quyết định chọn)
Chọn Lựa chọn B. Hệ thống tự động từ chối cược (SKIP) nếu Diversification Score $< 0.85$ và tự động phân rã vốn tối ưu chéo dựa trên Correlation Graph.

## Consequences (Hệ quả)
* Giảm thiểu tối đa mức sụt giảm vốn lớn nhất (MDD). Trong kiểm thử Epoch 2 (giai đoạn thị trường bất lợi), trong khi cược cố định mất 100% vốn, Kelly Portfolio chỉ mất 0.55% vốn ban đầu.
* Phân bổ vốn an toàn, tuân thủ nguyên lý đa dạng hóa danh mục đầu tư.
