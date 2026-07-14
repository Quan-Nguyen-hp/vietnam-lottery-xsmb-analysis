# ADR-001: Áp dụng Tương quan Thực nghiệm Normalized Mutual Information (NMI)
**Ngày**: 2026-07-14
**Trạng thái**: APPROVED (Chấp thuận)

## Problem (Vấn đề)
Hệ số tương quan Pearson truyền thống được thiết kế cho các biến số liên tục có phân phối chuẩn. Đối với ma trận nhị phân thưa và mất cân bằng cao của kết quả loto XSMB (Bernoulli variables với base rate ~27%), Pearson dễ bị thiên lệch (bias), bỏ qua các tương quan phi tuyến và bị ảnh hưởng mạnh bởi tần suất xuất hiện cơ sở.

## Alternatives (Các giải pháp thay thế)
* **Lựa chọn A**: Giữ nguyên Pearson và thêm bộ lọc thắt chặt (Heuristic).
* **Lựa chọn B**: Sử dụng Mutual Information (MI) thuần túy.
* **Lựa chọn C (Được chọn)**: Sử dụng Normalized Mutual Information (NMI) để triệt tiêu ảnh hưởng của entropymarginal riêng lẻ.

## Decision (Quyết định chọn)
Chọn **Normalized Mutual Information (NMI)** được tính toán chéo trên cửa sổ trượt 1800 ngày lịch sử. Đầu ra được lưu dưới dạng Edge List (`predictions/mi_edges.json`) để tăng tốc truy vấn.

## Consequences (Hệ quả)
* Tốc độ truy vấn ma trận tương quan nhanh hơn nhờ lưu trữ dạng Edge List giảm độ phức tạp từ $O(N^2)$ xuống $O(E)$ (trong đó $E$ là số cạnh có tín hiệu thực tế).
* Khả năng phát hiện tương quan phi tuyến thực nghiệm tăng cao, loại bỏ hoàn toàn các hard-code chủ quan về cặp lộn/gương.
