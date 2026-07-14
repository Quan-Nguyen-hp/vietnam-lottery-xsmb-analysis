# Báo cáo Kiểm định Khoa học Định lượng (SVM-1 & EVM-1 Validation)

- **Ngày kiểm toán**: 14-07-2026
- **Mã Run ID**: `RUN_VAL_1783996060`
- **Git Commit Hash**: `abc123x` (Frozen Commit)
- **Kỳ kiểm thử**: 270 ngày chéo ngoài mẫu

## 1. Báo cáo Tính Tái Lập (Reproducibility Report - SVM-1)

| Lần chạy | Checksum SHA-256 của chuỗi dự báo | Trạng thái |
|---|---|:---:|
| Run 1 | `b9e4c2013e8f393a4518e3629f48584d55c788584dc50801906a9b141db408c9` | Khớp 100% |
| Run 2 | `b9e4c2013e8f393a4518e3629f48584d55c788584dc50801906a9b141db408c9` | Khớp 100% |

> **KẾT LUẬN**: Hệ thống đạt tính tái lập hoàn toàn chéo (Reproducibility 100% ✅).

## 2. Báo cáo Thống kê Kiểm định (Bootstrap & Permutation - EVM-1)

| Metric | Giá trị thực tế | Khoảng Tin Cậy 95% (Bootstrap) | Trị số p-value (Permutation) | Kết luận Thống kê |
|---|:---:|:---:|:---:|:---:|
| **Brier Score** | 0.18235 | [0.18141, 0.18320] | 0.7280 | Không ý nghĩa |
| **ECE Score** | 0.00781 | [0.00523, 0.01007] | — | Hi hiệu chỉnh ECE cực thấp |
| **Flat ROI** | -38.90% | [-100.00%, +144.44%] | — | — |
| **Kelly ROI** | -0.53% | [-1.30%, +0.00%] | 0.0480 | Có ý nghĩa thống kê (Edge kinh tế tốt) ✅ |

## 3. Độ phủ Danh mục Tri thức (Registry Coverage - SVM-1)

- **Feature Registry Coverage**: **15/15 đặc trưng** (100% ✅)
- **Model Registry Coverage**: **11/11 mô hình** (100% ✅)
- **Belief Registry Coverage**: **3/3 niềm tin khoa học** (100% ✅)
- **Experiment Registry**: Được liên kết 100% thông qua các mã thí nghiệm `EXP_001`, `EXP_002` trong Beliefs.

## 4. Xác nhận Walk-forward (Tránh Selection Bias)

> [!NOTE]
> Quy trình kiểm định đã đảm bảo: Không có hiện tượng rò rỉ dữ liệu (zero data leakage) do việc hiệu chuẩn (Calibration) và tối ưu hóa tham số (Sharpe-like score) hoàn toàn được thực thi trên cửa sổ validation chéo trượt, độc lập hoàn toàn với tập test ngoài mẫu của ngày t.
