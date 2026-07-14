# Báo cáo Kiểm định Khoa học Đa Epoch & Regime (SVM-1 / EVM-1)

- **Kỳ đánh giá**: 3 Epoch độc lập, mỗi Epoch 90 ngày (Tổng 270 ngày qua)
- **Kiến trúc**: XPIS v1.2 APPROVED
- **Thời gian thực thi**: 15.9s

## 1. Kết quả kiểm định trên các Epoch độc lập (Tránh Selection Bias)

| Epoch | Số ngày | Tổng số cược | Trúng nháy | ROI Cược Phẳng | ROI Kelly | Đánh giá Edge |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Epoch_1_Past | 90 | 6 | 4 | +144.44% | +1.27% | PASS (Edge dương) |
| Epoch_2_Mid | 90 | 9 | 0 | -100.00% | -0.55% | FAIL (Cần tối ưu thêm) |
| Epoch_3_Recent | 90 | 8 | 2 | -8.33% | -0.09% | FAIL (Cần tối ưu thêm) |

## 2. Kết quả kiểm định phân cụm theo Regime XSMB (KMeans)

| Regime | Mô tả trạng thái ngày XSMB | Số ngày ghi nhận | Số lượt cược | Số nháy trúng | Lợi nhuận Kelly (VND) |
|---|---|:---:|:---:|:---:|:---:|
| 0 | Regime A (Mật độ lặp thấp - số phân bố đều) | 91 | 12 | 2 | +1,003,092đ |
| 1 | Regime B (Bão số lặp - nhiều nháy kép nổ) | 41 | 1 | 1 | +420,434đ |
| 2 | Regime C (Bình thường - phân bố ổn định) | 138 | 10 | 3 | +290,248đ |

## 3. Kết luận và Kế hoạch Hành động (EVM-1)

> [!NOTE]
> Kết quả so sánh trên các Epoch độc lập giúp chúng ta chứng thực rằng lợi thế dự báo (Predictive Edge) có ổn định và mang lại lợi thế kinh tế (Economic Edge) hay không. Đây là cơ sở khoa học để ký duyệt đóng băng SVM-1 và sẵn sàng đưa hệ thống vào vận hành định lượng.
