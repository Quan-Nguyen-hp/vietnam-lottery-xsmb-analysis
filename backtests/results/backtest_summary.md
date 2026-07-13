# Báo cáo kết quả Backtest các phương pháp dự đoán XSMB

- **Thời điểm kiểm thử**: 05-10-2023 đến 12-07-2026
- **Tổng số ngày kiểm thử**: 1000 ngày
- **Cài đặt**: Chọn Top 5 số lô mỗi ngày. Giá mua: 27,000đ/điểm. Trúng giải: 99,000đ/nháy.

## Bảng thống kê hiệu năng chi tiết

| Phương pháp | Tỷ lệ thắng ngày (Win Rate) | Số nháy trúng TB | Nháy trúng nhiều nhất/ngày | Tổng Chi phí (đ) | Tổng Thu về (đ) | Lợi nhuận ròng (đ) | ROI (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Max Delay (Lô Khan) | 76.00% | 1.221 | 5 | 135,000,000đ | 135,729,000đ | 729,000đ | 0.54% |
| Bạc Nhớ (Conditional Similarity) | 76.50% | 1.177 | 4 | 135,000,000đ | 130,878,000đ | -4,122,000đ | -3.05% |
| Xích Markov (Markov Chain) | 75.20% | 1.166 | 4 | 135,000,000đ | 131,868,000đ | -3,132,000đ | -2.32% |
| Tần suất Động lượng (Frequency Momentum - 30 ngày) | 77.30% | 1.225 | 5 | 135,000,000đ | 137,313,000đ | 2,313,000đ | 1.71% |
| Ước lượng Poisson (Poisson Estimator - 180 ngày) | 75.80% | 1.227 | 4 | 135,000,000đ | 135,828,000đ | 828,000đ | 0.61% |
| Đồng thuận Ensemble (Weighted Voting) | 76.00% | 1.235 | 4 | 135,000,000đ | 138,798,000đ | 3,798,000đ | 2.81% |

## Nhận định chi tiết về các phương pháp

> [!NOTE]
> **Xác suất cơ sở toán học**:
> - Khi chọn 5 số ngẫu nhiên mỗi ngày trong số 100 số, tỷ lệ thắng ngày mong đợi là khoảng 73.1% (có ít nhất 1 số trúng).
> - Số nháy trúng mong đợi mỗi ngày là: 5 số * 27 giải / 100 = 1.35 nháy.
> - ROI mong đợi của lô tô Việt Nam là: (1.35 * 99,000) / (5 * 27,000) - 1 = 133,650 / 135,000 - 1 = -1.00%.

### Phân tích hiệu năng thực tế:
- **Max Delay (Lô Khan)**:
  - Đạt tỷ lệ thắng ngày: **76.00%** và số nháy trúng trung bình **1.221**.
  - ROI thực tế: **0.54%** (Lợi nhuận ròng: 729,000đ).
- **Bạc Nhớ (Conditional Similarity)**:
  - Đạt tỷ lệ thắng ngày: **76.50%** và số nháy trúng trung bình **1.177**.
  - ROI thực tế: **-3.05%** (Lợi nhuận ròng: -4,122,000đ).
- **Xích Markov (Markov Chain)**:
  - Đạt tỷ lệ thắng ngày: **75.20%** và số nháy trúng trung bình **1.166**.
  - ROI thực tế: **-2.32%** (Lợi nhuận ròng: -3,132,000đ).
- **Tần suất Động lượng (Frequency Momentum - 30 ngày)**:
  - Đạt tỷ lệ thắng ngày: **77.30%** và số nháy trúng trung bình **1.225**.
  - ROI thực tế: **1.71%** (Lợi nhuận ròng: 2,313,000đ).
- **Ước lượng Poisson (Poisson Estimator - 180 ngày)**:
  - Đạt tỷ lệ thắng ngày: **75.80%** và số nháy trúng trung bình **1.227**.
  - ROI thực tế: **0.61%** (Lợi nhuận ròng: 828,000đ).
- **Đồng thuận Ensemble (Weighted Voting)**:
  - Đạt tỷ lệ thắng ngày: **76.00%** và số nháy trúng trung bình **1.235**.
  - ROI thực tế: **2.81%** (Lợi nhuận ròng: 3,798,000đ).

## Kết luận

🎉 Phương pháp mang lại hiệu quả cao nhất trong thời gian thử nghiệm là **Đồng thuận Ensemble (Weighted Voting)** với lợi nhuận ròng là **3,798,000đ**.
