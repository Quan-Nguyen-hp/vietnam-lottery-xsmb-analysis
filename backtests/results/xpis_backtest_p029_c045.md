# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.29 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 170.8s

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 3,699,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-135,000đ** | **-264,462đ** |
| ROI tổng | **-3.65%** | **-0.10%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-33.88%, +28.18%]** | — |
| Xác suất ROI dương (bootstrap) | **39.8%** | — |
| Win Rate ngày | **8.2%** (36/365 ngày có trúng) | — |
| Tổng số lần cược | 137 số | 108 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 170/365 | — |
| Vốn cuối kỳ | — | **269,735,538đ** (9,990.2 điểm) |

- **Calibration được chọn**: sigmoid 6 lần; isotonic 7 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. Trọng số Fusion tối ưu hiện tại của 11 Models (Layer 5)

| Model | Trọng số |
|---|:---:|
| lightgbm_classifier | 12.12% |
| day_of_week | 9.49% |
| markov_chain | 9.42% |
| ewma_probability | 9.38% |
| conditional_probability | 9.37% |
| frequency_momentum | 9.34% |
| bayesian_predictor | 9.17% |
| loto_repeat | 8.96% |
| inverted_pairs | 8.77% |
| max_delay | 6.99% |
| poisson_estimator | 6.99% |

## 3. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
| 2026-02-21 | 77, 09 | 3 | +243,000đ | 77(64đ), 09(28đ) | +9,388,149đ | 1.00 |
| 2026-03-31 | 91 | 2 | +171,000đ | 91(125đ) | +21,431,734đ | 1.00 |
| 2025-08-09 | 33, 49 | 2 | +144,000đ | 33(21đ), 49(19đ) | +2,656,368đ | 1.00 |
| 2025-09-02 | 62, 96 | 2 | +144,000đ | 62(77đ), 96(41đ) | +8,483,191đ | 1.00 |
| 2026-02-26 | 09, 01 | 2 | +144,000đ | 09(12đ), 01(29đ) | +2,950,355đ | 1.00 |
| 2025-07-13 | 29 | 1 | +72,000đ | 29(59đ) | +4,241,139đ | 1.00 |
| 2025-07-26 | 11 | 1 | +72,000đ | 11(101đ) | +7,293,061đ | 1.00 |
| 2025-08-11 | 53 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-08-17 | 49 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-09-15 | 82 | 1 | +72,000đ | 82(18đ) | +1,285,176đ | 1.00 |
| 2025-10-27 | 02 | 1 | +72,000đ | 02(29đ) | +2,119,022đ | 1.00 |
| 2025-12-31 | 22 | 1 | +72,000đ | 22(20đ) | +1,469,687đ | 1.00 |
| 2026-02-10 | 77 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2026-02-20 | 77 | 1 | +72,000đ | 77(23đ) | +1,660,317đ | 1.00 |
| 2026-03-23 | 16 | 1 | +72,000đ | 16(18đ) | +1,329,345đ | 1.00 |
