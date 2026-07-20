# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 730 ngày (2024-07-08 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 421.0s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật; calibration toàn bộ models = Tắt; constrained stacking = Bật

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 3,213,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **+450,000đ** | **+1,014,808đ** |
| ROI tổng | **+14.01%** | **+0.38%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-21.69%, +52.78%]** | — |
| Xác suất ROI dương (bootstrap) | **75.9%** | — |
| Win Rate ngày | **4.2%** (37/730 ngày có trúng) | — |
| Tổng số lần cược | 119 số | 103 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 0/730 | — |
| Vốn cuối kỳ | — | **271,014,808đ** (10,037.6 điểm) |

- **Calibration được chọn**: sigmoid 25 lần; isotonic 0 lần.

- **Constrained stacking weights ở lần retrain cuối**:
  - lightgbm_classifier: 10.5422%
  - day_of_week: 10.4347%
  - markov_chain: 10.4120%
  - conditional_probability: 10.3087%
  - loto_repeat: 9.8462%
  - ewma_probability: 9.8437%
  - frequency_momentum: 9.5519%
  - inverted_pairs: 8.8535%
  - bayesian_predictor: 8.2853%
  - poisson_estimator: 6.5334%
  - max_delay: 5.3884%
- **Stacking objective ở lần retrain cuối**: 0.365668

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 119 | +14.01% | [-21.69%, +52.78%] |
| Ngẫu nhiên (seed 42) | 119 | +7.84% | [-28.04%, +46.05%] |
| Tần suất lịch sử | 119 | -16.81% | [-47.22%, +16.67%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 2b. Ranking-only diagnostic (không threshold, không Kelly)

> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.

| Chỉ số | Giá trị |
|---|---:|
| Bets giả lập | 2920 |
| Hits | 752 |
| Hits trung bình/ngày | 1.0301 |
| ROI ranking-only | -5.57% |
| Bootstrap CI95 | [-12.23%, +1.08%] |
| P(ROI>0) | 5.0% |

## 2c. So sánh Fusion với Uniform Fusion

| Cấu hình | Brier ↓ | ECE ↓ | Precision@K ↑ | Top-K hits | Ranking ROI | CI95 |
|---|---:|---:|---:|---:|---:|---|
| Constrained stacking | 0.182037 | 0.004663 | 0.2288 | 752 | -5.57% | [-12.23%, +1.08%] |
| Uniform fusion | 0.182867 | 0.018122 | 0.2315 | 752 | -5.57% | [-11.97%, +1.08%] |

- Uniform fusion P(ROI>0): 5.1%

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | +450,000đ |
| PnL hoán vị trung bình | +108,193đ |
| PnL hoán vị 95% | [-936,000đ, +1,143,000đ] |
| p-value một phía | 0.2993 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| inverted_pairs | 119 | +7.84% | [-26.05%, +44.82%] |
| day_of_week | 119 | +4.76% | [-28.18%, +39.95%] |
| ewma_probability | 119 | +4.76% | [-28.04%, +39.53%] |
| frequency_momentum | 119 | +1.68% | [-29.49%, +34.93%] |
| markov_chain | 119 | -13.73% | [-42.61%, +17.09%] |
| loto_repeat | 119 | -13.73% | [-41.91%, +15.93%] |
| poisson_estimator | 119 | -19.89% | [-47.62%, +9.09%] |
| bayesian_predictor | 119 | -19.89% | [-48.99%, +12.31%] |
| lightgbm_classifier | 119 | -19.89% | [-51.11%, +14.76%] |
| max_delay | 119 | -29.13% | [-55.03%, -1.49%] |
| conditional_probability | 119 | -35.29% | [-62.59%, -4.72%] |

## 5. Trọng số Fusion tối ưu hiện tại của 11 Models (Layer 5)

| Model | Trọng số |
|---|:---:|
| lightgbm_classifier | 10.54% |
| day_of_week | 10.43% |
| markov_chain | 10.41% |
| conditional_probability | 10.31% |
| loto_repeat | 9.85% |
| ewma_probability | 9.84% |
| frequency_momentum | 9.55% |
| inverted_pairs | 8.85% |
| bayesian_predictor | 8.29% |
| poisson_estimator | 6.53% |
| max_delay | 5.39% |

## 6. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
| 2026-01-31 | 58, 49 | 3 | +243,000đ | SKIP | +0đ | 1.00 |
| 2024-07-26 | 66 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2025-11-24 | 62 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-03-31 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-07-09 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2024-07-29 | 22 | 1 | +72,000đ | 22(41đ) | +2,978,488đ | 1.00 |
| 2024-08-01 | 89 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-10-05 | 87 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-03-23 | 47 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-04-07 | 25 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-07-07 | 21 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-07-26 | 11 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-08-29 | 51 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-09-02 | 62 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-11-14 | 49 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
