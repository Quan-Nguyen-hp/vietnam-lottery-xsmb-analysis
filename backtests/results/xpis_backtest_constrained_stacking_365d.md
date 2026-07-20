# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 210.7s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật; calibration toàn bộ models = Tắt; constrained stacking = Bật

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 2,646,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **+225,000đ** | **-833,314đ** |
| ROI tổng | **+8.50%** | **-0.31%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-29.79%, +50.16%]** | — |
| Xác suất ROI dương (bootstrap) | **65.4%** | — |
| Win Rate ngày | **6.6%** (29/365 ngày có trúng) | — |
| Tổng số lần cược | 98 số | 84 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 0/365 | — |
| Vốn cuối kỳ | — | **269,166,686đ** (9,969.1 điểm) |

- **Calibration được chọn**: sigmoid 13 lần; isotonic 0 lần.

- **Constrained stacking weights ở lần retrain cuối**:
  - lightgbm_classifier: 10.6098%
  - markov_chain: 10.4931%
  - day_of_week: 10.4620%
  - conditional_probability: 10.3530%
  - loto_repeat: 9.8595%
  - ewma_probability: 9.8235%
  - frequency_momentum: 9.6389%
  - inverted_pairs: 8.4604%
  - bayesian_predictor: 7.8585%
  - poisson_estimator: 6.7255%
  - max_delay: 5.7159%
- **Stacking objective ở lần retrain cuối**: 0.366791

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 98 | +8.50% | [-29.79%, +50.16%] |
| Ngẫu nhiên (seed 42) | 98 | -10.20% | [-41.49%, +23.47%] |
| Tần suất lịch sử | 98 | +1.02% | [-35.07%, +40.70%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 2b. Ranking-only diagnostic (không threshold, không Kelly)

> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.

| Chỉ số | Giá trị |
|---|---:|
| Bets giả lập | 1460 |
| Hits | 380 |
| Hits trung bình/ngày | 1.0411 |
| ROI ranking-only | -4.57% |
| Bootstrap CI95 | [-13.61%, +4.22%] |
| P(ROI>0) | 15.3% |

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | +225,000đ |
| PnL hoán vị trung bình | +264,699đ |
| PnL hoán vị 95% | [-666,000đ, +1,314,000đ] |
| p-value một phía | 0.5587 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| frequency_momentum | 98 | +8.50% | [-27.43%, +48.41%] |
| inverted_pairs | 98 | +1.02% | [-34.65%, +41.03%] |
| day_of_week | 98 | -2.72% | [-35.92%, +31.45%] |
| ewma_probability | 98 | -2.72% | [-38.28%, +36.67%] |
| poisson_estimator | 98 | -10.20% | [-44.20%, +24.82%] |
| loto_repeat | 98 | -10.20% | [-43.88%, +26.30%] |
| bayesian_predictor | 98 | -10.20% | [-43.84%, +26.75%] |
| markov_chain | 98 | -17.69% | [-49.17%, +17.18%] |
| lightgbm_classifier | 98 | -28.91% | [-58.84%, +4.76%] |
| max_delay | 98 | -32.65% | [-61.59%, -2.22%] |
| conditional_probability | 98 | -55.10% | [-80.50%, -24.90%] |

## 5. Trọng số Fusion tối ưu hiện tại của 11 Models (Layer 5)

| Model | Trọng số |
|---|:---:|
| lightgbm_classifier | 10.61% |
| markov_chain | 10.49% |
| day_of_week | 10.46% |
| conditional_probability | 10.35% |
| loto_repeat | 9.86% |
| ewma_probability | 9.82% |
| frequency_momentum | 9.64% |
| inverted_pairs | 8.46% |
| bayesian_predictor | 7.86% |
| poisson_estimator | 6.73% |
| max_delay | 5.72% |

## 6. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
| 2026-01-31 | 58, 49 | 3 | +243,000đ | SKIP | +0đ | 1.00 |
| 2026-03-31 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-07-09 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2025-11-24 | 62, 83 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2025-07-26 | 11 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-09-02 | 62 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-11-14 | 49 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-11-16 | 88 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-12-15 | 38 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-12-18 | 49 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2026-01-11 | 68 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2026-02-20 | 58 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2026-02-26 | 09 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2026-04-06 | 52 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2026-04-15 | 30 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
