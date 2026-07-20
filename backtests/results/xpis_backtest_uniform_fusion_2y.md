# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 730 ngày (2024-07-08 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 373.3s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật; calibration toàn bộ models = Tắt; constrained stacking = Tắt; uniform fusion policy = Bật

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 47,682,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-3,825,000đ** | **-8,212,242đ** |
| ROI tổng | **-8.02%** | **-3.04%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-16.40%, +0.60%]** | — |
| Xác suất ROI dương (bootstrap) | **3.4%** | — |
| Win Rate ngày | **42.1%** (443/730 ngày có trúng) | — |
| Tổng số lần cược | 1766 số | 648 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 204/730 | — |
| Vốn cuối kỳ | — | **261,787,758đ** (9,695.8 điểm) |

- **Calibration được chọn**: sigmoid 25 lần; isotonic 0 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 1766 | -8.02% | [-16.40%, +0.60%] |
| Ngẫu nhiên (seed 42) | 1766 | +1.32% | [-7.55%, +10.23%] |
| Tần suất lịch sử | 1766 | +0.49% | [-7.97%, +9.19%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 2b. Ranking-only diagnostic (không threshold, không Kelly)

> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.

| Chỉ số | Giá trị |
|---|---:|
| Bets giả lập | 2920 |
| Hits | 752 |
| Hits trung bình/ngày | 1.0301 |
| ROI ranking-only | -5.57% |
| Bootstrap CI95 | [-11.97%, +1.08%] |
| P(ROI>0) | 5.1% |

## 2c. So sánh Fusion với Uniform Fusion

| Cấu hình | Brier ↓ | ECE ↓ | Precision@K ↑ | Top-K hits | Ranking ROI | CI95 |
|---|---:|---:|---:|---:|---:|---|
| Uniform fusion policy | 0.182867 | 0.018122 | 0.2315 | 752 | -5.57% | [-11.97%, +1.08%] |
| Uniform fusion | 0.182867 | 0.018122 | 0.2315 | 752 | -5.57% | [-11.97%, +1.08%] |

- Uniform fusion P(ROI>0): 5.1%

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | -3,825,000đ |
| PnL hoán vị trung bình | +424,001đ |
| PnL hoán vị 95% | [-3,726,000đ, +4,689,000đ] |
| p-value một phía | 0.9810 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| day_of_week | 1766 | +2.57% | [-6.08%, +11.42%] |
| inverted_pairs | 1766 | -0.76% | [-9.36%, +7.86%] |
| max_delay | 1766 | -1.59% | [-9.99%, +7.18%] |
| frequency_momentum | 1766 | -2.21% | [-10.72%, +6.65%] |
| ewma_probability | 1766 | -2.83% | [-10.99%, +5.52%] |
| conditional_probability | 1766 | -3.66% | [-11.87%, +4.65%] |
| bayesian_predictor | 1766 | -3.66% | [-12.07%, +4.70%] |
| poisson_estimator | 1766 | -3.87% | [-11.98%, +4.50%] |
| loto_repeat | 1766 | -5.32% | [-14.18%, +3.64%] |
| markov_chain | 1766 | -5.95% | [-14.61%, +2.88%] |
| lightgbm_classifier | 1766 | -9.68% | [-18.50%, -0.65%] |

## 5. Trọng số Fusion hiện tại của 11 Models (Layer 5)

| Model | Trọng số |
|---|:---:|
| max_delay | 9.09% |
| conditional_probability | 9.09% |
| markov_chain | 9.09% |
| frequency_momentum | 9.09% |
| poisson_estimator | 9.09% |
| loto_repeat | 9.09% |
| inverted_pairs | 9.09% |
| day_of_week | 9.09% |
| bayesian_predictor | 9.09% |
| ewma_probability | 9.09% |
| lightgbm_classifier | 9.09% |

## 6. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
| 2026-02-02 | 83, 12, 88 | 4 | +315,000đ | SKIP | +0đ | 1.00 |
| 2024-09-08 | 16, 34, 11, 60 | 4 | +288,000đ | SKIP | +0đ | 1.00 |
| 2024-10-05 | 81, 87, 34, 39 | 4 | +288,000đ | SKIP | +0đ | 1.00 |
| 2025-06-01 | 58 | 3 | +270,000đ | SKIP | +0đ | 1.00 |
| 2026-01-20 | 05, 92 | 3 | +243,000đ | SKIP | +0đ | 1.00 |
| 2024-12-14 | 91, 57, 58 | 3 | +216,000đ | SKIP | +0đ | 1.00 |
| 2025-04-20 | 24, 15, 53 | 3 | +216,000đ | SKIP | +0đ | 1.00 |
| 2025-07-23 | 29, 51, 78 | 3 | +216,000đ | SKIP | +0đ | 1.00 |
| 2026-02-04 | 25, 88, 65 | 3 | +216,000đ | SKIP | +0đ | 1.00 |
| 2024-07-20 | 89, 67, 53, 58 | 3 | +189,000đ | SKIP | +0đ | 1.00 |
| 2024-07-23 | 25, 69, 45, 89 | 3 | +189,000đ | SKIP | +0đ | 1.00 |
| 2024-07-26 | 07, 99, 77, 66 | 3 | +189,000đ | SKIP | +0đ | 1.00 |
| 2024-08-14 | 89, 81, 85, 80 | 3 | +189,000đ | SKIP | +0đ | 1.00 |
| 2024-08-25 | 07, 13, 77, 90 | 3 | +189,000đ | SKIP | +0đ | 1.00 |
| 2024-08-29 | 25, 77, 46, 40 | 3 | +189,000đ | SKIP | +0đ | 1.00 |
