# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 730 ngày (2024-07-08 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 1 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 360.2s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật; calibration toàn bộ models = Tắt; constrained stacking = Tắt; uniform fusion policy = Tắt; Meta gate + uniform ranking = Tắt

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 9,504,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **+297,000đ** | **+2,801,535đ** |
| ROI tổng | **+3.12%** | **+1.04%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-15.47%, +22.91%]** | — |
| Xác suất ROI dương (bootstrap) | **61.7%** | — |
| Win Rate ngày | **12.5%** (99/730 ngày có trúng) | — |
| Tổng số lần cược | 352 số | 352 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 92/730 | — |
| Vốn cuối kỳ | — | **272,801,535đ** (10,103.8 điểm) |

- **Calibration được chọn**: sigmoid 25 lần; isotonic 0 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 352 | +3.12% | [-15.47%, +22.91%] |
| Ngẫu nhiên (seed 42) | 352 | -16.67% | [-34.98%, +2.22%] |
| Tần suất lịch sử | 352 | -2.08% | [-21.79%, +18.35%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 2b. Ranking-only diagnostic (không threshold, không Kelly)

> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.

| Chỉ số | Giá trị |
|---|---:|
| Bets giả lập | 730 |
| Hits | 206 |
| Hits trung bình/ngày | 0.2822 |
| ROI ranking-only | +3.47% |
| Bootstrap CI95 | [-10.09%, +17.53%] |
| P(ROI>0) | 67.8% |

## 2c. So sánh Fusion với Uniform Fusion

| Cấu hình | Brier ↓ | ECE ↓ | Precision@K ↑ | Top-K hits | Ranking ROI | CI95 |
|---|---:|---:|---:|---:|---:|---|
| MetaFusion | 0.182261 | 0.009445 | 0.2534 | 206 | +3.47% | [-10.09%, +17.53%] |
| Uniform fusion | 0.182867 | 0.018122 | 0.2288 | 187 | -6.07% | [-19.13%, +7.99%] |

- Uniform fusion P(ROI>0): 17.9%

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | +297,000đ |
| PnL hoán vị trung bình | +252,806đ |
| PnL hoán vị 95% | [-1,584,000đ, +2,178,000đ] |
| p-value một phía | 0.4941 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| inverted_pairs | 352 | +11.46% | [-8.33%, +32.35%] |
| max_delay | 352 | +4.17% | [-16.19%, +25.74%] |
| frequency_momentum | 352 | +0.00% | [-19.10%, +20.37%] |
| day_of_week | 352 | -2.08% | [-20.69%, +17.20%] |
| poisson_estimator | 352 | -3.12% | [-23.30%, +18.35%] |
| ewma_probability | 352 | -4.17% | [-23.07%, +15.68%] |
| bayesian_predictor | 352 | -7.29% | [-26.25%, +12.33%] |
| lightgbm_classifier | 352 | -7.29% | [-26.47%, +12.98%] |
| loto_repeat | 352 | -10.42% | [-28.15%, +8.04%] |
| markov_chain | 352 | -12.50% | [-30.21%, +5.77%] |
| conditional_probability | 352 | -16.67% | [-34.13%, +1.73%] |

## 5. Trọng số Fusion hiện tại của 11 Models (Layer 5)

| Model | Trọng số |
|---|:---:|
| day_of_week | 9.77% |
| markov_chain | 9.68% |
| lightgbm_classifier | 9.64% |
| ewma_probability | 9.64% |
| conditional_probability | 9.62% |
| frequency_momentum | 9.57% |
| bayesian_predictor | 9.44% |
| loto_repeat | 9.23% |
| inverted_pairs | 9.03% |
| max_delay | 7.19% |
| poisson_estimator | 7.18% |

## 6. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
| 2025-06-01 | 58 | 3 | +270,000đ | SKIP | +0đ | 1.00 |
| 2024-07-26 | 66 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2024-08-14 | 22 | 2 | +171,000đ | 22(30đ) | +5,192,131đ | 1.00 |
| 2025-08-13 | 19 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2025-11-24 | 62 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-03-31 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-07-09 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2024-07-24 | 46 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-07-28 | 69 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-07-29 | 22 | 1 | +72,000đ | 22(49đ) | +3,508,523đ | 1.00 |
| 2024-08-01 | 89 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-08-02 | 97 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-08-28 | 49 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-08-29 | 25 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-09-02 | 13 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
