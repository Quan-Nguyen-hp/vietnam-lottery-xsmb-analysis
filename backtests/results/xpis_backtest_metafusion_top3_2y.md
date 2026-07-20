# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 730 ngày (2024-07-08 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 3 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 383.5s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật; calibration toàn bộ models = Tắt; constrained stacking = Tắt; uniform fusion policy = Tắt; Meta gate + uniform ranking = Tắt

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 12,393,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-612,000đ** | **+7,352,423đ** |
| ROI tổng | **-4.94%** | **+2.72%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-20.72%, +11.59%]** | — |
| Xác suất ROI dương (bootstrap) | **27.0%** | — |
| Win Rate ngày | **14.4%** (119/730 ngày có trúng) | — |
| Tổng số lần cược | 459 số | 351 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 1/730 | — |
| Vốn cuối kỳ | — | **277,352,423đ** (10,272.3 điểm) |

- **Calibration được chọn**: sigmoid 25 lần; isotonic 0 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 459 | -4.94% | [-20.72%, +11.59%] |
| Ngẫu nhiên (seed 42) | 459 | +1.45% | [-15.82%, +19.36%] |
| Tần suất lịch sử | 459 | -2.54% | [-18.70%, +14.28%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 2b. Ranking-only diagnostic (không threshold, không Kelly)

> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.

| Chỉ số | Giá trị |
|---|---:|
| Bets giả lập | 2190 |
| Hits | 554 |
| Hits trung bình/ngày | 0.7589 |
| ROI ranking-only | -7.25% |
| Bootstrap CI95 | [-14.78%, +0.62%] |
| P(ROI>0) | 3.5% |

## 2c. So sánh Fusion với Uniform Fusion

| Cấu hình | Brier ↓ | ECE ↓ | Precision@K ↑ | Top-K hits | Ranking ROI | CI95 |
|---|---:|---:|---:|---:|---:|---|
| MetaFusion | 0.182261 | 0.009445 | 0.2269 | 554 | -7.25% | [-14.78%, +0.62%] |
| Uniform fusion | 0.182867 | 0.018122 | 0.2288 | 560 | -6.24% | [-13.61%, +1.46%] |

- Uniform fusion P(ROI>0): 5.7%

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | -612,000đ |
| PnL hoán vị trung bình | +276,446đ |
| PnL hoán vị 95% | [-1,800,000đ, +2,457,000đ] |
| p-value một phía | 0.7972 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| inverted_pairs | 459 | +3.85% | [-13.24%, +21.71%] |
| max_delay | 459 | -0.15% | [-16.74%, +17.61%] |
| day_of_week | 459 | -0.15% | [-16.39%, +16.40%] |
| ewma_probability | 459 | -0.94% | [-17.65%, +16.63%] |
| poisson_estimator | 459 | -4.14% | [-20.43%, +13.13%] |
| frequency_momentum | 459 | -4.94% | [-21.06%, +12.06%] |
| lightgbm_classifier | 459 | -6.54% | [-23.16%, +10.74%] |
| bayesian_predictor | 459 | -7.33% | [-23.30%, +8.93%] |
| loto_repeat | 459 | -8.13% | [-23.87%, +8.13%] |
| markov_chain | 459 | -8.93% | [-24.65%, +7.36%] |
| conditional_probability | 459 | -17.72% | [-32.91%, -1.79%] |

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
| 2026-01-31 | 49, 58 | 3 | +243,000đ | SKIP | +0đ | 1.00 |
| 2024-07-26 | 66 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2024-08-14 | 22 | 2 | +171,000đ | 22(31đ) | +5,230,521đ | 1.00 |
| 2025-08-13 | 19 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-03-31 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-07-09 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2024-07-28 | 69, 22 | 2 | +144,000đ | 22(28đ) | +1,996,328đ | 1.00 |
| 2024-08-28 | 49, 13 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2025-04-14 | 25, 69 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2025-11-24 | 62, 83 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2025-04-16 | 91, 50, 53 | 2 | +117,000đ | SKIP | +0đ | 1.00 |
| 2024-08-01 | 89 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-08-02 | 97 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-09-02 | 13 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
