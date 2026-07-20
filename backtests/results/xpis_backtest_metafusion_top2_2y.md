# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 730 ngày (2024-07-08 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 2 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 383.5s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật; calibration toàn bộ models = Tắt; constrained stacking = Tắt; uniform fusion policy = Tắt; Meta gate + uniform ranking = Tắt

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 11,934,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-252,000đ** | **+7,546,211đ** |
| ROI tổng | **-2.11%** | **+2.79%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-18.24%, +14.79%]** | — |
| Xác suất ROI dương (bootstrap) | **39.5%** | — |
| Win Rate ngày | **14.4%** (118/730 ngày có trúng) | — |
| Tổng số lần cược | 442 số | 351 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 17/730 | — |
| Vốn cuối kỳ | — | **277,546,211đ** (10,279.5 điểm) |

- **Calibration được chọn**: sigmoid 25 lần; isotonic 0 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 442 | -2.11% | [-18.24%, +14.79%] |
| Ngẫu nhiên (seed 42) | 442 | -7.09% | [-24.05%, +10.63%] |
| Tần suất lịch sử | 442 | -2.94% | [-19.55%, +14.15%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 2b. Ranking-only diagnostic (không threshold, không Kelly)

> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.

| Chỉ số | Giá trị |
|---|---:|
| Bets giả lập | 1460 |
| Hits | 368 |
| Hits trung bình/ngày | 0.5041 |
| ROI ranking-only | -7.58% |
| Bootstrap CI95 | [-16.62%, +1.96%] |
| P(ROI>0) | 5.7% |

## 2c. So sánh Fusion với Uniform Fusion

| Cấu hình | Brier ↓ | ECE ↓ | Precision@K ↑ | Top-K hits | Ranking ROI | CI95 |
|---|---:|---:|---:|---:|---:|---|
| MetaFusion | 0.182261 | 0.009445 | 0.2288 | 368 | -7.58% | [-16.62%, +1.96%] |
| Uniform fusion | 0.182867 | 0.018122 | 0.2315 | 379 | -4.82% | [-14.36%, +4.98%] |

- Uniform fusion P(ROI>0): 15.8%

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | -252,000đ |
| PnL hoán vị trung bình | +278,838đ |
| PnL hoán vị 95% | [-1,737,000đ, +2,421,000đ] |
| p-value một phía | 0.6961 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| inverted_pairs | 442 | +4.52% | [-12.74%, +22.74%] |
| max_delay | 442 | -1.28% | [-18.61%, +17.26%] |
| day_of_week | 442 | -1.28% | [-17.79%, +15.79%] |
| ewma_probability | 442 | -1.28% | [-18.61%, +16.89%] |
| frequency_momentum | 442 | -3.77% | [-20.29%, +13.68%] |
| poisson_estimator | 442 | -6.26% | [-23.07%, +11.63%] |
| bayesian_predictor | 442 | -7.92% | [-24.00%, +8.55%] |
| lightgbm_classifier | 442 | -7.92% | [-24.64%, +9.20%] |
| loto_repeat | 442 | -8.75% | [-24.72%, +7.94%] |
| markov_chain | 442 | -10.41% | [-26.19%, +6.05%] |
| conditional_probability | 442 | -17.87% | [-33.26%, -1.73%] |

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
| 2024-08-01 | 89 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-08-02 | 97 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-09-02 | 13 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-09-05 | 06 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
