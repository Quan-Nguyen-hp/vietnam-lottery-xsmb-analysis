# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 151.8s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 6,750,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-18,000đ** | **-1,117,554đ** |
| ROI tổng | **-0.27%** | **-0.41%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-21.12%, +22.22%]** | — |
| Xác suất ROI dương (bootstrap) | **48.4%** | — |
| Win Rate ngày | **17.0%** (68/365 ngày có trúng) | — |
| Tổng số lần cược | 250 số | 190 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 0/365 | — |
| Vốn cuối kỳ | — | **268,882,446đ** (9,958.6 điểm) |

- **Calibration được chọn**: sigmoid 13 lần; isotonic 0 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 250 | -0.27% | [-21.12%, +22.22%] |
| Ngẫu nhiên (seed 42) | 250 | -1.73% | [-25.47%, +23.64%] |
| Tần suất lịch sử | 250 | +2.67% | [-19.36%, +25.87%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | -18,000đ |
| PnL hoán vị trung bình | +465,793đ |
| PnL hoán vị 95% | [-1,107,000đ, +2,160,000đ] |
| p-value một phía | 0.7399 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| poisson_estimator | 250 | -0.27% | [-21.72%, +22.70%] |
| frequency_momentum | 250 | -3.20% | [-25.50%, +19.54%] |
| loto_repeat | 250 | -3.20% | [-25.47%, +19.70%] |
| markov_chain | 250 | -4.67% | [-26.36%, +17.52%] |
| ewma_probability | 250 | -4.67% | [-27.52%, +18.58%] |
| inverted_pairs | 250 | -6.13% | [-28.52%, +18.13%] |
| day_of_week | 250 | -13.47% | [-33.47%, +7.39%] |
| lightgbm_classifier | 250 | -16.40% | [-37.65%, +6.11%] |
| max_delay | 250 | -17.87% | [-36.68%, +1.69%] |
| bayesian_predictor | 250 | -17.87% | [-37.10%, +2.98%] |
| conditional_probability | 250 | -19.33% | [-40.86%, +3.77%] |

## 5. Trọng số Fusion tối ưu hiện tại của 11 Models (Layer 5)

| Model | Trọng số |
|---|:---:|
| day_of_week | 9.75% |
| markov_chain | 9.69% |
| lightgbm_classifier | 9.65% |
| ewma_probability | 9.65% |
| conditional_probability | 9.63% |
| frequency_momentum | 9.60% |
| bayesian_predictor | 9.43% |
| loto_repeat | 9.21% |
| inverted_pairs | 9.01% |
| max_delay | 7.19% |
| poisson_estimator | 7.18% |

## 6. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
| 2026-01-31 | 58, 49 | 3 | +243,000đ | SKIP | +0đ | 1.00 |
| 2025-08-13 | 19 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-03-31 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-07-09 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2025-11-24 | 62, 83 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2025-07-14 | 13 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-07-15 | 06 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-07-25 | 11 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-07-29 | 48 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-07-31 | 56 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-08-09 | 28 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-08-22 | 75 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-08-29 | 51 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-08-31 | 81 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-10-15 | 28 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
