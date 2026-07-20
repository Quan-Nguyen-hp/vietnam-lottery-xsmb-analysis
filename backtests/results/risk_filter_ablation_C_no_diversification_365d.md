# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 154.2s
- **Ablation**: Kelly chọn cược = Có; diversification = Tắt

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 432,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-135,000đ** | **-1,117,554đ** |
| ROI tổng | **-31.25%** | **-0.41%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-100.00%, +52.78%]** | — |
| Xác suất ROI dương (bootstrap) | **19.6%** | — |
| Win Rate ngày | **0.8%** (3/365 ngày có trúng) | — |
| Tổng số lần cược | 16 số | 15 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 0/365 | — |
| Vốn cuối kỳ | — | **268,882,446đ** (9,958.6 điểm) |

- **Calibration được chọn**: sigmoid 13 lần; isotonic 0 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 16 | -31.25% | [-100.00%, +52.78%] |
| Ngẫu nhiên (seed 42) | 16 | +37.50% | [-61.40%, +153.85%] |
| Tần suất lịch sử | 16 | -31.25% | [-100.00%, +52.78%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | -135,000đ |
| PnL hoán vị trung bình | +54,229đ |
| PnL hoán vị 95% | [-333,000đ, +558,000đ] |
| p-value một phía | 0.8726 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| frequency_momentum | 16 | +60.42% | [-59.26%, +208.77%] |
| poisson_estimator | 16 | +14.58% | [-77.08%, +129.17%] |
| bayesian_predictor | 16 | +14.58% | [-78.43%, +138.33%] |
| ewma_probability | 16 | +14.58% | [-78.43%, +137.25%] |
| lightgbm_classifier | 16 | +14.58% | [-79.63%, +137.28%] |
| max_delay | 16 | -8.33% | [-100.00%, +103.70%] |
| day_of_week | 16 | -8.33% | [-79.63%, +83.33%] |
| inverted_pairs | 16 | -31.25% | [-100.00%, +83.33%] |
| conditional_probability | 16 | -54.17% | [-100.00%, +15.79%] |
| markov_chain | 16 | -54.17% | [-100.00%, +22.22%] |
| loto_repeat | 16 | -54.17% | [-100.00%, +22.22%] |

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
| 2025-11-16 | 88 | 1 | +72,000đ | 88(17đ) | +1,203,773đ | 1.00 |
| 2026-01-08 | 68 | 1 | +72,000đ | 68(20đ) | +1,459,513đ | 1.00 |
| 2026-05-16 | 91 | 1 | +72,000đ | 91(13đ) | +915,042đ | 1.00 |
| 2025-07-12 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-13 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-14 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-15 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-16 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-17 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-18 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-19 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-20 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-21 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-22 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-23 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
