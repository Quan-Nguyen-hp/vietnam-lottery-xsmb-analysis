# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 199.8s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật; calibration toàn bộ models = Bật

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 0đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **+0đ** | **+0đ** |
| ROI tổng | **+0.00%** | **+0.00%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[+0.00%, +0.00%]** | — |
| Xác suất ROI dương (bootstrap) | **0.0%** | — |
| Win Rate ngày | **0.0%** (0/365 ngày có trúng) | — |
| Tổng số lần cược | 0 số | 0 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 0/365 | — |
| Vốn cuối kỳ | — | **270,000,000đ** (10,000.0 điểm) |

- **Calibration được chọn**: sigmoid 0 lần; isotonic 0 lần.

- **Component calibration ở lần retrain cuối**: identity=0, platt=9, isotonic=2.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 0 | +0.00% | [+0.00%, +0.00%] |
| Ngẫu nhiên (seed 42) | 0 | +0.00% | [+0.00%, +0.00%] |
| Tần suất lịch sử | 0 | +0.00% | [+0.00%, +0.00%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 2b. Ranking-only diagnostic (không threshold, không Kelly)

> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.

| Chỉ số | Giá trị |
|---|---:|
| Bets giả lập | 1460 |
| Hits | 343 |
| Hits trung bình/ngày | 0.9397 |
| ROI ranking-only | -13.86% |
| Bootstrap CI95 | [-22.40%, -5.07%] |
| P(ROI>0) | 0.1% |

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | +0đ |
| PnL hoán vị trung bình | +0đ |
| PnL hoán vị 95% | [+0đ, +0đ] |
| p-value một phía | 1.0000 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| max_delay | 0 | +0.00% | [+0.00%, +0.00%] |
| conditional_probability | 0 | +0.00% | [+0.00%, +0.00%] |
| markov_chain | 0 | +0.00% | [+0.00%, +0.00%] |
| frequency_momentum | 0 | +0.00% | [+0.00%, +0.00%] |
| poisson_estimator | 0 | +0.00% | [+0.00%, +0.00%] |
| loto_repeat | 0 | +0.00% | [+0.00%, +0.00%] |
| inverted_pairs | 0 | +0.00% | [+0.00%, +0.00%] |
| day_of_week | 0 | +0.00% | [+0.00%, +0.00%] |
| bayesian_predictor | 0 | +0.00% | [+0.00%, +0.00%] |
| ewma_probability | 0 | +0.00% | [+0.00%, +0.00%] |
| lightgbm_classifier | 0 | +0.00% | [+0.00%, +0.00%] |

## 5. Trọng số Fusion tối ưu hiện tại của 11 Models (Layer 5)

| Model | Trọng số |
|---|:---:|
| markov_chain | 9.18% |
| max_delay | 9.14% |
| loto_repeat | 9.14% |
| poisson_estimator | 9.11% |
| ewma_probability | 9.10% |
| frequency_momentum | 9.10% |
| conditional_probability | 9.09% |
| day_of_week | 9.05% |
| lightgbm_classifier | 9.04% |
| bayesian_predictor | 9.03% |
| inverted_pairs | 9.02% |

## 6. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
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
| 2025-07-24 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-25 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2025-07-26 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
