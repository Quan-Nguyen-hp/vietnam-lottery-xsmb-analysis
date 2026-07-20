# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 162.7s

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 3,024,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-153,000đ** | **+238,333đ** |
| ROI tổng | **-5.06%** | **+0.09%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-38.89%, +30.45%]** | — |
| Xác suất ROI dương (bootstrap) | **37.3%** | — |
| Win Rate ngày | **6.6%** (29/365 ngày có trúng) | — |
| Tổng số lần cược | 112 số | 90 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 6/365 | — |
| Vốn cuối kỳ | — | **270,238,333đ** (10,008.8 điểm) |

- **Calibration được chọn**: sigmoid 6 lần; isotonic 7 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 112 | -5.06% | [-38.89%, +30.45%] |
| Ngẫu nhiên (seed 42) | 112 | -11.61% | [-41.03%, +21.00%] |
| Tần suất lịch sử | 112 | +1.49% | [-31.02%, +38.25%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 3. Trọng số Fusion tối ưu hiện tại của 11 Models (Layer 5)

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

## 4. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
| 2026-02-21 | 77, 09 | 3 | +243,000đ | 77(64đ), 09(28đ) | +9,446,228đ | 1.00 |
| 2026-03-31 | 91 | 2 | +171,000đ | 91(125đ) | +21,412,086đ | 1.00 |
| 2025-08-09 | 33, 49 | 2 | +144,000đ | 33(21đ), 49(19đ) | +2,671,188đ | 1.00 |
| 2025-09-02 | 62, 96 | 2 | +144,000đ | 62(77đ), 96(42đ) | +8,546,198đ | 1.00 |
| 2025-07-13 | 29 | 1 | +72,000đ | 29(59đ) | +4,243,540đ | 1.00 |
| 2025-07-26 | 11 | 1 | +72,000đ | 11(101đ) | +7,298,407đ | 1.00 |
| 2025-08-11 | 53 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2025-10-27 | 02 | 1 | +72,000đ | 02(30đ) | +2,129,213đ | 1.00 |
| 2026-02-20 | 77 | 1 | +72,000đ | 77(23đ) | +1,670,588đ | 1.00 |
| 2026-02-26 | 09 | 1 | +72,000đ | 09(12đ) | +870,292đ | 1.00 |
| 2026-03-23 | 16 | 1 | +72,000đ | 16(18đ) | +1,328,126đ | 1.00 |
| 2026-03-28 | 91 | 1 | +72,000đ | 91(10đ) | +715,490đ | 1.00 |
| 2026-04-15 | 30 | 1 | +72,000đ | 30(14đ) | +1,016,908đ | 1.00 |
| 2026-05-12 | 91 | 1 | +72,000đ | 91(26đ) | +1,886,202đ | 1.00 |
| 2026-05-13 | 91 | 1 | +72,000đ | 91(16đ) | +1,158,035đ | 1.00 |
