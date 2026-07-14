# Báo cáo hiệu suất kiến trúc XPIS v1.1 (Dynamic Fusion & Calibration)

- **Kỳ kiểm thử**: 180 ngày qua
- **Hệ thống**: XPIS v1.1 (11 Models + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 59.7s

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 216,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-117,000đ** | **+557,946đ** |
| ROI tổng | **-54.17%** | **+0.21%** (Tăng trưởng vốn) |
| Win Rate ngày | **0.6%** (1/180 ngày có trúng) | — |
| Tổng số lần cược | 8 số | 7 ngày cược |
| Vốn cuối kỳ | — | **270,557,946đ** (10,020.7 điểm) |

## 2. Trọng số Fusion tối ưu hiện tại của 11 Models (Layer 5)

| Model | Trọng số |
|---|:---:|
| lightgbm_classifier | 12.22% |
| markov_chain | 9.43% |
| day_of_week | 9.43% |
| conditional_probability | 9.39% |
| frequency_momentum | 9.37% |
| ewma_probability | 9.34% |
| bayesian_predictor | 9.28% |
| loto_repeat | 8.99% |
| inverted_pairs | 8.84% |
| max_delay | 6.92% |
| poisson_estimator | 6.80% |

## 3. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)

| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |
|---|---|:---:|---|---|---|:---:|
| 2026-05-16 | 91 | 1 | +72,000đ | 91(59đ) | +4,254,600đ | 1.00 |
| 2026-01-11 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-12 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-13 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-14 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-15 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-16 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-17 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-18 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-19 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-20 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-21 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-22 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-23 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
| 2026-01-24 | SKIP | 0 | +0đ | SKIP | +0đ | 1.00 |
