# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)

- **Kỳ kiểm thử**: 730 ngày (2024-07-08 đến 2026-07-15)
- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)
- **Tham số**: Top K: 4 | Min Prob: 0.31 | Min Conf: 0.45 | Min Diversification: 0.85
- **Thời gian chạy**: 345.8s
- **Ablation**: Kelly chọn cược = Không; diversification = Bật; calibration toàn bộ models = Tắt; constrained stacking = Tắt; uniform fusion policy = Tắt; Meta gate + uniform ranking = Bật

> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, nhưng làm mới mỗi 30 ngày để giới hạn chi phí backtest.

## 1. Kết quả Tổng Hợp

| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |
|---|:---:|:---:|
| Tổng vốn chi | 12,420,000đ | Thay đổi theo ngày |
| Lợi nhuận ròng | **-1,431,000đ** | **+7,352,423đ** |
| ROI tổng | **-11.52%** | **+2.72%** (Tăng trưởng vốn) |
| ROI bootstrap 95% | **[-26.83%, +4.99%]** | — |
| Xác suất ROI dương (bootstrap) | **7.9%** | — |
| Win Rate ngày | **13.3%** (111/730 ngày có trúng) | — |
| Tổng số lần cược | 460 số | 351 ngày cược |
| Ngày bị Top K cắt bớt ứng viên | 0/730 | — |
| Vốn cuối kỳ | — | **277,352,423đ** (10,272.3 điểm) |

- **Calibration được chọn**: sigmoid 25 lần; isotonic 0 lần.

- **Statistical Edge Gate**: **FAIL** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.

> Lưu ý: cột Kelly vẫn là phân bổ của MetaFusion champion và không phải kết quả của danh sách gated-uniform. Chỉ cột Flat Bet và paired comparison được dùng để đánh giá challenger này.

### So sánh ghép cặp với MetaFusion champion

> MetaFusion quyết định ngày và số lượng cược; uniform fusion chỉ thay thứ tự chọn số. Hai policy có exposure từng ngày giống hệt nhau.

| Cấu hình | Lượt cược | ROI | CI95 |
|---|---:|---:|---:|
| Gated uniform ranking | 460 | -11.52% | [-26.83%, +4.99%] |
| MetaFusion champion | 460 | -5.14% | [-20.90%, +11.32%] |

- **ΔROI challenger − champion**: -6.38%, paired CI95 [-23.06%, +10.27%], P(ΔROI>0)=21.1%.

## 2. So sánh baseline cùng mức exposure

| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| XPIS | 460 | -11.52% | [-26.83%, +4.99%] |
| Ngẫu nhiên (seed 42) | 460 | -13.12% | [-29.03%, +3.38%] |
| Tần suất lịch sử | 460 | -2.75% | [-18.86%, +13.99%] |
| Không cược | 0 | 0.00% | [0.00%, 0.00%] |

## 2b. Ranking-only diagnostic (không threshold, không Kelly)

> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.

| Chỉ số | Giá trị |
|---|---:|
| Bets giả lập | 2920 |
| Hits | 737 |
| Hits trung bình/ngày | 1.0096 |
| ROI ranking-only | -7.45% |
| Bootstrap CI95 | [-13.86%, -0.80%] |
| P(ROI>0) | 1.4% |

## 2c. So sánh Fusion với Uniform Fusion

| Cấu hình | Brier ↓ | ECE ↓ | Precision@K ↑ | Top-K hits | Ranking ROI | CI95 |
|---|---:|---:|---:|---:|---:|---|
| MetaFusion | 0.182261 | 0.009445 | 0.2271 | 737 | -7.45% | [-13.86%, -0.80%] |
| Uniform fusion | 0.182867 | 0.018122 | 0.2315 | 752 | -5.57% | [-11.97%, +1.08%] |

- Uniform fusion P(ROI>0): 5.1%

## 3. Permutation test (5.000 lần)

Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.

| Chỉ số | Giá trị |
|---|---:|
| PnL quan sát | -1,431,000đ |
| PnL hoán vị trung bình | +115,261đ |
| PnL hoán vị 95% | [-2,025,000đ, +2,331,000đ] |
| p-value một phía | 0.9268 |

## 4. Ablation leaderboard cùng mức exposure

| Model | Lượt cược | ROI | ROI bootstrap 95% |
|---|---:|---:|---:|
| inverted_pairs | 460 | +3.62% | [-13.43%, +21.43%] |
| max_delay | 460 | -0.36% | [-16.88%, +17.36%] |
| day_of_week | 460 | -0.36% | [-16.48%, +16.10%] |
| ewma_probability | 460 | -1.16% | [-17.82%, +16.45%] |
| poisson_estimator | 460 | -4.35% | [-20.65%, +12.94%] |
| frequency_momentum | 460 | -5.14% | [-21.15%, +11.84%] |
| lightgbm_classifier | 460 | -5.94% | [-22.50%, +11.27%] |
| loto_repeat | 460 | -6.74% | [-22.85%, +9.84%] |
| bayesian_predictor | 460 | -7.54% | [-23.50%, +8.64%] |
| markov_chain | 460 | -9.13% | [-24.84%, +7.01%] |
| conditional_probability | 460 | -17.90% | [-33.06%, -2.00%] |

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
| 2024-09-05 | 57 | 3 | +270,000đ | SKIP | +0đ | 1.00 |
| 2026-05-29 | 46, 14, 98 | 3 | +216,000đ | 91(7đ) | -194,131đ | 1.00 |
| 2025-02-13 | 51 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2025-04-20 | 50 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2025-08-13 | 19 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2026-03-31 | 91 | 2 | +171,000đ | SKIP | +0đ | 1.00 |
| 2024-11-10 | 82, 53 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2025-04-14 | 69, 25 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2025-11-24 | 62, 22 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2026-01-10 | 01, 14 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2026-01-31 | 08, 58 | 2 | +144,000đ | SKIP | +0đ | 1.00 |
| 2025-04-16 | 50, 91, 61 | 2 | +117,000đ | SKIP | +0đ | 1.00 |
| 2024-07-22 | 08 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-08-01 | 89 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
| 2024-08-02 | 27 | 1 | +72,000đ | SKIP | +0đ | 1.00 |
