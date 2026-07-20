# Báo cáo Model Slimming — Ablation ngoài mẫu toàn diện

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 00:00:00 đến 2026-07-15 00:00:00)
- **Tham số**: top_k=4 | min_prob=0.31 | min_conf=0.45
- **Thời gian chạy**: 174.4s

## 1. Bảng đóng góp ngoài mẫu từng model

| Model | Brier ↓ | LogLoss ↓ | ECE ↓ | Prec@K ↑ | Bets | Hits | ROI | CI95 | P(ROI>0) | Brier>Med | ROI<Freq | Trạng thái |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|:---:|:---:|:---:|
| poisson_estimator | 0.369519 | 1.046526 | 0.388928 | 0.2555 | 250 | 68 | -0.27% | [-21.72%, +22.70%] | 48.5% | Yes | Yes | ⚠️ ỨNG VIÊN |
| frequency_momentum | 0.188173 | 0.569029 | 0.057502 | 0.2500 | 250 | 66 | -3.20% | [-25.50%, +19.54%] | 38.3% | No | Yes | ✅ GIỮ |
| loto_repeat | 0.201019 | 0.638276 | 0.122201 | 0.2288 | 250 | 66 | -3.20% | [-25.47%, +19.70%] | 38.1% | Yes | Yes | ⚠️ ỨNG VIÊN |
| markov_chain | 0.181822 | 0.549858 | 0.000113 | 0.2233 | 250 | 65 | -4.67% | [-26.36%, +17.52%] | 33.0% | No | Yes | ✅ GIỮ |
| ewma_probability | 0.186050 | 0.562113 | 0.045229 | 0.2390 | 250 | 65 | -4.67% | [-27.52%, +18.58%] | 33.4% | No | Yes | ✅ GIỮ |
| inverted_pairs | 0.208479 | 0.696799 | 0.147750 | 0.2336 | 250 | 64 | -6.13% | [-28.52%, +18.13%] | 29.9% | Yes | Yes | ⚠️ ỨNG VIÊN |
| day_of_week | 0.181950 | 0.550202 | 0.000431 | 0.2308 | 250 | 59 | -13.47% | [-33.47%, +7.39%] | 9.8% | No | Yes | ✅ GIỮ |
| lightgbm_classifier | 0.181724 | 0.549588 | 0.000143 | 0.2404 | 250 | 57 | -16.40% | [-37.65%, +6.11%] | 7.6% | No | Yes | ✅ GIỮ |
| max_delay | 0.226229 | 1.524793 | 0.164900 | 0.2473 | 250 | 56 | -17.87% | [-36.68%, +1.69%] | 3.6% | Yes | Yes | ⚠️ ỨNG VIÊN |
| bayesian_predictor | 0.197583 | 0.640953 | 0.082675 | 0.2425 | 250 | 56 | -17.87% | [-37.10%, +2.98%] | 4.4% | Yes | Yes | ⚠️ ỨNG VIÊN |
| conditional_probability | 0.181851 | 0.549934 | 0.000046 | 0.2308 | 250 | 55 | -19.33% | [-40.86%, +3.77%] | 5.0% | No | Yes | ✅ GIỮ |

- **Median Brier**: 0.188173
- **Frequency baseline ROI**: +2.67%

### Tiêu chí sàng lọc model
1. **ROI < frequency baseline** (+2.67%) — không tốt hơn chọn số theo tần suất
2. **Brier > median Brier** — xác suất kém chất lượng hơn trung bình
3. Cả 2 điều kiện chỉ tạo **ứng viên loại**; chỉ áp dụng khi CI95 của chênh lệch ROI ghép cặp Pruned-fixed − Full > 0.

### Models ứng viên loại: **['loto_repeat']**

### Models đề xuất giữ: **['poisson_estimator', 'frequency_momentum', 'markov_chain', 'ewma_probability', 'inverted_pairs', 'day_of_week', 'lightgbm_classifier', 'max_delay', 'bayesian_predictor', 'conditional_probability']**

## 2. So sánh baselines

| Baseline | Bets | Hits | ROI | CI95 |
|---|---:|---:|---:|---:|
| random | 250 | 67 | -1.73% | [-25.50%, +23.73%] |
| frequency | 250 | 70 | +2.67% | [-19.55%, +25.59%] |
| ensemble_11 | 250 | 68 | -0.27% | [-21.12%, +22.22%] |

## 3. Phân tích tương quan

- **Spearman rank (Brier vs ROI)**: ρ = 0.311, p = 0.3515
- Không có bằng chứng về tương quan có ý nghĩa thống kê giữa Brier và ROI trong mẫu này.

## 4. Trọng số Fusion hiện tại

| Model | Trọng số | Giữ/Loại |
|---|:---:|:---:|
| poisson_estimator | 7.18% | Giữ |
| frequency_momentum | 9.60% | Giữ |
| loto_repeat | 9.21% | Ứng viên loại |
| markov_chain | 9.69% | Giữ |
| ewma_probability | 9.65% | Giữ |
| inverted_pairs | 9.01% | Giữ |
| day_of_week | 9.75% | Giữ |
| lightgbm_classifier | 9.65% | Giữ |
| max_delay | 7.19% | Giữ |
| bayesian_predictor | 9.43% | Giữ |
| conditional_probability | 9.63% | Giữ |

## 5. So sánh Pruned vs Full Ensemble

> **Phương pháp**: Full baseline từ main walk-forward loop (chính xác). Pruned dùng dynamic weights từ main backtest, redistribute weights proportionally trên models giữ lại.
> **Pruned-fixed**: Cùng số bets/ngày với full ensemble (same exposure), chỉ thay fusion — so sánh công bằng nhất.

### Full (11 models)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 250 |
| Tổng nháy | 68 |
| ROI | -0.27% |
| CI95 | [-21.12%, +22.22%] |
| P(ROI>0) | 48.4% |

### Pruned (10 models)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 1081 |
| Tổng nháy | 284 |
| ROI | -3.67% |
| CI95 | [-14.29%, +7.51%] |
| P(ROI>0) | 25.3% |

### Pruned-fixed (10 models, same exposure)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 250 |
| Tổng nháy | 65 |
| ROI | -4.67% |
| CI95 | [-25.62%, +17.92%] |
| P(ROI>0) | 32.6% |

### Kiểm định chênh lệch ghép cặp Pruned-fixed − Full

| Chỉ số | Giá trị |
|---|---:|
| ΔROI quan sát | -4.40% |
| Bootstrap CI95 của ΔROI | [-15.03%, +5.87%] |
| P(ΔROI>0) | 16.1% |
| Quyết định loại model | KHÔNG ĐẠT |

- CI95 của ΔROI vẫn chứa 0; chưa đủ bằng chứng để loại ứng viên khỏi production.

### Phân tích exposure gap

- Full ensemble cược trung bình 250/365 ngày = 0.68 bets/ngày
- Pruned (tự do) cược 1081/365 ngày = 2.96 bets/ngày
- Pruned-fixed (cùng exposure) cược 250/365 ngày
- Bỏ ứng viên `loto_repeat` làm fusion probability spread khác → confidence/probability thresholds cho qua nhiều hơn → số bets tăng 831 lượt.
- Pruned-fixed đo thay đổi xếp hạng ở cùng exposure; quyết định chỉ dựa trên CI95 của ΔROI ghép cặp ở trên.

