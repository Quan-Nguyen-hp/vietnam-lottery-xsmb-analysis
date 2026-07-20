# Báo cáo Model Slimming — Ablation ngoài mẫu toàn diện

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 00:00:00 đến 2026-07-15 00:00:00)
- **Tham số**: top_k=4 | min_prob=0.31 | min_conf=0.45
- **Thời gian chạy**: 158.6s

## 1. Bảng đóng góp ngoài mẫu từng model

| Model | Brier ↓ | LogLoss ↓ | ECE ↓ | Prec@K ↑ | Bets | Hits | ROI | CI95 | P(ROI>0) | Brier>Med | ROI<Freq | Trạng thái |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|:---:|:---:|:---:|
| frequency_momentum | 0.188173 | 0.569029 | 0.057502 | 0.2500 | 16 | 7 | +60.42% | [-59.26%, +208.77%] | 81.5% | No | No | ✅ GIỮ |
| poisson_estimator | 0.369519 | 1.046526 | 0.388928 | 0.2555 | 16 | 5 | +14.58% | [-77.08%, +129.17%] | 57.0% | Yes | No | ✅ GIỮ |
| bayesian_predictor | 0.197583 | 0.640953 | 0.082675 | 0.2425 | 16 | 5 | +14.58% | [-78.43%, +138.33%] | 57.3% | Yes | No | ✅ GIỮ |
| ewma_probability | 0.186050 | 0.562113 | 0.045229 | 0.2390 | 16 | 5 | +14.58% | [-78.43%, +137.25%] | 57.0% | No | No | ✅ GIỮ |
| lightgbm_classifier | 0.181724 | 0.549588 | 0.000143 | 0.2404 | 16 | 5 | +14.58% | [-79.63%, +137.28%] | 57.3% | No | No | ✅ GIỮ |
| max_delay | 0.226229 | 1.524793 | 0.164900 | 0.2473 | 16 | 4 | -8.33% | [-100.00%, +103.70%] | 38.6% | Yes | No | ✅ GIỮ |
| day_of_week | 0.181950 | 0.550202 | 0.000431 | 0.2308 | 16 | 4 | -8.33% | [-79.63%, +83.33%] | 39.2% | No | No | ✅ GIỮ |
| inverted_pairs | 0.208479 | 0.696799 | 0.147750 | 0.2336 | 16 | 3 | -31.25% | [-100.00%, +83.33%] | 24.6% | Yes | No | ✅ GIỮ |
| conditional_probability | 0.181851 | 0.549934 | 0.000046 | 0.2308 | 16 | 2 | -54.17% | [-100.00%, +15.79%] | 5.7% | No | Yes | ✅ GIỮ |
| markov_chain | 0.181822 | 0.549858 | 0.000113 | 0.2233 | 16 | 2 | -54.17% | [-100.00%, +22.22%] | 5.8% | No | Yes | ✅ GIỮ |
| loto_repeat | 0.201019 | 0.638276 | 0.122201 | 0.2288 | 16 | 2 | -54.17% | [-100.00%, +22.22%] | 5.8% | Yes | Yes | ⚠️ ỨNG VIÊN |

- **Median Brier**: 0.188173
- **Frequency baseline ROI**: -31.25%

### Tiêu chí sàng lọc model
1. **ROI < frequency baseline** (-31.25%) — không tốt hơn chọn số theo tần suất
2. **Brier > median Brier** — xác suất kém chất lượng hơn trung bình
3. Cả 2 điều kiện chỉ tạo **ứng viên loại**; chỉ áp dụng khi CI95 của chênh lệch ROI ghép cặp Pruned-fixed − Full > 0.

### Models ứng viên loại: **['loto_repeat']**

### Models đề xuất giữ: **['frequency_momentum', 'poisson_estimator', 'bayesian_predictor', 'ewma_probability', 'lightgbm_classifier', 'max_delay', 'day_of_week', 'inverted_pairs', 'conditional_probability', 'markov_chain']**

## 2. So sánh baselines

| Baseline | Bets | Hits | ROI | CI95 |
|---|---:|---:|---:|---:|
| random | 16 | 6 | +37.50% | [-54.17%, +152.08%] |
| frequency | 16 | 3 | -31.25% | [-100.00%, +46.67%] |
| ensemble_11 | 16 | 3 | -31.25% | [-100.00%, +52.78%] |

## 3. Phân tích tương quan

- **Spearman rank (Brier vs ROI)**: ρ = 0.122, p = 0.7199
- Không có bằng chứng về tương quan có ý nghĩa thống kê giữa Brier và ROI trong mẫu này.

## 4. Trọng số Fusion hiện tại

| Model | Trọng số | Giữ/Loại |
|---|:---:|:---:|
| frequency_momentum | 9.60% | Giữ |
| poisson_estimator | 7.18% | Giữ |
| bayesian_predictor | 9.43% | Giữ |
| ewma_probability | 9.65% | Giữ |
| lightgbm_classifier | 9.65% | Giữ |
| max_delay | 7.19% | Giữ |
| day_of_week | 9.75% | Giữ |
| inverted_pairs | 9.01% | Giữ |
| conditional_probability | 9.63% | Giữ |
| markov_chain | 9.69% | Giữ |
| loto_repeat | 9.21% | Ứng viên loại |

## 5. So sánh Pruned vs Full Ensemble

> **Phương pháp**: Full baseline từ main walk-forward loop (chính xác). Pruned dùng dynamic weights từ main backtest, redistribute weights proportionally trên models giữ lại.
> **Pruned-fixed**: Cùng số bets/ngày với full ensemble (same exposure), chỉ thay fusion — so sánh công bằng nhất.

### Full (11 models)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 16 |
| Tổng nháy | 3 |
| ROI | -31.25% |
| CI95 | [-100.00%, +52.78%] |
| P(ROI>0) | 19.6% |

### Pruned (10 models)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 326 |
| Tổng nháy | 80 |
| ROI | -10.02% |
| CI95 | [-28.77%, +10.11%] |
| P(ROI>0) | 15.3% |

### Pruned-fixed (10 models, same exposure)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 16 |
| Tổng nháy | 5 |
| ROI | +14.58% |
| CI95 | [-66.67%, +109.52%] |
| P(ROI>0) | 61.8% |

### Kiểm định chênh lệch ghép cặp Pruned-fixed − Full

| Chỉ số | Giá trị |
|---|---:|
| ΔROI quan sát | +45.83% |
| Bootstrap CI95 của ΔROI | [+0.00%, +115.79%] |
| P(ΔROI>0) | 86.5% |
| Quyết định loại model | KHÔNG ĐẠT |

- CI95 của ΔROI vẫn chứa 0; chưa đủ bằng chứng để loại ứng viên khỏi production.

### Phân tích exposure gap

- Full ensemble cược trung bình 16/365 ngày = 0.04 bets/ngày
- Pruned (tự do) cược 326/365 ngày = 0.89 bets/ngày
- Pruned-fixed (cùng exposure) cược 16/365 ngày
- Bỏ ứng viên `loto_repeat` làm fusion probability spread khác → confidence/probability thresholds cho qua nhiều hơn → số bets tăng 310 lượt.
- Pruned-fixed đo thay đổi xếp hạng ở cùng exposure; quyết định chỉ dựa trên CI95 của ΔROI ghép cặp ở trên.

