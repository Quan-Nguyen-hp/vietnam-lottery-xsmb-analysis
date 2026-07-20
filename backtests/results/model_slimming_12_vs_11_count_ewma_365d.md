# Báo cáo Model Slimming — Ablation ngoài mẫu toàn diện

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 00:00:00 đến 2026-07-15 00:00:00)
- **Tham số**: top_k=4 | min_prob=0.31 | min_conf=0.45
- **Thời gian chạy**: 195.6s

## 1. Bảng đóng góp ngoài mẫu từng model

| Model | Brier ↓ | LogLoss ↓ | ECE ↓ | Prec@K ↑ | Bets | Hits | ROI | CI95 | P(ROI>0) | Brier>Med | ROI<Freq | Trạng thái |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|:---:|:---:|:---:|
| poisson_estimator | 0.369519 | 1.046526 | 0.388928 | 0.2555 | 169 | 52 | +12.82% | [-15.21%, +43.04%] | 80.5% | Yes | No | ✅ GIỮ |
| inverted_pairs | 0.208479 | 0.696799 | 0.147750 | 0.2336 | 169 | 51 | +10.65% | [-17.28%, +40.43%] | 76.0% | Yes | No | ✅ GIỮ |
| frequency_momentum | 0.188173 | 0.569029 | 0.057502 | 0.2500 | 169 | 46 | -0.20% | [-27.15%, +28.45%] | 48.0% | Yes | Yes | ⚠️ ỨNG VIÊN |
| count_ewma_poisson | 0.182358 | 0.551382 | 0.005438 | 0.2452 | 169 | 45 | -2.37% | [-31.74%, +28.45%] | 42.5% | No | Yes | ✅ GIỮ |
| loto_repeat | 0.201019 | 0.638276 | 0.122201 | 0.2288 | 169 | 43 | -6.71% | [-32.70%, +20.73%] | 30.5% | Yes | Yes | ⚠️ ỨNG VIÊN |
| markov_chain | 0.181822 | 0.549858 | 0.000113 | 0.2233 | 169 | 42 | -8.88% | [-34.70%, +18.75%] | 25.5% | No | Yes | ✅ GIỮ |
| day_of_week | 0.181950 | 0.550202 | 0.000431 | 0.2308 | 169 | 42 | -8.88% | [-33.73%, +17.25%] | 24.4% | No | Yes | ✅ GIỮ |
| ewma_probability | 0.186050 | 0.562113 | 0.045229 | 0.2390 | 169 | 42 | -8.88% | [-34.68%, +18.08%] | 25.0% | No | Yes | ✅ GIỮ |
| lightgbm_classifier | 0.181724 | 0.549588 | 0.000143 | 0.2404 | 169 | 41 | -11.05% | [-39.28%, +19.33%] | 22.3% | No | Yes | ✅ GIỮ |
| max_delay | 0.226229 | 1.524793 | 0.164900 | 0.2473 | 169 | 39 | -15.38% | [-39.26%, +9.58%] | 10.8% | Yes | Yes | ⚠️ ỨNG VIÊN |
| bayesian_predictor | 0.197583 | 0.640953 | 0.082675 | 0.2425 | 169 | 38 | -17.55% | [-41.24%, +8.39%] | 8.7% | Yes | Yes | ⚠️ ỨNG VIÊN |
| conditional_probability | 0.181851 | 0.549934 | 0.000046 | 0.2308 | 169 | 29 | -37.08% | [-59.26%, -12.32%] | 0.2% | No | Yes | ✅ GIỮ |

- **Median Brier**: 0.187112
- **Frequency baseline ROI**: +6.31%

### Tiêu chí sàng lọc model
1. **ROI < frequency baseline** (+6.31%) — không tốt hơn chọn số theo tần suất
2. **Brier > median Brier** — xác suất kém chất lượng hơn trung bình
3. Cả 2 điều kiện chỉ tạo **ứng viên loại**; chỉ áp dụng khi CI95 của chênh lệch ROI ghép cặp Pruned-fixed − Full > 0.

### Models ứng viên loại: **['count_ewma_poisson']**

### Models đề xuất giữ: **['poisson_estimator', 'inverted_pairs', 'frequency_momentum', 'loto_repeat', 'markov_chain', 'day_of_week', 'ewma_probability', 'lightgbm_classifier', 'max_delay', 'bayesian_predictor', 'conditional_probability']**

## 2. So sánh baselines

| Baseline | Bets | Hits | ROI | CI95 |
|---|---:|---:|---:|---:|
| random | 169 | 40 | -13.21% | [-40.00%, +15.12%] |
| frequency | 169 | 49 | +6.31% | [-22.03%, +36.95%] |
| ensemble_12 | 169 | 45 | -2.37% | [-29.49%, +27.42%] |

## 3. Phân tích tương quan

- **Spearman rank (Brier vs ROI)**: ρ = 0.437, p = 0.1558
- Không có bằng chứng về tương quan có ý nghĩa thống kê giữa Brier và ROI trong mẫu này.

## 4. Trọng số Fusion hiện tại

| Model | Trọng số | Giữ/Loại |
|---|:---:|:---:|
| poisson_estimator | 6.54% | Giữ |
| inverted_pairs | 8.20% | Giữ |
| frequency_momentum | 8.74% | Giữ |
| count_ewma_poisson | 8.99% | Ứng viên loại |
| loto_repeat | 8.38% | Giữ |
| markov_chain | 8.82% | Giữ |
| day_of_week | 8.88% | Giữ |
| ewma_probability | 8.78% | Giữ |
| lightgbm_classifier | 8.78% | Giữ |
| max_delay | 6.54% | Giữ |
| bayesian_predictor | 8.58% | Giữ |
| conditional_probability | 8.76% | Giữ |

## 5. So sánh Pruned vs Full Ensemble

> **Phương pháp**: Full baseline từ main walk-forward loop. Pruned dùng trọng số cuối kỳ rồi redistribute trên models giữ lại.
> **Cảnh báo**: Pruned/Pruned-fixed chỉ là diagnostic fixed-weight; kết luận 12-vs-11 phải dùng hai lần chạy full walk-forward độc lập.

### Full (12 models)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 169 |
| Tổng nháy | 45 |
| ROI | -2.37% |
| CI95 | [-29.49%, +27.42%] |
| P(ROI>0) | 42.9% |

### Pruned (11 models)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 286 |
| Tổng nháy | 79 |
| ROI | +1.28% |
| CI95 | [-18.23%, +21.79%] |
| P(ROI>0) | 54.4% |

### Pruned-fixed (11 models, same exposure)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 169 |
| Tổng nháy | 41 |
| ROI | -11.05% |
| CI95 | [-37.55%, +18.43%] |
| P(ROI>0) | 21.7% |

### Kiểm định chênh lệch ghép cặp Pruned-fixed − Full

| Chỉ số | Giá trị |
|---|---:|
| ΔROI quan sát | -8.68% |
| Bootstrap CI95 của ΔROI | [-18.00%, -1.98%] |
| P(ΔROI>0) | 0.0% |
| Quyết định loại model | KHÔNG ĐẠT |

- CI95 của ΔROI vẫn chứa 0; chưa đủ bằng chứng để loại ứng viên khỏi production.

### Phân tích exposure gap

- Full ensemble cược trung bình 169/365 ngày = 0.46 bets/ngày
- Pruned (tự do) cược 286/365 ngày = 0.78 bets/ngày
- Pruned-fixed (cùng exposure) cược 169/365 ngày
- Bỏ ứng viên `count_ewma_poisson` làm fusion probability spread khác → confidence/probability thresholds cho qua nhiều hơn → số bets tăng 117 lượt.
- Pruned-fixed đo thay đổi xếp hạng ở cùng exposure; quyết định chỉ dựa trên CI95 của ΔROI ghép cặp ở trên.

