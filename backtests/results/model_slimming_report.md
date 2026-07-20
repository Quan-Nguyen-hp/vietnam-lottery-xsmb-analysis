# Báo cáo Model Slimming — Ablation ngoài mẫu toàn diện

- **Kỳ kiểm thử**: 365 ngày (2025-07-12 00:00:00 đến 2026-07-15 00:00:00)
- **Tham số**: top_k=4 | min_prob=0.31 | min_conf=0.45
- **Thời gian chạy**: 195.2s

## 1. Bảng đóng góp ngoài mẫu từng model

| Model | Brier ↓ | LogLoss ↓ | ECE ↓ | Prec@K ↑ | Bets | Hits | ROI | CI95 | P(ROI>0) | Brier>Med | ROI<Freq | Loại? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|:---:|:---:|:---:|
| bayesian_predictor | 0.197583 | 0.640953 | 0.082675 | 0.2425 | 112 | 38 | +24.40% | [-14.11%, +66.39%] | 88.4% | Yes | No | ⚠️ |
| frequency_momentum | 0.188173 | 0.569029 | 0.057502 | 0.2500 | 112 | 36 | +17.86% | [-23.00%, +61.76%] | 79.2% | No | No | ⚠️ |
| inverted_pairs | 0.208479 | 0.696799 | 0.147750 | 0.2336 | 112 | 34 | +11.31% | [-24.02%, +50.53%] | 71.7% | Yes | No | ⚠️ |
| lightgbm_classifier | 0.196017 | 0.607713 | 0.065751 | 0.2411 | 112 | 32 | +4.76% | [-29.86%, +43.48%] | 58.8% | No | No | ⚠️ |
| max_delay | 0.226229 | 1.524793 | 0.164900 | 0.2473 | 112 | 31 | +1.49% | [-30.16%, +35.09%] | 51.8% | Yes | No | ⚠️ |
| conditional_probability | 0.181851 | 0.549934 | 0.000046 | 0.2308 | 112 | 31 | +1.49% | [-33.93%, +39.20%] | 51.8% | No | No | ⚠️ |
| poisson_estimator | 0.369519 | 1.046526 | 0.388928 | 0.2555 | 112 | 31 | +1.49% | [-30.33%, +34.91%] | 52.6% | Yes | No | ⚠️ |
| day_of_week | 0.181950 | 0.550202 | 0.000431 | 0.2308 | 112 | 31 | +1.49% | [-31.86%, +37.07%] | 51.8% | No | No | ⚠️ |
| markov_chain | 0.181822 | 0.549858 | 0.000113 | 0.2233 | 112 | 29 | -5.06% | [-38.89%, +32.70%] | 38.0% | No | Yes | ⚠️ |
| loto_repeat | 0.201019 | 0.638276 | 0.122201 | 0.2288 | 112 | 29 | -5.06% | [-38.89%, +32.70%] | 38.0% | Yes | Yes | ❌ LOẠI |
| ewma_probability | 0.186050 | 0.562113 | 0.045229 | 0.2390 | 112 | 29 | -5.06% | [-39.45%, +32.31%] | 37.7% | No | Yes | ⚠️ |

- **Median Brier**: 0.196017
- **Frequency baseline ROI**: +1.49%

### Tiêu chí loại model
1. **ROI < frequency baseline** (+1.49%) — không tốt hơn chọn số theo tần suất
2. **Brier > median Brier** — xác suất kém chất lượng hơn trung bình
3. Cả 2 điều kiện → **LOẠI** (❌)

### Models đề xuất loại: **['loto_repeat']**

### Models đề xuất giữ: **['bayesian_predictor', 'frequency_momentum', 'inverted_pairs', 'lightgbm_classifier', 'max_delay', 'conditional_probability', 'poisson_estimator', 'day_of_week', 'markov_chain', 'ewma_probability']**

## 2. So sánh baselines

| Baseline | Bets | Hits | ROI | CI95 |
|---|---:|---:|---:|---:|
| random | 112 | 27 | -11.61% | [-40.96%, +21.07%] |
| frequency | 112 | 31 | +1.49% | [-32.21%, +38.30%] |
| ensemble_11 | 112 | 29 | -5.06% | [-38.89%, +30.45%] |

## 3. Phân tích tương quan

- **Spearman rank (Brier vs ROI)**: ρ = 0.272, p = 0.4176
-  Mô hình có Brier thấp (xác suất chính xác) thường có ROI cao hơn.

## 4. Trọng số Fusion hiện tại

| Model | Trọng số | Giữ/Loại |
|---|:---:|:---:|
| bayesian_predictor | 9.17% | Giữ |
| frequency_momentum | 9.34% | Giữ |
| inverted_pairs | 8.77% | Giữ |
| lightgbm_classifier | 12.12% | Giữ |
| max_delay | 6.99% | Giữ |
| conditional_probability | 9.37% | Giữ |
| poisson_estimator | 6.99% | Giữ |
| day_of_week | 9.49% | Giữ |
| markov_chain | 9.42% | Giữ |
| loto_repeat | 8.96% | Loại |
| ewma_probability | 9.38% | Giữ |

## 5. So sánh Pruned vs Full Ensemble

> **Phương pháp**: Full baseline từ main walk-forward loop (chính xác). Pruned dùng dynamic weights từ main backtest, redistribute weights proportionally trên models giữ lại.
> **Pruned-fixed**: Cùng số bets/ngày với full ensemble (same exposure), chỉ thay fusion — so sánh công bằng nhất.

### Full (11 models)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 112 |
| Tổng nháy | 29 |
| ROI | -5.06% |
| CI95 | [-38.89%, +30.45%] |
| P(ROI>0) | 37.3% |

### Pruned (10 models)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 440 |
| Tổng nháy | 120 |
| ROI | +0.00% |
| CI95 | [-16.49%, +17.07%] |
| P(ROI>0) | 49.2% |

### Pruned-fixed (10 models, same exposure)

| Metric | Giá trị |
|---|---:|
| Tổng cược | 112 |
| Tổng nháy | 32 |
| ROI | +4.76% |
| CI95 | [-26.67%, +38.44%] |
| P(ROI>0) | 60.3% |

### Phân tích exposure gap

- Full ensemble cược trung bình 112/365 ngày = 0.31 bets/ngày
- Pruned (tự do) cược 440/365 ngày = 1.21 bets/ngày
- Pruned-fixed (cùng exposure) cược 112/365 ngày
- **Loại `loto_repeat` làm fusion probability spread khác** → confidence/probability thresholds cho qua nhiều hơn → số bets tăng 328 lượt.
- Khi khóa cùng exposure, ROI pruned-fixed cho thấy **độ chính xác của số được chọn** thay vì chỉ do nhiều bets hơn.

