# Exact Production Pipeline Backtest

- **Kỳ kiểm thử**: 30 ngày (2026-06-16 đến 2026-07-15)
- **Pipeline**: gọi trực tiếp `daily_predict.run_shared_prediction_pipeline()` mỗi ngày
- **Top K**: 4

| Chỉ số | Giá trị |
|---|---:|
| Số lượt cược | 1 |
| Số nháy trúng | 0 |
| ROI | -100.00% |
| ROI bootstrap 95% | [-100.00%, +0.00%] |
| Xác suất ROI dương (bootstrap) | 0.0% |

> Đây là exact-mode: calibration, fusion, EMA và quyết định dùng đúng code production; không ghi prediction log production.
