# Vietnam Lottery (XSMB) Analysis & Prediction v2.0

Sử dụng GitHub Action để tự động hóa thu thập và phân tích kết quả xổ số hàng ngày của Việt Nam, kết hợp các mô hình học máy nâng cao để dự báo xác suất xuất hiện của các số loto.

Dự án này được tạo bởi [Khiêm Đoàn](https://github.com/khiemdoan) và phát triển nâng cấp hệ thống dự đoán học máy v2.0. Mục đích chính là nghiên cứu khoa học dữ liệu và thống kê số học.

---

## 📊 Bảng kết quả xổ số hàng ngày (Cập nhật tự động)

| Lottery (Xổ số) | Loto (Lô tô) |
| :------------: | :----------: |
| <table><tr><td>Date (Ngày)</td><td>13-07-2026</td></tr><tr><td>Special (Giải đặc biệt)</td><td>74299</td></tr><tr><td>First (Giải nhất)</td><td>93956</td></tr><tr><td>Second (Giải nhì)</td><td>52860, 61224</td></tr><tr><td rowspan="2">Third (Giải ba)</td><td>56764, 65767, 54685</td></tr><tr><td>07842, 95097, 33930</td></tr><tr><td>Fourth (Giải tư)</td><td>6331, 4632, 6150, 0553</td></tr><tr><td rowspan="2">Fifth (Giải năm)</td><td>9892, 1455, 2364</td></tr><tr><td>0413, 0001, 5503</td></tr><tr><td>Sixth (Giải sáu)</td><td>889, 268, 080</td></tr><tr><td>Seventh (Giải bảy)</td><td>97, 76, 05, 75</td></tr></table> | <table><tr><td>First (Đầu)</td><td>Last (Đuôi)</td></tr><tr><td>0</td><td>1, 3, 5</td></tr><tr><td>1</td><td>3</td></tr><tr><td>2</td><td>4</td></tr><tr><td>3</td><td>0, 1, 2</td></tr><tr><td>4</td><td>2</td></tr><tr><td>5</td><td>0, 3, 5, 6</td></tr><tr><td>6</td><td>0, 4, 4, 7, 8</td></tr><tr><td>7</td><td>5, 6</td></tr><tr><td>8</td><td>0, 5, 9</td></tr><tr><td>9</td><td>2, 7, 7, 9</td></tr></table> |

---

## 📁 Dữ liệu (Data)

| Loại dữ liệu | CSV | JSON | Parquet |
|---|---|---|---|
| **Raw (Dữ liệu gốc)** | [xsmb.csv](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.csv) | [xsmb.json](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.json) | [xsmb.parquet](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.parquet) |
| **2-digits (Hai số cuối)** | [xsmb-2-digits.csv](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb-2-digits.csv) | [xsmb-2-digits.json](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb-2-digits.json) | [xsmb-2-digits.parquet](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb-2-digits.parquet) |
| **Sparse (Ma trận thưa)** | [xsmb-sparse.csv](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb-sparse.csv) | [xsmb-sparse.json](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb-sparse.json) | [xsmb-sparse.parquet](https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb-sparse.parquet) |

---

---

## 🏗️ XPIS v1.0 — Kiến trúc 8 Tầng (Architecture Freeze)

**XPIS** (XSMB Prediction Intelligence System) là kiến trúc thế hệ mới, được thiết kế để phát triển bền vững trong nhiều năm mà không cần thay đổi nền tảng.

```
Raw Data (xsmb-2-digits.csv)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 1 — DATA LAYER                               │
│  DataLoader: load → validate → sparse matrix        │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2 — EVIDENCE LAYER  [IMMUTABLE]              │
│  EvidenceBuilder → EvidenceStore (Parquet)          │
│  100 số × raw observations: delay, freq, markov...  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 3 — FEATURE LAYER  [VERSIONED]               │
│  6 Extractors → Hybrid FeatureStore                 │
│  RAM cache (daily) + Parquet snapshots (backtest)   │
│  FeatureStore/v1/snapshots/YYYY-MM-DD.parquet       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 4 — PROBABILITY MODEL LAYER  (10 Models)     │
│  Mỗi model độc lập, có thể backtest riêng           │
│                                                     │
│  ①MaxDelay  ②Conditional  ③Markov  ④Momentum       │
│  ⑤Poisson   ⑥Repeat      ⑦Inverted ⑧DayOfWeek     │
│  ⑨Bayesian  ⑩EWMA Multi-scale                      │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 5 — META LEARNING LAYER                      │
│  LightGBM: 250 features + 10 probabilities → P(num) │
│  → ProbabilityCalibrator (Platt / Isotonic)         │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 6 — DECISION INTELLIGENCE  [QUAN TRỌNG NHẤT] │
│  Probability + Confidence + Kelly → BET / SKIP      │
│  ConfidenceEngine: đo đồng thuận giữa 10 models     │
│  KellyCriterion: tối ưu vốn theo odds & confidence  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 7 — EVALUATION LAYER                         │
│  ROI · WinRate · LogLoss · Brier · MDD · Drift      │
│  Daily / Weekly / Monthly Reports                   │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 8 — RESEARCH LAYER  [CÁCH LY PRODUCTION]     │
│  Idea → Hypothesis → Backtest → Validate → Approve  │
└─────────────────────────────────────────────────────┘
```

### Feature Catalog (Layer 3)

| Nhóm | Features | Số lượng |
|---|---|:---:|
| **Delay** | delay, delay², zscore, rank, percentile, momentum, velocity, acceleration, volatility, ewma, mean, std | 12 |
| **Frequency** | freq_3d/7d/14d/30d/60d/90d/120d/180d/365d, momentum_short/long/ultra, ratio, std, skew, kurt, all | 18 |
| **Markov** | order1, order2, entropy, persistence, stationary, run_length | 6 |
| **Bayesian** | cond_prob_yesterday, cond_prob_head_cam, cond_prob_tail_cam, cond_prob_inverted, cond_prob_mirror, bayesian_posterior | 7 |
| **Pair** | is_twin, head, tail, digit_sum/diff/product, dist_to_inverted/mirror, is_even, head_even, tail_even | 12 |
| **Time** | weekday, day_of_month, month, is_weekend, week_of_month, quarter, sin/cos encodings, holiday flags | 14 |
| **Tổng** | | **~78** |

### Versioning (Reproducibility)

Mỗi lần Backtest / Prediction đều ghi rõ version của từng layer:

```json
{
  "date": "2026-07-14",
  "evidence_version": "v1.0",
  "feature_version": "v1",
  "model_version": "v1.0",
  "decision_version": "v1.0"
}
```

---

## 📅 Roadmap & Tiến độ (XPIS v1.0)

| Giai đoạn | Nội dung | Trạng thái |
| :---: |---|:---:|
| **Phase 1** | Scaffold 8 layers: Data, Evidence, Feature, Probability, Meta, Decision, Evaluation, Research | ✅ Hoàn thành |
| **Phase 2** | Implement Evidence Layer đầy đủ (EvidenceBuilder, EvidenceStore) | ✅ Hoàn thành |
| **Phase 3** | Implement Feature Layer đầy đủ (6 extractors + Hybrid FeatureStore v1) | ✅ Hoàn thành |
| **Phase 4** | Implement Model 9 (Bayesian Predictor) & Model 10 (EWMA Multi-scale) | ✅ Hoàn thành |
| **Phase 5** | Migrate Models 1–8 từ `src/methods/` → `src/probability/` | 🔄 Tiếp theo |
| **Phase 6** | Train LightGBM Meta Learner trên XPIS FeatureStore | 📝 Lên kế hoạch |
| **Phase 7** | Probability Calibration + ConfidenceEngine + DecisionEngine | 📝 Lên kế hoạch |
| **Phase 8** | Walk-forward Backtest + EvaluationMetrics + Report | 📝 Lên kế hoạch |
| **Research** | Layer 8: Experiment tracking, hypothesis testing framework | 📝 Lên kế hoạch |

---

## ⚙️ Cài đặt & Sử dụng (XPIS v1.0)

```sh
pip install lightgbm scikit-learn numpy pandas pyarrow
```

### Chạy Integration Test
```sh
python tests/test_xpis_integration.py
```

### Chạy Mini Backtest (90 ngày, Model 9 vs Model 10)
```sh
python tests/test_backtest_90d.py
```

### Dự đoán hàng ngày (legacy)
```sh
python daily_predict.py --method lightgbm --top-k 10
```


Hệ thống v2.0 đã chuyển đổi từ mô hình quyết định theo ngưỡng ma trận thủ công sang kiến trúc **Meta-Learning** thông minh:

```
[Dữ liệu lịch sử 1000 ngày]
           │
           ▼
┌──────────────────────────────────────┐
│    FeatureEngine (100+ đặc trưng)     │
│  - Delay, Freq Multi-Timeframe       │
│  - Markov Chain, Bayesian Co-occur   │
│  - Cặp số lộn, Kép, Gương, Thời gian │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│     LightGBM Meta-Predictor          │
│  - Tự học quy luật và sự kết hợp     │
│  - Dự báo xác suất (Probability)     │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│    Probability Calibration & Kelly   │
│  - Hiệu chỉnh xác suất thực tế       │
│  - Đi tiền Kelly tối ưu hóa lợi nhuận│
└──────────────────────────────────────┘
```

### 1. Tập đặc trưng mở rộng (Feature Catalog)
*   **Delay Features:** Khoảng trễ (lô khan), Z-score trễ, Percentile trễ, Rank trễ, Volatility trễ, EWMA trễ.
*   **Multi-Timeframe Frequency:** Tần suất xuất hiện trên các cửa sổ trượt ngắn/trung/dài hạn (3, 7, 14, 30, 60, 90, 120, 180, 365 ngày). Động lượng tần suất.
*   **Markov Chain:** Xác suất chuyển đổi trạng thái bậc 1 & bậc 2, Entropy chuyển đổi trạng thái, độ bền vững trạng thái (persistence).
*   **Bayesian & Bạc nhớ:** Ma trận co-occurrence $100 \times 100$, xác suất có điều kiện dựa trên Đầu/Đuôi câm hôm trước, tổng số đề đặc biệt.
*   **Cặp số & Thuộc tính số:** Số kép, Cặp lộn (Inverted pair), Số gương (Mirror pair), khoảng cách số học.
*   **Thời gian:** Thứ trong tuần, ngày trong tháng, tháng, cuối tuần.

### 2. Mô hình Meta Predictor (LightGBM)
Mô hình LightGBM sẽ tự động học các liên kết đặc trưng phi tuyến tính phức tạp (Ví dụ: Trễ Z-score cao + Động lượng ngắn hạn tăng mạnh + Markov entropy thấp $\to$ Xác suất nổ = 68%), loại bỏ việc thiết lập quy tắc thủ công.

---

## 📅 Roadmap & Tiến độ công việc (Roadmap & Progress Tracker)

Dưới đây là lịch trình phát triển XSMB v2.0 và trạng thái thực tế:

| Phase | Mục tiêu | Mức độ ưu tiên | Trạng thái | Chi tiết |
| :---: |---| :---: | :---: |---|
| **Phase 1** | Mở rộng Feature Catalog lên 100+ đặc trưng | ⭐⭐⭐⭐⭐ |  Đang kiểm thử | Trích xuất đặc trưng Delay, Freq, Markov, Bayesian, Pairs, Time. |
| **Phase 2** | Multi-Timeframe Feature Engine | ⭐⭐⭐⭐⭐ |  Đang kiểm thử | Hỗ trợ tính toán song song các cửa sổ thời gian từ 3 ngày đến 365 ngày. |
| **Phase 3** | LightGBM Meta Predictor & Backtester | ⭐⭐⭐⭐⭐ |  Đang kiểm thử | Tích hợp LightGBM Classifier và hệ thống Backtest walk-forward. |
| **Phase 4** | Probability Calibration (Hiệu chỉnh xác suất) | ⭐⭐⭐⭐ | 📝 Lên kế hoạch | Platt Scaling & Isotonic Regression khớp xác suất mô hình với thực tế. |
| **Phase 5** | Dynamic Context-Aware Ensemble | ⭐⭐⭐⭐ | 📝 Lên kế hoạch | Tự động thay đổi trọng số mô hình thành phần theo điều kiện thị trường. |
| **Phase 6** | Confidence & Explainability Engine | ⭐⭐⭐⭐ | 📝 Lên kế hoạch | Đánh giá độ tin cậy và lý do khuyên chọn số của mô hình. |
| **Phase 7** | Kelly Position Sizing (Quản lý vốn Kelly) | ⭐⭐⭐ | 📝 Lên kế hoạch | Tự động phân bổ số vốn đánh cho từng số theo xác suất tối ưu. |
| **Phase 8** | Online Learning & Daily Retraining | ⭐⭐⭐ | 📝 Lên kế hoạch | Huấn luyện lại mô hình tự động hàng ngày khi có kết quả mới. |

---

## ⚙️ Cài đặt & Sử dụng (XSMB v2.0)

Yêu cầu cài đặt thêm các thư viện học máy:
```sh
pip install lightgbm scikit-learn numpy pandas
```

### 1. Chạy Backtest hồi cứu mô hình
```sh
python backtests/lightgbm_backtest.py --days 365 --top-k 10
```

### 2. Dự đoán hàng ngày (với LightGBM)
```sh
python daily_predict.py --method lightgbm --top-k 10
```

### 3. Cập nhật kết quả & Tự động cập nhật mô hình
```sh
python daily_update.py
```