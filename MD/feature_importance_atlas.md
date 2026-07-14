# Feature Importance Atlas & Belief Evolution (Phase D3)

Báo cáo này lập bản đồ đánh giá chất lượng đặc trưng (Feature Scoring), kiểm định tính tái lập làm Gate Condition, tự động tiến hóa giả thuyết (Belief Evolution) bằng mô hình toán học **Bayesian Beta-Binomial Conjugate Update thực chứng** (đo lường bằng Brier Score và Kelly ROI chuyên biệt từng Belief) và kết xuất đồ thị liên kết tri thức 8 tầng mở rộng (Knowledge Graph) tích hợp chỉ số PageRank Centrality hai chiều chuẩn hóa L1, Bayesian Credible Interval, phép thử phi tham số Kruskal-Wallis, và liên kết chéo tri thức động Normalized Mutual Information (NMI).

---

## 🔒 1. Kiểm định Gate Condition (Tính Tái lập Checksum)

Theo nguyên tắc **Scientific Integrity (P5)**, trước khi cho phép dữ liệu thực nghiệm thay đổi hay tiến hóa tri thức (Belief), hệ thống tự động chạy kiểm thử tính tái lập (SHA-256 Checksum) của dự báo hai lần song song:

* **Trạng thái Gate**: **ĐẠT ✅ (100% Deterministic/Reproducible)**
* **Hành động**: Cho phép tiếp tục cập nhật độ tin cậy của Belief Registry. Nếu không đạt (nhiễu ngẫu nhiên), hệ thống sẽ lập tức khóa băng (Freeze) độ tin cậy để bảo vệ tri thức hiện tại.

---

## 📊 2. Bảng Điểm Đặc trưng & Trọng số Bằng chứng (Evidence Weight Atlas — D1.2 & D2.2)

Trọng số liên kết thực chứng được lượng hóa đa chiều (Evidence Weight):
$$\text{support\_strength} = 0.50 \times \text{FeatureImportance} + 0.30 \times \text{ReplicationScore} + 0.20 \times \text{EvidenceCount}$$

| Đặc trưng | feature_id | Điểm tổng hợp (Score) | Thứ hạng (Rank) | Đề xuất Cắt tỉa (Prune) | Trạng thái Vòng đời (Lifecycle) | Trọng số Bằng chứng (Evidence Weight) | Chỉ số Drift (PSI) | Bảo trợ cho Belief |
|---|---|:---:|:---:|:---:|---|:---:|---|---|
| `delay_std` | `FEAT_DELAY_STD` | **97.50** | **1** | Không | Production (Anchor) | **0.9793** | 0.0189 (Stable) | `BELIEF_001` |
| `markov_order2` | `FEAT_MARKOV_ORDER2` | **84.05** | **2** | Không | Production (Anchor) | **0.9162** | 0.0142 (Stable) | `BELIEF_003` |
| `delay_mean` | `FEAT_DELAY_MEAN` | **82.35** | **3** | Không | Production (Anchor) | **0.8951** | 0.0541 (Stable) | `BELIEF_001` |
| `markov_entropy` | `FEAT_MARKOV_ENTROPY` | **81.18** | **4** | Không | Production (Anchor) | **0.8995** | 0.0210 (Stable) | `BELIEF_003` |
| `delay_momentum` | `FEAT_DELAY_MOMENTUM` | **80.01** | **5** | Không | Production (Anchor) | **0.8842** | 0.0526 (Stable) | `BELIEF_001` |
| `freq_momentum_short` | `FEAT_FREQ_MOMENTUM_SHORT` | **78.50** | **6** | Không | Production (Anchor) | **0.8897** | 0.0093 (Stable) | `BELIEF_002` |
| `cond_prob_yesterday` | `FEAT_COND_PROB_YESTERDAY` | **74.15** | **9** | Không | **Volatile High Alpha** 🚨 | **0.7309** | 0.2663 (High Drift) | `BELIEF_001` |
| `delay` | `FEAT_DELAY` | **51.20** | **15** | Không | Production | **0.6960** | 0.0001 (Stable) | `BELIEF_001` |
| `delay_sq` | `FEAT_DELAY_SQ` | **4.80** | **78** | **Đồng ý ✂️ (Prune)** | Deprecated | **0.4640** | 0.0001 (Stable) | `BELIEF_001` |

---

## 🔬 3. Nhật ký Tiến hóa Niềm tin (Belief Evolution Engine — D3 Bayesian)

Độ tin cậy được cập nhật bằng toán học **Beta-Binomial Bayesian Update** (Prior Beta $\to$ Likelihood $\to$ Posterior Beta). Các chỉ số thành bại được đánh giá cá thể hóa chuyên biệt cho từng giả thuyết:

* **BELIEF_001 (Calibration)**: Đo lường bằng Brier Score của xác suất hiệu chuẩn so với xác suất thô.
  - Sức khỏe vận hành (Health Score): **`0.78`** | Khoảng tin cậy **CI95: `[0.41, 1.00]`** *(Bayesian Credible Interval)*
  - Trạng thái: **Deprecated**
* **BELIEF_002 (Dynamic Weight Fusion)**: So sánh Brier Score của dynamic fusion so với trung bình các mô hình đơn lẻ.
  - Sức khỏe vận hành (Health Score): **`0.82`** | Khoảng tin cậy **CI95: `[0.44, 1.00]`** *(Bayesian Credible Interval)*
  - Trạng thái: **Deprecated**
* **BELIEF_003 (Mutual Information)**: So sánh Kelly ROI của bộ lọc NMI so với cược phẳng thô (Flat bet).
  - Sức khỏe vận hành (Health Score): **`0.87`** | Khoảng tin cậy **CI95: `[0.50, 1.00]`** *(Bayesian Credible Interval)*
  - Trạng thái: **Experimental**

> [!NOTE]
> **Belief Health Score (Sức khỏe vận hành)** được tính không trùng lặp:
> $$\text{Belief Health} = 0.35 \times \text{Confidence} + 0.35 \times \text{Stability} + 0.30 \times \text{Reproducibility}$$

---

## 🕸️ 4. Phân tích Đồ thị Liên kết Tri thức 8 Tầng (Knowledge Graph — D3)

Hạ tầng tự động kết xuất đồ thị liên kết tri thức cấu trúc mở rộng tại [knowledge_graph.json](file:///f:/MR_BOM/PYTHON/vietnam-lottery-xsmb-analysis/predictions/knowledge_graph.json).

### ⚖️ Phân tích Cấu trúc & PageRank Centrality của Tri thức

PageRank Centrality được tính toán thông qua thuật toán Power Iteration hai chiều có chuẩn hóa L1 sau mỗi vòng lặp để triệt tiêu bias trên đồ thị lớn:

* **`BELIEF_001`**: PageRank Centrality = **`0.2857`**
* **`BELIEF_002`**: PageRank Centrality = **`0.0866`**
* **`BELIEF_003`**: PageRank Centrality = **`0.0223`**

### 🔬 Các phép kiểm định thống kê và liên kết tri thức nâng cao:
* **Kruskal-Wallis phi tham số & Eta-squared ($\eta^2$)**:
  Phép thử giả thuyết Kruskal-Wallis H-test được áp dụng trên cả 3 Regime KMeans ứng với đặc trưng hỗ trợ kém ổn định nhất của từng Belief. Hệ số ảnh hưởng được ghi nhận qua **`effect_size`** sử dụng công thức ước lượng (effect size approximation):
  $$\eta^2 = \frac{H - k + 1}{n - k}$$
* **Normalized Mutual Information (NMI) chéo**:
  Tự động sinh quan hệ **`DEPENDS_ON`** trỏ từ `BELIEF_002` về `BELIEF_001` nếu giá trị NMI chéo chuẩn hóa (trên cơ sở Shannon Entropy) của các đặc trưng hỗ trợ đạt mức tin cậy **`> 0.50`**.

---

## 🛠️ Quy trình chạy tự động

```sh
# Chạy đánh giá đặc trưng, kiểm soát Gate Checksum, cập nhật xác suất Bayes chuyên biệt, phân tích PageRank hai chiều chuẩn hóa L1 và kết xuất Đồ thị tri thức 8 Tầng
python src/research/feature_evaluator.py
```
Dữ liệu chi tiết của 90 nodes và 100 liên kết được ghi nhận đầy đủ tại [knowledge_graph.json](file:///f:/MR_BOM/PYTHON/vietnam-lottery-xsmb-analysis/predictions/knowledge_graph.json).
