# AGENTS.md — XPIS v1.2 System Guide

> Tài liệu tham khảo cho AI agent. Mô tả toàn bộ kiến trúc, pipeline, data flow, quy ước và trạng thái dự án.
> Cập nhật lần cuối: 2026-07-19

---

## 1. Tổng quan dự án

**XPIS** (Xổ số Probability Intelligence System) v1.2 — Hệ thống dự báo định lượng xổ số miền Bắc (XSMB) 2-chữ-số cuối (00–99).

- **Ngôn ngữ**: Python 3.14, kiến trúc 8 tầng (layered pipeline)
- **Chế độ hiện tại**: `paper_trade` — KHÔNG đặt tiền thật
- **Kinh tế cược**:
  - Chi phí: **27.000 đ/số** cược
  - Thù lao: **99.000 đ/số trúng** (odds = 99/27 ≈ 3.666x)
  - 27 số trúng/ngày (từ 7 giải, lấy 2 chữ số cuối % 100)
- **Dữ liệu**: ~7.492 ngày (2005-10-01 → 2026-07-15), xoso.com.vn

### Nguyên tắc cốt lõi

1. **Không dùng LLM để chọn số** — LLM không tạo thêm tín hiệu thống kê.
2. **Chỉ thay đổi policy khi vượt Edge Gate** — Bootstrap 95% CI không chứa 0.
3. **Holdout khóa tuyệt đối** — Không tune model/ngưỡng trên holdout.

---

## 2. Cấu trúc thư mục

```
vietnam-lottery-xsmb-analysis/
├── AGENTS.md                    # ← File này
├── ROADMAP.md                   # Roadmap kiểm định lợi thế
├── MD/out_of_sample_roadmap.md  # Tiêu chí tốt nghiệp EVM-1
├── pyproject.toml               # Dependencies & project metadata
│
├── daily_predict.py             # Pipeline production chính
├── daily_update.py             # Fetch kết quả & cập nhật log
├── auto_runner.py              # Tự động chạy theo lịch (Actions/Scheduler)
├── prediction_dashboard.py     # Dashboard ASCII thống kê + holdout status
│
├── src/                         # === 8-LAYER ARCHITECTURE ===
│   ├── dtos.py                 # Pydantic models (Result, ResultList)
│   ├── lottery.py              # Fetch từ xoso.com.vn + generate CSV/JSON/Parquet
│   ├── fetch.py                 # Script fetch dữ liệu mới
│   ├── analyze.py              # Visualizations (seaborn/matplotlib)
│   ├── feature_engine.py       # FeatureEngine cũ (LightGBM training)
│   │
│   ├── data/                   # L1 — Data Layer
│   │   └── loader.py           # DataLoader
│   ├── evidence/               # L2 — Evidence Layer
│   │   ├── base.py             # EvidenceObject (dataclass)
│   │   ├── builder.py          # EvidenceBuilder
│   │   └── store.py            # EvidenceStore (Parquet immutable)
│   ├── features/               # L3 — Feature Layer
│   │   ├── base.py             # BaseFeatureExtractor (ABC)
│   │   ├── feature_store.py    # FeatureStore (RAM + Parquet hybrid)
│   │   ├── delay_features.py   # DelayFeatureExtractor
│   │   ├── frequency_features.py # FrequencyFeatureExtractor
│   │   ├── markov_features.py  # MarkovFeatureExtractor
│   │   ├── bayesian_features.py # BayesianFeatureExtractor
│   │   ├── pair_features.py    # PairFeatureExtractor
│   │   └── time_features.py    # TimeFeatureExtractor
│   ├── probability/            # L4 — Probability Model Layer
│   │   ├── base.py             # BaseProbabilityModel (ABC)
│   │   ├── max_delay.py        # Model 1
│   │   ├── conditional.py      # Model 2
│   │   ├── markov.py           # Model 3
│   │   ├── momentum.py         # Model 4
│   │   ├── poisson.py          # Model 5
│   │   ├── repeat.py           # Model 6
│   │   ├── inverted_pairs.py   # Model 7
│   │   ├── day_of_week.py      # Model 8
│   │   ├── bayesian.py         # Model 9
│   │   ├── ewma_prob.py        # Model 10
│   │   └── lgb_model.py        # Model 11 (LightGBM)
│   ├── meta/                   # L5 — Meta Learning Layer
│   │   ├── base.py             # BaseMetaLearner (ABC)
│   │   ├── fusion.py           # MetaFusion (dynamic weights)
│   │   ├── calibration.py      # ProbabilityCalibrator (Platt/Isotonic)
│   │   └── lightgbm_meta.py   # LightGBMMetaLearner (walk-forward)
│   ├── decision/               # L6 — Decision Intelligence Layer
│   │   ├── engine.py           # DecisionEngine, NumberDecision, DayDecision
│   │   ├── confidence.py       # ConfidenceEngine
│   │   ├── kelly.py            # KellyCriterion
│   │   └── risk_filters.py     # RiskFilters (NMI correlation)
│   ├── evaluation/             # L7 — Evaluation Layer
│   │   ├── metrics.py          # EvaluationMetrics
│   │   └── reporter.py         # ReportGenerator
│   ├── registry/               # L8 — Registry Layer
│   │   ├── belief_registry.py  # BeliefRegistry
│   │   ├── feature_registry.py # FeatureRegistry
│   │   └── model_registry.py  # ModelRegistry
│   ├── methods/                # Legacy predictors (BasePredictor interface)
│   │   ├── ensemble.py, frequency_momentum.py, max_delay.py,
│   │   ├── markov_chain.py, conditional_prob.py, poisson_estimator.py,
│   │   ├── loto_repeat.py, day_of_week.py, inverted_pairs.py,
│   │   ├── matrix_decision.py, lightgbm_predictor.py
│   ├── research/               # R&D tools
│   │   ├── feature_evaluator.py  # Đánh giá đóng góp feature (40KB)
│   │   ├── stacking_experiment.py
│   │   └── experiment_tracker.py
│   └── templates/              # Jinja2 templates
│       └── README.j2
│
├── backtests/                   # Backtesting engine & results
│   ├── engine.py               # Base engine (5 predictors + ensemble)
│   ├── xpis_backtest.py        # XPIS v1.2 walk-forward (CHÍNH)
│   ├── run_locked_holdout.py   # Holdout đã khóa (180 kỳ)
│   ├── holdout_status.py        # Trạng thái holdout chỉ đọc (Markdown + JSON)
│   ├── lightgbm_backtest.py    # LightGBM standalone
│   ├── lightgbm_train.py       # One-time LightGBM training
│   ├── matrix_optimizer.py     # Matrix Decision optimizer (random search)
│   ├── matrix_backtest_1year.py # Matrix Decision 1-year backtest
│   ├── parameter_sweep.py      # Grid search (p × confidence)
│   ├── scientific_validation.py # Reproducibility + bootstrap + permutation
│   ├── multi_epoch_benchmark.py # 3-epoch + regime analysis
│   └── results/                # 18 báo cáo Markdown
│
├── data/                        # Raw data (3 formats × 3 datasets)
│   ├── xsmb-2-digits.csv       # ← PRIMARY (27 cột, %100)
│   ├── xsmb-2-digits.json
│   ├── xsmb-2-digits.parquet
│   ├── xsmb-sparse.csv/json/parquet  # Sparse format (100 cột count)
│   └── xsmb.csv/json/parquet         # Full results (all tiers)
│
├── predictions/                 # Runtime artifacts
│   ├── evaluation_policy.json  # ← POLICY (mode, holdout, thresholds)
│   ├── prediction_log.json     # ← DAILY LOG (tất cả dự báo)
│   ├── belief_registry.json
│   ├── feature_registry.json
│   ├── model_registry.json
│   ├── adaptive_weights.json
│   ├── matrix_rules.json       # Matrix Decision optimized rules
│   ├── knowledge_graph.json
│   ├── mi_edges.json           # NMI edge list
│   ├── feature_importance_atlas.json
│   ├── EvidenceStore/           # Parquet snapshots (1 subdir/date)
│   ├── FeatureStore/           # Versioned Parquet (v1/snapshots/)
│   ├── models/                 # Saved LightGBM models (.pkl)
│   ├── experiments/            # Experiment tracking
│   └── rfcs/                   # RFC documents
│
├── tests/                       # Integration tests (5 files, manual run)
│   ├── test_all_10_models.py
│   ├── test_backtest_90d.py
│   ├── test_xpis_integration.py
│   ├── test_meta_learner.py
│   └── test_models_9_10.py
│
├── images/                      # 7 JPG visualizations
├── index.html                   # Web dashboard (Chart.js, dark theme)
└── setup_scheduler.bat          # Windows Task Scheduler setup
```

---

## 3. Kiến trúc 8 tầng — Chi tiết API

### L1 — Data Layer (`src/data/loader.py`)

```python
class DataLoader:
    """Đọc CSV, trả DataFrame sạch + ma trận nhị phân."""
    DEFAULT_CSV = "data/xsmb-2-digits.csv"

    def load(self) -> "DataLoader"             # Load CSV, sort by date, build S
    def slice_history(self, up_to_idx: int) -> tuple[DataFrame, ndarray]  # History trước idx (exclusive)

    @property df -> DataFrame                 # Full DataFrame (date + 27 prize cols)
    @property S -> ndarray                    # Binary matrix shape (N, 100), int8
    @property total_days -> int               # len(df)
    def prize_cols() -> list[str]            # 27 cột giải thưởng
```

**Ma trận S**: `S[day, num] = 1` nếu số `num` (00-99) xuất hiện ngày `day`. ~7.492 × 100.

---

### L2 — Evidence Layer (`src/evidence/`)

```python
@dataclass
class EvidenceObject:
    """Snapshot raw observation của 1 số tại 1 thời điểm."""
    number: int              # 0-99
    date: str                # YYYY-MM-DD
    history_days: int
    # Delay
    current_delay: int        # Ngày kể từ lần xuất hiện cuối
    historical_delays: list[int]
    # Frequency counts (9 windows)
    count_3d, count_7d, count_14d, count_30d, count_60d,
    count_90d, count_120d, count_180d, count_365d, count_all: int
    # Markov
    state_yesterday: int     # 0 | 1
    state_day_before: int    # 0 | 1
    # Number properties
    head, tail: int          # Chữ số đầu/cuối
    is_twin: bool            # Số kép (00, 11, ...)
    inverted: int            # Số lộn (12→21)
    mirror: int              # Số gương (12→67)
    # Context
    yesterday_actives: list[int]
    yesterday_head_cam: list[int]
    yesterday_tail_cam: list[int]
    # Time
    weekday: int              # 0=Mon..6=Sun
    day_of_month: int
    month: int
    is_weekend: bool

    def to_dict() / to_json() / from_dict() / from_json()

class EvidenceBuilder:
    """Build EvidenceObject cho 100 số từ history."""
    def __init__(self, store: EvidenceStore | None = None)
    def build_all(self, df_history, S_history, target_date, save=True) -> DataFrame
        # Trả DataFrame 100 rows, tự cache vào EvidenceStore

class EvidenceStore:
    """Parquet immutable cache. Một khi ghi, không bao giờ ghi đè."""
    EVIDENCE_VERSION = "v1.0"
    # Cấu trúc: predictions/EvidenceStore/YYYY-MM-DD/evidence.parquet
    def __init__(self, store_root: Path | None = None)
    def exists(self, date_str) -> bool
    def save(self, date_str, df_evidence, overwrite=False) -> Path
    def load(self, date_str) -> DataFrame | None
    def list_dates() -> list[str]
```

---

### L3 — Feature Layer (`src/features/`)

```python
class BaseFeatureExtractor(ABC):
    """Interface chuẩn. Input: Evidence(100 rows) → Output: Feature columns(100 rows)."""
    @abstractmethod name -> str
    @abstractmethod version -> str
    @abstractmethod extract(self, df_evidence: DataFrame) -> DataFrame
    def feature_names(self, df_evidence) -> list[str]

# 6 Extractors cụ thể:
class DelayFeatureExtractor        # delay_zscore, delay_percentile, delay_momentum, ...
class FrequencyFeatureExtractor    # freq_momentum_short/long, freq_skew_30, freq_kurt_30, ...
class MarkovFeatureExtractor(S)    # Cần S_history. markov_features từ S
class BayesianFeatureExtractor(S)  # Cần S_history. cond_prob từ S
class PairFeatureExtractor         # inverted_appeared_yesterday, mirror_appeared_yesterday, ...
class TimeFeatureExtractor         # weekday, day_of_month, month, is_weekend

class FeatureStore:
    """Hybrid: RAM cache + Parquet snapshot. Versioned (v1/, v2/, ...)."""
    FEATURE_STORE_VERSION = "v1"

    def __init__(self, store_root=None, version="v1",
                 S_history=None, evidence_store=None)
    def build(self, df_evidence, date_str, S=None,
              use_cache=True, save_parquet=True) -> DataFrame
        # Priority: RAM cache → Parquet snapshot → Compute from Evidence
        # Trả DataFrame 100 rows × N_features cols (có 'number', 'date')
    def load(self, date_str) -> DataFrame | None
    def exists(self, date_str) -> bool
    def list_dates() -> list[str]
    def clear_ram_cache() -> None
    @property version -> str
    def feature_count(self, sample_date) -> int | None
```

---

### L4 — Probability Model Layer (`src/probability/`)

```python
class BaseProbabilityModel(ABC):
    """Interface chuẩn. Input: FeatureVector(100 rows) → Output: proba(100,)."""
    @abstractmethod name -> str
    @abstractmethod version -> str
    @abstractmethod predict_proba(self, df_features, df_history=None, S_history=None) -> ndarray
        # Returns ndarray(100,) — xác suất độc lập mỗi số, KHÔNG sum=1
    def top_k(self, proba, k=10) -> list[int]
    def evaluate_on_day(self, proba, actual_numbers, k=10) -> dict

# 11 Models theo thứ tự chuẩn (get_all_models()):
#  1. MaxDelayPredictor         — Dựa vào current_delay
#  2. ConditionalPredictor      — P(num | yesterday_actives)
#  3. MarkovPredictor          — Markov chain order 1/2
#  4. MomentumPredictor        — Short-term vs long-term frequency
#  5. PoissonPredictor          — Poisson estimation (window 180)
#  6. RepeatPredictor           — Lặp lại từ các giai đoạn
#  7. InvertedPairsPredictor    — Dựa vào cặp lộn
#  8. DayOfWeekPredictor        — Xác suất theo ngày trong tuần
#  9. BayesianPredictor         — Naive Bayes kết hợp đa bằng chứng
# 10. EWMAPredictor(multi_scale=True) — Exponential Weighted Moving Average
# 11. LightGBMProbabilityModel  — ML classifier, walk-forward train

def get_all_models() -> list[BaseProbabilityModel]
    # Trả về 11 models theo thứ tự chuẩn
```

**LightGBMProbabilityModel** (`lgb_model.py`):
```python
class LightGBMProbabilityModel:
    def __init__(self)
    def fit(self, X, y, feature_names=None) -> None    # Train LightGBM
    def predict_proba(self, df_features, df_history=None, S_history=None) -> ndarray
    def feature_importance(self, top_n=20) -> DataFrame
```

---

### L5 — Meta Learning Layer (`src/meta/`)

```python
class MetaFusion:
    """Tổng hợp xác suất đa model với trọng số động."""
    def compute_dynamic_weights(
        self,
        historical_predictions: dict[str, ndarray],  # {model_name: (n_days, 100)}
        historical_labels: ndarray,                  # (n_days, 100)
    ) -> dict[str, float]
        # Quality = 0.35*(1-Brier) + 0.35*exp(-LogLoss) + 0.20*Precision@10 + 0.10*max(0,ROI)
        # Weights = Quality / sum(Quality)

    def fuse(self, model_probas: dict[str, ndarray]) -> ndarray
        # Weighted average, fallback uniform nếu chưa có weights

class ProbabilityCalibrator:
    """Hiệu chỉnh xác suất (Platt scaling hoặc Isotonic regression)."""
    def __init__(self, method: "platt" | "isotonic" = "isotonic")
    def fit(self, proba_raw: ndarray, y_true: ndarray) -> self
    def calibrate(self, proba_raw: ndarray) -> ndarray
    def ece_score(self, proba_raw, y_true, n_bins=10) -> float

class LightGBMMetaLearner(BaseMetaLearner):
    """Meta Learner — kết hợp features + model probabilities → P(num)."""
    def __init__(self, train_window=365, retrain_every=30, calibrate=True)
    def build_training_data(self, feature_snapshots, labels) -> tuple[ndarray, ndarray]
    def train(self, X, y, feature_names=None, **kwargs) -> None
        # LGBMClassifier(n_estimators=80, lr=0.05, num_leaves=15)
        # Nếu calibrate → CalibratedClassifierCV(isotonic, cv=5)
    def predict_proba(self, X) -> ndarray
    def feature_importance(self, top_n=20) -> DataFrame
    def save(self, path=None) -> Path
    def load(self, path: str) -> None
```

---

### L6 — Decision Intelligence Layer (`src/decision/`)

```python
@dataclass
class NumberDecision:
    """Output Contract cho 1 số."""
    number: int
    probability: float
    confidence: float
    risk: str              # "LOW" | "MEDIUM" | "HIGH"
    allocation: float      # Kelly adjusted fraction
    decision: str          # "BET" | "SKIP" | "WATCH"
    explanation: dict      # {feature_states: {feat: "Zσ"}, approximate_contributions: {feat: "±val"}}

@dataclass
class DayDecision:
    """Tổng hợp 1 ngày."""
    date: str
    run_id: str
    decisions: list[NumberDecision]
    diversification_score: float     # 1 - mean(NMI correlation)
    rank_stability_index: float      # Spearman Rank Correlation PSI
    decision_summary: dict          # Số qualified/rejected theo từng cổng
    # Pipeline Metadata
    feature_version, model_version, belief_version, evidence_version, git_commit, random_seed

    @property bets -> list[NumberDecision]    # Chỉ những số decision="BET"
    @property top_numbers -> list[int]
    def to_dict() -> dict                     # Data Contract chuẩn

class DecisionEngine:
    """Tổng hợp Probability + Confidence + Risk → Decision."""
    DECISION_VERSION = "v1.2"
    def __init__(self, min_probability=0.31, min_confidence=0.45,
                 top_k=10, kelly_odds=3.666, kelly_fraction=0.20,
                 min_diversification=0.85)
    def decide(self, date, meta_proba, model_probas=None, S_history=None,
               df_features=None, feature_importance_df=None, ...) -> DayDecision

class ConfidenceEngine:
    """Đồng thuận giữa models: 70% variance-based + 30% agreement-based."""
    def compute(self, model_probas, meta_proba) -> ndarray  # (100,) range [0,1]
    def agreement_matrix(self, model_probas, top_k=10) -> DataFrame

class KellyCriterion:
    """f* = (p*b - q) / b, với fractional Kelly."""
    def __init__(self, odds=70.0, kelly_fraction=0.25, max_fraction=0.10, min_prob=0.30)
    def compute(self, proba, confidence=None) -> ndarray  # (100,) fraction vốn
    def capital_allocation(self, kelly_fractions, total_capital, top_k=10) -> dict

class RiskFilters:
    """Portfolio risk: NMI correlation, head/tail exposure limits."""
    def __init__(self, max_head_exposure=0.20, max_tail_exposure=0.20, min_diversification=0.85)
    def build_empirical_correlation(self, S_history) -> None  # NMI matrix → mi_edges.json
    def get_correlation(self, num1, num2) -> float
    def compute_diversification_score(self, numbers: list[int]) -> float
    def optimize_allocations(self, raw_kelly, top_numbers) -> ndarray
```

---

### L7 — Evaluation Layer (`src/evaluation/`)

```python
class EvaluationMetrics:
    """Tất cả metrics kiểm định."""
    def __init__(self, odds=3.666, cost_per_bet=27.0)
    def roi(self, results: DataFrame) -> float
    def brier_score(self, proba_all, y_all) -> float
    def log_loss(self, proba_all, y_all) -> float
    def auc_roc(self, proba_all, y_all) -> float
    def ece_score(self, proba_all, y_all, n_bins=10) -> float
    def precision_at_k(self, proba_matrix, y_matrix, k=10) -> float
    def recall_at_k(self, proba_matrix, y_matrix, k=10) -> float
    def max_drawdown(self, cumulative_pnl) -> float
    def compute_full(self, results, proba_history=None, y_history=None) -> dict

class ReportGenerator:
    """Tạo báo cáo Markdown (daily/weekly/monthly)."""
    def daily_report(self, date, decisions, actual_numbers, meta_proba, version_info=None) -> str
    def summary_report(self, results, title, version_info=None) -> Path
```

---

### L8 — Registry Layer (`src/registry/`)

```python
class BeliefRegistry:
    """Knowledge Graph — quản lý beliefs khoa học."""
    # predictions/belief_registry.json
    def register(self, belief_id, title, hypothesis, status="Experimental",
                 confidence=0.50, ...) -> None
    def get_belief(self, belief_id) -> dict | None
    def get_active_beliefs(self, status="Validated") -> list[dict]

class FeatureRegistry:
    """Feature Catalog — audit features."""
    # predictions/feature_registry.json
    def register(self, name, group, formula, description, version="1.0", status="active") -> None
    def get_features(self, group=None, status="active") -> list[str]
    def is_valid(self, name) -> bool

class ModelRegistry:
    """Model Catalog — audit models."""
    # predictions/model_registry.json
    def register(self, name, model_type, version, description, parameters=None, status="active") -> None
    def get_model_meta(self, name) -> dict | None
    def get_active_models(self, model_type=None) -> list[str]
```

---

## 4. Pipeline chung — `run_shared_prediction_pipeline()`

**File**: `daily_predict.py:126-380`

Đây là pipeline DUY NHẤT dùng cho cả production và backtest exact-mode.

```
DataLoader.load()
    │
    ├─ slice_history(last_hist_idx) → df_hist, S_hist
    │
    ├─ EvidenceBuilder.build_all(df_hist, S_hist, target_date)
    │   └─ EvidenceStore cache (immutable Parquet)
    │
    ├─ FeatureStore.build(df_ev, date_str, S=S_hist)
    │   └─ RAM cache → Parquet snapshot → 6 extractors
    │
    ├─ Train LightGBM trên 365 ngày (train_start..train_end)
    │   train_end = last_hist_idx - 90  (tránh optimistic bias)
    │   train_start = max(50, train_end - 365)
    │
    ├─ 11 Models predict_proba(df_feat) → model_probas dict
    │
    ├─ Calibration — Nested Split trên 90 ngày validation:
    │   ├─ Nửa đầu (45 ngày): fit Sigmoid + fit Isotonic
    │   ├─ Nửa sau (45 ngày): chọn winner bằng composite score
    │   │   score = 0.5*Brier + 0.3*LogLoss + 0.2*ECE
    │   └─ Áp dụng best calibrator lên LightGBM probs
    │
    ├─ MetaFusion — dynamic weights trên 45 ngày selection (đã calibrate):
    │   ├─ compute_dynamic_weights(eval_preds, eval_labels)
    │   └─ EMA smoothing: α = min(0.9, n_samples/50)
    │       smoothed = α*old + (1-α)*new  (chuẩn hóa sum=1)
    │
    ├─ DecisionEngine.decide():
    │   ├─ RiskFilters.build_empirical_correlation(S_hist)  → NMI matrix
    │   ├─ ConfidenceEngine.compute(model_probas, meta_proba)
    │   ├─ Lọc eligible: p >= 0.31 AND confidence >= 0.45
    │   ├─ Top-K = 2
    │   ├─ Diversification score (1 - mean NMI)
    │   ├─ KellyCriterion.compute(meta_proba, confidence)
    │   ├─ RiskFilters.optimize_allocations() → head/tail exposure limits
    │   └─ Package: NumberDecision + DayDecision
    │
    └─ Output Contract (to_dict()):
        ├─ pipeline_metadata (run_id, git, SHA256 hashes, env info)
        ├─ ensemble_proba, model_probas, raw_lgb_proba, calibrated_lgb_proba
        ├─ dynamic_weights, best_calibration_method
        ├─ decision_summary, diversification_score, rank_stability_index
        └─ bets[] với explanation (feature_states Zσ, approximate_contributions)
```

---

## 5. Scripts Root

### `daily_predict.py` — Production Pipeline

```bash
python daily_predict.py                        # Dự đoán hôm nay (tự fetch data)
python daily_predict.py --date 2026-07-14      # Ngày cụ thể
python daily_predict.py --top-k 2             # Số lượng cược tối đa
python daily_predict.py --dry-run             # Chạy thử, không lưu log
python daily_predict.py --no-fetch            # Bỏ qua fetch
```

- `run_predict(date, top_k)` → `run_shared_prediction_pipeline(date, top_k)`
- Ghi kết quả vào `predictions/prediction_log.json`
- In bảng quyết định ra terminal (Unicode box drawing)

### `daily_update.py` — Cập nhật kết quả

```bash
python daily_update.py                        # Cập nhật hôm nay
python daily_update.py --date 2026-07-13      # Ngày cụ thể
```

- Fetch kết quả thực tế từ web
- Tính hits, revenue, PnL cho mỗi bet
- Cập nhật `prediction_log.json` entry tương ứng
- In thống kê tổng hợp (win rate, ROI)

### `auto_runner.py` — Tự động hóa

```bash
python auto_runner.py
```

- **Trước 18:35 VN**: Dự đoán hôm nay (nếu chưa có)
- **Sau 18:35 VN**: Cập nhật kết quả hôm nay + dự đoán ngày mai
- Log tại `predictions/auto_runner.log`
- Dùng `uv run` nếu có, fallback `python`
- Gọi bởi GitHub Actions (01:00 UTC, 12:05 UTC) hoặc Windows Task Scheduler

### `prediction_dashboard.py` — Dashboard dòng lệnh

```bash
python prediction_dashboard.py           # Toàn bộ
python prediction_dashboard.py --days 30 # 30 ngày gần nhất
```

- ASCII box drawing dashboard
- Thống kê: tổng PnL, ROI, win rate, chuỗi thắng/thua
- Phân phối PnL theo tháng (bar chart ASCII)
- Lịch sử 10 ngày gần nhất
- Khối trạng thái locked holdout: ngày bắt đầu, số ngày hợp lệ và prediction log hợp lệ

### `index.html` — Web dashboard

- Giao diện web tĩnh dùng `predictions/prediction_log.json` và `predictions/matrix_rules.json`.
- Hiển thị cùng khối locked holdout từ `backtests/results/locked_holdout_status.json`.
- `auto_runner.py` cập nhật JSON holdout; workflow deploy phải đưa file này vào artifact/hosting.

### `backtests/holdout_status.py` — Holdout monitor

```bash
python backtests/holdout_status.py
```

- Chỉ đọc policy, dữ liệu và prediction log; không tune model/ngưỡng/top_k và không sửa Edge Gate.
- Ghi `backtests/results/locked_holdout_status.md` và `.json`.
- Đánh dấu rõ các log trước ngày holdout để không bị tính nhầm.

### `fetch.py` / `lottery.py` — Data Fetching

- `lottery.py`: `Lottery` class — load JSON → fetch từ xoso.com.vn (CloudScraper) → dump CSV/JSON/Parquet
- `fetch.py`: Script one-shot fetch dữ liệu mới
- 3 output formats: `xsmb` (raw), `xsmb-2-digits` (%100), `xsmb-sparse` (count matrix)

### `analyze.py` — Visualization

- Tạo 7 JPG images trong `images/`:
  - Heatmap frequency, delta, distribution, top-10
  - Special prize delta heatmap
- Cập nhật `README.md` từ Jinja2 template

---

## 6. Backtests

### XPIS v1.2 Backtest (`backtests/xpis_backtest.py`)

```python
def run_exact_production_backtest(n_test_days=30, top_k=2, report_name=None)
    # Gọi run_shared_prediction_pipeline() mỗi ngày — CHẬM nhưng chính xác 100%

def run_xpis_backtest(n_test_days=180, min_prob=0.31, min_conf=0.45, top_k=4)
    # Walk-forward: retrain LGBM mỗi 30 ngày, calibrate trên 90-day window

def _calibration_score(brier, logloss, ece) -> float
    # 0.5*Brier + 0.3*LogLoss + 0.2*ECE

def _bootstrap_roi_interval(results, n_bootstrap=20000) -> tuple[float, float]
    # 95% CI bằng resampling

def _permutation_test(results, n_permutations=5000) -> dict
    # Xáo kết quả giữa các ngày, so PnL
```

### Locked Holdout (`backtests/run_locked_holdout.py`)

- Đọc `evaluation_policy.json`: holdout từ **2026-07-21**, minimum 180 ngày
- Kiểm tra holdout là cửa sổ mới nhất
- Gọi `run_exact_production_backtest()` với parameters đã khóa
- Exit code 0 nếu chưa đủ dữ liệu (chờ bình thường)
- Trạng thái hiện tại: `0/180` ngày, policy `top_k=2`, chưa đủ dữ liệu để chạy exact holdout.

### Base Engine (`backtests/engine.py`)

- 5 statistical predictors + weighted ensemble
- Cost: 27k/number, Payout: 99k/hit
- Kết quả 1000 ngày: Ensemble ROI +2.81%

### Matrix Decision (`backtests/matrix_optimizer.py` + `matrix_backtest_1year.py`)

- Random search 2000 iterations
- Input: feature matrix (delay, poisson, markov, momentum, repeat, pairs, cond_prob, dow)
- Best: ROI +11.83% (180 ngày optimize), +9.40% (365 ngày backtest)

### Parameter Sweep (`backtests/parameter_sweep.py`)

- Grid search: `min_probability` [0.26–0.32] × `min_confidence` [0.40–0.60]
- 35 combinations, flat betting + Kelly
- Kết quả: tất cả 35 config có Kelly ROI âm

### Scientific Validation (`backtests/scientific_validation.py`)

- Reproducibility: 2 independent backtests → SHA-256 hash so sánh
- Bootstrap 1000 iterations
- Permutation 1000 iterations
- ECE/Brier, Registry coverage

### Multi-Epoch Benchmark (`backtests/multi_epoch_benchmark.py`)

- 3 epochs 90 ngày độc lập
- KMeans(n_clusters=3) phát hiện regimes (A=uniform, B=repeat storm, C=normal)
- Epoch 1 PASS (Kelly ROI +1.27%), Epoch 2 FAIL (-0.55%), Epoch 3 FAIL (-0.09%)

---

## 7. Dữ liệu

### Primary: `data/xsmb-2-digits.csv`

| Thuộc tính | Giá trị |
|---|---|
| Format | CSV (cũng có JSON, Parquet) |
| Date range | 2005-10-01 → 2026-07-15 |
| Rows | ~7.492 ngày |
| Columns | `date` + 27 prize columns |
| Values | 2-digit integers (00-99), đã `% 100` |
| Columns: | `special`, `prize1`, `prize2_1`, `prize2_2`, `prize3_1..6`, `prize4_1..4`, `prize5_1..6`, `prize6_1..3`, `prize7_1..4` |

### Ma trận nhị phân S

- Shape: `(total_days, 100)`, dtype `int8`
- `S[t, n] = 1` nếu số `n` xuất hiện ngày `t`
- Mỗi ngày có ~27 số trúng → ~27% sparsity

### Sparse format (`data/xsmb-sparse.csv`)

- 100 cột (số 00-99), giá trị = count xuất hiện trong ngày
- Hữu ích cho thống kê tần suất nhanh

---

## 8. Cấu hình & Policy

### `predictions/evaluation_policy.json`

```json
{
  "mode": "paper_trade",           // KHÔNG đặt tiền thật
  "locked_holdout": {
    "start_date": "2026-07-21",   // Bắt đầu holdout Top-2
    "minimum_days": 180            // Chờ đủ 180 kỳ
  },
  "decision_policy": {
    "top_k": 2,                   // Tối đa 2 số/ngày
    "min_probability": 0.31,       // Ngưỡng xác suất
    "min_confidence": 0.45,       // Ngưỡng đồng thuận models
    "min_diversification": 0.85   // Ngưỡng đa dạng hóa
  },
  "promotion_gate": {
    "required": "bootstrap_roi_lower_95_gt_zero",
    "on_failure": "paper_trade"
  }
}
```

### `predictions/prediction_log.json`

Array của daily prediction entries. Mỗi entry:
```json
{
  "pipeline_metadata": {
    "run_id": "RUN_20260714_abc123",
    "date": "2026-07-14",
    "schema_version": "2.0",
    "prediction_engine_version": "1.3.1",
    "git_commit": "abc123...",
    "experiment": {
      "id": "XPIS-EVM-1",
      "dataset_hash": "sha256:...",
      "feature_count": 88,
      "environment": { "python": "3.14", "numpy": "2.4.1", ... }
    }
  },
  "diversification_score": 0.92,
  "rank_stability_index": 0.85,
  "ensemble_proba": [...100 values...],
  "model_probas": { "model_name": [...100 values...] },
  "raw_lgb_proba": [...],
  "calibrated_lgb_proba": [...],
  "pipeline_metadata": {
    "best_calibration_method": "sigmoid",
    "dynamic_weights": { "model1": 0.15, ... }
  },
  "bets": [
    {
      "number": 42,
      "probability": 0.35,
      "confidence": 0.72,
      "risk": "LOW",
      "allocation": 0.05,
      "decision": "BET",
      "explanation": {
        "feature_states": { "delay": "+1.5σ", ... },
        "approximate_contributions": { "delay": "+0.0012", ... }
      }
    }
  ],
  "actual_results": [12, 42, 55, ...],    // Được thêm bởi daily_update.py
  "ensemble_hits": 1,
  "revenue_k": 99,
  "pnl_k": -9
}
```

---

## 9. Roadmap — Trạng thái hiện tại

| # | Mục | Tiến độ |
|---|-----|---------|
| 1 | Pipeline chung production/backtest | ✅ (exact holdout chờ đủ dữ liệu) |
| 2 | Khóa holdout | ✅ (Top-2 từ 2026-07-21, chờ đủ 180 ngày) |
| 3 | Baseline bắt buộc | ✅ HOÀN THÀNH |
| 4 | Kiểm định tín hiệu | ✅ HOÀN THÀNH |
| 5 | Làm gọn mô hình | ✅ Không loại model; giữ 11 models |
| 6 | Paper-trade Edge Gate | ✅ FAIL, giữ paper-trade |
| 7 | Kelly minh bạch | ✅ Sizing-only |
| 8–12 | Count challenger, stacking, fusion và holdout monitor | ✅/research-only; không promote |

### Kết quả định lượng (exploratory 365 ngày)

| Chỉ số | Giá trị |
|---|---|
| XPIS flat ROI | **-5.06%** (112 bets, 29 win days) |
| XPIS Kelly ROI | +0.09% |
| Bootstrap 95% CI | **[-38.89%, +30.45%]** — chứa 0 → Edge Gate FAIL |
| Permutation p-value | **0.7630** — không có tín hiệu |
| Random baseline ROI | -11.61% |
| Frequency baseline ROI | +1.49% |
| No-bet ROI | 0.00% |

**Kết luận**: Không có bằng chứng thống kê về lợi thế. Giữ `paper_trade`.

---

## 10. Tiêu chí Tốt nghiệp EVM-1 (Phase 3)

Yêu cầu **300–500 kỳ holdout ngoài mẫu** + tất cả:

| Metric | Ngưỡng |
|---|---|
| Brier Score | ≤ 0.2200 |
| ECE | ≤ 0.0800 |
| ΔBrier vs Best Component | ≤ -0.0100, CI95 loại 0 |
| Kelly ROI | Dương sau chi phí |
| Max Drawdown | ≤ 35% vốn Kelly |

**Baselines đối chiếu bắt buộc**:
1. Uniform Probability (ngẫu nhiên)
2. Best Single Component (model đơn lẻ tốt nhất)
3. Previous Production Version

---

## 11. Quy ước Code

### Project Setup

- **Python**: 3.14 (`requires-python = "==3.14.*"`)
- **Package manager**: `uv`
- **Linting**: `ruff` + `isort`
- **Line length**: 120
- **Quote style**: Single quotes
- **Indent**: Spaces

### Kiến trúc

- `src/` — 8-layer modules (production code)
- Root — Scripts, backtests, config
- `sys.path.insert(0, "src")` ở các root scripts

### Ngôn ngữ

- Code: English (identifiers, docstrings)
- Bình luận: **Tiếng Việt** (giải thích business logic)
- Reports/Output: Tiếng Việt

### Nguyên tắc Pipeline

- **Evidence immutable**: Một khi tạo, không bao giờ ghi đè
- **Feature versioned**: Mỗi version có thư mục riêng (v1/, v2/)
- **Model versioned**: Lưu .pkl + .json metadata
- **Pipeline Metadata**: Mỗi prediction record chứa git commit, SHA256 hashes, environment info
- **Output Contract**: `DayDecision.to_dict()` — schema cố định, version-tagged
- **Random seed**: 42 (mặc định, có thể override)

### Testing

- Manual execution (không pytest framework, nhưng có `pytest` trong dev deps)
- `tests/` — 5 integration test scripts
- Kiểm tra từng layer độc lập, sau đó full pipeline

---

## 12. Dependencies

```
# Core
numpy==2.4.1
pandas==3.0.0
pyarrow==23.0.0          # Parquet I/O

# ML
scikit-learn>=1.9.0
lightgbm>=4.6.0

# Data fetching
cloudscraper==1.2.71
beautifulsoup4==4.14.3
lxml==6.0.2
tenacity==9.1.2

# Validation/Serialization
pydantic==2.12.5
pydantic-settings==2.12.0
tzdata==2025.3

# Visualization
matplotlib==3.10.8
seaborn==0.13.2

# Templates
jinja2==3.1.6

# Dev (optional)
pytest, pytest-asyncio, ruff, isort
```

---

## 13. Data Flow Summary

```
xoso.com.vn
    │
    ▼ fetch.py / lottery.py
data/xsmb-2-digits.csv (7492 days × 28 cols)
    │
    ▼ DataLoader.load()
DataFrame + Ma trận S (N × 100, int8)
    │
    ▼ EvidenceBuilder.build_all() [L2]
EvidenceObject × 100 → Parquet cache
    │
    ▼ FeatureStore.build() [L3]
FeatureVector (100 × N_features) → RAM/Parquet cache
    │
    ├──▶ 10 Static Models [L4] ──┐
    ├──▶ LightGBM [L4 train]   ──┤
    │                            ▼
    │                    Calibration [L5]
    │                    (sigmoid/isotonic, 45/45 split)
    │                            │
    │                    MetaFusion [L5]
    │                    (dynamic weights + EMA)
    │                            │
    │                            ▼
    │                    DecisionEngine [L6]
    │                    (confidence + kelly + risk)
    │                            │
    │                            ▼
    │                    DayDecision (Output Contract)
    │                            │
    ▼                            ▼
predictions/prediction_log.json ──▶ daily_update.py (fetch actuals, compute PnL)
                                        │
                                        ▼
                                prediction_dashboard.py (ROI, win rate, streaks)
                                index.html (web dashboard)
```
