# XPIS v2 Count-First Research Subsystem — Architecture & Statistical Design Specification

- **Document Version**: 2.1.0-REVISED
- **Date**: 2026-08-30
- **Status**: DESIGN SPECIFICATION (LOCKED FOR CHATGPT CONTROL PLANE REVIEW)
- **Authoring Agent**: Repositorially Grounded Design Spec Author (Gemini 3.7 Flash, Reasoning: High)
- **Canonical Repository**: `Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis`
- **Design Base Commit**: `067296fbe4d38edbe60fee9a062434c6e06396c4`
- **Design Base Tree**: `57a1771d2f36a8653c4fd19958f7079707e9b51f`
- **Design Branch**: `codex/design/xpis-v2-count-first-20260830`
- **Target Specification Path**: `docs/superpowers/specs/2026-08-30-xpis-v2-count-first-design.md`

---

## 1. Executive Summary & Purpose

### 1.1 Purpose & Scientific Objective
XPIS (Xổ số Probability Intelligence System) v2 is a minimal, parallel, **COUNT-FIRST** research subsystem. Its primary scientific objective is to resolve the fundamental inquiry:

$$\text{Does stable predictive information exist in the occurrence-count target beyond simple count baselines?}$$

XPIS v2 reformulates lottery forecasting from binary occurrence classification ($S \in \{0, 1\}$) to direct integer count modeling ($C \in \{0, 1, 2, \dots, 27\}$). This transition aligns the mathematical modeling domain with the physical draw generation process of Northern Vietnam Lottery (XSMB), which produces 27 prize numbers each day across 100 possible two-digit outcomes (00–99).

### 1.2 Scope & Non-Goals
Slice 1 is exclusively a **Development / Exploratory Count Research Core** (`SLICE_1 = DEVELOPMENT / EXPLORATORY COUNT RESEARCH CORE`). Its implementation scope is strictly confined to minimal count contracts, dataset validation, initial baselines/models, an exploratory research adapter, and correctness unit testing.

While this architecture document defines the requirements for future confirmatory trials, Slice 1 itself does **NOT** implement the full confirmatory ledger, remote receipt timestamping, or prospective execution framework.

Explicitly forbidden and deferred in Slice 1:
- **NO** LightGBM classification or regression models.
- **NO** model ensembling or stacking.
- **NO** dynamic weight optimizers or MetaFusion layers.
- **NO** FeatureStore or feature extraction pipelines.
- **NO** EvidenceStore or historical snapshot caches from v1.2.
- **NO** Knowledge Graph or Belief Registry integrations.
- **NO** Kelly Criterion capital allocations or portfolio risk optimizers.
- **NO** production pipeline integration or automated daily betting runners.
- **NO** modifications to the existing XPIS v1.2 probability, meta, or decision stack.
- **NO** tuning or alteration of XPIS v1.2 thresholds.

---

## 2. Target Definition & Mathematical Coherence

### 2.1 Canonical Count Target
For any target date $t$ and two-digit number $n \in \{00, 01, \dots, 99\}$:

$$C[t, n] = \sum_{p=1}^{27} \mathbb{I}(\text{draw}[t, p] = n)$$

where $\text{draw}[t, p]$ denotes the $p$-th two-digit prize outcome drawn on date $t$.

The canonical count matrix $C$ satisfies the following structural invariants:
1. **Dimensionality**: $C \in \mathbb{Z}_{\ge 0}^{N \times 100}$ for $N$ observation dates.
2. **Non-negativity**: $C[t, n] \ge 0 \quad \forall t, n$.
3. **Integrality**: $C[t, n] \in \{0, 1, 2, \dots, 27\} \quad \forall t, n$.
4. **Exact Count Conservation**: $\sum_{n=0}^{99} C[t, n] = 27 \quad \forall t$.

### 2.2 Distinction from Binary Occurrence
The binary occurrence matrix $S \in \{0, 1\}^{N \times 100}$ is defined as:

$$S[t, n] = \mathbb{I}(C[t, n] > 0)$$

In XPIS v2, $S$ is strictly a **derived diagnostic quantity**. Binary occurrence is **NOT** the canonical prediction target. All modeling, estimation, evaluation, and economic scoring are anchored natively on $C$.

### 2.3 Forecast Coherence Invariant
Every valid point forecast produced by an XPIS v2 model represents the conditional expected count vector $\hat{\mu}_t \in \mathbb{R}_{\ge 0}^{100}$ for target date $t$, where:

$$\hat{\mu}_t[n] = \mathbb{E}\left[C[t, n] \mid \mathcal{H}_{<t}\right]$$

Every coherent mean forecast must satisfy:
1. **Non-negativity**: $\hat{\mu}_t[n] \ge 0 \quad \forall n \in \{0, \dots, 99\}$.
2. **Sum-to-27 Invariant**:

$$\left| \sum_{n=0}^{99} \hat{\mu}_t[n] - 27.0 \right| \le \epsilon_{\text{tol}}$$

where $\epsilon_{\text{tol}} = 10^{-6}$ is a frozen numerical floating-point tolerance.

---

## 3. Initial Model Portfolio (Slice 1 Models)

Slice 1 restricts the model portfolio to exactly two baselines and two primary count models:

```text
B0: Uniform Count Baseline
B1: Rolling Count Baseline
M1: Exponentially Weighted Moving Average (EWMA) Count Model
M2: Dirichlet-Shrinkage Multinomial Model
```

### 3.1 Model B0: Uniform Count Baseline
The theoretical structural null model assuming independent, identically distributed uniform draws across all 100 states:

$$\hat{\mu}_{\text{B0}}[t, n] = \frac{27}{100} = 0.27 \quad \forall n \in \{0, \dots, 99\}$$

- **Sum Invariant**: $\sum_{n=0}^{99} 0.27 = 27.0$ (exact).
- **Parameters**: None (parameter-free structural reference).

### 3.2 Model B1: Rolling Count Baseline
An empirical trailing sample mean estimator computed over a fixed rolling historical window of $W$ draw days:

$$\hat{\mu}_{\text{B1}}[t, n; W] = \frac{1}{W} \sum_{i=1}^{W} C[t - i, n]$$

- **Sum Invariant**:

$$\sum_{n=0}^{99} \hat{\mu}_{\text{B1}}[t, n; W] = \frac{1}{W} \sum_{i=1}^{W} \sum_{n=0}^{99} C[t - i, n] = \frac{1}{W} \sum_{i=1}^{W} 27 = 27.0$$

- **Development Parameter**: Window size $W \in \mathbb{N}$. The exact window size for confirmatory comparison is evaluated during development and frozen in `preregistration.json` prior to confirmatory execution.

### 3.3 Model M1: EWMA Count Model
An exponentially weighted moving average count estimator with smoothing parameter $\alpha = 1 - \exp(-\ln(2) / H)$ corresponding to half-life $H > 0$ draw days:

$$\hat{\mu}_{\text{M1}}[t, n; H] = \alpha \sum_{k=0}^{\infty} (1 - \alpha)^k C[t - 1 - k, n]$$

In practical finite histories of length $K$, weights are normalized:

$$w_k = (1 - \alpha)^k, \quad \tilde{w}_k = \frac{w_k}{\sum_{j=0}^{K-1} w_j}, \quad \hat{\mu}_{\text{M1}}[t, n; H] = \sum_{k=0}^{K-1} \tilde{w}_k C[t - 1 - k, n]$$

- **Sum Invariant**: $\sum_{n=0}^{99} \hat{\mu}_{\text{M1}}[t, n; H] = \sum_{k=0}^{K-1} \tilde{w}_k (27) = 27.0$.
- **Development Parameter**: Half-life $H > 0$. The exact half-life is evaluated during development and frozen in `preregistration.json` prior to confirmatory execution.

### 3.4 Model M2: Dirichlet-Shrinkage Multinomial Model
A Bayesian count model treating the 27 daily draws as generated from a categorical distribution parameterized by simplex probabilities $\mathbf{p}_t = (p_{t, 0}, \dots, p_{t, 99})$ with $\sum_{n=0}^{99} p_{t, n} = 1$:

$$\text{Prior: } \mathbf{p}_t \sim \text{Dirichlet}(\boldsymbol{\alpha}_0), \quad \alpha_{0, n} = \frac{\beta}{100}$$

where $\beta > 0$ represents the total prior pseudo-count weight shrunk toward uniformity. Given observed counts $\mathbf{k}_{\text{obs}}[n] = \sum_{i=1}^{W} C[t - i, n]$ over window $W$:

$$\text{Posterior Mean: } \mathbb{E}[p_{t, n} \mid \text{data}] = \frac{\alpha_{0, n} + \mathbf{k}_{\text{obs}}[n]}{\sum_{m=0}^{99} (\alpha_{0, m} + \mathbf{k}_{\text{obs}}[m])} = \frac{\frac{\beta}{100} + \mathbf{k}_{\text{obs}}[n]}{\beta + 27 W}$$

$$\hat{\mu}_{\text{M2}}[t, n; W, \beta] = 27 \times \mathbb{E}[p_{t, n} \mid \text{data}]$$

- **Semantics**: Posterior mean probabilities scaled by total daily draw count 27.
- **Explicit Clarification**: M2 is strictly a Dirichlet-shrinkage Multinomial estimator over multinomial state probabilities. It does **NOT** assume or require that the true empirical data-generating process follows an overdispersed Dirichlet-Multinomial distribution.

---

## 4. System Architecture & Dependency Boundaries

```text
+-----------------------------------------------------------------------------+
|                          LEGACY XPIS REPOSITORY                             |
|                                                                             |
|   +-------------------+        +----------------------------------------+   |
|   |    DataLoader     |        |   src/probability/**, src/meta/**      |   |
|   | (builds df and S) |        |   src/decision/**, predictions/**      |   |
|   +---------+---------+        +----------------------------------------+   |
+-------------|---------------------------------------------------------------+
              |
              | DataLoader.df (Plain dates + 27 draw columns)
              v
+-----------------------------------------------------------------------------+
|                     XPIS v2 RESEARCH ADAPTER LAYER                          |
|             (extracts dates and draws, strips legacy S)                     |
+-------------------------------------+---------------------------------------+
                                      |
                                      | RawDrawBatch (Pure arrays)
                                      v
+-----------------------------------------------------------------------------+
|                          XPIS v2 COUNT CORE                                 |
|                                                                             |
|   +-----------------------+              +------------------------------+   |
|   |  src/count_v2/        |              |  src/count_v2/models/        |   |
|   |    contracts.py       |              |    uniform.py (B0)           |   |
|   |    dataset.py         |              |    rolling.py (B1)           |   |
|   +-----------+-----------+              |    ewma.py (M1)              |   |
|               |                          |    dirichlet_shrinkage.py(M2)|   |
|               +------------------------->+------------------------------+   |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   |  research_artifacts/xpis_v2_count_first/ (Isolated Artifact Store)  |   |
+---+---------------------------------------------------------------------+---+
```

### 4.1 Strict Core Isolation Rule
`COUNT_V2_CORE_DEPENDS_ON_V1 = NO`

The core research package `src/count_v2/**` must **NEVER** import from:
- `src/probability/**`
- `src/meta/**`
- `src/decision/**`
- `src/features/**`
- `src/evidence/**`
- `src/registry/**`
- `predictions/**`

`src/count_v2/**` must remain a standalone, self-contained Python package depending only on Python standard libraries, `numpy`, and `pandas`.

### 4.2 Research Adapter Architecture
`RESEARCH_ADAPTER_MAY_READ = DataLoader.df`

Legacy `DataLoader.load()` constructs both `df` (raw draws) and binary matrix `S` as a side effect. To avoid code duplication in exploratory development while preserving architectural isolation:
1. The exploratory development adapter may invoke `DataLoader` to retrieve `DataLoader.df`.
2. The adapter must immediately extract raw dates and the 27 prize columns, discarding `S`.
3. The adapter converts raw tabular data into validated `RawDrawBatch` objects.
4. `src/count_v2/` core modules must **NOT** import `DataLoader` directly.

### 4.3 Confirmatory Ingestion Boundary
Confirmatory ingestion does **NOT** read `DataLoader.df` from the mutable local working tree. Confirmatory execution resolves raw data explicitly via immutable snapshot manifests anchored by cryptographic commit hashes and blob hashes (see Section 23).

### 4.4 Intended Slice 1 File Layout
```text
src/count_v2/
    __init__.py
    contracts.py
    dataset.py
    models/
        __init__.py
        uniform.py
        rolling.py
        ewma.py
        dirichlet_shrinkage.py

backtests/
    count_v2_research.py

tests/
    count_v2/
        test_contracts.py
        test_dataset.py
        test_models.py
        test_research_adapter.py
```

*(Note: These files represent the intended Slice 1 development scope and are NOT created during design spec gates).*

---

## 5. Data Contracts & Type Specifications

All data exchanges within XPIS v2 are governed by explicit dataclass / TypedDict contracts.

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

@dataclass(frozen=True)
class RawDrawBatch:
    """Raw draw observations ingested from lottery records."""
    dates: np.ndarray          # Shape: (N,), dtype: '<U10' (YYYY-MM-DD, sorted strictly ascending)
    draws: np.ndarray          # Shape: (N, 27), dtype: int8/int16, range: [0, 99]

@dataclass(frozen=True)
class CountHistory:
    """Historical aggregate count matrix strictly preceding target evaluation."""
    dates: np.ndarray          # Shape: (H,), dtype: '<U10'
    counts: np.ndarray         # Shape: (H, 100), dtype: int8/int16, row sums == 27

@dataclass(frozen=True)
class CountOutcome:
    """Observed count outcome for a single target date."""
    target_date: str           # 'YYYY-MM-DD'
    observed_counts: np.ndarray # Shape: (100,), dtype: int8/int16, sum == 27

@dataclass(frozen=True)
class CountForecast:
    """Point and distributional forecast emitted by a count model."""
    target_date: str           # 'YYYY-MM-DD'
    history_start: str         # Earliest date in model-visible history
    history_end: str           # Latest date in model-visible history (strictly < target_date)
    expected_count: np.ndarray # Shape: (100,), float64, sum == 27.0 +/- 1e-6
    model_identity: str        # e.g., Model identifier string
    mean_standard_error: Optional[np.ndarray] = None # Shape: (100,), SE of estimated mean
    mean_lower_bound: Optional[np.ndarray] = None    # Shape: (100,), LCB of mean
    mean_upper_bound: Optional[np.ndarray] = None    # Shape: (100,), UCB of mean
    predictive_distribution: Optional[Dict[str, Any]] = None # Optional distributional params
    prediction_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None # (lower, upper) on draws
```

---

## 6. Dataset Validation & Integrity Rules

### 6.1 Fail-Closed Ingestion Rules
Dataset construction from raw observations must fail immediately and raise an explicit runtime exception (e.g., `DatasetValidationError`) upon encountering any of the following defects:
1. **Duplicate Dates**: Multiple rows with identical date strings.
2. **Non-Monotonic Chronology**: Dates not strictly sorted in ascending chronological order ($t_i \ge t_{i+1}$).
3. **Invalid Draw Count**: Any row containing fewer or more than exactly 27 draw entries.
4. **Out-of-Range Draws**: Any prize number $n < 0$ or $n > 99$.
5. **Non-Integral Draws**: Any missing (`NaN`), null, or non-integer value.
6. **Dimensional Inconsistency**: Mismatched array dimensions between dates and draw matrices.
7. **Row Count Sum Violation**: Transformed count row where $\sum_{n=0}^{99} C[t, n] \ne 27$.

### 6.2 Prohibition of Silent Corrections
The validation layer must **NEVER**:
- Silently drop malformed or incomplete rows.
- Impute or forward-fill missing draws.
- Clip or wrap out-of-range numbers into $[0, 99]$.
- Silently re-normalize an invalid count row to sum to 27.

### 6.3 Explicit Exception Handling
Runtime validation must not rely on Python `assert` statements (which may be optimized away via `-O` compiler flags). All checks must evaluate explicit conditionals and raise structured errors.

---

## 7. Chronology & Temporal Leakage Contract

### 7.1 Strict Precedence Rule
To eliminate lookahead bias and temporal leakage, model-visible history is governed by strict inequality:

$$\forall d \in \mathcal{H}_{\text{visible}}(t): \quad \text{date}(d) < t$$

### 7.2 Calendar Gaps and Non-Adjacency
Models must **NOT** assume calendar adjacency (e.g., $\text{date}(t) - 1\text{ day}$). Calendar gaps (e.g., Lunar New Year closures or missed draws) are valid. The history slice comprises all recorded draw observations strictly prior to target date $t$, regardless of calendar elapsed days.

### 7.3 Separation of Model API and Target Outcome
The model evaluation API must strictly decouple forecast generation from outcome evaluation:
1. `model.predict_count(history: CountHistory, target_date: str) -> CountForecast`
2. `evaluator.evaluate(forecast: CountForecast, outcome: CountOutcome)`

The model API does not accept, observe, or receive the target outcome.

### 7.4 Date Matching Invariant
During evaluation, the evaluation harness must assert:

$$\text{forecast}.\text{target\_date} == \text{outcome}.\text{target\_date}$$

Any discrepancy or inclusion of target-date observations in `CountHistory` constitutes an **Integrity Invalidation** event.

---

## 8. Forecast Representation & Distributional Transformations

### 8.1 Canonical Mean Representation
The primary output of every XPIS v2 model is the conditional expected count vector $\hat{\mu}_t \in \mathbb{R}_{\ge 0}^{100}$ summing to 27.0.

### 8.2 Non-Equivalence to Binary Exceedance
A point forecast of expected count does **NOT** universally imply binary hit probability via Poisson approximation:

$$P(C[t, n] > 0) \ne 1 - \exp(-\hat{\mu}_t[n]) \quad \text{(in general)}$$

The count data permits repeated two-digit outcomes within one date ($C[t, n] > 1$). The transformation $P(C > 0) = 1 - e^{-\mu}$ is mathematically exact **only** if the marginal count distribution is strictly Poisson. Because the draw process generates 27 total outcomes satisfying $\sum_{n=0}^{99} C[t, n] = 27$, the expected-count mean alone does not determine the full marginal predictive distribution, and empirical marginal count distributions may exhibit multi-state or departure from Poisson characteristics.

### 8.3 Distributional Probability Exposure Policy
A model may expose an exceedance probability $P(C[t, n] > 0)$ **only if** that model provides an explicit, theoretically justified, and preregistered predictive distribution $\mathcal{P}_n(c)$. In all other cases, models remain strictly point-mean count estimators.

---

## 9. Uncertainty Semantics & Estimator Variance

To avoid statistical conflation, the design rigorously distinguishes between two orthogonal uncertainties:

$$\begin{aligned}
\text{Uncertainty of Conditional Mean: } & \text{SE}(\hat{\mu}_t[n]) = \sqrt{\text{Var}\left(\hat{\mu}_t[n] \mid \mathcal{H}_{<t}\right)} \\
\text{Predictive Observation Uncertainty: } & \text{Dispersion of discrete draws } C[t, n] \sim \mathcal{P}_{t, n}(c)
\end{aligned}$$

A single ambiguous standard deviation $\sigma$ must **NEVER** be used.

```text
+-----------------------------------------------------------------------------+
|                            UNCERTAINTY TAXONOMY                             |
+-----------------------------------------------------------------------------+
|  1. ESTIMATOR UNCERTAINTY (SE of Estimated Conditional Mean)                |
|     - Measures statistical sampling error of the estimated parameter \mu   |
|     - Approaches 0 as sample size N -> \infty                               |
|     - Used for LCB selection rules: LCB_n = \hat{\mu}_n - z * SE(\hat{\mu}_n)|
+-----------------------------------------------------------------------------+
|  2. PREDICTIVE UNCERTAINTY (Observation Dispersion)                         |
|     - Measures intrinsic randomness/dispersion of future discrete draws     |
|     - Non-zero even with infinite historical data                           |
|     - Evaluated only under explicitly specified predictive distributions    |
+-----------------------------------------------------------------------------+
```

### 9.1 Baseline B0 Uncertainty
Model B0 is a fixed theoretical structural null ($\mu_n \equiv 0.27$).
- **Mean Estimation SE**: $\text{SE}(\hat{\mu}_{\text{B0}}[n]) = 0.0$ (exact).
- **Prohibition**: $\sqrt{0.27}$ is the Poisson standard deviation of a single draw observation, **NOT** the standard error of the estimated mean parameter.

### 9.2 Baseline B1 Uncertainty
For a rolling window of $W$ observations:
- **Mean Estimation SE**:

$$\text{SE}(\hat{\mu}_{\text{B1}}[n]) = \frac{s_n(W)}{\sqrt{W}}$$

where $s_n(W)$ is the sample standard deviation of $C[\tau, n]$ across the window $\tau \in [t - W, t - 1]$. The exact variance estimation method (e.g., sample variance vs. HAC estimator) is selected during development and frozen before confirmatory execution.

### 9.3 Model M1 Uncertainty
For finite normalized EWMA weights $\tilde{w}_k = \frac{w_k}{\sum_{j=0}^{K-1} w_j}$, the effective sample size is:

$$n_{\text{eff}} = \frac{1}{\sum_{k=0}^{K-1} \tilde{w}_k^2}$$

*(Note: As $K \to \infty$ under infinite geometric weighting, $n_{\text{eff}} \to \frac{2 - \alpha}{\alpha}$ as a limiting closed form).*

The exploratory variance approximation:

$$\text{SE}(\hat{\mu}_{\text{M1}}[n]) \approx \sqrt{\frac{\hat{\mu}_{\text{M1}}[n]}{n_{\text{eff}}}}$$

is classified as **Development Evidence Only**. If empirical diagnostics indicate residual autocorrelation or overdispersion, the uncertainty estimator must be updated during development and frozen in `preregistration.json` prior to confirmatory evaluation.

### 9.4 Model M2 Uncertainty
For the Dirichlet-shrinkage Multinomial model, the posterior covariance over simplex probabilities $\mathbf{p}$ is:

$$\text{Var}(p_n \mid \text{data}) = \frac{\tilde{\alpha}_n (\tilde{\alpha}_0 - \tilde{\alpha}_n)}{\tilde{\alpha}_0^2 (\tilde{\alpha}_0 + 1)}$$

where $\tilde{\alpha}_n = \alpha_{0, n} + \mathbf{k}_{\text{obs}}[n]$ and $\tilde{\alpha}_0 = \sum_{m=0}^{99} \tilde{\alpha}_m = \beta + 27 W$.
The standard error of the conditional expected count is:

$$\text{SE}(\hat{\mu}_{\text{M2}}[n]) = 27 \times \sqrt{\text{Var}(p_n \mid \text{data})}$$

---

## 10. Scientific Classification of Existing Count Research

All empirical count analyses, historical backtests, and exploratory experiments executed prior to the formal preregistration of XPIS v2 are formally classified as:

$$\text{EVIDENCE\_CLASS} = \text{DEVELOPMENT / EXPLORATORY}$$

This classification includes, but is not limited to:
- Historical 3-year count distribution studies.
- Rolling-window parameter sweep experiments.
- EWMA half-life sweeps and historical parameter findings.
- Pre-existing Lower Confidence Bound (LCB) ranking experiments.
- Previous permutation test runs and bootstrap estimations.
- Retrospective prospective logs from XPIS v1.2.

**Scientific Mandate**: Pre-existing positive confidence intervals or simulated returns do **NOT** constitute confirmatory out-of-sample evidence. Model architectures, half-lives, and selection thresholds developed using past data reflect post-hoc discovery and must be prospectively validated on unobserved future draws under frozen preregistration.

---

## 11. Development vs Confirmatory Separation

```text
+-----------------------------------------------------------------------------+
|                          DEVELOPMENT PHASE (EXPLORATORY)                    |
|                                                                             |
|  - Historical data up to T_freeze                                           |
|  - Model exploration (B0, B1, M1, M2)                                       |
|  - Hyperparameter tuning (Window W, Half-life H, Prior strength \beta)      |
|  - Uncertainty estimator calibration & diagnostic residual checks           |
|  - Candidate selection rule design & threshold exploration                  |
|  - Resampling methodology selection                                         |
|                                                                             |
|  OUTPUT: Preregistration Manifest & Freeze Manifest                         |
+-------------------------------------+---------------------------------------+
                                      |
                         [PREREGISTRATION FREEZE GATE]
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                        CONFIRMATORY PHASE (OUT-OF-SAMPLE)                   |
|                                                                             |
|  - Strictly prospective draws generated AFTER T_freeze                      |
|  - Immutable code, models, rules, thresholds, endpoints                     |
|  - Daily pre-outcome remote timestamped forecast persistence                |
|  - Matched comparator execution                                             |
|  - Append-only artifact ledger                                              |
|  - Dual-Axis Evaluation: Absolute Edge + Incremental Edge                   |
+-----------------------------------------------------------------------------+
```

### 11.1 Development Phase Scope
The development phase permits iterative data exploration, model comparison, diagnostic analysis, and hyperparameter optimization on historical data up to the freeze boundary $T_{\text{freeze}}$. All statistical conclusions drawn during this phase are strictly hypothesis-generating.

### 11.2 Confirmatory Phase Requirements
The confirmatory phase begins on the first draw date strictly after the completion of:
1. Architecture and design specification lock.
2. Source code implementation freeze.
3. Model and hyperparameter freeze.
4. Selection rule and threshold freeze.
5. Statistical preregistration commit.

### 11.3 Prohibition of Backfills and Reused Holdouts
Confirmatory evaluation strictly forbids:
- Backfilling historical dates as prospective holdout.
- Generating retroactive forecasts for past dates.
- Reusing the XPIS v1.2 180-day holdout dataset as confirmatory evidence for v2.
- Admitting any draw known prior to preregistration as confirmatory out-of-sample data.

---

## 12. Dual Independent Evaluation Axes

XPIS v2 enforces two orthogonal, non-hierarchical evaluation axes:

```text
AXIS 1: FORECAST VALIDITY  (Coherence, Calibration, Global Accuracy)
AXIS 2: SELECTIVE EDGE     (Economic Profitability, Matched Superiority)
```

```text
+-----------------------------------------------------------------------------+
|                        INDEPENDENT EVALUATION AXES                          |
+-------------------------------------+---------------------------------------+
|        AXIS 1: FORECAST VALIDITY    |         AXIS 2: SELECTIVE EDGE        |
|                                     |                                       |
|  - Global Mean Absolute Error (MAE) |  - Absolute Economic Edge:            |
|  - Global Root Mean Squared Error   |    LCB(Challenger PnL) > 0            |
|  - Poisson Deviance / Scoring Rule  |  - Incremental Matched Edge:          |
|  - Discrete Mean Calibration        |    LCB(PnL_chal - PnL_B1_match) > 0   |
|  - Evaluates all 100 numbers        |  - Evaluates Top-K Selected Subset    |
+-------------------------------------+---------------------------------------+
|  STATUS: PASS / FAIL / NOT_EVALUABLE|  STATUS: PASS / FAIL / NOT_EVALUABLE  |
+-------------------------------------+---------------------------------------+
```

### 12.1 Non-Hierarchical Architecture
The system must **NOT** implement evaluation as a sequential conditional gate (i.e., "Axis 1 must pass before Axis 2 is evaluated"). A model may have slightly higher global deviance across all 100 numbers while isolating sharp, statistically valid signal in the extreme upper tail ($k \le 2$). Both axes must be tracked, computed, and reported independently.

---

## 13. Axis 1: Forecast Validity Protocol

### 13.1 Validity Metrics
Forecast validity evaluates point and distributional forecast coherence across all 100 state dimensions:
1. **Count Mean Absolute Error (MAE)**:

$$\text{MAE}(t) = \frac{1}{100} \sum_{n=0}^{99} \left| \hat{\mu}_t[n] - C[t, n] \right|$$

2. **Count Root Mean Squared Error (RMSE)**:

$$\text{RMSE}(t) = \sqrt{\frac{1}{100} \sum_{n=0}^{99} (\hat{\mu}_t[n] - C[t, n])^2}$$

3. **Poisson Deviance**:

$$D(\mathbf{C}_t, \hat{\boldsymbol{\mu}}_t) = 2 \sum_{n=0}^{99} \left[ C[t, n] \ln\left(\frac{C[t, n]}{\hat{\mu}_t[n]}\right) - (C[t, n] - \hat{\mu}_t[n]) \right]$$

*(with the standard convention $0 \ln(0 / \mu) = 0$)*.

### 13.2 Proper Role of Poisson Deviance
Poisson deviance measures relative goodness-of-fit under a Poisson likelihood assumption. It is **NOT** an absolute arbiter of statistical truth. XPIS v2 explicitly forbids requiring that a model must achieve lower global Poisson deviance than B0 as an absolute prerequisite for testing selective tail edge.

### 13.3 Quantified Blocking Thresholds
Any performance threshold capable of triggering `forecast_validity_status = FAIL` (e.g., severe calibration collapse or gross mean divergence) must be:
1. Explicitly quantified (numerical bound).
2. Preregistered in `preregistration.json`.
3. Calibrated solely using development data.

No discretionary or qualitative failure criteria may be introduced during or after confirmatory holdout execution.

---

## 14. Axis 2: Selective Economic Edge Protocol

### 14.1 Economic Payoff Function
The empirical payout structure for Northern Vietnam Lottery (2-digit loto) is governed by:
- **Cost per bet**: 27,000 VND per number per day.
- **Payout per hit**: 99,000 VND per hit occurrence.

For a selected number $n$ on target date $t$ with observed count $C[t, n] \in \{0, 1, 2, \dots\}$:

$$\text{PnL}[t, n] = 99 \times C[t, n] - 27 \quad (\text{in thousands VND})$$

$$\text{ROI}[t, n] = \frac{99 \times C[t, n] - 27}{27} = \frac{11}{3} C[t, n] - 1$$

### 14.2 Multiplicity Preservation
The economic payoff function strictly preserves multiplicity. If number $n$ appears twice ($C[t, n] = 2$), $\text{PnL} = 99(2) - 27 = +171\text{k VND}$ ($\text{ROI} = +633.3\%$). It must **NEVER** be collapsed into binary hit/miss indicator $S[t, n]$.

### 14.3 Selection Rule & Lower Confidence Bound (LCB)
A candidate selection policy identifies eligible numbers for target date $t$ using a conservative estimator:

$$\text{LCB}_t[n] = \hat{\mu}_t[n] - z_{\text{crit}} \times \text{SE}(\hat{\mu}_t[n])$$

Numbers are ranked descending by $\text{LCB}_t[n]$. A portfolio of top-$K$ numbers meeting threshold $\theta_{\text{qual}}$ is selected:

$$\mathcal{S}_t = \{ n \in \text{argsort}(\text{LCB}_t)[-K:] \mid \text{LCB}_t[n] \ge \theta_{\text{qual}} \}$$

All policy parameters $(K, z_{\text{crit}}, \theta_{\text{qual}}, \text{SE method})$ must be evaluated on development data and frozen in `preregistration.json` prior to confirmatory execution.

---

## 15. Selective Edge Pass: Dual Requirement

To achieve a confirmatory `PASS` on Axis 2, the challenger model must satisfy the **Dual Edge Requirement**:

$$\text{SELECTIVE\_EDGE\_PASS} = \text{ABSOLUTE\_EDGE\_PASS} \land \text{INCREMENTAL\_EDGE\_PASS}$$

```text
+-----------------------------------------------------------------------------+
|                         DUAL SELECTIVE EDGE GATE                            |
+-----------------------------------------------------------------------------+
|                                                                             |
|   1. ABSOLUTE EDGE REQUIREMENT:                                             |
|      LCB_{1-\alpha}\left( \text{Challenger ROI Estimand} \right) > 0.0      |
|                                                                             |
|                             AND                                             |
|                                                                             |
|   2. INCREMENTAL MATCHED EDGE REQUIREMENT:                                  |
|      LCB_{1-\alpha}\left( \text{Challenger PnL} - \text{Matched B1 PnL}     |
|                    \right) > 0.0                                            |
|                                                                             |
+-----------------------------------------------------------------------------+
```

### 15.1 Absolute Edge Requirement
The lower bound of the $(1 - \alpha)$ confidence interval for the challenger's mean economic ROI must strictly exceed zero:

$$\text{LCB}_{1-\alpha}(\mathbb{E}[\text{ROI}_{\text{challenger}}]) > 0$$

### 15.2 Incremental Matched Edge Requirement
The lower bound of the $(1 - \alpha)$ confidence interval for the paired daily PnL difference against the primary matched comparator must strictly exceed zero:

$$\text{LCB}_{1-\alpha}\left( \mathbb{E}\left[ \text{PnL}_{\text{challenger}} - \text{PnL}_{\text{comparator}} \right] \right) > 0$$

### 15.3 Rejection of "Losing Less" as Economic Edge
A challenger model that outperforms a losing comparator but remains unprofitable overall does **NOT** possess an economic edge:

$$\text{Example: } \text{ROI}_{\text{challenger}} = -2.0\%, \quad \text{ROI}_{\text{comparator}} = -10.0\%, \quad \Delta = +8.0\%$$

In this scenario, incremental superiority is observed ($\Delta > 0$), but absolute edge is negative. Therefore:

$$\text{SELECTIVE\_EDGE\_PASS} = \text{FAIL}$$

---

## 16. Matched Exposure & Comparator Design

### 16.1 Target-Level Exposure Matching ($k_t$)
If the challenger model selects $k_t \in \{0, 1, \dots, K\}$ numbers on date $t$, every confirmatory comparator must be evaluated on exactly $k_t$ selections on date $t$.

### 16.2 Primary Matched B1 Comparator
The primary comparator is Model B1 (Rolling Count Mean) with matching exposure $k_t$:
- Rank numbers under B1 ranking (or $\text{LCB}_{\text{B1}, t}[n]$).
- Select the top $k_t$ numbers under B1 ranking.
- Compute paired daily difference: $\Delta_t = \sum_{n \in \mathcal{S}_{\text{chal}, t}} \text{PnL}[t, n] - \sum_{m \in \mathcal{S}_{\text{B1}, t}} \text{PnL}[t, m]$.

### 16.3 Random Matched Exposure Comparator
A secondary benchmark selecting a subset of $k_t$ unique numbers uniformly at random from $\{00, \dots, 99\}$ using a pre-committed, reproducible pseudorandom seed.

### 16.4 Uniform B0 Randomization Protocol
Because B0 assigns identical point forecasts ($0.27$) to all 100 numbers, ranked selection is degenerate. Any matched comparison against B0 must employ a preregistered deterministic tie-breaking or random sampling protocol.

### 16.5 Observation-Level Paired Differences
Statistical inference must be performed directly on observation-level or daily-level paired differences $\Delta_t$, rather than computing two uncoupled bootstrap distributions of aggregate ROI.

---

## 17. Pre-Outcome Persistence of Hypotheses & Comparators

To prevent post-hoc selection and cherry-picking, **ALL** confirmatory decision objects must be persisted to the immutable artifact ledger before the draw outcome is published.

```text
research_artifacts/xpis_v2_count_first/confirmatory/<experiment_id>/
  forecasts/
    <target_date>/
      <hypothesis_id>.json
```

The persisted daily bundle must contain:
1. Challenger full expected count vector $\hat{\boldsymbol{\mu}}_{\text{chal}, t}$.
2. Challenger selected numbers $\mathcal{S}_{\text{chal}, t}$ ($k_t$ items).
3. Primary matched B1 selected numbers $\mathcal{S}_{\text{B1}, t}$ ($k_t$ items).
4. Random matched selected numbers $\mathcal{S}_{\text{rand}, t}$ ($k_t$ items).
5. Any additional preregistered comparator selections.

**Strict Prohibition**: Comparator selections must **NEVER** be reconstructed retrospectively after observing the target outcome.

---

## 18. Multiple-Testing Governance & Statistical Inference

### 18.1 Primary Confirmatory Hypothesis Mandate
To maintain maximal statistical power and prevent family-wise error rate inflation, the confirmatory experiment must designate:

$$\text{EXACTLY ONE PRIMARY CONFIRMATORY HYPOTHESIS } (H_1^{\text{primary}})$$

### 18.2 Family-Wise Error Rate (FWER) Control & Claim Enumeration
If secondary confirmatory claims are preregistered, multiple-testing correction is mandatory across the complete confirmatory family $\mathcal{F}$:

$$\mathcal{F} = \{ \text{Challengers} \} \times \{ \text{Comparators} \} \times \{ \text{Endpoints} \}$$

- **Nominal Significance Level**: $\alpha = 0.05$.
- **Default Multiplicity Adjustment**: Holm-Bonferroni step-down procedure.
- **Claim-Level Preregistration**: The preregistration manifest must explicitly enumerate all individual inferential claims comprising $\mathcal{F}$, recording:
  - `claim_id`
  - `family_id`
  - `hypothesis_id`
  - `challenger_id`
  - `endpoint_id`
  - `comparator_id_or_absolute_null`
  - `primary_or_secondary`
- **Family Definition**: The membership of $\mathcal{F}$ must be fully frozen in `preregistration.json`. No endpoint or comparator may be added to or removed from $\mathcal{F}$ after the freeze.

---

## 19. Temporal Dependence & Resampling Methodology

### 19.1 Empirical Dependence Diagnostics
During development, the time series of paired daily differences $\Delta_t$ must be evaluated for temporal dependence using documented statistical diagnostic methods appropriate to the paired economic estimand (e.g., sample autocorrelation analysis, serial correlation tests).

### 19.2 Resampling Method Selection Protocol
Based on development diagnostic findings, development selects the inferential and resampling methodology:
1. If serial dependence is negligible, **Paired IID Bootstrap** may serve as the primary inferential tool, with Block Bootstrap reporting sensitivity checks.
2. If meaningful temporal dependence is present, **Moving Block Bootstrap (MBB)** or **Stationary Bootstrap** must be employed.

Development also selects the precision/power target, block selection rule (if applicable), and resample simulation count.

### 19.3 Pre-Freezing Resampling Strategy
The chosen resampling method, block length rule, simulation count, and pseudorandom seed protocol must be selected exclusively on development data and frozen in `preregistration.json`. Confirmatory results must never influence resampling selection.

---

## 20. Distributional Calibration Diagnostics

### 20.1 Mean Calibration Diagnostics
For point-mean models, calibration is evaluated by partitioning predicted expected counts into $M$ sorting bins and comparing bin-average predictions with observed average counts:

$$\text{ECE}_{\text{count}} = \sum_{m=1}^{M} \frac{|B_m|}{100 \cdot T} \left| \bar{\hat{\mu}}_{B_m} - \bar{C}_{B_m} \right|$$

### 20.2 Discrete Distributional Diagnostics
For models that output an explicit full predictive distribution $\mathcal{P}_{t, n}(c)$:
- **Randomized Probability Integral Transform (Randomized PIT)**: To handle discrete integer counts, uniform jitter is applied within the probability mass interval:

$$U_t[n] = F_{t, n}(C[t, n] - 1) + V \times P_{t, n}(C[t, n]), \quad V \sim \text{Uniform}(0, 1)$$

- **Prediction Interval Empirical Coverage**: Verification that nominal $(1 - \alpha)$ prediction intervals achieve nominal empirical coverage.
- **Prohibition**: Standard continuous PIT without randomized jitter must **NEVER** be applied to discrete count outcomes.

---

## 21. Experiment Duration, Power, and Stopping Rules

### 21.1 Minimum Duration Floor
$$\text{MINIMUM\_DAY\_FLOOR} = 180 \text{ prospective draw days}$$

The 180-day threshold is an operational floor, **NOT** an automatic trigger for statistical success.

### 21.2 Power and Precision Sizing
Prior to confirmatory launch, development simulations establish the statistical sample size and exposure required to bound the confidence interval width within target precision ($T_{\text{power\_target}}$).

### 21.3 Preregistered Stopping Rule
The confirmatory experiment terminates when:

$$T \ge \max\left(\text{MINIMUM\_DAY\_FLOOR}, T_{\text{power\_target}}\right)$$

Interim inspections during the holdout period are strictly observational. Early stopping for perceived efficacy or early failure is strictly forbidden.

---

## 22. Confirmatory Monitoring & Anti-Peeking Governance

### 22.1 Permissible Operational Monitoring
During the execution of a prospective confirmatory trial, the execution harness may monitor only:
- Successful draw data arrival and ingestion.
- Integrity of cryptographic hash validations.
- JSON schema and contract conformance.
- Pipeline execution health and logging.

### 22.2 Strict Prohibition of Retuning
The following modifications are strictly forbidden while an experiment is active:
- Altering model architecture, window size $W$, half-life $H$, or prior $\beta$.
- Modifying selection threshold $\theta_{\text{qual}}$ or exposure cap $K$.
- Changing uncertainty calculation methods.
- Swapping primary or secondary comparators.
- Modifying the primary hypothesis or confirmatory family $\mathcal{F}$.
- Prematurely stopping or restarting the experiment due to unfavorable interim PnL.

### 22.3 Consequences of Hypothesis Alteration
Any unauthorized modification to code, models, rules, or manifests immediately marks the experiment as:

$$\text{primary\_hypothesis\_status} = \text{INVALIDATED}$$

and invalidates all confirmatory claims.

---

## 23. Separation of Code Freeze, Data Feed, and Artifact Ledger

To reconcile immutable code execution with continuous daily data arrival and ledger appending, XPIS v2 establishes a tripartite authority separation:

```text
+-----------------------------------------------------------------------------+
|                          TRIPARTITE AUTHORITY MODEL                         |
+-----------------------------------------------------------------------------+
|  1. IMPLEMENTATION CODE AUTHORITY (FROZEN BEFORE CONFIRMATORY LAUNCH)       |
|     - Recorded upon completion and verification of implementation           |
|     - Immutable across entire confirmatory trial duration                   |
+-----------------------------------------------------------------------------+
|  2. DATA FEED AUTHORITY (EXPLICIT VERSIONED SNAPSHOTS)                      |
|     - Sourced from canonical data repository via commit-pinned snapshots    |
|     - Verified against source blob hash and history payload hash            |
|     - Does NOT require merging moving main into implementation worktree     |
+-----------------------------------------------------------------------------+
|  3. ARTIFACT LEDGER AUTHORITY (APPEND-ONLY MOVING HEAD)                     |
|     - Branch: codex/research/xpis-v2-count-first-<experiment_id>            |
|     - Appends daily forecasts, outcomes, and deviation records              |
|     - Protected against rebase, force-push, and deletion                    |
+-----------------------------------------------------------------------------+
```

### 23.1 Confirmatory Data Snapshot Contract
Confirmatory runs must not read untracked or dirty local CSV files. Each data ingestion event must record and verify:

```json
{
  "source_repository": "Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis",
  "source_data_commit": "<SHA1_commit_of_canonical_data>",
  "source_data_path": "data/xsmb-2-digits.csv",
  "source_blob_hash": "<git_blob_sha1>",
  "history_start": "2005-10-01",
  "history_end": "YYYY-MM-DD",
  "history_data_hash": "<sha256_of_canonical_history_bytes>",
  "ingested_at": "YYYY-MM-DDTHH:MM:SSZ"
}
```

---

## 24. Confirmatory Artifact Namespace & Ledger Storage

### 24.1 Directory Structure
All XPIS v2 research artifacts reside in a dedicated namespace outside legacy directories:

```text
research_artifacts/
  xpis_v2_count_first/
    development/
      <run_id>/
        run_manifest.json
        exploratory_metrics.json

    confirmatory/
      <experiment_id>/
        freeze_manifest.json
        preregistration.json
        forecasts/
          <target_date>/
            <hypothesis_id>.json
        outcomes/
          <target_date>/
            <outcome_record_id>.json
        protocol_deviations/
          <deviation_id>.json
        evaluation_summary.json
```

### 24.2 Derived Status of Summary Documents
`evaluation_summary.json` is strictly a generated downstream reporting artifact. The authoritative sources of truth are the individual immutable forecast and outcome record envelopes.

### 24.3 Confirmatory Ledger Branch
Confirmatory records are committed to a dedicated tracking branch:

$$\text{Branch: } \texttt{codex/research/xpis-v2-count-first-<experiment\_id>}$$

### 24.4 Remote Protection Enforcement
Before prospective confirmatory data collection begins, remote branch protection rules must be established and verified to prevent:
- Force-pushing (`git push --force`).
- Deleting the ledger branch.
- Rewriting commit history.

---

## 25. Cryptographic Hashing & Canonical Envelope Serialization

### 25.1 Envelope Pattern
To prevent self-referential hashing paradoxes, all persistent JSON records use an envelope structure where the hash covers only the inner `payload`:

```json
{
  "envelope_version": "2.0",
  "payload": {
    "target_date": "YYYY-MM-DD",
    "hypothesis_id": "<hypothesis_id>",
    "expected_count": [0.27, 0.27],
    "selected_numbers": [12, 85],
    "matched_b1_numbers": [12, 44],
    "matched_random_numbers": [3, 91],
    "created_at_utc": "YYYY-MM-DDTHH:MM:SSZ"
  },
  "payload_sha256": "<sha256_hex_digest>"
}
```

### 25.2 Single Deterministic Canonical Serialization Standard
Every `serialization_version` must define **EXACTLY ONE** deterministic canonical serialization algorithm. Before confirmatory launch, this algorithm must be fully specified, versioned, tested, and frozen in `freeze_manifest.json`.

The canonical serialization specification must unambiguously define:
1. **Character Encoding**: UTF-8 without byte order mark.
2. **Key Ordering**: Strictly lexicographical sort (`sort_keys=True`).
3. **Number Representation**: Exact, unambiguous numeric formatting without alternative representations. Non-finite values (`NaN`, `Infinity`, `-Infinity`) are strictly prohibited in payloads.
4. **Unicode Normalization**: Unicode Normalization Form C (NFC).
5. **Whitespace & Delimiters**: Compact JSON formatting (`separators=(',', ':')`) with no extraneous whitespace.
6. **Newline Handling**: Deterministic newline policy (trailing newline stripped before hashing).

**Prerequisite Gate**: The exact formal canonical serialization specification is:

$$\text{DEFERRED TO A LATER CONFIRMATORY-INTEGRITY DESIGN GATE}$$

Confirmatory data collection is strictly **BLOCKED** until this exact algorithm is specified and verified.

---

## 26. Pre-Outcome Remote Provenance & Timestamping

### 26.1 Server-Controlled Remote Timestamp Requirement
Local filesystem modification times and local Git commit timestamps are client-controlled and cannot serve as proof of pre-outcome generation.

$$\text{accepted\_forecast} \iff \text{Remote Timestamp} < \text{forecast\_cutoff\_utc}$$

A forecast is admissible as confirmatory evidence **only if** a trusted, immutable remote receipt verifies that the forecast payload was received before the preregistered daily draw cutoff time.

### 26.2 Forecast Cutoff Rule
XSMB lottery draws commence daily at 18:15:00 UTC+7 (11:15:00 UTC).
- **Preregistered Cutoff Rule**: An explicit `forecast_cutoff_utc` timestamp rule is selected during development and frozen in `freeze_manifest.json` prior to confirmatory launch (strictly preceding draw commencement).
- **Enforcement**: Any forecast record lacking a verified remote receipt timestamped prior to `forecast_cutoff_utc` on target date $t$ is strictly rejected from confirmatory evaluation.

### 26.3 Prerequisite Gate
The concrete remote timestamping infrastructure (e.g., GitHub Actions runner log with cryptographic attestation, commit timestamp verified by GitHub API, or RFC 3161 Time-Stamp Protocol) is:

$$\text{DEFERRED TO A LATER CONFIRMATORY-INTEGRITY DESIGN GATE}$$

**Prerequisite Condition**: Confirmatory data collection cannot transition to `CONFIRMATORY_RUNNING` until this remote receipt mechanism is fully specified, implemented, and verified in an authorized follow-on gate.

---

## 27. Write Semantics & Outcome Lifecycle

### 27.1 Atomic Create-Once Forecast Writes
Forecast persistence must enforce create-once write semantics:
- If a record for `(experiment_id, target_date, hypothesis_id)` already exists in the ledger, any subsequent write attempt is **REJECTED** and raises `DuplicateForecastError`.
- No forecast file may be overwritten, amended, or re-committed.

### 27.2 Immutable Outcome Ledger
Daily draw outcome records are written to `outcomes/<target_date>/<outcome_record_id>.json` and include full source provenance:
- `target_date`: 'YYYY-MM-DD'
- `observed_counts`: Array of shape (100,)
- `source_repository`: Canonical data repository identifier
- `source_data_commit`: Git commit SHA of raw data
- `source_data_path`: Path of raw data file
- `source_blob_hash`: SHA1 blob hash of data source
- `retrieved_at`: ISO 8601 UTC timestamp
- `payload_sha256`: Cryptographic hash of payload
- `outcome_record_id`: Unique identifier for outcome record

### 27.3 Explicit Superseding Outcome Correction Protocol
If an upstream data provider publishes an erroneous draw result and later corrects it:
1. The original outcome file is **NEVER** overwritten, modified, or deleted.
2. A new correction record `outcomes/<target_date>/<new_record_id>.json` is created.
3. The correction record must include full provenance metadata:
   - `supersedes_outcome_record_id`: ID of the superseded record.
   - `correction_reason`: Detailed explanation of upstream correction.
   - `source_repository`, `source_data_commit`, `source_data_path`, `source_blob_hash`.
   - `retrieved_at`, `payload_sha256`, `outcome_record_id`.
4. Downstream evaluation engines resolve outcomes using the latest valid correction record according to a frozen resolution rule while preserving the complete audit trail.

---

## 28. Experiment Status Model & Invalidation Protocol

### 28.1 Multi-Dimensional Status Taxonomy
XPIS v2 tracks three independent status dimensions:

```text
forecast_validity_status:
  - NOT_STARTED
  - RUNNING
  - PASS
  - FAIL
  - NOT_EVALUABLE

selective_edge_status:
  - NOT_STARTED
  - RUNNING
  - PASS
  - FAIL
  - NOT_EVALUABLE

primary_hypothesis_status:
  - NOT_STARTED
  - RUNNING
  - SUPPORTED
  - NOT_SUPPORTED
  - INVALIDATED
```

### 28.2 Statistical Failure vs. Integrity Invalidation
A clear boundary separates statistical performance outcomes from integrity violations:

```text
+-----------------------------------------------------------------------------+
|                     FAILURE VS INVALIDATION TAXONOMY                        |
+-----------------------------------------------------------------------------+
|  1. STATISTICAL OUTCOME (Valid Experiment, Hypothesis Rejected)             |
|     - Lower confidence bound fails to exceed 0 (LCB <= 0)                   |
|     - Incremental performance fails to beat matched B1                      |
|     - Model produces higher error than preregistered threshold              |
|     RESULT: selective_edge_status = FAIL, hypothesis = NOT_SUPPORTED        |
+-----------------------------------------------------------------------------+
|  2. INTEGRITY INVALIDATION (Experiment Compromised / Broken Protocol)       |
|     - Temporal lookahead leakage (history contains target date)             |
|     - Late forecast (remote timestamp >= cutoff)                            |
|     - Payload cryptographic hash mismatch                                   |
|     - Data contract or row-sum conservation violation                       |
|     - Code, rule, or hyperparameter drift during holdout                    |
|     - Unauthorized preregistration mutation                                 |
|     RESULT: selective_edge_status = NOT_EVALUABLE, hypothesis = INVALIDATED |
+-----------------------------------------------------------------------------+
```

### 28.3 Diagnostic Status Matrix

| Forecast Validity | Selective Edge | Protocol Integrity | Primary Hypothesis Status | Scientific Interpretation |
|---|---|---|---|---|
| `PASS` | `PASS` | `VALID` | `SUPPORTED` | Dual edge confirmed; valid point forecasts. |
| `FAIL` | `PASS` | `VALID` | `SUPPORTED` (Conditional) | Tail selection holds despite global metric failure. |
| `PASS` | `FAIL` | `VALID` | `NOT_SUPPORTED` | Coherent forecasts, but no economic edge. |
| `FAIL` | `FAIL` | `VALID` | `NOT_SUPPORTED` | Complete statistical failure on both axes. |
| `NOT_EVALUABLE` | `NOT_EVALUABLE` | `COMPROMISED` | `INVALIDATED` | Experiment void due to protocol or integrity breach. |

---

## 29. Freeze Manifest & Preregistration Specifications

### 29.1 Freeze Manifest Structure (`freeze_manifest.json`)
*(Non-normative schema illustration; concrete parameter values must be calibrated on development data and frozen in preregistration)*

```json
{
  "experiment_id": "<experiment_id>",
  "family_id": "<family_id>",
  "primary_hypothesis_id": "<primary_hypothesis_id>",
  "frozen_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "implementation": {
    "commit": "<accepted_implementation_commit_sha>",
    "tree_hash": "<accepted_implementation_tree_sha>",
    "code_repository": "Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis"
  },
  "models": {
    "challenger": {
      "id": "<challenger_model_id>",
      "version": "1.0.0",
      "parameters": {},
      "spec_hash": "<sha256>"
    },
    "primary_comparator": {
      "id": "<comparator_model_id>",
      "version": "1.0.0",
      "parameters": {},
      "spec_hash": "<sha256>"
    }
  },
  "selection_rule": {
    "method": "LCB_RANKING",
    "parameters": {},
    "spec_hash": "<sha256>"
  },
  "serialization": {
    "canonical_version": "v2_canonical_standard",
    "hash_algorithm": "SHA256"
  },
  "protocol": {
    "forecast_cutoff_utc": "<preregistered_cutoff_time_rule>",
    "minimum_day_floor": 180,
    "power_target_n": "<preregistered_sample_size>",
    "correction_resolution_rule": "USE_LATEST_SUPERSEDING"
  }
}
```

### 29.2 Preregistration Structure (`preregistration.json`)
*(Non-normative schema illustration; concrete parameter values must be calibrated on development data and frozen in preregistration)*

```json
{
  "preregistration_version": "1.0.0",
  "experiment_id": "<experiment_id>",
  "primary_hypothesis": {
    "id": "<primary_hypothesis_id>",
    "statement": "<formal_scientific_hypothesis_statement>",
    "challenger_model": "<challenger_model_id>",
    "primary_matched_comparator": "<primary_matched_comparator_id>",
    "absolute_edge_estimand": "MEAN_PORTFOLIO_ROI",
    "incremental_edge_estimand": "MEAN_PAIRED_DAILY_PNL_DIFF",
    "nominal_alpha": 0.05
  },
  "confirmatory_family": [
    {
      "claim_id": "CLAIM-01-ABSOLUTE",
      "family_id": "<family_id>",
      "hypothesis_id": "<primary_hypothesis_id>",
      "challenger_id": "<challenger_model_id>",
      "endpoint_id": "MEAN_PORTFOLIO_ROI",
      "comparator_id_or_absolute_null": "ABSOLUTE_NULL_ZERO",
      "primary_or_secondary": "PRIMARY"
    },
    {
      "claim_id": "CLAIM-02-INCREMENTAL",
      "family_id": "<family_id>",
      "hypothesis_id": "<primary_hypothesis_id>",
      "challenger_id": "<challenger_model_id>",
      "endpoint_id": "MEAN_PAIRED_DAILY_PNL_DIFF",
      "comparator_id_or_absolute_null": "<primary_matched_comparator_id>",
      "primary_or_secondary": "PRIMARY"
    }
  ],
  "multiplicity_correction": "HOLM_BONFERRONI",
  "statistical_inference": {
    "resampling_method": "<preregistered_resampling_method>",
    "resample_count": "<preregistered_resample_count>",
    "block_size_rule": "<preregistered_block_rule>",
    "random_seed": "<preregistered_seed>"
  },
  "blocking_thresholds": {
    "max_acceptable_mean_ece": "<preregistered_threshold>",
    "max_acceptable_mae": "<preregistered_threshold>"
  },
  "invalidation_criteria": [
    "LEAKAGE_TARGET_DATE_IN_HISTORY",
    "REMOTE_TIMESTAMP_AFTER_CUTOFF",
    "PAYLOAD_HASH_MISMATCH",
    "IMMUTABLE_FILE_OVERWRITE_ATTEMPT",
    "DATASET_SUM_CONSERVATION_VIOLATION"
  ]
}
```

---

## 30. Verification & Test Architecture

### 30.1 Slice 1 Correctness Test Suite
Slice 1 tests verify mathematical, structural, and contract correctness only. They do **NOT** test or assert economic edge.

Required test areas:
1. **Contract Invariants**: Serialization/deserialization round-trips for `RawDrawBatch`, `CountHistory`, `CountForecast`, and `CountOutcome`.
2. **Dataset Validation**: Strict fail-closed verification on duplicate dates, missing draws, non-integral values, and out-of-range numbers.
3. **Sum Conservation**: Proof that all transformed count matrices and model expected count vectors satisfy $\sum C = 27$ and $\sum \hat{\mu} = 27.0 \pm 10^{-6}$.
4. **Model Determinism**: Exact reproducibility of point predictions for B0, B1, M1, and M2 across repeated evaluations.
5. **Chronological Insulation**: Verification that `CountHistory` rejects target-date data and tolerates historical calendar gaps.
6. **Research Adapter Isolation**: Verification that the research adapter functions correctly using `DataLoader.df` without importing `src/probability/`, `src/meta/`, or `src/decision/`.

### 30.2 Prohibition of Edge Assertions in Unit Tests
Unit and integration test suites must **NEVER** contain assertions asserting profitability or empirical dominance, such as:
- Asserting model ROI exceeds zero.
- Asserting model PnL exceeds comparator PnL.
- Asserting specific historical picks recur.

All correctness tests must assert only mathematical invariants, numerical tolerances, determinism, and contract schema validity.

### 30.3 Confirmatory Integrity Test Suite
Prior to transitioning any confirmatory experiment to `CONFIRMATORY_RUNNING`, the follow-on confirmatory test harness must verify the full integrity test suite:
1. **Forecast Overwrite Prevention**: Attempting to overwrite an existing forecast record is rejected.
2. **Duplicate Forecast Identity**: Attempting to submit duplicate forecasts for `(experiment_id, target_date, hypothesis_id)` is rejected.
3. **Strict Chronological Leakage**: Slices containing `history_date >= target_date` are rejected.
4. **Date Consistency**: Forecast target date mismatch against outcome target date is rejected.
5. **Cutoff Compliance**: Forecasts lacking trusted remote receipts prior to `forecast_cutoff_utc` are rejected.
6. **Late Forecast Rejection**: Any late submission is rejected.
7. **Preregistration Immutability**: Any mutation of `preregistration.json` after freeze is rejected.
8. **Manifest Immutability**: Any mutation of `freeze_manifest.json` after freeze is rejected.
9. **Payload Hash Integrity**: Any payload SHA256 digest mismatch is rejected.
10. **Canonical Serialization Version**: Any mismatched serialization version is rejected.
11. **Code & Rule Hash Consistency**: Unknown implementation, model, or rule hashes are rejected.
12. **Confirmatory Family Consistency**: Mismatches between submitted claims and preregistered family $\mathcal{F}$ are rejected.
13. **Comparator Bundle Completeness**: Incomplete comparator submissions for target date $t$ are rejected.
14. **Unversioned Snapshot Rejection**: Data snapshots lacking full provenance commits are rejected.
15. **Blob Hash Verification**: Data commits with mismatched blob hashes are rejected.
16. **History Hash Verification**: Data snapshots with mismatched `history_data_hash` digests are rejected.
17. **Outcome Overwrite Prevention**: Attempting to overwrite existing outcome records is rejected.
18. **Superseding Correction Integrity**: Outcome corrections lacking explicit `supersedes_outcome_record_id` links and full provenance are rejected.

**Failure Semantics**: Any violation of these integrity rules produces:

$$\text{primary\_hypothesis\_status} = \text{INVALIDATED}, \quad \text{axis\_status} = \text{NOT\_EVALUABLE}$$

Integrity violations must **NEVER** be reported as ordinary statistical `FAIL`.

---

## 31. Operational Lifecycles & Production Lock

### 31.1 Git & Worktree Lifecycle
- **Design Base**: Anchored strictly to `067296fbe4d38edbe60fee9a062434c6e06396c4` (Tree: `57a1771d2f36a8653c4fd19958f7079707e9b51f`).
- **Implementation Code**: The future accepted implementation commit and tree hash will be recorded and frozen in `freeze_manifest.json` before confirmatory launch, remaining immutable across the holdout.
- **Data Ingestion**: Pinned via explicit snapshot metadata referencing upstream canonical commits.
- **Ledger Storage**: Append-only updates on dedicated research branch `codex/research/xpis-v2-count-first-<experiment_id>`.
- **Branch Synchronization**: No `git pull`, silent rebase, or merge from `main` into the implementation worktree is permitted during an active slice. Any code synchronization requires an explicit reconciliation gate adjudicated by the ChatGPT Control Plane.

### 31.2 Session Lifecycle & Agent Routing
To preserve conversational context integrity, all future engineering activities are partitioned into dedicated sessions:

```text
1. Architecture Design:            ChatGPT Control Plane Session (Completed)
2. Design Spec Authoring:           Current Gemini 3.7 Flash Session (Active)
3. Design Spec Revision:            Continue current session (Documentation only)
4. Implementation Planning:         NEW SESSION REQUIRED
5. Implementation Slice 1:          NEW SESSION REQUIRED
6. Slice 1 Fixes / Unit Testing:    Continue Slice 1 session
7. Subsequent Major Slices:         NEW SESSION REQUIRED
8. Any Repo / SHA / Path Ambiguity: STOP -> NEW SESSION REQUIRED
```

### 31.3 Production Lock & Zero Live Action
XPIS v2 is strictly an off-line research system.
- `PRODUCTION_INTEGRATION = FORBIDDEN`
- `LIVE_ACTION = FORBIDDEN`
- `AUTOMATIC_PROMOTION = FORBIDDEN`

Even in the event of `primary_hypothesis_status = SUPPORTED` with simultaneous passes on Axis 1 and Axis 2, XPIS v2 code and models remain quarantined in research. Any production deployment or live operational execution requires a separate, formal, multi-party authorization gate.

---

## 32. Deferred Mechanisms & Future Milestones

### 32.1 Explicitly Deferred Mechanisms
To maintain a focused and achievable Slice 1 scope, the following mechanisms are intentionally deferred:

1. **Automated Remote Timestamp & Receipt Protocol**:
   - **Status**: DEFERRED TO A LATER CONFIRMATORY-INTEGRITY DESIGN GATE.
   - **Prerequisite**: Must be designed, verified, and demonstrated before any prospective confirmatory experiment enters `CONFIRMATORY_RUNNING`.

2. **Single Deterministic Canonical Serialization Specification**:
   - **Status**: DEFERRED TO A LATER CONFIRMATORY-INTEGRITY DESIGN GATE.
   - **Prerequisite**: Must be fully specified and tested before confirmatory data collection is unblocked.

3. **Machine-Learned Count Models**:
   - **Status**: DEFERRED.
   - **Prerequisite**: Machine-learned count models are deferred. Any later ML slice requires a separate ChatGPT Control Plane design / authorization gate. It is NOT part of Slice 1.

---

## 33. Design Spec Self-Review Summary

- **Placeholders**: Verified zero unresolved placeholders or stub tags across all sections. All deferred components are explicitly assigned to follow-on design gates with specified prerequisites.
- **Internal Consistency**: Verified full alignment across count target invariants ($\sum C = 27$), dual-axis independence, dual selective edge criteria ($\text{Absolute} \land \text{Incremental}$), failure vs. invalidation taxonomy, payload hashing envelopes, and tripartite authority separation.
- **Scope Integrity**: Strictly constrained to minimal development research infrastructure (B0, B1, M1, M2). Complex ML, ensembling, and production features are quarantined.
- **Ambiguity Elimination**: All statistical choices affecting confirmatory inference are either fixed by this specification or mandated to be frozen in preregistration manifests prior to holdout launch.
