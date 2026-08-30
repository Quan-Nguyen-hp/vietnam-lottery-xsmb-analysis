# XPIS v2 Count-First Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal XPIS v2 count-first development/exploratory core with leakage-safe count contracts, B0/B1/M1/M2 models, a DataLoader research adapter, historical development evaluation, and correctness tests.

**Architecture:** `src/count_v2/**` is independent from XPIS v1 semantics. Only the research adapter may read `DataLoader.df`, converting raw 27-observation rows into canonical `C[N,100]` count data. Models produce coherent expected-count vectors summing to 27 and remain development/research-only.

**Tech Stack:** Python, NumPy, Pandas, pytest, existing repository evaluation utilities where compatible.

**Spec:** `docs/superpowers/specs/2026-08-30-xpis-v2-count-first-design.md`

## Global Constraints

- `SLICE_1 = DEVELOPMENT / EXPLORATORY COUNT RESEARCH CORE` and every historical result must carry `EVIDENCE_CLASS = DEVELOPMENT / EXPLORATORY`.
- `COUNT_V2_CORE_DEPENDS_ON_V1 = NO`: `src/count_v2/**` may depend only on the Python standard library, NumPy, and Pandas.
- `src/count_v2/**` must not import `src.probability`, `src.meta`, `src.decision`, `src.features`, `src.evidence`, `src.registry`, `src.data.loader`, `predictions`, or `DataLoader`.
- Only `backtests/count_v2_research.py` may import `DataLoader`; it may read `DataLoader.df` and `DataLoader.prize_cols()` but must not consume `DataLoader.S`.
- Do not modify `src/data/loader.py`, XPIS v1.2 models, thresholds, policies, prediction logs, or production behavior.
- The canonical raw batch has `dates.shape == (N,)`, `draws.shape == (N, 27)`, strictly increasing canonical `YYYY-MM-DD` dates, and integral draws in `[0, 99]`.
- The canonical count matrix has `C.shape == (N, 100)`, non-negative integral values, and every row sums exactly to `27`; repeated values in one raw row are valid.
- Derived `S = C > 0` is diagnostic only and is not a model input.
- Validation fails closed with explicit exceptions; it must not sort, drop, impute, clip, wrap, or normalize invalid data.
- Every model-visible history date must be strictly less than the target date. Calendar gaps are valid and calendar adjacency is never required.
- Forecast generation and outcome evaluation remain separate: `predict_count(history, target_date)` never receives the outcome.
- Every `CountForecast.expected_count` has shape `(100,)`, dtype `float64`, non-negative finite values, and absolute sum error at most `1e-6` from `27.0`.
- B0, B1, M1, and M2 must conserve 27 by construction. No generic post-hoc scaling may conceal a broken model; violations raise `ModelContractError` with code `MODEL_CONTRACT_FAILURE`.
- Expected count does not imply `P(any) = 1 - exp(-mu)`. Slice 1 exposes no generic binary probability conversion.
- `mean_standard_error`, mean bounds, `predictive_distribution`, and `prediction_interval` remain distinct optional fields. Mean-estimation uncertainty must not be labeled predictive uncertainty.
- B1 window `W` and M2 window `W` count historical draw rows, not calendar days. M1 half-life `H` also operates over historical draw rows.
- B1 development windows, M1 half-lives, M2 windows, prior strengths, uncertainty choices, and research Top-K values are exploratory parameters, not confirmatory selections.
- M2 terminology is exactly “Dirichlet-shrinkage Multinomial”; it does not assert a Dirichlet-Multinomial empirical data-generating process.
- Development artifacts may be written only below `research_artifacts/xpis_v2_count_first/development/<run_id>/`; no XPIS v2 artifact may be written below `predictions/**`.
- Slice 1 computes exploratory metrics without confirmatory adjudication. Tests must not assert positive ROI, model dominance, or recurring historical picks.
- Slice 1 excludes LightGBM, Negative Binomial ML, ensembles, MetaFusion, FeatureStore, EvidenceStore, registry work, Kelly sizing, allocation optimization, production integration, live betting, confirmatory receipts/ledgers, branch protection, prospective confirmatory execution, DataLoader refactoring, and preserved-WIP integration.
- Every Git command must use `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" ...` and the implementation session must not pull, merge, rebase, cherry-pick, push, or use `wip/test-harness-compat-20260830`.

---

## Execution Preconditions

- **Design authority:** branch `codex/design/xpis-v2-count-first-20260830`, plan/spec worktree `G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-design-20260830`, frozen design commit `58f2e19ca30cf5766aaf07df1f1f65133e67c757`.
- **Implementation branch:** `codex/feature/xpis-v2-count-first-slice1-20260830`.
- **Implementation worktree:** `G:\\MEGAsync\\MR_BOM\\XPIS_WORKTREES\\xpis-v2-count-first-slice1-20260830`.
- **Implementation base:** the exact accepted plan commit produced by this planning gate.
- **Required setup skill before Task 1:** `superpowers:using-git-worktrees`.
- `DESIGN_BRANCH_CODE_COMMITS = FORBIDDEN`.
- `IMPLEMENTATION_WORKTREE_BASE_MUST_EQUAL_ACCEPTED_PLAN_COMMIT = YES`.
- The future execution session must verify that the implementation branch and worktree do not already exist. If either exists, STOP without overwriting or reusing it unless ChatGPT Control Plane explicitly adjudicates that state.
- The future execution session creates the implementation worktree only after this revised plan is accepted. This planning-remediation gate must not create the branch or worktree.
- Before Task 1, run `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" branch --show-current`, `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" rev-parse HEAD`, and `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" status --short`. Require the branch to equal `codex/feature/xpis-v2-count-first-slice1-20260830`, HEAD to equal the `PLAN_COMMIT` accepted by ChatGPT Control Plane, and status output to be empty.

---

## FILE STRUCTURE DECISION

### CREATE

- `src/count_v2/__init__.py` — public exports for contracts, dataset conversion, and the four models.
- `src/count_v2/contracts.py` — frozen array contracts, canonical date validation, structured exceptions, serialization, chronology checks, and forecast conservation enforcement.
- `src/count_v2/dataset.py` — fail-closed DataFrame-to-raw conversion, raw-to-count transformation, history slicing, and outcome extraction.
- `src/count_v2/models/__init__.py` — `CountModel` protocol and exports for exactly B0/B1/M1/M2.
- `src/count_v2/models/uniform.py` — parameter-free B0 uniform expected-count baseline.
- `src/count_v2/models/rolling.py` — B1 trailing-row sample-mean baseline.
- `src/count_v2/models/ewma.py` — M1 finite normalized EWMA and development-only mean-SE approximation.
- `src/count_v2/models/dirichlet_shrinkage.py` — M2 uniform-prior Dirichlet-shrinkage Multinomial posterior mean and mean SE.
- `backtests/count_v2_research.py` — the only legacy DataLoader adapter, leakage-safe walk-forward development runner, metric aggregation, CLI, and isolated artifact writer.
- `tests/count_v2/test_contracts.py` — contract, serialization, dtype, chronology, optional uncertainty, and forecast conservation tests.
- `tests/count_v2/test_dataset.py` — exhaustive raw ingestion and canonical count transformation tests.
- `tests/count_v2/test_models.py` — deterministic mathematical tests for exactly B0/B1/M1/M2.
- `tests/count_v2/test_research_adapter.py` — DataLoader adapter, leakage, metrics, artifacts, CLI boundary, and static core-isolation tests.

### MODIFY

- None. Repository inspection found no need to alter an existing implementation, test, config, or policy file.

### READ ONLY

- `src/data/loader.py` — actual DataLoader API, Timestamp date representation, DataFrame layout, and `prize_cols()` convention.
- `src/probability/count_poisson.py` — legacy Poisson conversion is explicitly not reused by the count-first core.
- `src/probability/count_expectation.py` — reference only for existing EWMA row-weighting and effective-sample-size semantics.
- `src/evaluation/metrics.py` — adapter-level reuse of count MAE, RMSE, Poisson deviance, count calibration error, break-even count, EV, and ROI semantics.
- `predictions/evaluation_policy.json` — read-only confirmation of 27/99 economics and production lock; it is not an XPIS v2 configuration source.
- `backtests/count_expectation_research.py` — legacy research CLI/artifact conventions and known leakage-safe slicing pattern.
- `tests/test_count_evaluation.py`, `tests/test_count_poisson_model.py`, `tests/test_model_12_count_poisson.py`, `tests/test_count_challenger.py`, `tests/test_count_challenger_evaluator.py` — bounded legacy count regression suite.
- `pyproject.toml` — Python 3.14, NumPy 2.4.1, Pandas 3.0.0, pytest, Ruff, and 120-column conventions.

No file outside the CREATE list may be changed. Any discovered need to modify another path is `PLAN_DEVIATION_REQUIRES_CONTROL_PLANE_APPROVAL` and stops the affected task.

## Locked Interfaces

```python
# src/count_v2/contracts.py
FORECAST_SUM_TOLERANCE: float = 1e-6

class CountContractError(ValueError): ...
class DatasetValidationError(CountContractError): ...
class ChronologyError(CountContractError): ...
class ModelContractError(CountContractError): ...
class EvaluationIntegrityError(CountContractError): ...

@dataclass(frozen=True)
class RawDrawBatch:
    dates: np.ndarray
    draws: np.ndarray
    def to_dict(self) -> dict[str, object]: ...
    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RawDrawBatch": ...

@dataclass(frozen=True)
class CountHistory:
    dates: np.ndarray
    counts: np.ndarray
    def to_dict(self) -> dict[str, object]: ...
    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CountHistory": ...

@dataclass(frozen=True)
class CountOutcome:
    target_date: str
    observed_counts: np.ndarray
    def to_dict(self) -> dict[str, object]: ...
    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CountOutcome": ...

@dataclass(frozen=True)
class CountForecast:
    target_date: str
    history_start: str
    history_end: str
    expected_count: np.ndarray
    model_identity: str
    mean_standard_error: np.ndarray | None = None
    mean_lower_bound: np.ndarray | None = None
    mean_upper_bound: np.ndarray | None = None
    predictive_distribution: dict[str, object] | None = None
    prediction_interval: tuple[np.ndarray, np.ndarray] | None = None
    def to_dict(self) -> dict[str, object]: ...
    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CountForecast": ...

def canonical_date(value: object, *, field_name: str) -> str: ...
def require_target_after_history(history: CountHistory, target_date: str) -> str: ...
```

```python
# src/count_v2/dataset.py
def raw_draw_batch_from_frame(
    frame: pd.DataFrame,
    *,
    draw_columns: Sequence[str],
    date_column: str = "date",
) -> RawDrawBatch: ...

def count_matrix_from_raw(batch: RawDrawBatch) -> np.ndarray: ...
def count_history_before(batch: RawDrawBatch, *, target_date: str) -> CountHistory: ...
def count_outcome_for_date(batch: RawDrawBatch, *, target_date: str) -> CountOutcome: ...
```

```python
# src/count_v2/models/__init__.py
class CountModel(Protocol):
    @property
    def model_identity(self) -> str: ...
    def predict_count(self, history: CountHistory, target_date: str) -> CountForecast: ...

class UniformCountModel: ...
class RollingCountModel:
    def __init__(self, window: int): ...
class EWMACountModel:
    EVIDENCE_CLASS = "DEVELOPMENT / EXPLORATORY"
    MEAN_SEMANTICS = "DEVELOPMENT_ONLY_POISSON_MEAN_APPROXIMATION"
    def __init__(self, half_life: float): ...
    def normalized_weights(self, history_rows: int) -> np.ndarray: ...
    def effective_sample_size(self, history_rows: int) -> float: ...
class DirichletShrinkageMultinomialModel:
    def __init__(self, window: int, prior_strength: float): ...
    def posterior_probabilities(self, history: CountHistory) -> np.ndarray: ...
```

```python
# backtests/count_v2_research.py
EVIDENCE_CLASS = "DEVELOPMENT / EXPLORATORY"
DEVELOPMENT_ARTIFACT_ROOT = ROOT_DIR / "research_artifacts" / "xpis_v2_count_first" / "development"

@dataclass(frozen=True)
class DevelopmentRunConfig:
    test_rows: int
    rolling_window: int
    ewma_half_life: float
    dirichlet_window: int
    prior_strength: float
    top_k: int

@dataclass(frozen=True)
class DevelopmentForecastSet:
    target_dates: np.ndarray
    observed_counts: np.ndarray
    model_identities: tuple[str, ...]
    expected_counts: np.ndarray

@dataclass(frozen=True)
class DevelopmentRunResult:
    forecast_set: DevelopmentForecastSet
    summary: dict[str, dict[str, float | int]]

def load_legacy_raw_draw_batch(csv_path: Path | None = None) -> RawDrawBatch: ...
def build_models(config: DevelopmentRunConfig) -> tuple[CountModel, ...]: ...
def generate_walk_forward_forecasts(
    batch: RawDrawBatch,
    config: DevelopmentRunConfig,
) -> DevelopmentForecastSet: ...
def evaluate_forecast(forecast: CountForecast, outcome: CountOutcome) -> dict[str, float]: ...
def evaluate_development_run(
    forecast_set: DevelopmentForecastSet,
    config: DevelopmentRunConfig,
) -> DevelopmentRunResult: ...
def write_development_artifacts(
    result: DevelopmentRunResult,
    config: DevelopmentRunConfig,
    *,
    run_id: str,
    artifact_root: Path = DEVELOPMENT_ARTIFACT_ROOT,
) -> Path: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

`DevelopmentForecastSet.expected_counts` has shape `(4, T, 100)` in the same order as `model_identities`; `observed_counts` has shape `(T, 100)`. Artifact output is exactly `summary.json` plus compressed `forecasts.npz` under one create-once development run directory.

---

### Task 1: Count contracts and validation primitives

**Files:**
- Create: `src/count_v2/__init__.py`
- Create: `src/count_v2/contracts.py`
- Create: `tests/count_v2/test_contracts.py`

**Interfaces:**
- Consumes: NumPy arrays and canonical `YYYY-MM-DD` strings.
- Produces: the five structured exceptions, `RawDrawBatch`, `CountHistory`, `CountOutcome`, `CountForecast`, `canonical_date()`, and `require_target_after_history()` exactly as locked above.

- [ ] **Step 1: Write exact failing contract tests**

Create tests that construct valid objects and JSON-compatible `to_dict()`/`from_dict()` round trips, then assert array equality and preservation of optional forecast fields. Add named tests for `(N,)`, `(N,27)`, `(H,100)`, `(100,)`, integral raw/count dtypes, float64 forecast dtype, strictly increasing dates, calendar gaps, `CountOutcome.observed_counts.shape == (100,)`, target strictly after every history date, non-negative finite forecasts, and sum-to-27 tolerance. Add failures for malformed dates, duplicate/non-monotonic history dates, wrong optional-array shapes, and a forecast sum of `26.999` raising `ModelContractError("MODEL_CONTRACT_FAILURE: expected_count must sum to 27 within 1e-6")`. Independently enforce `history_start <= history_end < target_date` in the `CountForecast` constructor and in `from_dict()` round trips, with named failures for `history_start > history_end`, `history_end == target_date`, and `history_end > target_date`.

```python
def make_count_row(draws: list[int]) -> np.ndarray:
    if len(draws) != 27:
        raise ValueError("test helper requires exactly 27 draws")
    row = np.zeros(100, dtype=np.int16)
    for value in draws:
        row[value] += 1
    return row

def test_calendar_gaps_are_valid_but_target_must_follow_every_history_date():
    history = CountHistory(
        dates=np.array(["2026-01-01", "2026-01-05"], dtype="<U10"),
        counts=np.vstack([
            make_count_row(list(range(27))),
            make_count_row(list(range(1, 28))),
        ]),
    )
    assert require_target_after_history(history, "2026-01-06") == "2026-01-06"
    with pytest.raises(ChronologyError, match="strictly after every history date"):
        require_target_after_history(history, "2026-01-05")

def test_forecast_contract_rejects_conservation_failure():
    with pytest.raises(ModelContractError, match="MODEL_CONTRACT_FAILURE"):
        CountForecast(
            target_date="2026-01-06",
            history_start="2026-01-01",
            history_end="2026-01-05",
            expected_count=np.full(100, 0.26999, dtype=np.float64),
            model_identity="broken",
        )

@pytest.mark.parametrize(
    ("history_start", "history_end", "target_date"),
    [
        ("2026-01-05", "2026-01-01", "2026-01-06"),
        ("2026-01-01", "2026-01-06", "2026-01-06"),
        ("2026-01-01", "2026-01-07", "2026-01-06"),
    ],
)
def test_forecast_constructor_rejects_invalid_chronology(
    history_start: str,
    history_end: str,
    target_date: str,
):
    with pytest.raises(ChronologyError):
        CountForecast(
            target_date=target_date,
            history_start=history_start,
            history_end=history_end,
            expected_count=np.full(100, 0.27, dtype=np.float64),
            model_identity="chronology_probe",
        )

def test_forecast_from_dict_reuses_constructor_chronology_validation():
    payload = {
        "target_date": "2026-01-06",
        "history_start": "2026-01-01",
        "history_end": "2026-01-06",
        "expected_count": [0.27] * 100,
        "model_identity": "invalid_round_trip",
    }
    with pytest.raises(ChronologyError):
        CountForecast.from_dict(payload)
```

- [ ] **Step 2: Run the focused tests and confirm the expected import failure**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_contracts.py -q`

Expected: FAIL during collection because `src.count_v2.contracts` does not exist.

- [ ] **Step 3: Add the minimal contract implementation**

Use frozen dataclasses with explicit `__post_init__` checks. Copy accepted arrays, normalize accepted dates only to canonical `<U10`, cast validated raw/count integers to `int16`, cast validated floating forecast arrays to `float64`, mark stored arrays read-only, and raise the locked exception types through explicit conditionals. Do not use `assert`. `CountForecast` canonicalizes all three dates and directly enforces `history_start <= history_end < target_date`; it also validates optional mean arrays separately and validates each prediction-interval endpoint as shape `(100,)`, finite, non-negative float64. `to_dict()` returns lists and scalars. Every `from_dict()` calls the public constructor so no chronology, shape, dtype, or conservation check can be bypassed, and it must not coerce invalid fractional raw/count values into integers.

```python
class ModelContractError(CountContractError):
    pass

def require_target_after_history(history: CountHistory, target_date: str) -> str:
    canonical = canonical_date(target_date, field_name="target_date")
    if np.any(history.dates >= canonical):
        raise ChronologyError("target_date must be strictly after every history date")
    return canonical

@dataclass(frozen=True)
class CountForecast:
    target_date: str
    history_start: str
    history_end: str
    expected_count: np.ndarray
    model_identity: str
    mean_standard_error: np.ndarray | None = None
    mean_lower_bound: np.ndarray | None = None
    mean_upper_bound: np.ndarray | None = None
    predictive_distribution: dict[str, object] | None = None
    prediction_interval: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        target = canonical_date(self.target_date, field_name="target_date")
        history_start = canonical_date(self.history_start, field_name="history_start")
        history_end = canonical_date(self.history_end, field_name="history_end")
        if history_start > history_end or history_end >= target:
            raise ChronologyError("forecast requires history_start <= history_end < target_date")
        expected = _validated_float_vector(self.expected_count, "expected_count")
        if abs(float(expected.sum()) - 27.0) > FORECAST_SUM_TOLERANCE:
            raise ModelContractError(
                "MODEL_CONTRACT_FAILURE: expected_count must sum to 27 within 1e-6"
            )
        object.__setattr__(self, "target_date", target)
        object.__setattr__(self, "history_start", history_start)
        object.__setattr__(self, "history_end", history_end)
        object.__setattr__(self, "expected_count", expected)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CountForecast":
        return cls(**dict(payload))
```

- [ ] **Step 4: Run the exact focused suite**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_contracts.py -q`

Expected: PASS.

- [ ] **Step 5: Run the relevant existing count-contract regression**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/test_count_evaluation.py -q`

Expected: PASS with no XPIS v1.2 behavior change.

- [ ] **Step 6: Check whitespace and review only this task's paths**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff --check`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff -- src/count_v2/__init__.py src/count_v2/contracts.py tests/count_v2/test_contracts.py`

Expected: no whitespace errors and no unrelated paths.

- [ ] **Step 7: Commit the bounded contract slice**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" add -- src/count_v2/__init__.py src/count_v2/contracts.py tests/count_v2/test_contracts.py`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" commit -m "feat: add XPIS v2 count contracts"`

### Task 2: Raw-draw to canonical count dataset

**Files:**
- Create: `src/count_v2/dataset.py`
- Create: `tests/count_v2/test_dataset.py`
- Modify: `src/count_v2/__init__.py`

**Interfaces:**
- Consumes: `pd.DataFrame`, an explicit ordered sequence of exactly 27 raw draw column names, `RawDrawBatch`, and canonical target dates.
- Produces: `raw_draw_batch_from_frame()`, `count_matrix_from_raw()`, `count_history_before()`, and `count_outcome_for_date()` exactly as locked above.

- [ ] **Step 1: Write exact failing dataset tests**

Define `DRAW_COLUMNS` and `make_frame()` exactly as below so every fixture has 27 explicit raw observations per row. Cover a valid row, repeated same number in one row producing a count greater than one, duplicate date rejection, descending date rejection, 26 and 28 draw-column rejection, missing column rejection, null rejection, `7.5` rejection, `-1` rejection, `100` rejection, mismatched row dimensions, exact `(N,100)` shape, `int16` output, row sums of 27, and a malformed middle row proving no row is silently dropped. Verify `count_history_before()` uses every recorded row strictly before the target despite calendar gaps, and `count_outcome_for_date()` returns exactly the target row as a 100-vector.

```python
DRAW_COLUMNS = tuple(f"draw_{index:02d}" for index in range(27))

def make_frame(
    dates: list[str],
    rows: list[list[int]] | None = None,
) -> pd.DataFrame:
    draw_rows = rows if rows is not None else [list(range(27)) for _ in dates]
    if len(draw_rows) != len(dates):
        raise ValueError("test helper requires one draw row per date")
    if any(len(row) != 27 for row in draw_rows):
        raise ValueError("test helper requires exactly 27 draws per row")
    return pd.DataFrame(
        [
            {"date": date, **dict(zip(DRAW_COLUMNS, row, strict=True))}
            for date, row in zip(dates, draw_rows, strict=True)
        ]
    )

def test_repeated_draw_is_valid_and_preserves_multiplicity():
    frame = make_frame(["2026-01-01"])
    frame.loc[0, list(DRAW_COLUMNS[:3])] = 7
    batch = raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS)
    counts = count_matrix_from_raw(batch)
    assert counts.shape == (1, 100)
    assert counts.dtype == np.int16
    assert counts[0, 7] == 3
    assert counts[0].sum() == 27

def test_invalid_middle_row_is_rejected_without_row_dropping():
    frame = make_frame(["2026-01-01", "2026-01-02", "2026-01-03"])
    frame.loc[1, DRAW_COLUMNS[8]] = np.nan
    with pytest.raises(DatasetValidationError, match="null draw"):
        raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS)
```

- [ ] **Step 2: Run the focused tests and confirm the expected import failure**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_dataset.py -q`

Expected: FAIL because `src.count_v2.dataset` does not exist.

- [ ] **Step 3: Implement fail-closed conversion and slicing**

`raw_draw_batch_from_frame()` must verify the explicit list has exactly 27 distinct columns and that all exist before extraction. Convert Timestamp/date-like values to canonical strings without reordering the frame. Reject nulls before numeric conversion; accept integer dtypes or finite numeric values equal to their integer truncation, reject booleans and fractional values, then range-check before casting to `int16`. Build counts with integer indexing so duplicate draws accumulate rather than collapse. Validate the resulting row sums and return a read-only `int16` matrix. History slicing is `batch.dates < target_date`; outcome lookup requires exactly one equal date.

```python
def count_matrix_from_raw(batch: RawDrawBatch) -> np.ndarray:
    counts = np.zeros((len(batch.dates), 100), dtype=np.int16)
    rows = np.repeat(np.arange(len(batch.dates)), batch.draws.shape[1])
    np.add.at(counts, (rows, batch.draws.reshape(-1)), 1)
    if not np.all(counts.sum(axis=1) == 27):
        raise DatasetValidationError("count rows must sum exactly to 27")
    counts.setflags(write=False)
    return counts

def count_history_before(batch: RawDrawBatch, *, target_date: str) -> CountHistory:
    target = canonical_date(target_date, field_name="target_date")
    mask = batch.dates < target
    if not np.any(mask):
        raise ChronologyError("history must contain at least one row before target_date")
    return CountHistory(batch.dates[mask], count_matrix_from_raw(batch)[mask])
```

- [ ] **Step 4: Run the exact focused suite**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_contracts.py tests/count_v2/test_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Run the relevant existing count regression**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/test_count_evaluation.py tests/test_count_challenger.py -q`

Expected: PASS.

- [ ] **Step 6: Check whitespace and review only dataset paths**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff --check`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff -- src/count_v2/__init__.py src/count_v2/dataset.py tests/count_v2/test_dataset.py`

- [ ] **Step 7: Commit the bounded dataset slice**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" add -- src/count_v2/__init__.py src/count_v2/dataset.py tests/count_v2/test_dataset.py`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" commit -m "feat: add canonical count dataset"`

### Task 3: B0 uniform and B1 rolling baselines

**Files:**
- Create: `src/count_v2/models/__init__.py`
- Create: `src/count_v2/models/uniform.py`
- Create: `src/count_v2/models/rolling.py`
- Create: `tests/count_v2/test_models.py`
- Modify: `src/count_v2/__init__.py`

**Interfaces:**
- Consumes: validated `CountHistory` and a target date strictly after every history row.
- Produces: `CountModel`, `UniformCountModel`, and `RollingCountModel(window: int)` with `predict_count(history, target_date) -> CountForecast`.

- [ ] **Step 1: Write exact failing B0/B1 tests**

Test B0 twice on the same gapped history and assert deterministic 100-vector output, every value exactly `0.27`, non-negativity, sum 27, `mean_standard_error` is a zero float64 vector, and identity `B0_UNIFORM`. For B1, use four distinguishable count rows with non-adjacent dates and `window=2`; assert only the last two rows contribute, exact mean, deterministic output, sum 27, identity `B1_ROLLING_W2`, and `mean_standard_error is None`. Assert `window <= 0` and history shorter than `window` fail explicitly. Give both models a target date equal to history end and assert `ChronologyError`.

```python
def history_from_draw_rows(
    dates: list[str],
    draw_rows: list[list[int]],
) -> CountHistory:
    if len(dates) != len(draw_rows):
        raise ValueError("test helper requires one draw row per date")
    if any(len(row) != 27 for row in draw_rows):
        raise ValueError("test helper requires exactly 27 draws per row")
    counts = np.vstack([
        np.bincount(np.asarray(row, dtype=np.int16), minlength=100)
        for row in draw_rows
    ]).astype(np.int16)
    return CountHistory(np.asarray(dates, dtype="<U10"), counts)

def test_rolling_uses_only_trailing_observation_rows_not_calendar_days():
    history = history_from_draw_rows(
        ["2026-01-01", "2026-01-04", "2026-02-10", "2026-02-28"],
        [[1] * 27, [2] * 27, [3] * 27, [4] * 27],
    )
    forecast = RollingCountModel(window=2).predict_count(history, "2026-03-15")
    np.testing.assert_allclose(forecast.expected_count, (history.counts[-2] + history.counts[-1]) / 2.0)
    assert np.isclose(forecast.expected_count.sum(), 27.0)
```

- [ ] **Step 2: Run the focused tests and confirm missing-model failures**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_models.py -q`

Expected: FAIL during import because the model package does not exist.

- [ ] **Step 3: Implement B0/B1 directly from conserving mathematics**

B0 returns `np.full(100, 27.0 / 100.0, dtype=np.float64)`. B1 requires at least `window` observed rows and returns `history.counts[-window:].mean(axis=0, dtype=np.float64)`. Both call `require_target_after_history()` before reading counts and construct `CountForecast` without any rescaling operation.

```python
class RollingCountModel:
    def __init__(self, window: int):
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer number of draw rows")
        self.window = window

    @property
    def model_identity(self) -> str:
        return f"B1_ROLLING_W{self.window}"

    def predict_count(self, history: CountHistory, target_date: str) -> CountForecast:
        target = require_target_after_history(history, target_date)
        if len(history.dates) < self.window:
            raise ChronologyError("rolling history has fewer rows than window")
        expected = history.counts[-self.window:].mean(axis=0, dtype=np.float64)
        return CountForecast(target, history.dates[-self.window], history.dates[-1], expected, self.model_identity)
```

- [ ] **Step 4: Run the exact focused suite**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_models.py -k "uniform or rolling" -q`

Expected: PASS.

- [ ] **Step 5: Run contract and dataset regressions**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_contracts.py tests/count_v2/test_dataset.py tests/count_v2/test_models.py -q`

Expected: PASS.

- [ ] **Step 6: Check whitespace and review B0/B1 paths**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff --check`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff -- src/count_v2/__init__.py src/count_v2/models tests/count_v2/test_models.py`

- [ ] **Step 7: Commit the bounded baseline slice**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" add -- src/count_v2/__init__.py src/count_v2/models/__init__.py src/count_v2/models/uniform.py src/count_v2/models/rolling.py tests/count_v2/test_models.py`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" commit -m "feat: add count-first baselines"`

### Task 4: M1 finite normalized EWMA count model

**Files:**
- Create: `src/count_v2/models/ewma.py`
- Modify: `src/count_v2/models/__init__.py`
- Modify: `src/count_v2/__init__.py`
- Modify: `tests/count_v2/test_models.py`

**Interfaces:**
- Consumes: all validated historical draw rows, a positive finite half-life, and a future target date.
- Produces: `EWMACountModel`, normalized finite weights, finite `n_eff = 1 / sum(w**2)`, and a `CountForecast` whose mean SE is explicitly development-only.

- [ ] **Step 1: Add exact failing M1 tests**

Assert `half_life <= 0`, boolean, NaN, and infinity are rejected. For a three-row gapped history and `half_life=2`, compute expected weights independently as `exp(-log(2) * ages / 2)`, normalize once, and assert exact weights, weight sum 1, finite expected counts, deterministic forecasts, non-negativity, sum 27, and exact `n_eff`. Verify the latest observation has the largest weight, target equality is rejected, `mean_standard_error == sqrt(expected_count / n_eff)`, the semantics constant is `DEVELOPMENT_ONLY_POISSON_MEAN_APPROXIMATION`, and no predictive distribution or prediction interval is emitted.

```python
def test_ewma_uses_finite_normalized_row_weights_and_finite_neff():
    model = EWMACountModel(half_life=2.0)
    weights = model.normalized_weights(history_rows=3)
    expected = np.exp(-np.log(2.0) * np.array([2.0, 1.0, 0.0]) / 2.0)
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected)
    assert np.isclose(model.effective_sample_size(3), 1.0 / np.sum(expected**2))
    assert np.isfinite(model.effective_sample_size(3))
```

- [ ] **Step 2: Run the M1 selection and confirm the import failure**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_models.py -k ewma -q`

Expected: FAIL because `EWMACountModel` is unavailable.

- [ ] **Step 3: Implement M1 without post-hoc normalization**

Use every row in `CountHistory`, with oldest-to-newest ages `H-1` through `0`. Normalize the finite weights, compute `expected = weights @ history.counts.astype(np.float64)`, compute exact finite `n_eff`, and expose `sqrt(expected / n_eff)` only as `mean_standard_error`. The model identity is `M1_EWMA_H<half_life:g>_DEVELOPMENT`; bounds and predictive fields stay `None`.

```python
def normalized_weights(self, history_rows: int) -> np.ndarray:
    if isinstance(history_rows, bool) or not isinstance(history_rows, int) or history_rows <= 0:
        raise ValueError("history_rows must be a positive integer")
    ages = np.arange(history_rows - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-np.log(2.0) * ages / self.half_life)
    return weights / weights.sum()

def effective_sample_size(self, history_rows: int) -> float:
    weights = self.normalized_weights(history_rows)
    return float(1.0 / np.sum(weights**2))
```

- [ ] **Step 4: Run the exact M1 tests**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_models.py -k ewma -q`

Expected: PASS.

- [ ] **Step 5: Run all core tests accumulated so far**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_contracts.py tests/count_v2/test_dataset.py tests/count_v2/test_models.py -q`

Expected: PASS.

- [ ] **Step 6: Check whitespace and review M1 paths**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff --check`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff -- src/count_v2/__init__.py src/count_v2/models/__init__.py src/count_v2/models/ewma.py tests/count_v2/test_models.py`

- [ ] **Step 7: Commit the bounded M1 slice**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" add -- src/count_v2/__init__.py src/count_v2/models/__init__.py src/count_v2/models/ewma.py tests/count_v2/test_models.py`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" commit -m "feat: add EWMA count model"`

### Task 5: M2 Dirichlet-shrinkage Multinomial model

**Files:**
- Create: `src/count_v2/models/dirichlet_shrinkage.py`
- Modify: `src/count_v2/models/__init__.py`
- Modify: `src/count_v2/__init__.py`
- Modify: `tests/count_v2/test_models.py`

**Interfaces:**
- Consumes: the trailing `window` count rows, positive finite total uniform prior strength `beta`, and a future target date.
- Produces: `DirichletShrinkageMultinomialModel`, `posterior_probabilities(history) -> np.ndarray`, and an M2 `CountForecast`.

- [ ] **Step 1: Add exact failing M2 tests**

Reject invalid windows and non-positive/non-finite prior strengths. Build 100 valid integral history rows containing 2,700 draws total, cycling through categories so each of the 100 categories occurs exactly 27 times. With `window=100`, assert a length-100 non-negative posterior vector summing to one and expected counts all `0.27`. With a separate history concentrated in one number, independently verify `(beta/100 + accumulated_count[n]) / (beta + 27*W)`, uniform shrinkage toward `0.01`, expected counts `27*p`, deterministic output, sum 27, target chronology, and exact posterior mean SE `27 * sqrt(alpha_n * (alpha_total-alpha_n) / (alpha_total**2 * (alpha_total+1)))`. Assert `predictive_distribution is None` so the posterior probability vector is not mislabeled as an empirical Dirichlet-Multinomial predictive law.

```python
def test_dirichlet_shrinkage_matches_uniform_prior_posterior_mean():
    draw_rows = [
        [(27 * row_index + offset) % 100 for offset in range(27)]
        for row_index in range(100)
    ]
    dates = pd.date_range("2025-01-01", periods=100, freq="D").strftime("%Y-%m-%d").tolist()
    history = history_from_draw_rows(dates, draw_rows)
    np.testing.assert_array_equal(
        history.counts.sum(axis=0),
        np.full(100, 27, dtype=np.int64),
    )
    model = DirichletShrinkageMultinomialModel(window=100, prior_strength=10.0)
    probabilities = model.posterior_probabilities(history)
    np.testing.assert_allclose(probabilities, np.full(100, 0.01))
    assert probabilities.shape == (100,)
    assert np.isclose(probabilities.sum(), 1.0)
    forecast = model.predict_count(history, "2025-04-11")
    np.testing.assert_allclose(forecast.expected_count, np.full(100, 0.27))
```

- [ ] **Step 2: Run the M2 selection and confirm the import failure**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_models.py -k dirichlet -q`

Expected: FAIL because the M2 class is unavailable.

- [ ] **Step 3: Implement the exact posterior mathematics**

Use `alpha_n = prior_strength / 100 + trailing_counts.sum(axis=0)` and `alpha_total = prior_strength + 27 * window`. Return `alpha_n / alpha_total`; never normalize the resulting vector again. `predict_count()` multiplies by 27, computes the stated posterior mean SE, and emits identity `M2_DIRICHLET_SHRINKAGE_MULTINOMIAL_W<window>_B<prior_strength:g>`. Require at least `window` historical rows.

```python
def posterior_probabilities(self, history: CountHistory) -> np.ndarray:
    if len(history.dates) < self.window:
        raise ChronologyError("Dirichlet history has fewer rows than window")
    accumulated = history.counts[-self.window:].sum(axis=0, dtype=np.float64)
    alpha = self.prior_strength / 100.0 + accumulated
    alpha_total = self.prior_strength + 27.0 * self.window
    probabilities = alpha / alpha_total
    if abs(float(probabilities.sum()) - 1.0) > FORECAST_SUM_TOLERANCE:
        raise ModelContractError("MODEL_CONTRACT_FAILURE: posterior probabilities must sum to 1")
    return probabilities
```

- [ ] **Step 4: Run the exact M2 tests**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_models.py -k dirichlet -q`

Expected: PASS.

- [ ] **Step 5: Run the complete core model suite**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_contracts.py tests/count_v2/test_dataset.py tests/count_v2/test_models.py -q`

Expected: PASS with exactly four exported model implementations.

- [ ] **Step 6: Check whitespace and review M2 paths**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff --check`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff -- src/count_v2/__init__.py src/count_v2/models/__init__.py src/count_v2/models/dirichlet_shrinkage.py tests/count_v2/test_models.py`

- [ ] **Step 7: Commit the bounded M2 slice**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" add -- src/count_v2/__init__.py src/count_v2/models/__init__.py src/count_v2/models/dirichlet_shrinkage.py tests/count_v2/test_models.py`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" commit -m "feat: add Dirichlet shrinkage count model"`

### Task 6: DataLoader development adapter and leakage-safe forecast generation

**Files:**
- Create: `backtests/count_v2_research.py`
- Create: `tests/count_v2/test_research_adapter.py`

**Interfaces:**
- Consumes: legacy `DataLoader.df`, `DataLoader.prize_cols()`, the four count models, and `DevelopmentRunConfig`.
- Produces: `DevelopmentRunConfig`, `DevelopmentForecastSet`, `load_legacy_raw_draw_batch()`, `build_models()`, and `generate_walk_forward_forecasts()` exactly as locked above.

- [ ] **Step 1: Write exact failing adapter and walk-forward tests**

Provide a fake DataLoader whose `.df` contains Timestamp dates and 27 raw columns, whose `prize_cols()` returns those columns, and whose `.S` property raises immediately if accessed. Assert `load_legacy_raw_draw_batch()` calls `.load()`, produces canonical `RawDrawBatch`, preserves repeated observations, and never touches `.S`. Add a 26-column failure. Assert `build_models()` returns identities in exact order B0, B1, M1, M2. For walk-forward, use a spy model or independently computed B1 values to assert every forecast for index `t` sees only `batch[:t]`, no target row enters history, gaps are allowed, output shapes are `(T,)`, `(T,100)`, and `(4,T,100)`, and generation is deterministic.

```python
LEGACY_DRAW_COLUMNS = tuple(f"draw_{index:02d}" for index in range(27))

def make_legacy_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(date),
                **{
                    column: (row_index * 27 + draw_index) % 100
                    for draw_index, column in enumerate(LEGACY_DRAW_COLUMNS)
                },
            }
            for row_index, date in enumerate(dates)
        ]
    )

def make_adapter_batch(row_count: int = 6) -> RawDrawBatch:
    dates = pd.date_range("2026-01-01", periods=row_count, freq="2D").strftime("%Y-%m-%d")
    draws = np.asarray(
        [
            [(row_index * 27 + offset) % 100 for offset in range(27)]
            for row_index in range(row_count)
        ],
        dtype=np.int16,
    )
    return RawDrawBatch(np.asarray(dates, dtype="<U10"), draws)

def test_adapter_reads_df_and_exact_draw_columns_without_binary_s(monkeypatch):
    frame = make_legacy_frame(["2026-01-01", "2026-01-03"])
    frame.loc[0, list(LEGACY_DRAW_COLUMNS[:3])] = 7

    class FakeLoader:
        load_calls = 0

        def __init__(self, csv_path: Path | None = None):
            self.csv_path = csv_path
        def load(self):
            type(self).load_calls += 1
            return self
        @property
        def df(self):
            return frame
        def prize_cols(self):
            return list(LEGACY_DRAW_COLUMNS)
        @property
        def S(self):
            raise AssertionError("binary S must not be read")

    monkeypatch.setattr(count_v2_research, "DataLoader", FakeLoader)
    batch = count_v2_research.load_legacy_raw_draw_batch()
    assert FakeLoader.load_calls == 1
    assert batch.draws.shape == (len(frame), 27)
    assert np.count_nonzero(batch.draws[0] == 7) >= 3

def test_walk_forward_forecasts_are_generated_from_strictly_prior_rows():
    batch = make_adapter_batch()
    config = DevelopmentRunConfig(
        test_rows=2,
        rolling_window=2,
        ewma_half_life=2.0,
        dirichlet_window=2,
        prior_strength=10.0,
        top_k=2,
    )
    result = generate_walk_forward_forecasts(batch, config)
    start = len(batch.dates) - config.test_rows
    b1_index = result.model_identities.index("B1_ROLLING_W2")
    np.testing.assert_allclose(
        result.expected_counts[b1_index, 0],
        count_matrix_from_raw(batch)[:start][-2:].mean(axis=0),
    )
```

- [ ] **Step 2: Run the focused adapter tests and confirm the missing-module failure**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_research_adapter.py -k "adapter or walk_forward or build_models" -q`

Expected: FAIL because `backtests.count_v2_research` does not exist.

- [ ] **Step 3: Implement the narrow adapter and forecast loop**

At module import, add repository root to `sys.path` using the existing backtest convention, then import `DataLoader` only in this adapter. `load_legacy_raw_draw_batch()` constructs `DataLoader(csv_path)` when a path is supplied or `DataLoader()` otherwise, calls `.load()`, obtains `draw_columns = loader.prize_cols()`, and immediately delegates `loader.df` to `raw_draw_batch_from_frame()`. `DevelopmentRunConfig.__post_init__` validates positive integer row/window/Top-K values, `top_k <= 100`, and positive finite float parameters. The first target index is `len(batch.dates) - test_rows`; reject configurations where it is smaller than `max(1, rolling_window, dirichlet_window)`. For each target index, create history from `batch.dates[:index]` and counts `C[:index]`, create the outcome separately from `C[index]`, then call all four models. Never pass the outcome to a model.

```python
def generate_walk_forward_forecasts(
    batch: RawDrawBatch,
    config: DevelopmentRunConfig,
) -> DevelopmentForecastSet:
    counts = count_matrix_from_raw(batch)
    start = len(batch.dates) - config.test_rows
    minimum_history = max(1, config.rolling_window, config.dirichlet_window)
    if start < minimum_history:
        raise ValueError("test_rows leave insufficient prior draw rows for configured windows")
    models = build_models(config)
    forecasts = np.empty((len(models), config.test_rows, 100), dtype=np.float64)
    for target_offset, target_index in enumerate(range(start, len(batch.dates))):
        history = CountHistory(batch.dates[:target_index], counts[:target_index])
        target_date = str(batch.dates[target_index])
        for model_index, model in enumerate(models):
            forecasts[model_index, target_offset] = model.predict_count(
                history, target_date
            ).expected_count
    return DevelopmentForecastSet(
        target_dates=batch.dates[start:],
        observed_counts=counts[start:],
        model_identities=tuple(model.model_identity for model in models),
        expected_counts=forecasts,
    )
```

- [ ] **Step 4: Run the exact focused adapter suite**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_research_adapter.py -k "adapter or walk_forward or build_models" -q`

Expected: PASS.

- [ ] **Step 5: Run all new correctness tests plus the bounded DataLoader regression**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2 -q`

Expected: PASS. Repository inspection found no dedicated legacy DataLoader unit-test file; the fake-loader and real-interface adapter tests in `test_research_adapter.py` are the relevant bounded DataLoader regression and must pass.

- [ ] **Step 6: Check whitespace and review adapter paths**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff --check`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff -- backtests/count_v2_research.py tests/count_v2/test_research_adapter.py`

- [ ] **Step 7: Commit the bounded adapter slice**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" add -- backtests/count_v2_research.py tests/count_v2/test_research_adapter.py`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" commit -m "feat: add count research adapter"`

### Task 7: Development metrics, artifacts, and CLI

**Files:**
- Modify: `backtests/count_v2_research.py`
- Modify: `tests/count_v2/test_research_adapter.py`

**Interfaces:**
- Consumes: `DevelopmentForecastSet`, `DevelopmentRunConfig`, `CountForecast`, `CountOutcome`, and adapter-level `EvaluationMetrics`.
- Produces: `DevelopmentRunResult`, `evaluate_forecast()`, `evaluate_development_run()`, `write_development_artifacts()`, and `main()` exactly as locked above.

- [ ] **Step 1: Add exact failing evaluation, artifact, and CLI tests**

Test target-date mismatch in `evaluate_forecast()` raises `EvaluationIntegrityError`. For two synthetic dates, verify each model summary contains existing count metric keys `count_mae`, `count_rmse`, `poisson_deviance`, `count_calibration_error`, and `break_even_expected_count`. Verify fixed Top-K economic fields preserve multiplicity: stable descending expected-count selection, `total_bets`, `total_hits`, `total_cost`, `total_payout`, `pnl`, and `roi`. Do not assert sign or dominance. With `tmp_path` and explicit `run_id="DEV_TEST_001"`, assert only `summary.json` and `forecasts.npz` are written under `<tmp>/DEV_TEST_001`, the evidence class is exact, forecast array shapes round-trip, an existing run directory is rejected, and no `predictions` path is touched. Test `main([...]) == 0` with monkeypatched loading/writing and required explicit development parameters.

```python
def make_development_config() -> DevelopmentRunConfig:
    return DevelopmentRunConfig(
        test_rows=2,
        rolling_window=2,
        ewma_half_life=2.0,
        dirichlet_window=2,
        prior_strength=10.0,
        top_k=2,
    )

def test_artifacts_are_create_once_and_development_classified(tmp_path):
    config = make_development_config()
    forecast_set = generate_walk_forward_forecasts(make_adapter_batch(), config)
    result = evaluate_development_run(forecast_set, config)
    run_dir = write_development_artifacts(
        result,
        config,
        run_id="DEV_TEST_001",
        artifact_root=tmp_path,
    )
    assert {path.name for path in run_dir.iterdir()} == {"summary.json", "forecasts.npz"}
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["evidence_class"] == "DEVELOPMENT / EXPLORATORY"
    with pytest.raises(FileExistsError):
        write_development_artifacts(result, config, run_id="DEV_TEST_001", artifact_root=tmp_path)
```

- [ ] **Step 2: Run the focused tests and confirm missing-interface failures**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_research_adapter.py -k "evaluate or economic or artifact or main" -q`

Expected: FAIL because evaluation and artifact interfaces are not implemented.

- [ ] **Step 3: Implement adapter-level evaluation and the minimal artifact schema**

Import `EvaluationMetrics` only in `backtests/count_v2_research.py`. Reuse its count metric semantics. For economic summaries, choose exactly `top_k` numbers per target using `np.argsort(-mu, kind="stable")[:top_k]`; sum observed counts without binary collapse; use cost 27 and payout 99. `summary.json` contains exactly `schema_version`, `evidence_class`, `run_id`, `config`, `date_range`, `target_rows`, `model_identities`, and `models`. `forecasts.npz` contains `target_dates`, `observed_counts`, `model_identities`, and `expected_counts`. Create the run directory with `exist_ok=False`. The CLI requires all development parameters, defaults artifact root only to the approved namespace, creates a UTC development run id, prints the created path, and returns zero. It emits no confirmatory status and never reads `predictions/evaluation_policy.json` as configuration.

```python
def _economic_summary(mu: np.ndarray, observed: np.ndarray, top_k: int) -> dict[str, float | int]:
    picks = np.stack([np.argsort(-row, kind="stable")[:top_k] for row in mu])
    hits = int(sum(observed[day, picks[day]].sum() for day in range(len(observed))))
    bets = int(len(observed) * top_k)
    total_cost = float(27.0 * bets)
    total_payout = float(99.0 * hits)
    pnl = total_payout - total_cost
    return {
        "top_k": top_k,
        "total_bets": bets,
        "total_hits": hits,
        "total_cost": total_cost,
        "total_payout": total_payout,
        "pnl": pnl,
        "roi": float(pnl / total_cost),
    }

def evaluate_forecast(forecast: CountForecast, outcome: CountOutcome) -> dict[str, float]:
    if forecast.target_date != outcome.target_date:
        raise EvaluationIntegrityError("forecast and outcome target dates must match")
    return EvaluationMetrics(
        cost_per_bet=27.0,
        payout_per_hit=99.0,
    ).count_forecast_metrics(forecast.expected_count, outcome.observed_counts)
```

- [ ] **Step 4: Run the exact focused evaluation/artifact suite**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_research_adapter.py -k "evaluate or economic or artifact or main" -q`

Expected: PASS.

- [ ] **Step 5: Run new tests and relevant existing metric/count-expectation regressions**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2 tests/test_count_evaluation.py tests/test_count_poisson_model.py tests/test_model_12_count_poisson.py tests/test_count_challenger.py tests/test_count_challenger_evaluator.py -q`

Expected: PASS with no artifact written outside pytest temporary directories.

- [ ] **Step 6: Check whitespace and review development harness paths**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff --check`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff -- backtests/count_v2_research.py tests/count_v2/test_research_adapter.py`

- [ ] **Step 7: Commit the bounded evaluation/artifact slice**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" add -- backtests/count_v2_research.py tests/count_v2/test_research_adapter.py`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" commit -m "feat: add development count evaluation"`

### Task 8: Core isolation and full Slice 1 regression gate

**Files:**
- Modify: `tests/count_v2/test_research_adapter.py`

**Interfaces:**
- Consumes: the complete `src/count_v2/**/*.py` source tree and the final research adapter.
- Produces: an automated AST-based proof of core dependency isolation and the final Slice 1 verification record in test output and Git history.

- [ ] **Step 1: Add exact failing permanent isolation tests**

Add permanent tests that call a not-yet-defined `collect_core_import_violations(source, filename)` helper. One test supplies `from src.data.loader import DataLoader` and requires violations for both the prohibited module and imported class. A parameterized test supplies `import data`, `import data.loader`, each prohibited XPIS v1 namespace, `import predictions`, and `import sklearn`, proving the policy rejects both legacy modules and any third-party dependency outside NumPy/Pandas. A positive test accepts standard-library imports, `numpy`, `pandas`, relative imports, and `src.count_v2` imports. A separate repository scan passes every real `src/count_v2/**/*.py` file through the same helper.

```python
def test_import_guard_detects_prohibited_loader_import():
    source = "from src.data.loader import DataLoader\n"
    violations = collect_core_import_violations(source, "synthetic_loader.py")
    assert "synthetic_loader.py: prohibited module src.data.loader" in violations
    assert "synthetic_loader.py: prohibited symbol DataLoader" in violations

@pytest.mark.parametrize(
    "source",
    [
        "import data\n",
        "import data.loader\n",
        "import src.probability\n",
        "import src.meta\n",
        "import src.decision\n",
        "import src.features\n",
        "import src.evidence\n",
        "import src.registry\n",
        "import predictions\n",
        "import sklearn\n",
    ],
)
def test_import_guard_rejects_legacy_and_unapproved_external_modules(source: str):
    assert collect_core_import_violations(source, "synthetic_external.py")

def test_count_v2_core_has_no_prohibited_imports():
    violations: list[str] = []
    for path in sorted((ROOT_DIR / "src" / "count_v2").rglob("*.py")):
        violations.extend(collect_core_import_violations(path.read_text(encoding="utf-8"), str(path)))
    assert violations == []
```

- [ ] **Step 2: Run the permanent guard tests and confirm the expected helper failure**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2/test_research_adapter.py -k "import_guard or prohibited_imports" -q`

Expected: FAIL with `NameError: name 'collect_core_import_violations' is not defined`; no repository source file is temporarily corrupted or removed.

- [ ] **Step 3: Implement the reusable AST guard and adapter boundary checks**

Implement the helper in the test module. Relative imports and explicit `src.count_v2` imports are allowed. Absolute imports are allowed only when their root is in `sys.stdlib_module_names`, `numpy`, or `pandas`. Independently flag the explicit prohibited prefixes `src.data`, `data`, `src.probability`, `src.meta`, `src.decision`, `src.features`, `src.evidence`, `src.registry`, and `predictions`, plus every imported symbol named `DataLoader`. Add an adapter AST test that permits its single `from src.data.loader import DataLoader` boundary but rejects any attribute access named `S`. Do not use comment/text matching.

```python
PROHIBITED_CORE_PREFIXES = (
    "src.data", "data", "src.probability", "src.meta", "src.decision",
    "src.features", "src.evidence", "src.registry", "predictions",
)
ALLOWED_EXTERNAL_ROOTS = frozenset({"numpy", "pandas"})

def collect_core_import_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [(alias.name, alias.name.rsplit(".", 1)[-1]) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue
            imports = [(node.module or "", alias.name) for alias in node.names]
        else:
            continue
        for module, symbol in imports:
            if symbol == "DataLoader":
                violations.append(f"{filename}: prohibited symbol DataLoader")
            if any(module == prefix or module.startswith(prefix + ".") for prefix in PROHIBITED_CORE_PREFIXES):
                violations.append(f"{filename}: prohibited module {module}")
                continue
            root = module.split(".", 1)[0]
            if module.startswith("src.count_v2") or root in ALLOWED_EXTERNAL_ROOTS or root in sys.stdlib_module_names:
                continue
            violations.append(f"{filename}: unapproved external module {module}")
    return violations

def test_research_adapter_is_the_only_loader_boundary_and_never_reads_binary_s():
    path = ROOT_DIR / "backtests" / "count_v2_research.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    loader_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.data.loader"
        and [alias.name for alias in node.names] == ["DataLoader"]
    ]
    binary_s_reads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "S"
    ]
    assert len(loader_imports) == 1
    assert binary_s_reads == []
```

- [ ] **Step 4: Run the complete new Slice 1 suite**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/count_v2 -q`

Expected: PASS.

- [ ] **Step 5: Run bounded legacy regressions and static compilation**

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" pytest tests/test_count_evaluation.py tests/test_count_poisson_model.py tests/test_model_12_count_poisson.py tests/test_count_challenger.py tests/test_count_challenger_evaluator.py -q`

Run: `uv run --directory "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" python -m compileall -q src/count_v2 backtests/count_v2_research.py`

Expected: all selected tests PASS and compilation exits zero. Do not run the research CLI against repository data during this correctness gate because that would create a development artifact unrelated to the test fixture.

- [ ] **Step 6: Run final diff, scope, and whitespace checks**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" diff --check`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" status --short`

Expected before the final task commit: only `tests/count_v2/test_research_adapter.py` is modified. Confirm no path under `predictions/**`, no existing XPIS v1.2 source/config path, and no `research_artifacts/**` path is present.

- [ ] **Step 7: Commit the final isolation gate and verify clean state**

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" add -- tests/count_v2/test_research_adapter.py`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" commit -m "test: enforce count core isolation"`

Run: `git -C "G:\MEGAsync\MR_BOM\XPIS_WORKTREES\xpis-v2-count-first-slice1-20260830" status --short`

Expected: empty output. Stop and report Slice 1 verification results; do not push, merge, create a PR, start confirmatory execution, or change production policy.

---

## Spec Coverage Map

| Slice 1 requirement | Plan coverage |
|---|---|
| Four canonical contracts, serialization, shapes, dtypes, chronology | Task 1 |
| Fail-closed raw ingestion, exact 27 columns, repeated observations, canonical `C[N,100]` | Task 2 |
| B0 uniform structural null and zero mean-estimation SE | Task 3 |
| B1 trailing observed-row window, gap tolerance, no target leakage | Task 3 |
| M1 finite normalized weights, row half-life, finite effective sample size, development-only mean SE | Task 4 |
| M2 uniform-prior Dirichlet-shrinkage Multinomial posterior mean and mean SE | Task 5 |
| No post-hoc normalization and model contract failure | Tasks 1, 3, 4, 5 |
| DataLoader.df adapter, exact raw observations, no S dependency | Task 6 |
| Walk-forward generation with strict prior history and separate outcomes | Task 6 |
| Count MAE/RMSE, Poisson deviance, count calibration, economics with multiplicity | Task 7 |
| Development-only create-once artifact namespace and minimal schema | Task 7 |
| No profitability/dominance assertions or confirmatory adjudication | Global Constraints, Tasks 7 and 8 |
| Automated core isolation proof | Task 8 |
| New suite, legacy count regressions, DataLoader boundary, compilation, diff check | Task 8 |

## Execution Boundary

This plan ends at the Slice 1 development correctness gate. Confirmatory manifests, remote receipts, append-only ledgers, protected branches, prospective execution, model promotion, and production integration require separate approved design and implementation gates.
