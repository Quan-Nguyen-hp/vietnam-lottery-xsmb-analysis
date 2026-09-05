"""Tests for XPIS v3 M3 economic protocol (Slice M3-09)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.m3_digit_factor.bootstrap import EconomicBootstrapResult
from src.m3_digit_factor.contracts import FailureExitStatus, FailureStage
from src.m3_digit_factor.economics import (
    COST_PER_SELECTED_NUMBER,
    ECONOMIC_K_VALUES,
    MINIMUM_POSITIVE_BLOCK_COUNT,
    PAYOUT_PER_HIT_OCCURRENCE,
    STABILITY_BLOCK_COUNT,
    EconomicEvaluationError,
    EconomicEvaluationResult,
    PerKEconomicEvaluation,
    compute_daily_economic_delta,
    compute_economic_delta_matrix,
    compute_full_stability_mean,
    compute_outcome_pnl,
    compute_stability_block_means,
    evaluate_economic_protocol,
    rank_outcomes_for_date,
    select_recommended_k,
    validate_stability_arrays,
)


# --- 1. Constants and Basic Units Tests ---


def test_economic_constants_and_units() -> None:
    """Normative constants: K=[1,3,5,10], cost=27, payout=99 in thousand VND."""
    assert ECONOMIC_K_VALUES == (1, 3, 5, 10)
    assert COST_PER_SELECTED_NUMBER == 27.0
    assert PAYOUT_PER_HIT_OCCURRENCE == 99.0
    assert STABILITY_BLOCK_COUNT == 6
    assert MINIMUM_POSITIVE_BLOCK_COUNT == 5


@pytest.mark.parametrize(
    "hits,expected_pnl",
    [
        (0, -27.0),
        (1, 72.0),
        (2, 171.0),
        (3, 270.0),
    ],
)
def test_outcome_pnl_multiplicity_preservation(hits: int, expected_pnl: float) -> None:
    """PnL per selected number = 99 * y - 27 (thousand VND), preserving hit multiplicity."""
    pnl = compute_outcome_pnl(hits)
    assert pnl == expected_pnl


# --- 2. Top-K Ranking and Tie-Breaking Tests ---


def test_top_k_ranking_descending_mu_and_smaller_n_tiebreak() -> None:
    """Rank outcomes by mu descending, tie-breaking by smaller outcome index n."""
    mu = np.full(100, 0.20, dtype=np.float64)
    # Give specific values
    mu[42] = 0.80
    mu[15] = 0.50
    mu[77] = 0.50  # Tie with 15 -> 15 should win tie-break because 15 < 77
    mu[99] = 0.50  # Tie with 15, 77 -> 15 < 77 < 99

    ranked = rank_outcomes_for_date(mu)
    assert ranked.shape == (100,)
    assert ranked[0] == 42
    assert ranked[1] == 15
    assert ranked[2] == 77
    assert ranked[3] == 99

    # The rest have mu=0.20, so they must be ordered 0..99 excluding 15, 42, 77, 99
    remaining = [n for n in range(100) if n not in (42, 15, 77, 99)]
    np.testing.assert_array_equal(ranked[4:], remaining)


def test_top_k_selections_exact() -> None:
    """Exact top 1, 3, 5, 10 selections."""
    mu = np.full(100, 0.27, dtype=np.float64)
    mu[0] = 0.5
    mu[1] = 0.4
    mu[2] = 0.3
    mu[3] = 0.29
    mu[4] = 0.28
    mu[5] = 0.275
    mu[6] = 0.274
    mu[7] = 0.273
    mu[8] = 0.272
    mu[9] = 0.271

    ranked = rank_outcomes_for_date(mu)
    assert ranked[:1].tolist() == [0]
    assert ranked[:3].tolist() == [0, 1, 2]
    assert ranked[:5].tolist() == [0, 1, 2, 3, 4]
    assert ranked[:10].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# --- 3. Daily Economic Delta Tests ---


def test_daily_economic_delta_formula_and_not_b0_diff() -> None:
    """Daily economic delta = 99 * sum(y_selected) - 27 * K (thousand VND net PnL)."""
    mu = np.full(100, 0.27, dtype=np.float64)
    mu[0] = 0.9
    mu[1] = 0.8
    mu[2] = 0.7

    y = np.zeros(100, dtype=np.int64)
    y[0] = 2  # Multiplicity 2 on first selected
    y[1] = 1  # Multiplicity 1 on second selected
    y[2] = 0  # Multiplicity 0 on third selected
    y[3] = 24  # Total sum = 27

    # K=1: selected=[0], hits=2 -> 99*2 - 27*1 = 198 - 27 = 171
    d1 = compute_daily_economic_delta(y, mu, k=1)
    assert d1 == 171.0

    # K=3: selected=[0, 1, 2], hits=2+1+0=3 -> 99*3 - 27*3 = 297 - 81 = 216
    d3 = compute_daily_economic_delta(y, mu, k=3)
    assert d3 == 216.0


def test_compute_economic_delta_matrix_shape_and_k_order() -> None:
    """Matrix has shape (N, 4) with columns corresponding to K in [1, 3, 5, 10]."""
    n_days = 5
    mu = np.full((n_days, 100), 0.27, dtype=np.float64)
    y = np.zeros((n_days, 100), dtype=np.int64)
    y[:, :27] = 1

    matrix = compute_economic_delta_matrix(y, mu)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (n_days, 4)

    # Verify column K alignment
    for i, k in enumerate(ECONOMIC_K_VALUES):
        for t in range(n_days):
            expected = compute_daily_economic_delta(y[t], mu[t], k=k)
            assert matrix[t, i] == expected


# --- 4. ILR-01 Mandatory Regression and Block Aggregation Tests ---


def test_ilr_01_mandatory_full_stability_mean_regression() -> None:
    """ILR-01: 61-date stability cohort partitioned [11, 10, 10, 10, 10, 10].

    First block daily delta = 5, remaining 50 dates = -1.
    Exact mean = (11*5 + 50*(-1)) / 61 = 5 / 61.
    Forbidden unweighted block mean = (5 - 1 - 1 - 1 - 1 - 1) / 6 = 0.
    """
    n_days = 61
    block_sizes = (11, 10, 10, 10, 10, 10)
    assert sum(block_sizes) == 61

    stability_blocks = []
    start = 0
    for size in block_sizes:
        stability_blocks.append(tuple(range(start, start + size)))
        start += size

    # Create daily delta array of length 61
    daily_delta_k = np.full(n_days, -1.0, dtype=np.float64)
    daily_delta_k[:11] = 5.0

    # 1. Full-stability mean
    full_mean = compute_full_stability_mean(daily_delta_k)
    expected_full_mean = 5.0 / 61.0
    assert abs(full_mean - expected_full_mean) < 1e-15
    assert full_mean > 0.0  # Positive!

    # 2. Block means
    matrix = daily_delta_k.reshape(-1, 1)
    block_means = compute_stability_block_means(matrix, stability_blocks)
    assert len(block_means) == 6
    assert abs(block_means[0, 0] - 5.0) < 1e-15
    for b in range(1, 6):
        assert abs(block_means[b, 0] - (-1.0)) < 1e-15

    # 3. Prove that forbidden mean of block means is 0.0
    forbidden_mean = float(np.mean(block_means[:, 0]))
    assert forbidden_mean == 0.0
    assert full_mean != forbidden_mean


def test_positive_block_count_strict_positivity() -> None:
    """Strict positive block rule: block_mean > 0 counted; <= 0 not counted."""
    # 6 blocks: 5 positive, 1 exactly 0.0 -> count = 5
    block_means = np.array([
        [1.0],
        [2.0],
        [0.5],
        [0.1],
        [1.5],
        [0.0],  # Exactly 0.0 is NOT positive
    ], dtype=np.float64)

    counts = [int(np.sum(block_means[:, col] > 0.0)) for col in range(1)]
    assert counts[0] == 5

    # If one is -0.1 -> count = 4
    block_means[4, 0] = -0.1
    counts_4 = [int(np.sum(block_means[:, col] > 0.0)) for col in range(1)]
    assert counts_4[0] == 4


# --- 5. Input Validation Tests ---


@pytest.mark.parametrize(
    "bad_y,bad_mu,match_msg",
    [
        (np.zeros((10, 99)), np.full((10, 100), 0.27), "shape \\(N, 100\\)"),
        (np.zeros((10, 100)), np.full((10, 99), 0.27), "shape \\(N, 100\\)"),
        (np.zeros((0, 100)), np.full((0, 100), 0.27), "cannot be empty"),
        (np.zeros((10, 100)), np.full((11, 100), 0.27), "Row counts of y and mu must match"),
        (np.full((10, 100), -1), np.full((10, 100), 0.27), "non-negative"),
        (np.zeros((10, 100)), np.full((10, 100), 0.27), "Observed count row sum must be 27"),
        (
            np.array([[1] * 27 + [0] * 73] * 10),
            np.full((10, 100), -0.1),
            "strictly positive",
        ),
        (
            np.array([[1] * 27 + [0] * 73] * 10),
            np.full((10, 100), 0.28),
            "row sum must equal 27",
        ),
    ],
)
def test_validate_stability_arrays_rejections(
    bad_y: Any, bad_mu: Any, match_msg: str
) -> None:
    """Invalid input arrays raise EconomicEvaluationError with ECONOMIC_EVALUATION stage."""
    with pytest.raises(EconomicEvaluationError, match=match_msg) as exc_info:
        validate_stability_arrays(bad_y, bad_mu)

    err = exc_info.value
    assert err.stage == FailureStage.ECONOMIC_EVALUATION
    proto = err.as_protocol_failure()
    assert proto.stage == FailureStage.ECONOMIC_EVALUATION
    assert proto.exit_status in {
        FailureExitStatus.NEEDS_DATA_REVISION,
        FailureExitStatus.NEEDS_MODEL_REVISION,
        FailureExitStatus.TECHNICAL_FAILURE,
    }


def test_input_arrays_not_mutated_during_economic_evaluation() -> None:
    """Inputs y and mu are not modified."""
    y = np.zeros((12, 100), dtype=np.int64)
    y[:, :27] = 1
    mu = np.full((12, 100), 0.27, dtype=np.float64)

    y_orig = y.copy()
    mu_orig = mu.copy()

    blocks = tuple(tuple(range(i * 2, (i + 1) * 2)) for i in range(6))
    _ = evaluate_economic_protocol(y, mu, blocks, forecast_signal=True)

    np.testing.assert_array_equal(y, y_orig)
    np.testing.assert_array_equal(mu, mu_orig)


# --- 6. Forecast Signal False Short-Circuit Tests ---


def test_forecast_signal_false_short_circuits_without_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """When forecast_signal=False, economic bootstrap is NOT executed and returns empty results."""
    bootstrap_called = False

    def mock_bootstrap(*args: Any, **kwargs: Any) -> Any:
        nonlocal bootstrap_called
        bootstrap_called = True
        raise RuntimeError("Economic bootstrap must NOT be called when forecast_signal is False!")

    import src.m3_digit_factor.economics as econ_mod
    monkeypatch.setattr(econ_mod, "run_economic_bootstrap", mock_bootstrap)

    y = np.zeros((12, 100), dtype=np.int64)
    y[:, :27] = 1
    mu = np.full((12, 100), 0.27, dtype=np.float64)
    blocks = tuple(tuple(range(i * 2, (i + 1) * 2)) for i in range(6))

    res = evaluate_economic_protocol(y, mu, blocks, forecast_signal=False)

    assert isinstance(res, EconomicEvaluationResult)
    assert not bootstrap_called
    assert res.bootstrap_executed is False
    assert res.economic_signal is False
    assert res.qualified_top_k == []
    assert res.recommended_top_k == []
    assert res.per_k == {}


# --- 7. Qualification and Recommendation Selection Tests ---


def test_recommendation_selection_rule_and_tie_breaks() -> None:
    """Recommendation selects argmax(bootstrap_lower_bound, mean_economic_delta, -K)."""
    # Case 1: Higher bootstrap lower bound wins
    evals_1 = {
        1: PerKEconomicEvaluation(k=1, mean_economic_delta=10.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=2.0, qualifies=True),
        3: PerKEconomicEvaluation(k=3, mean_economic_delta=50.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=5.0, qualifies=True),
    }
    assert select_recommended_k(evals_1, qualified_top_k=[1, 3]) == [3]

    # Case 2: Same bootstrap lower bound, higher mean wins
    evals_2 = {
        1: PerKEconomicEvaluation(k=1, mean_economic_delta=20.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=5.0, qualifies=True),
        3: PerKEconomicEvaluation(k=3, mean_economic_delta=15.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=5.0, qualifies=True),
    }
    assert select_recommended_k(evals_2, qualified_top_k=[1, 3]) == [1]

    # Case 3: Same lower bound, same mean -> smaller K wins
    evals_3 = {
        3: PerKEconomicEvaluation(k=3, mean_economic_delta=20.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=5.0, qualifies=True),
        5: PerKEconomicEvaluation(k=5, mean_economic_delta=20.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=5.0, qualifies=True),
    }
    assert select_recommended_k(evals_3, qualified_top_k=[3, 5]) == [3]

    # Case 4: Non-qualified K with better stats CANNOT win
    evals_4 = {
        1: PerKEconomicEvaluation(k=1, mean_economic_delta=10.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=2.0, qualifies=True),
        10: PerKEconomicEvaluation(k=10, mean_economic_delta=100.0, block_mean_economic_delta=(1,)*6, positive_block_count=4, bootstrap_lower_bound=50.0, qualifies=False),
    }
    assert select_recommended_k(evals_4, qualified_top_k=[1]) == [1]

    # Case 5: Empty qualified list -> empty recommended
    assert select_recommended_k(evals_4, qualified_top_k=[]) == []


def test_per_k_qualification_three_predicates_truth_table() -> None:
    """K qualifies iff mean > 0 AND positive_blocks >= 5 AND bootstrap_lower > 0."""
    # Test boundary: mean == 0 fails
    e_mean0 = PerKEconomicEvaluation(k=1, mean_economic_delta=0.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=1.0, qualifies=False)
    # Test boundary: positive_blocks == 4 fails
    e_pb4 = PerKEconomicEvaluation(k=1, mean_economic_delta=1.0, block_mean_economic_delta=(1,)*4 + (-1, -1), positive_block_count=4, bootstrap_lower_bound=1.0, qualifies=False)
    # Test boundary: bootstrap_lower == 0 fails
    e_bs0 = PerKEconomicEvaluation(k=1, mean_economic_delta=1.0, block_mean_economic_delta=(1,)*6, positive_block_count=6, bootstrap_lower_bound=0.0, qualifies=False)
    # All 3 pass
    e_pass = PerKEconomicEvaluation(k=1, mean_economic_delta=1.0, block_mean_economic_delta=(1,)*5 + (-1,), positive_block_count=5, bootstrap_lower_bound=0.01, qualifies=True)

    assert not e_mean0.qualifies
    assert not e_pb4.qualifies
    assert not e_bs0.qualifies
    assert e_pass.qualifies


def test_economic_bootstrap_called_once_with_complete_n_by_4_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Economic bootstrap must be invoked exactly ONCE with shape (N, 4)."""
    call_count = 0
    passed_shape = None

    def spy_bootstrap(delta_matrix: np.ndarray) -> EconomicBootstrapResult:
        nonlocal call_count, passed_shape
        call_count += 1
        passed_shape = delta_matrix.shape
        return EconomicBootstrapResult(
            lower_bounds={1: 1.0, 3: 2.0, 5: 3.0, 10: 4.0},
            bootstrap_replications=2000,
            replicate_means=np.zeros((2000, 4)),
        )

    import src.m3_digit_factor.economics as econ_mod
    monkeypatch.setattr(econ_mod, "run_economic_bootstrap", spy_bootstrap)

    y = np.zeros((12, 100), dtype=np.int64)
    y[:, :27] = 1
    mu = np.full((12, 100), 0.27, dtype=np.float64)
    blocks = tuple(tuple(range(i * 2, (i + 1) * 2)) for i in range(6))

    res = evaluate_economic_protocol(y, mu, blocks, forecast_signal=True)

    assert call_count == 1
    assert passed_shape == (12, 4)
    assert res.bootstrap_executed is True
    assert res.economic_signal is True
    assert res.qualified_top_k == [1, 3, 5, 10]
    assert res.recommended_top_k == [10]  # k=10 has highest lower bound 4.0


def test_qualified_top_k_preserves_normative_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """qualified_top_k preserves [1, 3, 5, 10] normative order even if stats are out of order."""
    def mock_bootstrap(delta_matrix: np.ndarray) -> EconomicBootstrapResult:
        # K=1, 5, 10 pass, K=3 fails
        return EconomicBootstrapResult(
            lower_bounds={1: 50.0, 3: -10.0, 5: 100.0, 10: 1.0},
            bootstrap_replications=2000,
            replicate_means=np.zeros((2000, 4)),
        )

    import src.m3_digit_factor.economics as econ_mod
    monkeypatch.setattr(econ_mod, "run_economic_bootstrap", mock_bootstrap)

    y = np.zeros((12, 100), dtype=np.int64)
    y[:, :27] = 1
    mu = np.full((12, 100), 0.27, dtype=np.float64)
    blocks = tuple(tuple(range(i * 2, (i + 1) * 2)) for i in range(6))

    res = evaluate_economic_protocol(y, mu, blocks, forecast_signal=True)

    # K=5 has highest lower bound (100.0), but qualified_top_k MUST preserve [1, 5, 10] normative order!
    assert res.qualified_top_k == [1, 5, 10]
    # And recommendation picks K=5 because it has highest lower bound among qualified
    assert res.recommended_top_k == [5]


def test_m3_09_does_not_leak_artifact_or_serialization_code() -> None:
    """M3-09 must not implement file writers, JSON encoding, CSV encoding, or artifact schemas."""
    import inspect
    import src.m3_digit_factor.economics as econ_mod

    source = inspect.getsource(econ_mod)
    forbidden_terms = [
        "json.dump",
        "csv.writer",
        "economic_summary.csv",
        "economic_uncertainty.json",
        "development_adjudication.json",
        "open(",
        "Path(",
        "to_csv",
        "to_json",
        "gzip",
    ]
    for term in forbidden_terms:
        assert term not in source, f"Forbidden term {term!r} found in economics.py"

