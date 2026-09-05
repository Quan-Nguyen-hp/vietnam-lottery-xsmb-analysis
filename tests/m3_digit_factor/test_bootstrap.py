"""Tests for XPIS v3 M3 exact stationary circular bootstrap (Slice M3-08)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.m3_digit_factor.bootstrap import (
    ECONOMIC_BOOTSTRAP_BIT_GENERATOR,
    ECONOMIC_BOOTSTRAP_K_VALUES,
    ECONOMIC_BOOTSTRAP_MEAN_BLOCK_LENGTH,
    ECONOMIC_BOOTSTRAP_QUANTILE,
    ECONOMIC_BOOTSTRAP_QUANTILE_METHOD,
    ECONOMIC_BOOTSTRAP_REPLICATIONS,
    ECONOMIC_BOOTSTRAP_RESTART_PROBABILITY,
    ECONOMIC_BOOTSTRAP_RNG_IMPLEMENTATION,
    ECONOMIC_BOOTSTRAP_SEED,
    FORECAST_BOOTSTRAP_BIT_GENERATOR,
    FORECAST_BOOTSTRAP_MEAN_BLOCK_LENGTH,
    FORECAST_BOOTSTRAP_QUANTILE,
    FORECAST_BOOTSTRAP_QUANTILE_METHOD,
    FORECAST_BOOTSTRAP_REPLICATIONS,
    FORECAST_BOOTSTRAP_RESTART_PROBABILITY,
    FORECAST_BOOTSTRAP_RNG_IMPLEMENTATION,
    FORECAST_BOOTSTRAP_SEED,
    SHARED_RESAMPLE_INDICES,
    EconomicBootstrapError,
    EconomicBootstrapResult,
    ForecastBootstrapError,
    ForecastBootstrapResult,
    generate_stationary_circular_indices,
    run_economic_bootstrap,
    run_forecast_bootstrap,
)
from src.m3_digit_factor.contracts import FailureExitStatus, FailureStage


# --- 1. Stub / Recording RNG for Structural Call-Order Verification ---


class StubRecordingRNG:
    """Mock RNG to strictly verify call order and consumption semantics."""

    def __init__(
        self,
        start_indices: list[int],
        restart_draws: list[float],
        restart_indices: list[int] | None = None,
    ) -> None:
        self._start_indices = list(start_indices)
        self._restart_draws = list(restart_draws)
        self._restart_indices = list(restart_indices or [])
        self.call_log: list[str] = []

    def integers(self, low: int, high: int) -> int:
        if not self.call_log or self.call_log[-1] not in ("random_restart",):
            self.call_log.append("integers_start")
            if not self._start_indices:
                raise RuntimeError("Exhausted start indices in StubRecordingRNG")
            return self._start_indices.pop(0)
        else:
            self.call_log.append("integers_restart")
            if not self._restart_indices:
                raise RuntimeError("Exhausted restart indices in StubRecordingRNG")
            return self._restart_indices.pop(0)

    def random(self) -> float:
        if not self._restart_draws:
            raise RuntimeError("Exhausted restart draws in StubRecordingRNG")
        draw = self._restart_draws.pop(0)
        # 1/30 = ~0.03333333333333333
        if draw < (1.0 / 30.0):
            self.call_log.append("random_restart")
        else:
            self.call_log.append("random_continuation")
        return draw


def test_recording_rng_exact_call_order_and_no_integers_on_continuation() -> None:
    """Verify that NO integers() call occurs on continuation steps."""
    # Setup: 1 replicate of length 5
    # start = 4
    # draws = [0.5, 0.5, 0.01, 0.5] -> cont, cont, restart (< 1/30), cont
    # restart integer = 0
    stub = StubRecordingRNG(
        start_indices=[4],
        restart_draws=[0.5, 0.5, 0.01, 0.5],
        restart_indices=[0],
    )

    indices = generate_stationary_circular_indices(n=5, replications=1, rng=stub)  # type: ignore[arg-type]
    assert indices.shape == (1, 5)

    # Path should be:
    # k=0: start -> 4
    # k=1: cont -> (4 + 1) % 5 = 0 (circular wraparound!)
    # k=2: cont -> (0 + 1) % 5 = 1
    # k=3: restart -> 0
    # k=4: cont -> (0 + 1) % 5 = 1
    np.testing.assert_array_equal(indices[0], np.array([4, 0, 1, 0, 1]))

    # Check exact calls
    assert stub.call_log == [
        "integers_start",
        "random_continuation",
        "random_continuation",
        "random_restart",
        "integers_restart",
        "random_continuation",
    ]


def test_circular_wraparound_at_right_edge() -> None:
    """Continuation across N-1 cleanly wraps to 0 without truncation."""
    stub = StubRecordingRNG(
        start_indices=[4],
        restart_draws=[0.5, 0.5, 0.5, 0.5],  # all continuation
    )
    indices = generate_stationary_circular_indices(n=5, replications=1, rng=stub)  # type: ignore[arg-type]
    np.testing.assert_array_equal(indices[0], np.array([4, 0, 1, 2, 3]))


def test_resampler_with_n_equals_one() -> None:
    """N=1 is valid circular behavior, producing an array of zeros."""
    rng = np.random.Generator(np.random.PCG64(12345))
    indices = generate_stationary_circular_indices(n=1, replications=10, rng=rng)
    assert indices.shape == (10, 1)
    np.testing.assert_array_equal(indices, np.zeros((10, 1), dtype=np.int64))


def test_resampler_index_domain_and_replicate_length() -> None:
    """Indices must strictly remain within [0, N-1] for all positions."""
    rng = np.random.Generator(np.random.PCG64(20260831))
    n = 25
    reps = 100
    indices = generate_stationary_circular_indices(n=n, replications=reps, rng=rng)
    assert indices.shape == (reps, n)
    assert np.all(indices >= 0)
    assert np.all(indices < n)


def test_resampler_rng_continuity_and_no_seed_reset() -> None:
    """A single Generator continues across all replicates without resetting."""
    rng1 = np.random.Generator(np.random.PCG64(42))
    idx1 = generate_stationary_circular_indices(n=10, replications=5, rng=rng1)

    rng2 = np.random.Generator(np.random.PCG64(42))
    idx2_parts = []
    for _ in range(5):
        idx2_parts.append(generate_stationary_circular_indices(n=10, replications=1, rng=rng2))
    idx2 = np.vstack(idx2_parts)

    np.testing.assert_array_equal(idx1, idx2)


# --- 2. Forecast Bootstrap Tests ---


def test_forecast_bootstrap_constants() -> None:
    """Forecast bootstrap constants strictly match the frozen specification."""
    assert FORECAST_BOOTSTRAP_RNG_IMPLEMENTATION == "numpy.random.Generator"
    assert FORECAST_BOOTSTRAP_BIT_GENERATOR == "PCG64"
    assert FORECAST_BOOTSTRAP_SEED == 20260831
    assert FORECAST_BOOTSTRAP_REPLICATIONS == 2000
    assert FORECAST_BOOTSTRAP_MEAN_BLOCK_LENGTH == 30
    assert math.isclose(FORECAST_BOOTSTRAP_RESTART_PROBABILITY, 1.0 / 30.0, abs_tol=1e-15)
    assert FORECAST_BOOTSTRAP_QUANTILE == 0.05
    assert FORECAST_BOOTSTRAP_QUANTILE_METHOD == "linear"


def test_forecast_golden_rng_trace_early_replicates() -> None:
    """Golden early replicates for N=10 under PCG64(20260831)."""
    rng = np.random.Generator(np.random.PCG64(FORECAST_BOOTSTRAP_SEED))
    indices = generate_stationary_circular_indices(n=10, replications=3, rng=rng)

    # Independent hand-verified expected indices for N=10 under PCG64(20260831)
    expected_rep_0 = [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
    expected_rep_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    expected_rep_2 = [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]

    np.testing.assert_array_equal(indices[0], expected_rep_0)
    np.testing.assert_array_equal(indices[1], expected_rep_1)
    np.testing.assert_array_equal(indices[2], expected_rep_2)


def test_forecast_bootstrap_deterministic_golden_lower_bound() -> None:
    """Golden synthetic lower bound on fixed input vector."""
    d = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3, 0.4, -0.5])
    res = run_forecast_bootstrap(d)

    assert isinstance(res, ForecastBootstrapResult)
    assert res.bootstrap_replications == 2000
    assert len(res.replicate_means) == 2000

    # Bitwise deterministic match under the frozen environment
    expected_lb = -0.0400000000000000
    assert abs(res.bootstrap_lower_bound - expected_lb) < 1e-14

    # Run again to verify reproducibility
    res2 = run_forecast_bootstrap(d)
    assert res.bootstrap_lower_bound == res2.bootstrap_lower_bound
    np.testing.assert_array_equal(res.replicate_means, res2.replicate_means)


@pytest.mark.parametrize(
    "bad_input,match_msg",
    [
        (np.array([]), "cannot be empty"),
        (np.zeros((10, 2)), "1-dimensional"),
        (np.array([0.1, np.nan, 0.3]), "finite"),
        (np.array([0.1, np.inf, 0.3]), "finite"),
        (np.array([0.1, -np.inf, 0.3]), "finite"),
    ],
)
def test_forecast_bootstrap_input_validation(bad_input: Any, match_msg: str) -> None:
    """Invalid input arrays raise ForecastBootstrapError with FORECAST_BOOTSTRAP stage."""
    with pytest.raises(ForecastBootstrapError, match=match_msg) as exc_info:
        run_forecast_bootstrap(bad_input)

    err = exc_info.value
    assert err.stage == FailureStage.FORECAST_BOOTSTRAP
    proto = err.as_protocol_failure()
    assert proto.stage == FailureStage.FORECAST_BOOTSTRAP
    assert proto.exit_status in {
        FailureExitStatus.NEEDS_DATA_REVISION,
        FailureExitStatus.TECHNICAL_FAILURE,
    }


def test_forecast_bootstrap_does_not_mutate_input() -> None:
    """Input vector remains bitwise unmodified."""
    d = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    d_copy = d.copy()
    _ = run_forecast_bootstrap(d)
    np.testing.assert_array_equal(d, d_copy)


# --- 3. Economic Bootstrap Tests ---


def test_economic_bootstrap_constants() -> None:
    """Economic bootstrap constants strictly match the frozen specification."""
    assert ECONOMIC_BOOTSTRAP_RNG_IMPLEMENTATION == "numpy.random.Generator"
    assert ECONOMIC_BOOTSTRAP_BIT_GENERATOR == "PCG64"
    assert ECONOMIC_BOOTSTRAP_SEED == 20260832
    assert ECONOMIC_BOOTSTRAP_SEED != FORECAST_BOOTSTRAP_SEED
    assert ECONOMIC_BOOTSTRAP_REPLICATIONS == 2000
    assert ECONOMIC_BOOTSTRAP_MEAN_BLOCK_LENGTH == 30
    assert math.isclose(ECONOMIC_BOOTSTRAP_RESTART_PROBABILITY, 1.0 / 30.0, abs_tol=1e-15)
    assert ECONOMIC_BOOTSTRAP_QUANTILE == 0.0125
    assert ECONOMIC_BOOTSTRAP_QUANTILE_METHOD == "linear"
    assert ECONOMIC_BOOTSTRAP_K_VALUES == (1, 3, 5, 10)
    assert SHARED_RESAMPLE_INDICES is True


def test_economic_golden_rng_trace_and_lower_bounds() -> None:
    """Golden early replicates and 4-column lower bounds under PCG64(20260832)."""
    rng = np.random.Generator(np.random.PCG64(ECONOMIC_BOOTSTRAP_SEED))
    indices = generate_stationary_circular_indices(n=10, replications=3, rng=rng)

    # Independent hand-verified expected indices for N=10 under PCG64(20260832)
    expected_rep_0 = [6, 7, 8, 9, 0, 1, 2, 3, 4, 5]
    expected_rep_1 = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    expected_rep_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    np.testing.assert_array_equal(indices[0], expected_rep_0)
    np.testing.assert_array_equal(indices[1], expected_rep_1)
    np.testing.assert_array_equal(indices[2], expected_rep_2)

    # Test synthetic (10, 4) delta matrix
    delta = np.zeros((10, 4), dtype=np.float64)
    delta[:, 0] = np.linspace(-10, 10, 10)
    delta[:, 1] = np.linspace(-5, 15, 10)
    delta[:, 2] = np.linspace(0, 20, 10)
    delta[:, 3] = np.linspace(5, 25, 10)

    res = run_economic_bootstrap(delta)
    assert isinstance(res, EconomicBootstrapResult)
    assert res.bootstrap_replications == 2000
    assert res.replicate_means.shape == (2000, 4)
    assert list(res.lower_bounds.keys()) == [1, 3, 5, 10]

    expected_lbs = {
        1: -3.336111111111112,
        3: 1.663888888888889,
        5: 6.663888888888889,
        10: 11.66388888888889,
    }
    for k in [1, 3, 5, 10]:
        assert abs(res.lower_bounds[k] - expected_lbs[k]) < 1e-12

    # Check tuple property
    assert res.lower_bounds_tuple == (
        res.lower_bounds[1],
        res.lower_bounds[3],
        res.lower_bounds[5],
        res.lower_bounds[10],
    )


def test_economic_bootstrap_shared_resample_indices_invariant() -> None:
    """Verify that all 4 portfolio sizes K share the exact same resampled rows."""
    # Construct delta matrix where column 1 is known, and columns 3, 5, 10 are scaled multiples
    n = 20
    delta = np.zeros((n, 4), dtype=np.float64)
    delta[:, 0] = np.arange(n, dtype=np.float64)
    delta[:, 1] = delta[:, 0] * 2.0
    delta[:, 2] = delta[:, 0] * 3.0
    delta[:, 3] = delta[:, 0] * 4.0

    res = run_economic_bootstrap(delta)

    # Because indices are shared, replicate_means[:, 1] must equal 2 * replicate_means[:, 0] exactly!
    np.testing.assert_allclose(res.replicate_means[:, 1], 2.0 * res.replicate_means[:, 0], rtol=1e-14)
    np.testing.assert_allclose(res.replicate_means[:, 2], 3.0 * res.replicate_means[:, 0], rtol=1e-14)
    np.testing.assert_allclose(res.replicate_means[:, 3], 4.0 * res.replicate_means[:, 0], rtol=1e-14)


@pytest.mark.parametrize(
    "bad_delta,match_msg",
    [
        (np.zeros((0, 4)), "cannot be empty"),
        (np.zeros((10, 3)), "shape \\(N, 4\\)"),
        (np.zeros((10, 5)), "shape \\(N, 4\\)"),
        (np.zeros(10), "shape \\(N, 4\\)"),
        (np.full((10, 4), np.nan), "finite"),
        (np.full((10, 4), np.inf), "finite"),
    ],
)
def test_economic_bootstrap_input_validation(bad_delta: Any, match_msg: str) -> None:
    """Invalid delta inputs raise EconomicBootstrapError with ECONOMIC_BOOTSTRAP stage."""
    with pytest.raises(EconomicBootstrapError, match=match_msg) as exc_info:
        run_economic_bootstrap(bad_delta)

    err = exc_info.value
    assert err.stage == FailureStage.ECONOMIC_BOOTSTRAP
    proto = err.as_protocol_failure()
    assert proto.stage == FailureStage.ECONOMIC_BOOTSTRAP
    assert proto.exit_status in {
        FailureExitStatus.NEEDS_DATA_REVISION,
        FailureExitStatus.TECHNICAL_FAILURE,
    }


def test_economic_bootstrap_does_not_mutate_input() -> None:
    """Input matrix remains bitwise unmodified."""
    delta = np.ones((10, 4), dtype=np.float64)
    delta_copy = delta.copy()
    _ = run_economic_bootstrap(delta)
    np.testing.assert_array_equal(delta, delta_copy)


# --- 4. Isolation and No-Leakage Verification ---


def test_m3_08_does_not_leak_economics_or_multiplicity_rules() -> None:
    """M3-08 must not implement top-K ranking, PnL math, or economic qualification."""
    import inspect
    import src.m3_digit_factor.bootstrap as b_mod

    source = inspect.getsource(b_mod)
    forbidden_terms = [
        "qualifies",
        "qualified_top_k",
        "recommended_top_k",
        "positive_block_count",
        "mean_economic_delta",
        "99 *",
        "99.0 *",
        "- 27",
        "BCa",
        "studentized",
        "bias_corrected",
    ]
    for term in forbidden_terms:
        assert term not in source, f"Forbidden term {term!r} found in bootstrap.py"
