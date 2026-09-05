"""Tests for XPIS v3 M3 digit-factor model representation (Slice M3-04)."""

import ast
import inspect
import math

import numpy as np
import pytest

from src.m3_digit_factor.model import (
    FORECAST_SUM_TOLERANCE,
    MODEL_FAMILY,
    THETA_LENGTH,
    ForecastContractViolation,
    ModelValidationError,
    compute_eta,
    compute_probabilities,
    digits_to_outcome,
    forecast_mu,
    outcome_to_digits,
    pack_parameters,
    unpack_parameters,
)


def test_constants() -> None:
    assert THETA_LENGTH == 41
    assert FORECAST_SUM_TOLERANCE == 1e-10
    assert MODEL_FAMILY == "RANK1_DIGIT_INTERACTION_MULTINOMIAL_LOGLINEAR"


@pytest.mark.parametrize(
    ("outcome", "expected_head", "expected_tail"),
    [
        (0, 0, 0),
        (9, 0, 9),
        (10, 1, 0),
        (42, 4, 2),
        (99, 9, 9),
    ],
)
def test_outcome_to_digits_pinned_values(outcome: int, expected_head: int, expected_tail: int) -> None:
    i, j = outcome_to_digits(outcome)
    assert (i, j) == (expected_head, expected_tail)
    assert digits_to_outcome(i, j) == outcome


def test_outcome_to_digits_complete_range() -> None:
    for n in range(100):
        i, j = outcome_to_digits(n)
        assert i == n // 10
        assert j == n % 10
        assert 0 <= i <= 9
        assert 0 <= j <= 9
        assert digits_to_outcome(i, j) == n


@pytest.mark.parametrize("invalid_n", [-1, 100, -10, 105, 999])
def test_outcome_to_digits_out_of_bounds_rejected(invalid_n: int) -> None:
    with pytest.raises((ValueError, ModelValidationError)):
        outcome_to_digits(invalid_n)


@pytest.mark.parametrize("invalid_type", ["42", 4.2, None, [42], (4, 2), True, False])
def test_outcome_to_digits_invalid_type_rejected(invalid_type: object) -> None:
    with pytest.raises((TypeError, ValueError, ModelValidationError)):
        outcome_to_digits(invalid_type)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_coord", [(-1, 5), (5, -1), (10, 0), (0, 10), (12, 15)])
def test_digits_to_outcome_out_of_bounds_rejected(invalid_coord: tuple[int, int]) -> None:
    i, j = invalid_coord
    with pytest.raises((ValueError, ModelValidationError)):
        digits_to_outcome(i, j)


@pytest.mark.parametrize("invalid_coord", [(True, 5), (5, False), ("1", 2), (1, 2.5)])
def test_digits_to_outcome_invalid_type_rejected(invalid_coord: tuple[object, object]) -> None:
    i, j = invalid_coord
    with pytest.raises((TypeError, ValueError, ModelValidationError)):
        digits_to_outcome(i, j)  # type: ignore[arg-type]


def test_pack_parameters_exact_order_and_slices() -> None:
    a = np.arange(10, dtype=np.float64)
    b = np.arange(10, 20, dtype=np.float64)
    u = np.arange(20, 30, dtype=np.float64)
    v = np.arange(30, 40, dtype=np.float64)
    gamma = 40.0

    theta = pack_parameters(a, b, u, v, gamma)

    assert isinstance(theta, np.ndarray)
    assert theta.shape == (41,)
    assert theta.dtype == np.float64
    assert np.array_equal(theta[0:10], a)
    assert np.array_equal(theta[10:20], b)
    assert np.array_equal(theta[20:30], u)
    assert np.array_equal(theta[30:40], v)
    assert theta[40] == 40.0


def test_unpack_parameters_exact_order_and_slices() -> None:
    expected_theta = np.arange(41, dtype=np.float64)
    a, b, u, v, gamma = unpack_parameters(expected_theta)

    assert isinstance(a, np.ndarray) and a.shape == (10,) and a.dtype == np.float64
    assert isinstance(b, np.ndarray) and b.shape == (10,) and b.dtype == np.float64
    assert isinstance(u, np.ndarray) and u.shape == (10,) and u.dtype == np.float64
    assert isinstance(v, np.ndarray) and v.shape == (10,) and v.dtype == np.float64
    assert isinstance(gamma, float)

    assert np.array_equal(a, expected_theta[0:10])
    assert np.array_equal(b, expected_theta[10:20])
    assert np.array_equal(u, expected_theta[20:30])
    assert np.array_equal(v, expected_theta[30:40])
    assert gamma == float(expected_theta[40])


def test_pack_unpack_roundtrip() -> None:
    rng = np.random.default_rng(20260906)
    a = rng.normal(size=10)
    b = rng.normal(size=10)
    u = rng.normal(size=10)
    v = rng.normal(size=10)
    gamma = float(rng.uniform(0.1, 5.0))

    theta = pack_parameters(a, b, u, v, gamma)
    a_out, b_out, u_out, v_out, gamma_out = unpack_parameters(theta)

    assert np.array_equal(a, a_out)
    assert np.array_equal(b, b_out)
    assert np.array_equal(u, u_out)
    assert np.array_equal(v, v_out)
    assert gamma == gamma_out


@pytest.mark.parametrize(
    "bad_a",
    [
        np.zeros(9),
        np.zeros(11),
        np.zeros((5, 2)),
    ],
)
def test_pack_parameters_mismatched_lengths_rejected(bad_a: np.ndarray) -> None:
    b = np.zeros(10)
    u = np.zeros(10)
    v = np.zeros(10)
    with pytest.raises((ValueError, ModelValidationError)):
        pack_parameters(bad_a, b, u, v, 1.0)


@pytest.mark.parametrize(
    "bad_theta",
    [
        np.zeros(40),
        np.zeros(42),
        np.zeros(0),
        np.zeros((41, 1)),
    ],
)
def test_unpack_parameters_mismatched_theta_rejected(bad_theta: np.ndarray) -> None:
    with pytest.raises((ValueError, ModelValidationError)):
        unpack_parameters(bad_theta)


@pytest.mark.parametrize(
    "non_finite_val",
    [np.nan, np.inf, -np.inf],
)
def test_pack_parameters_rejects_non_finite(non_finite_val: float) -> None:
    a = np.zeros(10)
    a[3] = non_finite_val
    b = np.zeros(10)
    u = np.zeros(10)
    v = np.zeros(10)
    with pytest.raises((ValueError, ModelValidationError)):
        pack_parameters(a, b, u, v, 0.0)

    # non-finite gamma
    with pytest.raises((ValueError, ModelValidationError)):
        pack_parameters(np.zeros(10), b, u, v, non_finite_val)


@pytest.mark.parametrize(
    "non_finite_val",
    [np.nan, np.inf, -np.inf],
)
def test_unpack_parameters_rejects_non_finite(non_finite_val: float) -> None:
    theta = np.zeros(41)
    theta[15] = non_finite_val
    with pytest.raises((ValueError, ModelValidationError)):
        unpack_parameters(theta)


def test_caller_arrays_not_mutated_during_pack() -> None:
    a = np.array([1.0] * 10)
    b = np.array([2.0] * 10)
    u = np.array([3.0] * 10)
    v = np.array([4.0] * 10)
    a_orig = a.copy()
    b_orig = b.copy()
    u_orig = u.copy()
    v_orig = v.copy()

    theta = pack_parameters(a, b, u, v, 5.0)
    theta[0] = 999.0
    theta[10] = 888.0

    assert np.array_equal(a, a_orig)
    assert np.array_equal(b, b_orig)
    assert np.array_equal(u, u_orig)
    assert np.array_equal(v, v_orig)


def test_unpack_does_not_permit_write_through_to_theta() -> None:
    theta = np.zeros(41, dtype=np.float64)
    a, b, u, v, gamma = unpack_parameters(theta)

    a[0] = 123.0
    b[1] = 456.0
    u[2] = 789.0
    v[3] = 987.0

    assert theta[0] == 0.0
    assert theta[11] == 0.0
    assert theta[22] == 0.0
    assert theta[33] == 0.0


def test_compute_eta_shape_and_hand_calculated_formula() -> None:
    a = np.array([0.1 * i for i in range(10)], dtype=np.float64)
    b = np.array([-0.05 * j for j in range(10)], dtype=np.float64)
    u = np.array([0.2 * (i - 4.5) for i in range(10)], dtype=np.float64)
    v = np.array([-0.3 * (j - 4.5) for j in range(10)], dtype=np.float64)
    gamma = 2.5

    theta = pack_parameters(a, b, u, v, gamma)
    eta = compute_eta(theta)

    assert eta.shape == (10, 10)
    assert eta.dtype == np.float64

    for i in range(10):
        for j in range(10):
            expected = a[i] + b[j] + gamma * u[i] * v[j]
            assert math.isclose(eta[i, j], expected, rel_tol=1e-14, abs_tol=1e-14)


def test_compute_eta_exact_gamma_interaction_contribution() -> None:
    a = np.zeros(10)
    b = np.zeros(10)
    u = np.zeros(10)
    v = np.zeros(10)
    u[4] = 0.5
    v[2] = -0.8
    gamma = 3.0

    theta = pack_parameters(a, b, u, v, gamma)
    eta = compute_eta(theta)

    for i in range(10):
        for j in range(10):
            if (i, j) == (4, 2):
                assert math.isclose(eta[i, j], 3.0 * 0.5 * (-0.8), rel_tol=1e-14, abs_tol=1e-14)
            else:
                assert eta[i, j] == 0.0


def test_negative_gamma_not_clipped() -> None:
    a = np.zeros(10)
    b = np.zeros(10)
    u = np.ones(10)
    v = np.ones(10)
    gamma = -2.5

    theta = pack_parameters(a, b, u, v, gamma)
    eta = compute_eta(theta)

    assert np.allclose(eta, -2.5)


def test_theta_not_mutated_by_compute_eta_or_forecast() -> None:
    rng = np.random.default_rng(20260906)
    theta = rng.normal(size=41)
    theta_orig = theta.copy()

    _ = compute_eta(theta)
    assert np.array_equal(theta, theta_orig)

    _ = compute_probabilities(theta)
    assert np.array_equal(theta, theta_orig)

    _ = forecast_mu(theta)
    assert np.array_equal(theta, theta_orig)


def test_uniform_model_regression() -> None:
    a = np.zeros(10)
    b = np.zeros(10)
    u = np.ones(10) / math.sqrt(10)
    v = np.ones(10) / math.sqrt(10)
    gamma = 0.0

    theta = pack_parameters(a, b, u, v, gamma)

    eta = compute_eta(theta)
    assert np.all(eta == 0.0)

    p = compute_probabilities(theta)
    assert p.shape == (10, 10)
    assert np.allclose(p, 0.01, atol=1e-15)

    mu = forecast_mu(theta)
    assert mu.shape == (100,)
    assert np.allclose(mu, 0.27, atol=1e-15)
    assert math.isclose(float(np.sum(mu)), 27.0, abs_tol=1e-10)


def test_matrix_to_outcome_ordering_pins_row_major_n_equals_10_i_plus_j() -> None:
    a = np.zeros(10)
    b = np.zeros(10)
    u = np.zeros(10)
    v = np.zeros(10)
    # Give cell (4, 2) uniquely high eta
    a[4] = 2.0
    b[2] = 3.0
    gamma = 0.0

    theta = pack_parameters(a, b, u, v, gamma)
    eta = compute_eta(theta)

    assert np.argmax(eta) == 42  # in C order (row-major)
    assert eta[4, 2] == 5.0

    p = compute_probabilities(theta)
    assert (p[4, 2] == np.max(p))

    mu = forecast_mu(theta)
    assert np.argmax(mu) == 42
    assert mu[42] == np.max(mu)


def test_global_100_cell_normalization_and_mu_properties() -> None:
    rng = np.random.default_rng(42)
    theta = rng.normal(size=41)

    p = compute_probabilities(theta)
    assert p.shape == (10, 10)
    assert math.isclose(float(np.sum(p)), 1.0, abs_tol=1e-14)
    assert np.all(p > 0.0)

    mu = forecast_mu(theta)
    assert mu.shape == (100,)
    assert np.all(np.isfinite(mu))
    assert np.all(mu > 0.0)
    assert abs(float(np.sum(mu)) - 27.0) <= FORECAST_SUM_TOLERANCE


def test_constant_shift_invariance() -> None:
    rng = np.random.default_rng(12345)
    theta = rng.normal(size=41)
    eta = compute_eta(theta)

    p_base = compute_probabilities(eta)

    for c in [-1000.0, -42.5, 0.0, 15.2, 500.0, 50000.0]:
        eta_shifted = eta + c
        p_shifted = compute_probabilities(eta_shifted)
        assert np.allclose(p_shifted, p_base, atol=1e-12, rtol=1e-11)


def test_numerical_stability_large_finite_eta() -> None:
    # Very large positive finite eta: in unstabilized softmax, exp(1500) overflows to +inf
    a = np.array([1000.0] * 10)
    b = np.array([500.0] * 10)
    u = np.zeros(10)
    v = np.zeros(10)
    gamma = 0.0

    # Add moderate variation within [0, 5]
    a[0] += 2.0
    b[3] -= 1.5

    theta = pack_parameters(a, b, u, v, gamma)
    p = compute_probabilities(theta)
    mu = forecast_mu(theta)

    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(mu))
    assert np.all(mu > 0.0)
    assert math.isclose(float(np.sum(p)), 1.0, abs_tol=1e-12)
    assert abs(float(np.sum(mu)) - 27.0) <= FORECAST_SUM_TOLERANCE


def test_no_clipping_behavioral_regression() -> None:
    # Construct eta where one cell has a very small, strictly positive probability
    # e.g., delta eta = -50.0 relative to others
    a = np.zeros(10)
    a[0] = -50.0
    b = np.zeros(10)
    u = np.zeros(10)
    v = np.zeros(10)
    gamma = 0.0

    theta = pack_parameters(a, b, u, v, gamma)
    p = compute_probabilities(theta)
    mu = forecast_mu(theta)

    # For cell (0, 0), probability is ~ exp(-50) / (10 * exp(-50) + 90 * exp(0))
    # exp(-50) ~ 1.9287e-22, denominator ~ 90, so p ~ 2.143e-24
    # mu ~ 27 * p ~ 5.786e-23
    # If legacy clipping or epsilon floor (e.g. 1e-9, 1e-12, 1e-15) were active,
    # mu[0] would be >= 1e-15.
    assert 0.0 < mu[0] < 1e-20
    assert 0.0 < p[0, 0] < 1e-20
    assert abs(float(np.sum(mu)) - 27.0) <= FORECAST_SUM_TOLERANCE


def test_no_clipping_ast_check() -> None:
    import src.m3_digit_factor.model as model_mod

    source = inspect.getsource(model_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            assert func_name not in ("clip", "maximum", "minimum"), (
                f"Forbidden clipping/capping function {func_name!r} detected in model.py"
            )


def test_no_posthoc_renormalization_exact_relation() -> None:
    rng = np.random.default_rng(999)
    theta = rng.normal(size=41)

    p = compute_probabilities(theta)
    mu = forecast_mu(theta)

    # mu must strictly equal 27.0 * p.ravel() with exact numerical equivalence
    expected_mu = 27.0 * p.ravel()
    assert np.array_equal(mu, expected_mu)


def test_underflow_producing_zero_mu_fails_closed() -> None:
    # If an extreme difference causes underflow to exactly 0.0
    # The specification mandates:
    # "Any numerical underflow producing mu[n] = 0.0 is fail-closed: ERROR_TYPE=ForecastContractViolation"
    a = np.zeros(10)
    a[0] = -1000.0  # exp(-1000) underflows to 0.0 in float64
    b = np.zeros(10)
    u = np.zeros(10)
    v = np.zeros(10)
    gamma = 0.0

    theta = pack_parameters(a, b, u, v, gamma)
    with pytest.raises(ForecastContractViolation):
        forecast_mu(theta)
