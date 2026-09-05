"""Tests for XPIS v3 M3 exact SLSQP model fitting (Slice M3-06)."""

import math
from typing import Any

import numpy as np
import pytest
import scipy.optimize

from src.m3_digit_factor.contracts import FailureExitStatus, FailureStage
from src.m3_digit_factor.fitting import (
    EQUALITY_CONSTRAINT_TOLERANCE,
    GAMMA_ZERO_TOLERANCE,
    SLSQP_FTOL,
    SLSQP_MAXITER,
    ConstraintViolation,
    FittedModelResult,
    ForecastContractViolation,
    ModelFitError,
    NonFiniteModelFit,
    OptimizerExecutionError,
    OptimizerNonConvergence,
    compute_constraint_jacobian,
    compute_constraint_residuals,
    compute_objective,
    compute_objective_jacobian,
    fit_m3_model,
)
from src.m3_digit_factor.initialization import initialize_m3_from_counts
from src.m3_digit_factor.model import (
    THETA_LENGTH,
    pack_parameters,
    unpack_parameters,
)


def test_fitting_constants() -> None:
    assert EQUALITY_CONSTRAINT_TOLERANCE == 1e-10
    assert GAMMA_ZERO_TOLERANCE == 1e-10
    assert SLSQP_MAXITER == 2000
    assert SLSQP_FTOL == 1e-12
    assert issubclass(OptimizerNonConvergence, ModelFitError)
    assert issubclass(OptimizerExecutionError, ModelFitError)
    assert issubclass(NonFiniteModelFit, ModelFitError)
    assert issubclass(ConstraintViolation, ModelFitError)
    assert issubclass(ForecastContractViolation, ModelFitError)


def test_objective_value_against_independent_formula() -> None:
    rng = np.random.default_rng(20260906)
    theta = rng.normal(size=41)
    theta[40] = abs(theta[40])  # gamma >= 0

    # 10x10 counts
    counts = rng.integers(1, 20, size=(10, 10))
    # Independent calculation of softmax and negative log likelihood
    a, b, u, v, gamma = unpack_parameters(theta)
    eta = np.empty((10, 10), dtype=np.float64)
    for i in range(10):
        for j in range(10):
            eta[i, j] = a[i] + b[j] + gamma * u[i] * v[j]

    max_eta = np.max(eta)
    exp_eta = np.exp(eta - max_eta)
    p_expected = exp_eta / np.sum(exp_eta)

    expected_nll = -float(np.sum(counts * np.log(p_expected)))

    obj_val = compute_objective(theta, counts)
    assert math.isclose(obj_val, expected_nll, rel_tol=1e-14, abs_tol=1e-14)


def central_finite_difference_gradient(
    theta: np.ndarray,
    counts: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Independent test oracle: central finite difference gradient for objective."""
    grad = np.empty_like(theta)
    for k in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[k] += eps
        theta_minus[k] -= eps
        L_plus = compute_objective(theta_plus, counts)
        L_minus = compute_objective(theta_minus, counts)
        grad[k] = (L_plus - L_minus) / (2.0 * eps)
    return grad


@pytest.mark.parametrize(
    "gamma_val",
    [2.5, 0.0],
    ids=["gamma_positive", "gamma_zero"],
)
def test_analytic_gradient_matches_finite_difference_oracle(gamma_val: float) -> None:
    rng = np.random.default_rng(12345)
    a = rng.normal(scale=0.5, size=10)
    a -= np.mean(a)
    b = rng.normal(scale=0.5, size=10)
    b -= np.mean(b)
    u = rng.normal(scale=0.5, size=10)
    u -= np.mean(u)
    u /= np.linalg.norm(u)
    v = rng.normal(scale=0.5, size=10)
    v -= np.mean(v)
    v /= np.linalg.norm(v)

    theta = pack_parameters(a, b, u, v, gamma_val)

    counts = rng.integers(5, 30, size=(10, 10))

    analytic_grad = compute_objective_jacobian(theta, counts)
    fd_grad = central_finite_difference_gradient(theta, counts, eps=1e-7)

    assert analytic_grad.shape == (41,)
    assert np.all(np.isfinite(analytic_grad))

    # All 41 coordinates must match within tight numerical differentiation tolerance
    assert np.allclose(analytic_grad, fd_grad, rtol=1e-5, atol=1e-5)


def test_constraint_residual_vector_exact_order_and_shape() -> None:
    a = np.array([1.0] * 10)  # sum = 10
    b = np.array([2.0] * 10)  # sum = 20
    u = np.array([0.5] * 10)  # sum = 5, dot = 10 * 0.25 = 2.5 -> dot - 1 = 1.5
    v = np.array([0.3] * 10)  # sum = 3, dot = 10 * 0.09 = 0.9 -> dot - 1 = -0.1
    gamma = 5.0

    theta = pack_parameters(a, b, u, v, gamma)
    residuals = compute_constraint_residuals(theta)

    assert residuals.shape == (6,)
    expected = np.array([10.0, 20.0, 5.0, 3.0, 1.5, -0.1], dtype=np.float64)
    assert np.allclose(residuals, expected, atol=1e-14)


def test_constraint_jacobian_shape_and_complete_exact_values() -> None:
    rng = np.random.default_rng(20260906)
    a = rng.normal(size=10)
    b = rng.normal(size=10)
    u = rng.normal(size=10)
    v = rng.normal(size=10)
    gamma = 3.7

    theta = pack_parameters(a, b, u, v, gamma)
    jac = compute_constraint_jacobian(theta)

    assert jac.shape == (6, 41)

    expected_jac = np.zeros((6, 41), dtype=np.float64)
    # Row 0: sum a = 0 -> 1 on coords 0..9
    expected_jac[0, 0:10] = 1.0
    # Row 1: sum b = 0 -> 1 on coords 10..19
    expected_jac[1, 10:20] = 1.0
    # Row 2: sum u = 0 -> 1 on coords 20..29
    expected_jac[2, 20:30] = 1.0
    # Row 3: sum v = 0 -> 1 on coords 30..39
    expected_jac[3, 30:40] = 1.0
    # Row 4: sum u^2 - 1 = 0 -> 2*u on coords 20..29
    expected_jac[4, 20:30] = 2.0 * u
    # Row 5: sum v^2 - 1 = 0 -> 2*v on coords 30..39
    expected_jac[5, 30:40] = 2.0 * v

    assert np.array_equal(jac, expected_jac)


def test_slsqp_call_parameters_and_spy_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    spy_calls: list[dict[str, Any]] = []

    def mock_minimize(
        fun: Any,
        x0: Any,
        method: Any = None,
        jac: Any = None,
        bounds: Any = None,
        constraints: Any = None,
        options: Any = None,
        **kwargs: Any,
    ) -> Any:
        spy_calls.append(
            {
                "fun": fun,
                "x0": x0.copy(),
                "method": method,
                "jac": jac,
                "bounds": bounds,
                "constraints": constraints,
                "options": options,
                "kwargs": kwargs,
            }
        )
        # Return a mock successful result
        res = scipy.optimize.OptimizeResult()
        res.success = True
        res.x = x0.copy()  # x0 already satisfies equality constraints
        res.fun = 100.0
        res.nit = 5
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    # Valid counts: W = 30 -> 27 * 30 = 810
    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1
    assert int(np.sum(counts)) == 27 * 30

    init_res = initialize_m3_from_counts(counts, W=30)
    _ = fit_m3_model(counts, W=30)

    assert len(spy_calls) == 1
    call = spy_calls[0]

    # Verify method
    assert call["method"] == "SLSQP"

    # Verify x0 is exact U-004 initialization theta
    assert np.array_equal(call["x0"], init_res.theta)

    # Verify bounds: 40 unbounded, gamma in [0, +inf)
    bounds = call["bounds"]
    assert len(bounds) == 41
    for k in range(40):
        low, high = bounds[k]
        assert low is None or np.isneginf(low)
        assert high is None or np.isposinf(high)
    gamma_low, gamma_high = bounds[40]
    assert gamma_low == 0.0
    assert gamma_high is None or np.isposinf(gamma_high)

    # Verify options
    options = call["options"]
    assert options["maxiter"] == 2000
    assert options["ftol"] == 1e-12
    assert options["disp"] is False
    assert "tol" not in options
    assert "tol" not in call["kwargs"]

    # Verify callables supplied
    assert callable(call["jac"])
    assert isinstance(call["constraints"], list) and len(call["constraints"]) == 1
    eq_cons = call["constraints"][0]
    assert eq_cons["type"] == "eq"
    assert callable(eq_cons["fun"])
    assert callable(eq_cons["jac"])


def test_independent_initialization_between_distinct_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    intercepted_x0s: list[np.ndarray] = []

    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        intercepted_x0s.append(x0.copy())
        res = scipy.optimize.OptimizeResult()
        res.success = True
        res.x = x0.copy()
        res.fun = 50.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    # Fit 1
    counts1 = np.full((10, 10), 8, dtype=np.int64)
    counts1[0, :] += 1
    init1 = initialize_m3_from_counts(counts1, W=30)
    _ = fit_m3_model(counts1, W=30)

    # Fit 2 with distinct counts
    counts2 = np.full((10, 10), 8, dtype=np.int64)
    counts2[5, :] += 1
    init2 = initialize_m3_from_counts(counts2, W=30)
    _ = fit_m3_model(counts2, W=30)

    assert len(intercepted_x0s) == 2
    assert np.array_equal(intercepted_x0s[0], init1.theta)
    assert np.array_equal(intercepted_x0s[1], init2.theta)
    assert not np.array_equal(intercepted_x0s[0], intercepted_x0s[1])


def test_optimizer_nonconvergence_failure_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = False
        res.message = "Iteration limit reached"
        res.x = x0.copy()
        res.fun = 100.0
        res.nit = 2000
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    with pytest.raises(OptimizerNonConvergence) as exc_info:
        fit_m3_model(counts, W=30)

    err = exc_info.value
    assert err.stage == FailureStage.MODEL_FIT
    assert err.error_type == "OptimizerNonConvergence"
    assert err.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION


@pytest.mark.parametrize(
    ("bad_x", "bad_fun"),
    [
        (np.full(41, np.nan), 10.0),
        (np.full(41, np.inf), 10.0),
        (np.zeros(41), np.nan),
        (np.zeros(41), np.inf),
    ],
)
def test_nonfinite_model_fit_failure_mapping(
    monkeypatch: pytest.MonkeyPatch,
    bad_x: np.ndarray,
    bad_fun: float,
) -> None:
    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        res.x = bad_x.copy()
        res.fun = bad_fun
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    with pytest.raises(NonFiniteModelFit) as exc_info:
        fit_m3_model(counts, W=30)

    err = exc_info.value
    assert err.stage == FailureStage.MODEL_FIT
    assert err.error_type == "NonFiniteModelFit"
    assert err.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION


def test_raw_negative_gamma_fails_constraint_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        res.x = x0.copy()
        res.x[40] = -1e-12  # raw negative gamma
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    with pytest.raises(ConstraintViolation) as exc_info:
        fit_m3_model(counts, W=30)

    err = exc_info.value
    assert err.stage == FailureStage.MODEL_FIT
    assert err.error_type == "ConstraintViolation"
    assert err.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION


def test_equality_feasibility_gate_and_no_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    # Case 1: residual exceeds 1e-10 -> ConstraintViolation
    def mock_infeasible(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        res.x = x0.copy()
        res.x[0] += 1.01e-10  # drift in sum(a)
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_infeasible)
    with pytest.raises(ConstraintViolation) as exc_info:
        fit_m3_model(counts, W=30)
    assert exc_info.value.error_type == "ConstraintViolation"

    # Case 2: residual within 1e-10 -> PASSES feasibility
    def mock_feasible(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        res.x = x0.copy()
        res.x[0] += 0.5e-10  # within tolerance
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_feasible)
    res = fit_m3_model(counts, W=30)
    assert isinstance(res, FittedModelResult)


@pytest.mark.parametrize(
    ("gamma_raw", "expected_gamma_is_zero"),
    [
        (0.5e-10, True),
        (1.0e-10, True),
        (1.01e-10, False),
    ],
    ids=["below_tol", "at_tol", "above_tol"],
)
def test_gamma_zero_canonicalization_boundary(
    monkeypatch: pytest.MonkeyPatch,
    gamma_raw: float,
    expected_gamma_is_zero: bool,
) -> None:
    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        res.x = x0.copy()
        res.x[40] = gamma_raw
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    fit_res = fit_m3_model(counts, W=30)

    z_canonical = np.array([9.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]) / math.sqrt(90.0)

    if expected_gamma_is_zero:
        assert fit_res.gamma == 0.0
        assert np.allclose(fit_res.u, z_canonical, atol=1e-14)
        assert np.allclose(fit_res.v, z_canonical, atol=1e-14)
    else:
        assert fit_res.gamma == gamma_raw


def test_nonzero_canonical_sign_rule_in_post_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that when gamma > 1e-10, if max |u| has negative sign, u and v are negated
    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        # Set u with negative dominant element
        a = np.zeros(10)
        b = np.zeros(10)
        u = np.array([-0.9] + [0.1] * 9)
        u -= np.mean(u)
        u /= np.linalg.norm(u)
        v = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0])
        v -= np.mean(v)
        v /= np.linalg.norm(v)
        gamma = 2.0
        res.x = pack_parameters(a, b, u, v, gamma)
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    fit_res = fit_m3_model(counts, W=30)
    # The dominant element was at index 0 and was negative, so after canonicalization u[0] must be > 0
    assert fit_res.u[0] > 0.0


def test_forecast_validation_gate_fails_closed_underflow(monkeypatch: pytest.MonkeyPatch) -> None:
    # If extreme parameters produce underflow in forecast_mu after passing equality feasibility:
    z_unit = np.array([9.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]) / math.sqrt(90.0)

    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        a = np.zeros(10)
        a[0] = -1000.0  # extreme negative causing underflow in softmax
        a[1:] = 1000.0 / 9.0  # sum(a) == 0.0 exactly
        b = np.zeros(10)
        u = z_unit.copy()  # sum(u) == 0, dot(u, u) == 1
        v = z_unit.copy()  # sum(v) == 0, dot(v, v) == 1
        gamma = 0.0
        res.x = pack_parameters(a, b, u, v, gamma)
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    with pytest.raises(ForecastContractViolation) as exc_info:
        fit_m3_model(counts, W=30)

    err = exc_info.value
    assert err.stage == FailureStage.MODEL_FIT
    assert err.error_type == "ForecastContractViolation"
    assert err.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION


def test_input_count_array_not_mutated_during_fit() -> None:
    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1
    orig_counts = counts.copy()

    _ = fit_m3_model(counts, W=30)
    assert np.array_equal(counts, orig_counts)


def test_real_slsqp_smoke_test_converges_and_returns_valid_fit() -> None:
    # Small synthetic deterministic fitting test using REAL scipy.optimize.minimize
    # W = 30 -> 27 * 30 = 810
    W = 30
    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, 0] += 5
    counts[0, 1] += 3
    counts[1, 0] += 2
    # Current sum: 800 + 10 = 810
    assert int(np.sum(counts)) == 27 * W

    fit_res = fit_m3_model(counts, W=W)

    assert isinstance(fit_res, FittedModelResult)
    assert np.all(np.isfinite(fit_res.theta))
    assert fit_res.gamma >= 0.0
    assert len(fit_res.theta) == THETA_LENGTH

    # Forecast mu checks
    assert fit_res.mu.shape == (100,)
    assert np.all(np.isfinite(fit_res.mu))
    assert np.all(fit_res.mu > 0.0)
    assert abs(float(np.sum(fit_res.mu)) - 27.0) <= 1e-10

    # Constraint feasibility of accepted result
    residuals = compute_constraint_residuals(fit_res.theta)
    assert np.max(np.abs(residuals)) <= EQUALITY_CONSTRAINT_TOLERANCE


def test_tied_max_sign_rule_in_post_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Construct u with tied max absolute values at index 1 and 4
    # u[1] = -0.5, u[4] = +0.5
    # Smallest index is 1, where value is negative -> must flip both u and v!
    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        a = np.zeros(10)
        b = np.zeros(10)
        u = np.zeros(10)
        u[1] = -1.0 / math.sqrt(2.0)
        u[4] = 1.0 / math.sqrt(2.0)
        v = np.zeros(10)
        v[2] = 1.0 / math.sqrt(2.0)
        v[3] = -1.0 / math.sqrt(2.0)
        gamma = 1.5
        res.x = pack_parameters(a, b, u, v, gamma)
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    fit_res = fit_m3_model(counts, W=30)
    # After flip, u[1] must be positive
    assert fit_res.u[1] > 0.0
    assert fit_res.u[4] < 0.0


def test_no_post_fit_recentering_or_renormalization(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verify that post-solver vectors are preserved without altering magnitudes
    z_unit = np.array([9.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]) / math.sqrt(90.0)

    # Slight perturbation within tolerance (e.g. 5e-11)
    a_raw = np.array([0.05e-10] * 10)  # sum(a) = 0.5e-10 <= 1e-10
    b_raw = np.zeros(10)
    u_raw = z_unit.copy()
    v_raw = z_unit.copy()
    gamma_raw = 2.0

    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        res.x = pack_parameters(a_raw, b_raw, u_raw, v_raw, gamma_raw)
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    fit_res = fit_m3_model(counts, W=30)

    # The implementation must NOT recenter a to sum to exactly 0.0 if within tolerance
    assert np.array_equal(fit_res.a, a_raw)
    assert np.array_equal(fit_res.u, u_raw)
    assert np.array_equal(fit_res.v, v_raw)


def test_post_fit_order_regression_infeasible_not_repaired_by_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If gamma <= 1e-10 but equality constraint was violated (e.g. sum(a) = 1e-5),
    # it must fail at Step 2 (feasibility) and NOT be masked by gamma-zero canonicalization!
    def mock_minimize(fun: Any, x0: Any, **kwargs: Any) -> Any:
        res = scipy.optimize.OptimizeResult()
        res.success = True
        a = np.array([1e-5] * 10)  # sum(a) = 1e-4 >> 1e-10
        b = np.zeros(10)
        u = np.zeros(10)
        v = np.zeros(10)
        gamma = 0.5e-10  # would enter gamma-zero branch if Step 2 didn't fail
        res.x = pack_parameters(a, b, u, v, gamma)
        res.fun = 10.0
        res.nit = 10
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", mock_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    with pytest.raises(ConstraintViolation) as exc_info:
        fit_m3_model(counts, W=30)
    assert exc_info.value.error_type == "ConstraintViolation"


def test_optimizer_execution_exception_boundary_fails_closed_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver_call_count = 0

    def mock_crashing_minimize(*args: Any, **kwargs: Any) -> Any:
        nonlocal solver_call_count
        solver_call_count += 1
        raise RuntimeError("Simulated low-level solver crash before returning result")

    monkeypatch.setattr(scipy.optimize, "minimize", mock_crashing_minimize)

    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1

    with pytest.raises(OptimizerExecutionError) as exc_info:
        fit_m3_model(counts, W=30)

    err = exc_info.value

    # Exact failure triplet
    assert err.stage == FailureStage.MODEL_FIT
    assert err.error_type == "OptimizerExecutionError"
    assert err.exit_status == FailureExitStatus.TECHNICAL_FAILURE

    # Protocol failure representation
    proto = err.as_protocol_failure()
    assert proto.stage == FailureStage.MODEL_FIT
    assert proto.error_type == "OptimizerExecutionError"
    assert proto.exit_status == FailureExitStatus.TECHNICAL_FAILURE

    # Distinct from OptimizerNonConvergence
    assert not isinstance(err, OptimizerNonConvergence)
    assert err.error_type != "OptimizerNonConvergence"
    assert err.exit_status != FailureExitStatus.NEEDS_MODEL_REVISION

    # Exactly one solver call, no retry, no fallback solver
    assert solver_call_count == 1


