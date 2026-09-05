"""Exact constrained SLSQP model fitting for XPIS v3 M3 (Slice M3-06).

This module implements the exact multinomial loglinear fitting objective,
analytic objective Jacobian, analytic 6x41 equality constraint Jacobian,
and the four-stage post-solver validation pipeline under the frozen XPIS v3 M3
canonical specification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import scipy.optimize

from .contracts import FailureExitStatus, FailureStage, ProtocolFailure
from .initialization import initialize_m3_from_counts
from .model import (
    N_HEADS,
    N_OUTCOMES,
    N_TAILS,
    ModelValidationError,
    compute_probabilities,
    forecast_mu,
    pack_parameters,
    unpack_parameters,
)


EQUALITY_CONSTRAINT_TOLERANCE = 1e-10
GAMMA_ZERO_TOLERANCE = 1e-10
SLSQP_MAXITER = 2000
SLSQP_FTOL = 1e-12

_CANONICAL_ZERO_VECTOR = (
    np.array([9.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64) / math.sqrt(90.0)
)


class ModelFitError(ModelValidationError):
    """Base error for all failures during model fitting."""

    stage: FailureStage = FailureStage.MODEL_FIT
    exit_status: FailureExitStatus = FailureExitStatus.NEEDS_MODEL_REVISION

    def __init__(self, message: str, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type

    def as_protocol_failure(self) -> ProtocolFailure:
        return ProtocolFailure(
            stage=self.stage,
            error_type=self.error_type,
            exit_status=self.exit_status,
        )


class OptimizerNonConvergence(ModelFitError):
    """Raised when SLSQP optimizer fails to converge (success=False)."""

    def __init__(self, message: str = "SLSQP optimizer failed to converge") -> None:
        super().__init__(message, error_type="OptimizerNonConvergence")


class NonFiniteModelFit(ModelFitError):
    """Raised when fitted objective or any parameter is non-finite."""

    def __init__(self, message: str = "Non-finite parameter or objective value returned by optimizer") -> None:
        super().__init__(message, error_type="NonFiniteModelFit")


class ConstraintViolation(ModelFitError):
    """Raised when post-solver equality feasibility or parameter bounds are violated."""

    def __init__(self, message: str = "Fitted parameters violate equality constraints or bounds") -> None:
        super().__init__(message, error_type="ConstraintViolation")


class ForecastContractViolation(ModelFitError):
    """Raised when final forecast fails authoritative contract validation."""

    def __init__(self, message: str = "Final forecast violates contract") -> None:
        super().__init__(message, error_type="ForecastContractViolation")


@dataclass(frozen=True)
class FittedModelResult:
    """Immutable representation of accepted fitted model."""

    a: np.ndarray
    b: np.ndarray
    u: np.ndarray
    v: np.ndarray
    gamma: float
    theta: np.ndarray
    mu: np.ndarray
    objective_value: float
    iterations: int

    def __post_init__(self) -> None:
        for arr in (self.a, self.b, self.u, self.v, self.theta, self.mu):
            arr.flags.writeable = False


def _validate_counts_matrix(counts: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate and reshape input counts to 10x10 float64 array."""
    if isinstance(counts, (str, bytes)):
        raise ModelValidationError("Counts cannot be string or bytes")
    try:
        arr = np.asarray(counts, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise ModelValidationError("Counts could not be converted to float64 array") from err

    if arr.ndim == 1 and arr.shape[0] == N_OUTCOMES:
        counts_2d = arr.reshape((N_HEADS, N_TAILS))
    elif arr.ndim == 2 and arr.shape == (N_HEADS, N_TAILS):
        counts_2d = arr
    else:
        raise ModelValidationError(
            f"Counts must have shape ({N_OUTCOMES},) or ({N_HEADS}, {N_TAILS}), got {arr.shape}"
        )

    if not np.all(np.isfinite(counts_2d)):
        raise ModelValidationError("Counts contain non-finite values (NaN or Inf)")

    if np.any(counts_2d < 0.0):
        raise ModelValidationError("Counts cannot contain negative values")

    return counts_2d


def compute_objective(theta: np.ndarray, counts: np.ndarray) -> float:
    """Compute negative log-likelihood fitting objective L(theta).

    Formula:
        L(theta) = - sum_{i=0}^9 sum_{j=0}^9 X[i, j] * ln(p[i, j])
    """
    counts_2d = _validate_counts_matrix(counts)
    p = compute_probabilities(theta)  # shape (10, 10)
    # Binary64 log evaluation
    val = -float(np.sum(counts_2d * np.log(p)))
    return val


def compute_objective_jacobian(theta: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Compute analytic objective gradient with respect to the 41 parameters.

    Formulas:
        T = sum_{i,j} X[i, j]
        G[i, j] = T * p[i, j] - X[i, j]
        dL/da[i] = sum_j G[i, j]
        dL/db[j] = sum_i G[i, j]
        dL/du[i] = gamma * sum_j G[i, j] * v[j]
        dL/dv[j] = gamma * sum_i G[i, j] * u[i]
        dL/dgamma = sum_{i,j} G[i, j] * u[i] * v[j]
    """
    counts_2d = _validate_counts_matrix(counts)
    p = compute_probabilities(theta)  # shape (10, 10)

    T = float(np.sum(counts_2d))
    G = T * p - counts_2d

    a, b, u, v, gamma = unpack_parameters(theta)

    dL_da = np.sum(G, axis=1)
    dL_db = np.sum(G, axis=0)
    dL_du = gamma * (G @ v)
    dL_dv = gamma * (G.T @ u)
    dL_dgamma = float(u @ (G @ v))

    grad = pack_parameters(dL_da, dL_db, dL_du, dL_dv, dL_dgamma)
    return grad


def compute_constraint_residuals(theta: np.ndarray) -> np.ndarray:
    """Compute exact 6-dimensional equality constraint residual vector c(theta).

    Order:
        c(theta) = [
            sum(a),
            sum(b),
            sum(u),
            sum(v),
            dot(u, u) - 1.0,
            dot(v, v) - 1.0
        ]
    """
    a, b, u, v, _ = unpack_parameters(theta)
    return np.array(
        [
            float(np.sum(a)),
            float(np.sum(b)),
            float(np.sum(u)),
            float(np.sum(v)),
            float(np.dot(u, u) - 1.0),
            float(np.dot(v, v) - 1.0),
        ],
        dtype=np.float64,
    )


def compute_constraint_jacobian(theta: np.ndarray) -> np.ndarray:
    """Compute exact 6x41 analytic Jacobian of the equality constraints.

    Rows:
        row 0 (sum a): 1.0 on coords 0..9
        row 1 (sum b): 1.0 on coords 10..19
        row 2 (sum u): 1.0 on coords 20..29
        row 3 (sum v): 1.0 on coords 30..39
        row 4 (dot u, u - 1): 2*u on coords 20..29
        row 5 (dot v, v - 1): 2*v on coords 30..39
    """
    _, _, u, v, _ = unpack_parameters(theta)
    jac = np.zeros((6, 41), dtype=np.float64)
    jac[0, 0:10] = 1.0
    jac[1, 10:20] = 1.0
    jac[2, 20:30] = 1.0
    jac[3, 30:40] = 1.0
    jac[4, 20:30] = 2.0 * u
    jac[5, 30:40] = 2.0 * v
    return jac


def fit_m3_model(
    counts: Sequence[float] | np.ndarray,
    W: int | None = None,
) -> FittedModelResult:
    """Execute deterministic SLSQP model fitting with U-005 validation.

    Execution sequence:
        1. Initialize from U-004 deterministic initialization (M3-05).
        2. Solve constrained optimization via scipy.optimize.minimize (SLSQP).
        3. Post-solver Step 1: Raw solver validity (success, finite params, gamma >= 0).
        4. Post-solver Step 2: Equality feasibility (max abs residual <= 1e-10).
        5. Post-solver Step 3: Interaction canonicalization (gamma-zero or sign rule).
        6. Post-solver Step 4: Final forecast validation (finite, mu > 0, sum == 27.0).
    """
    counts_2d = _validate_counts_matrix(counts)

    # 1. Exact U-004 initialization for this fit's own data
    init_res = initialize_m3_from_counts(counts_2d, W=W)
    x0 = np.array(init_res.theta, dtype=np.float64, copy=True)

    # SLSQP parameter bounds: 40 unbounded, gamma in [0, +inf)
    bounds = [(None, None)] * 40 + [(0.0, None)]

    constraints = [
        {
            "type": "eq",
            "fun": compute_constraint_residuals,
            "jac": compute_constraint_jacobian,
        }
    ]

    # 2. Exact SLSQP invocation
    try:
        opt_res = scipy.optimize.minimize(
            fun=compute_objective,
            x0=x0,
            args=(counts_2d,),
            method="SLSQP",
            jac=compute_objective_jacobian,
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": SLSQP_MAXITER,
                "ftol": SLSQP_FTOL,
                "disp": False,
            },
        )
    except Exception as err:
        raise ModelFitError(f"SLSQP optimizer raised an unexpected exception: {err}", error_type="TechnicalFailure") from err

    # 3. Post-solver Step 1: Raw solver validity
    if not opt_res.success:
        raise OptimizerNonConvergence(f"SLSQP optimization failed to converge: {opt_res.message}")

    theta_raw = np.asarray(opt_res.x, dtype=np.float64)
    if not np.all(np.isfinite(theta_raw)) or not math.isfinite(float(opt_res.fun)):
        raise NonFiniteModelFit("Fitted parameter vector or objective value is non-finite")

    gamma_raw = float(theta_raw[40])
    if gamma_raw < 0.0:
        raise ConstraintViolation(f"Raw fitted gamma is negative: {gamma_raw}")

    # 4. Post-solver Step 2: Equality feasibility
    c_raw = compute_constraint_residuals(theta_raw)
    max_res = float(np.max(np.abs(c_raw)))
    if max_res > EQUALITY_CONSTRAINT_TOLERANCE:
        raise ConstraintViolation(
            f"Fitted equality residuals exceed tolerance: {max_res:.3e} > {EQUALITY_CONSTRAINT_TOLERANCE:.1e}"
        )

    # 5. Post-solver Step 3: Interaction canonicalization
    a_raw, b_raw, u_raw, v_raw, _ = unpack_parameters(theta_raw)

    if gamma_raw <= GAMMA_ZERO_TOLERANCE:
        gamma = 0.0
        u = _CANONICAL_ZERO_VECTOR.copy()
        v = _CANONICAL_ZERO_VECTOR.copy()
        a = a_raw.copy()
        b = b_raw.copy()
    else:
        gamma = gamma_raw
        a = a_raw.copy()
        b = b_raw.copy()
        u = u_raw.copy()
        v = v_raw.copy()

        abs_u = np.abs(u)
        max_abs_u = float(np.max(abs_u))
        k = int(np.where(abs_u == max_abs_u)[0][0])
        if u[k] < 0.0:
            u = -u
            v = -v

    theta_canonical = pack_parameters(a, b, u, v, gamma)

    # 6. Post-solver Step 4: Final forecast validation
    try:
        mu = forecast_mu(theta_canonical)
    except Exception as err:
        raise ForecastContractViolation(f"Post-fit forecast contract failed: {err}") from err

    return FittedModelResult(
        a=a,
        b=b,
        u=u,
        v=v,
        gamma=gamma,
        theta=theta_canonical,
        mu=mu,
        objective_value=float(opt_res.fun),
        iterations=int(getattr(opt_res, "nit", 0)),
    )
