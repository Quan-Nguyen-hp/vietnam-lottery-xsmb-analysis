"""Deterministic U-004 model initialization for XPIS v3 M3 (Slice M3-05).

This module implements deterministic SVD-based initialization from historical
empirical counts under the frozen XPIS v3 M3 canonical specification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .contracts import FailureExitStatus, FailureStage, ProtocolFailure
from .model import (
    N_HEADS,
    N_OUTCOMES,
    N_TAILS,
    ModelValidationError,
    pack_parameters,
)


SVD_ZERO_TOLERANCE = 1e-10
SVD_LEADING_GAP_THRESHOLD = 1e-8
CENTERED_VECTOR_NORM_TOLERANCE = 1e-12

# Exact frozen canonical zero-interaction vector: (9, -1, ..., -1) / sqrt(90)
_CANONICAL_ZERO_VECTOR = (
    np.array([9.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64) / math.sqrt(90.0)
)


class ModelInitializationError(ModelValidationError):
    """Base error for all failures during model initialization."""

    stage: FailureStage = FailureStage.MODEL_INITIALIZATION
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


class ZeroDigitMarginal(ModelInitializationError):
    """Raised when any head or tail digit marginal count/probability is non-positive."""

    def __init__(self, message: str = "Zero or non-positive digit marginal encountered") -> None:
        super().__init__(message, error_type="ZeroDigitMarginal")


class SVDLeadingSubspaceAmbiguity(ModelInitializationError):
    """Raised when relative gap between leading singular values is <= 1e-8."""

    def __init__(self, message: str = "SVD leading subspace relative gap <= 1e-8") -> None:
        super().__init__(message, error_type="SVDLeadingSubspaceAmbiguity")


class CenteredSingularVectorDegeneracy(ModelInitializationError):
    """Raised when norm of centered leading singular vector is <= 1e-12."""

    def __init__(self, message: str = "Centered singular vector norm <= 1e-12") -> None:
        super().__init__(message, error_type="CenteredSingularVectorDegeneracy")


@dataclass(frozen=True)
class ModelInitializationResult:
    """Immutable representation of deterministic model initialization."""

    a: np.ndarray
    b: np.ndarray
    u: np.ndarray
    v: np.ndarray
    gamma: float
    theta: np.ndarray

    def __post_init__(self) -> None:
        for arr in (self.a, self.b, self.u, self.v, self.theta):
            arr.flags.writeable = False


def compute_empirical_distribution(
    counts: Sequence[float] | np.ndarray,
    W: int | None = None,
) -> np.ndarray:
    """Compute empirical 10x10 joint distribution q from historical draw counts.

    Formula:
        q[i, j] = X[10 * i + j] / (27 * W)
    """
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

    total_count = float(np.sum(counts_2d))

    if W is not None:
        if isinstance(W, bool) or not isinstance(W, (int, np.integer)):
            raise ModelValidationError(f"Window W must be an integer, got {type(W).__name__}")
        if W <= 0:
            raise ModelValidationError(f"Window W must be positive, got {W}")
        expected_total = 27.0 * W
        if not math.isclose(total_count, expected_total, abs_tol=1e-6):
            raise ModelValidationError(
                f"Sum of counts {total_count} does not match expected 27 * W = {expected_total}"
            )
        denominator = expected_total
    else:
        if total_count <= 0.0:
            raise ModelValidationError("Total count must be strictly positive")
        denominator = total_count

    q = counts_2d / denominator
    return q


def compute_digit_marginals(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute head digit marginals r and tail digit marginals c.

    Formula:
        r[i] = sum_j q[i, j]
        c[j] = sum_i q[i, j]

    Enforces strict positivity (no zero marginals, no epsilon repair).
    """
    r = np.sum(q, axis=1)
    c = np.sum(q, axis=0)

    if np.any(r <= 0.0) or np.any(c <= 0.0):
        raise ZeroDigitMarginal(
            f"Encountered non-positive digit marginals (min_r={np.min(r)}, min_c={np.min(c)})"
        )
    return r, c


def compute_residual_matrix(q: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Construct double-centered residual interaction matrix R.

    Formula:
        R[i, j] = q[i, j] - r[i] * c[j]
    """
    R = q - r[:, None] * c[None, :]
    return R


def initialize_m3_from_counts(
    counts: Sequence[float] | np.ndarray,
    W: int | None = None,
) -> ModelInitializationResult:
    """Execute deterministic U-004 model initialization from historical count totals.

    Steps:
        1. Compute empirical joint distribution q = X / (27 * W)
        2. Compute digit marginals r[i] = sum_j q[i, j], c[j] = sum_i q[i, j]
        3. Check r[i] > 0 and c[j] > 0; fail with ZeroDigitMarginal if not
        4. Compute initial main effects:
           a0[i] = ln(r[i]) - mean(ln(r))
           b0[j] = ln(c[j]) - mean(ln(c))
        5. Compute residual matrix R = q - r * c
        6. Compute SVD: U, s, Vt = numpy.linalg.svd(R, full_matrices=False)
        7. If s[0] <= 1e-10:
           gamma = 0.0
           u = v = (9, -1, ..., -1) / sqrt(90)
        8. Else:
           Require (s[0] - s[1]) / s[0] > 1e-8; fail with SVDLeadingSubspaceAmbiguity if not
           Center raw leading vectors: u_c = U[:,0] - mean, v_c = Vt[0,:] - mean
           Require ||u_c|| > 1e-12 and ||v_c|| > 1e-12; fail with CenteredSingularVectorDegeneracy if not
           Normalize: u = u_c / ||u_c||, v = v_c / ||v_c||
           Rescale: gamma = s[0] * ||u_c|| * ||v_c||
           Canonical sign: let k = min { i : |u[i]| = max |u| }. If u[k] < 0, negate u and v
        9. Pack into theta using authoritative 41-parameter order
    """
    q = compute_empirical_distribution(counts, W=W)
    r, c = compute_digit_marginals(q)

    # Initial main effects: zero-sum deviation of log marginals
    ln_r = np.log(r)
    ln_c = np.log(c)
    a0 = np.array(ln_r - np.mean(ln_r), dtype=np.float64, copy=True)
    b0 = np.array(ln_c - np.mean(ln_c), dtype=np.float64, copy=True)

    R = compute_residual_matrix(q, r, c)

    # Exact SVD: numpy.linalg.svd with full_matrices=False
    U, s, Vt = np.linalg.svd(R, full_matrices=False)

    s0 = float(s[0])

    if s0 <= SVD_ZERO_TOLERANCE:
        # Zero-interaction canonical branch
        gamma = 0.0
        u = _CANONICAL_ZERO_VECTOR.copy()
        v = _CANONICAL_ZERO_VECTOR.copy()
    else:
        # Nonzero interaction branch
        s1 = float(s[1])
        relative_gap = (s0 - s1) / s0
        if relative_gap <= SVD_LEADING_GAP_THRESHOLD:
            raise SVDLeadingSubspaceAmbiguity(
                f"Relative singular value gap {relative_gap:.3e} <= {SVD_LEADING_GAP_THRESHOLD:.1e}"
            )

        u_raw = U[:, 0]
        v_raw = Vt[0, :]

        u_c = u_raw - np.mean(u_raw)
        v_c = v_raw - np.mean(v_raw)

        nu = float(np.linalg.norm(u_c))
        nv = float(np.linalg.norm(v_c))

        if nu <= CENTERED_VECTOR_NORM_TOLERANCE or nv <= CENTERED_VECTOR_NORM_TOLERANCE:
            raise CenteredSingularVectorDegeneracy(
                f"Centered singular vector norm degenerate (nu={nu:.3e}, nv={nv:.3e} <= {CENTERED_VECTOR_NORM_TOLERANCE:.1e})"
            )

        u = np.array(u_c / nu, dtype=np.float64, copy=True)
        v = np.array(v_c / nv, dtype=np.float64, copy=True)
        gamma = float(s0 * nu * nv)

        # Canonical sign rule: find smallest index attaining max |u|
        abs_u = np.abs(u)
        max_abs_u = np.max(abs_u)
        k = int(np.where(abs_u == max_abs_u)[0][0])
        if u[k] < 0.0:
            u = -u
            v = -v

    theta = pack_parameters(a0, b0, u, v, gamma)

    return ModelInitializationResult(
        a=a0,
        b=b0,
        u=u,
        v=v,
        gamma=gamma,
        theta=theta,
    )
