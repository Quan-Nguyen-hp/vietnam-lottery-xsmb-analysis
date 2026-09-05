"""XPIS v3 M3 rank-1 digit-interaction model representation (Slice M3-04).

This module implements the exact rank-1 digit interaction multinomial loglinear model,
41-parameter packing/unpacking, numerically stabilized global softmax, and
forecast construction under the frozen XPIS v3 M3 specification.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


THETA_LENGTH = 41
FORECAST_SUM_TOLERANCE = 1e-10
MODEL_FAMILY = "RANK1_DIGIT_INTERACTION_MULTINOMIAL_LOGLINEAR"
N_HEADS = 10
N_TAILS = 10
N_OUTCOMES = 100
EXPECTED_FORECAST_SUM = 27.0


class ModelValidationError(ValueError):
    """Raised when model input parameters or coordinates fail validation."""


class ForecastContractViolation(ModelValidationError):
    """Raised when a generated forecast violates the authoritative M3 contract."""


def outcome_to_digits(outcome: int) -> tuple[int, int]:
    """Map outcome index n in {0..99} to head digit i and tail digit j.

    Defined as:
        i = floor(n / 10)
        j = n mod 10
    such that n = 10 * i + j, with i, j in {0..9}.
    """
    if isinstance(outcome, bool) or not isinstance(outcome, (int, np.integer)):
        raise ModelValidationError(f"Outcome must be an integer, got {type(outcome).__name__}")
    n = int(outcome)
    if n < 0 or n > 99:
        raise ModelValidationError(f"Outcome must be in range [0, 99], got {n}")
    return n // 10, n % 10


def digits_to_outcome(head: int, tail: int) -> int:
    """Map head digit i in {0..9} and tail digit j in {0..9} to outcome index n.

    Defined as:
        n = 10 * i + j
    """
    if isinstance(head, bool) or not isinstance(head, (int, np.integer)):
        raise ModelValidationError(f"Head digit must be an integer, got {type(head).__name__}")
    if isinstance(tail, bool) or not isinstance(tail, (int, np.integer)):
        raise ModelValidationError(f"Tail digit must be an integer, got {type(tail).__name__}")
    i = int(head)
    j = int(tail)
    if i < 0 or i > 9:
        raise ModelValidationError(f"Head digit must be in range [0, 9], got {i}")
    if j < 0 or j > 9:
        raise ModelValidationError(f"Tail digit must be in range [0, 9], got {j}")
    return 10 * i + j


def _validate_vector_component(val: object, name: str, length: int) -> np.ndarray:
    """Validate and convert a parameter component into a 1D float64 array."""
    if isinstance(val, (str, bytes)):
        raise ModelValidationError(f"Component {name} cannot be string or bytes")
    try:
        arr = np.asarray(val, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise ModelValidationError(f"Component {name} could not be converted to float64 array") from err

    if arr.ndim != 1 or arr.shape[0] != length:
        raise ModelValidationError(
            f"Component {name} must be 1D array of length {length}, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ModelValidationError(f"Component {name} contains non-finite values (NaN or Inf)")
    return arr


def pack_parameters(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    u: Sequence[float] | np.ndarray,
    v: Sequence[float] | np.ndarray,
    gamma: float | int | np.ndarray,
) -> np.ndarray:
    """Pack parameter components into the exact 41-parameter vector theta.

    Parameter vector layout:
        theta = [
            a[0], ..., a[9],
            b[0], ..., b[9],
            u[0], ..., u[9],
            v[0], ..., v[9],
            gamma
        ]
    """
    arr_a = _validate_vector_component(a, "a", N_HEADS)
    arr_b = _validate_vector_component(b, "b", N_TAILS)
    arr_u = _validate_vector_component(u, "u", N_HEADS)
    arr_v = _validate_vector_component(v, "v", N_TAILS)

    if isinstance(gamma, (str, bytes)):
        raise ModelValidationError("gamma cannot be string or bytes")
    try:
        arr_gamma = np.asarray(gamma, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise ModelValidationError("gamma could not be converted to float64") from err

    if arr_gamma.ndim != 0 and (arr_gamma.ndim != 1 or arr_gamma.shape[0] != 1):
        raise ModelValidationError(f"gamma must be a scalar, got shape {arr_gamma.shape}")

    val_gamma = float(arr_gamma.item())
    if not math.isfinite(val_gamma):
        raise ModelValidationError(f"gamma must be finite, got {val_gamma}")

    theta = np.empty(THETA_LENGTH, dtype=np.float64)
    theta[0:10] = arr_a
    theta[10:20] = arr_b
    theta[20:30] = arr_u
    theta[30:40] = arr_v
    theta[40] = val_gamma
    return theta


def unpack_parameters(
    theta: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Unpack 41-parameter vector theta into (a, b, u, v, gamma).

    Returns defensive copies of vectors to ensure no unexpected write-through mutation.
    """
    if isinstance(theta, (str, bytes)):
        raise ModelValidationError("theta cannot be string or bytes")
    try:
        arr_theta = np.asarray(theta, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise ModelValidationError("theta could not be converted to float64 array") from err

    if arr_theta.ndim != 1 or arr_theta.shape[0] != THETA_LENGTH:
        raise ModelValidationError(
            f"theta must be 1D array of length {THETA_LENGTH}, got shape {arr_theta.shape}"
        )
    if not np.all(np.isfinite(arr_theta)):
        raise ModelValidationError("theta contains non-finite values (NaN or Inf)")

    a = np.array(arr_theta[0:10], dtype=np.float64, copy=True)
    b = np.array(arr_theta[10:20], dtype=np.float64, copy=True)
    u = np.array(arr_theta[20:30], dtype=np.float64, copy=True)
    v = np.array(arr_theta[30:40], dtype=np.float64, copy=True)
    gamma = float(arr_theta[40])
    return a, b, u, v, gamma


def compute_eta(theta: Sequence[float] | np.ndarray) -> np.ndarray:
    """Compute 10x10 latent loglinear score matrix eta from parameter vector theta.

    Formula:
        eta[i, j] = a[i] + b[j] + gamma * u[i] * v[j]
    for all i, j in {0..9}.
    """
    a, b, u, v, gamma = unpack_parameters(theta)
    # Vectorized computation:
    # a[:, None] has shape (10, 1)
    # b[None, :] has shape (1, 10)
    # u[:, None] * v[None, :] has shape (10, 10)
    eta = a[:, None] + b[None, :] + gamma * (u[:, None] * v[None, :])
    return eta


def compute_probabilities(theta_or_eta: Sequence[float] | np.ndarray) -> np.ndarray:
    """Compute 10x10 probability matrix p via numerically stabilized global softmax.

    Accepts either theta of shape (41,) or precomputed eta of shape (10, 10).

    Formula:
        max_eta = max over all 100 cells eta[r, s]
        p[i, j] = exp(eta[i, j] - max_eta) / sum_{r, s} exp(eta[r, s] - max_eta)
    """
    if isinstance(theta_or_eta, (str, bytes)):
        raise ModelValidationError("Input cannot be string or bytes")
    arr = np.asarray(theta_or_eta, dtype=np.float64)

    if arr.ndim == 1 and arr.shape[0] == THETA_LENGTH:
        eta = compute_eta(arr)
    elif arr.ndim == 2 and arr.shape == (N_HEADS, N_TAILS):
        if not np.all(np.isfinite(arr)):
            raise ModelValidationError("eta contains non-finite values (NaN or Inf)")
        eta = arr
    else:
        raise ModelValidationError(
            f"Input must be theta shape ({THETA_LENGTH},) or eta shape ({N_HEADS}, {N_TAILS}), got {arr.shape}"
        )

    max_eta = float(np.max(eta))
    shifted_eta = eta - max_eta
    exp_eta = np.exp(shifted_eta)
    sum_exp = float(np.sum(exp_eta))

    p = exp_eta / sum_exp
    return p


def forecast_mu(theta: Sequence[float] | np.ndarray) -> np.ndarray:
    """Construct expected outcome count forecast mu of length 100 from parameter vector theta.

    Formula:
        p = compute_probabilities(theta)  # shape (10, 10)
        mu[10 * i + j] = 27.0 * p[i, j]

    Validates:
        - mu.shape == (100,)
        - all mu finite
        - all mu > 0.0 strictly
        - abs(sum(mu) - 27.0) <= 1e-10
    """
    p = compute_probabilities(theta)
    # Flatten in exact row-major order: n = 10 * i + j
    p_flat = p.ravel()
    mu = EXPECTED_FORECAST_SUM * p_flat

    if not np.all(np.isfinite(mu)):
        raise ForecastContractViolation("Non-finite values encountered in mu forecast")

    if not np.all(mu > 0.0):
        raise ForecastContractViolation(
            "Numerical underflow or non-positive value encountered in mu forecast"
        )

    sum_mu = float(np.sum(mu))
    if abs(sum_mu - EXPECTED_FORECAST_SUM) > FORECAST_SUM_TOLERANCE:
        raise ForecastContractViolation(
            f"Forecast sum drift exceeds tolerance: abs({sum_mu} - 27.0) > {FORECAST_SUM_TOLERANCE}"
        )

    return mu
