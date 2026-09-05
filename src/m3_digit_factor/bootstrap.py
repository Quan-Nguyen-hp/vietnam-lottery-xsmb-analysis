"""Exact stationary circular block bootstrap for XPIS v3 M3 (Slice M3-08).

This module implements:
1. Exact Politis & Romano (1994) stationary circular resampling with strictly
   controlled RNG consumption order and PCG64 stream continuity.
2. Forecast bootstrap lower bound on daily deviance differences (VAL stage).
3. Economic shared-index bootstrap lower bounds across portfolio sizes K in {1, 3, 5, 10}
   (STABILITY stage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import FailureExitStatus, FailureStage, ProtocolFailure
from .model import ModelValidationError


FORECAST_BOOTSTRAP_RNG_IMPLEMENTATION: str = "numpy.random.Generator"
FORECAST_BOOTSTRAP_BIT_GENERATOR: str = "PCG64"
FORECAST_BOOTSTRAP_SEED: int = 20260831
FORECAST_BOOTSTRAP_REPLICATIONS: int = 2000
FORECAST_BOOTSTRAP_MEAN_BLOCK_LENGTH: int = 30
FORECAST_BOOTSTRAP_RESTART_PROBABILITY: float = 1.0 / 30.0
FORECAST_BOOTSTRAP_QUANTILE: float = 0.05
FORECAST_BOOTSTRAP_QUANTILE_METHOD: str = "linear"

ECONOMIC_BOOTSTRAP_RNG_IMPLEMENTATION: str = "numpy.random.Generator"
ECONOMIC_BOOTSTRAP_BIT_GENERATOR: str = "PCG64"
ECONOMIC_BOOTSTRAP_SEED: int = 20260832
ECONOMIC_BOOTSTRAP_REPLICATIONS: int = 2000
ECONOMIC_BOOTSTRAP_MEAN_BLOCK_LENGTH: int = 30
ECONOMIC_BOOTSTRAP_RESTART_PROBABILITY: float = 1.0 / 30.0
ECONOMIC_BOOTSTRAP_QUANTILE: float = 0.0125
ECONOMIC_BOOTSTRAP_QUANTILE_METHOD: str = "linear"
ECONOMIC_BOOTSTRAP_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
SHARED_RESAMPLE_INDICES: bool = True


class BootstrapError(ModelValidationError):
    """Base error for all bootstrap failures."""

    def __init__(
        self,
        message: str,
        stage: FailureStage,
        error_type: str = "BootstrapError",
        exit_status: FailureExitStatus = FailureExitStatus.NEEDS_DATA_REVISION,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.error_type = error_type
        self.exit_status = exit_status

    def as_protocol_failure(self) -> ProtocolFailure:
        return ProtocolFailure(
            stage=self.stage,
            error_type=self.error_type,
            exit_status=self.exit_status,
        )


class ForecastBootstrapError(BootstrapError):
    """Raised on invalid input or failure during forecast bootstrap."""

    def __init__(
        self,
        message: str,
        error_type: str = "ForecastBootstrapError",
        exit_status: FailureExitStatus = FailureExitStatus.NEEDS_DATA_REVISION,
    ) -> None:
        super().__init__(
            message=message,
            stage=FailureStage.FORECAST_BOOTSTRAP,
            error_type=error_type,
            exit_status=exit_status,
        )


class EconomicBootstrapError(BootstrapError):
    """Raised on invalid input or failure during economic bootstrap."""

    def __init__(
        self,
        message: str,
        error_type: str = "EconomicBootstrapError",
        exit_status: FailureExitStatus = FailureExitStatus.NEEDS_DATA_REVISION,
    ) -> None:
        super().__init__(
            message=message,
            stage=FailureStage.ECONOMIC_BOOTSTRAP,
            error_type=error_type,
            exit_status=exit_status,
        )


@dataclass(frozen=True)
class ForecastBootstrapResult:
    """Immutable result of the forecast bootstrap qualification analysis."""

    bootstrap_lower_bound: float
    bootstrap_replications: int
    replicate_means: np.ndarray


@dataclass(frozen=True)
class EconomicBootstrapResult:
    """Immutable result of the economic bootstrap evaluation across K in {1, 3, 5, 10}."""

    lower_bounds: dict[int, float]
    bootstrap_replications: int
    replicate_means: np.ndarray

    @property
    def lower_bounds_tuple(self) -> tuple[float, float, float, float]:
        """Return the lower bounds in frozen K order (1, 3, 5, 10)."""
        return (
            self.lower_bounds[1],
            self.lower_bounds[3],
            self.lower_bounds[5],
            self.lower_bounds[10],
        )


def generate_stationary_circular_indices(
    n: int,
    replications: int,
    rng: np.random.Generator,
    p_restart: float = 1.0 / 30.0,
) -> np.ndarray:
    """Generate stationary circular block bootstrap resample indices.

    Consumption protocol:
      - For each replicate b:
        - index[0] = rng.integers(0, n)
        - For m = 1..n-1:
          - u = rng.random()
          - if u < p_restart:
              index[m] = rng.integers(0, n)
            else:
              index[m] = (index[m-1] + 1) % n

    Note:
      No rng.integers() call is consumed on a continuation step.
      Continuation across n-1 wraps to 0 circularly.
    """
    if n <= 0:
        raise ValueError(f"Sample length n must be strictly positive, got {n}")
    if replications <= 0:
        raise ValueError(f"Replications must be strictly positive, got {replications}")

    indices = np.empty((replications, n), dtype=np.int64)

    for b in range(replications):
        indices[b, 0] = int(rng.integers(0, n))
        for m in range(1, n):
            u = float(rng.random())
            if u < p_restart:
                indices[b, m] = int(rng.integers(0, n))
            else:
                indices[b, m] = (indices[b, m - 1] + 1) % n

    return indices


def run_forecast_bootstrap(
    d: np.ndarray | Sequence[float],
) -> ForecastBootstrapResult:
    """Run the exact stationary circular bootstrap on daily deviance differences.

    d[t] = PD_B0[t] - PD_M3[t]
    Uses PCG64(seed=20260831), 2000 replications, p_restart=1/30, quantile=0.05 linear.
    """
    d_arr = np.asarray(d, dtype=np.float64)

    if d_arr.ndim != 1:
        raise ForecastBootstrapError(
            f"Input deviance differences must be 1-dimensional, got shape {d_arr.shape}",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if d_arr.size == 0:
        raise ForecastBootstrapError(
            "Input deviance differences vector cannot be empty",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if not np.all(np.isfinite(d_arr)):
        raise ForecastBootstrapError(
            "Input deviance differences vector must contain finite values",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )

    n = len(d_arr)
    rng = np.random.Generator(np.random.PCG64(FORECAST_BOOTSTRAP_SEED))

    indices = generate_stationary_circular_indices(
        n=n,
        replications=FORECAST_BOOTSTRAP_REPLICATIONS,
        rng=rng,
        p_restart=FORECAST_BOOTSTRAP_RESTART_PROBABILITY,
    )

    replicate_means = np.mean(d_arr[indices], axis=1)

    lower_bound = float(
        np.quantile(
            replicate_means,
            FORECAST_BOOTSTRAP_QUANTILE,
            method=FORECAST_BOOTSTRAP_QUANTILE_METHOD,
        )
    )

    return ForecastBootstrapResult(
        bootstrap_lower_bound=lower_bound,
        bootstrap_replications=FORECAST_BOOTSTRAP_REPLICATIONS,
        replicate_means=replicate_means,
    )


def run_economic_bootstrap(
    economic_delta: np.ndarray | Sequence[Sequence[float]],
) -> EconomicBootstrapResult:
    """Run the exact stationary circular bootstrap on economic deltas across K in {1, 3, 5, 10}.

    Input shape: (N, 4) in order K=1, 3, 5, 10.
    Uses PCG64(seed=20260832), 2000 replications, p_restart=1/30, quantile=0.0125 linear.
    Shared resample indices: ONE index vector is generated per replicate and applied to all 4 K columns.
    """
    delta_arr = np.asarray(economic_delta, dtype=np.float64)

    if delta_arr.ndim != 2 or delta_arr.shape[1] != len(ECONOMIC_BOOTSTRAP_K_VALUES):
        raise EconomicBootstrapError(
            f"Economic delta matrix must have shape (N, {len(ECONOMIC_BOOTSTRAP_K_VALUES)}), got {delta_arr.shape}",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if delta_arr.shape[0] == 0:
        raise EconomicBootstrapError(
            "Economic delta matrix cannot be empty",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if not np.all(np.isfinite(delta_arr)):
        raise EconomicBootstrapError(
            "Economic delta matrix must contain finite values",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )

    n = delta_arr.shape[0]
    rng = np.random.Generator(np.random.PCG64(ECONOMIC_BOOTSTRAP_SEED))

    indices = generate_stationary_circular_indices(
        n=n,
        replications=ECONOMIC_BOOTSTRAP_REPLICATIONS,
        rng=rng,
        p_restart=ECONOMIC_BOOTSTRAP_RESTART_PROBABILITY,
    )

    replicate_means = np.empty((ECONOMIC_BOOTSTRAP_REPLICATIONS, 4), dtype=np.float64)
    lower_bounds: dict[int, float] = {}

    for k_idx, k_val in enumerate(ECONOMIC_BOOTSTRAP_K_VALUES):
        # Apply shared resample indices to column k_idx
        col_means = np.mean(delta_arr[:, k_idx][indices], axis=1)
        replicate_means[:, k_idx] = col_means
        lower_bounds[k_val] = float(
            np.quantile(
                col_means,
                ECONOMIC_BOOTSTRAP_QUANTILE,
                method=ECONOMIC_BOOTSTRAP_QUANTILE_METHOD,
            )
        )

    return EconomicBootstrapResult(
        lower_bounds=lower_bounds,
        bootstrap_replications=ECONOMIC_BOOTSTRAP_REPLICATIONS,
        replicate_means=replicate_means,
    )
