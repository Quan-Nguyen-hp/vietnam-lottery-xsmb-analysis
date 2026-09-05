"""Exact forecast metrics, B0 baseline, and DEV/VAL gate composition for XPIS v3 M3 (Slice M3-07).

This module implements:
1. Daily Poisson deviance, MAE, and RMSE with exact zero-count Poisson cell semantics.
2. Stage and block metric aggregation with pooled root-mean-square error.
3. Uniform B0 baseline forecast creation.
4. Lexicographic DEV candidate winner selection across the frozen window universe.
5. Deterministic VAL primary, secondary, and complete three-part forecast gate evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .contracts import (
    CandidateID,
    FailureExitStatus,
    FailureStage,
    ModelID,
    ProtocolFailure,
)
from .model import FORECAST_SUM_TOLERANCE, ModelValidationError


COMPARISON_TOLERANCE: float = 0.0
B0_VALUE: float = 0.27
B0_MODEL_ID: ModelID = ModelID.B0
B0_CANDIDATE_ID: CandidateID = CandidateID.B0_UNIFORM
EXPECTED_FORECAST_SUM: float = 27.0
N_OUTCOMES: int = 100

CANDIDATE_WINDOWS: tuple[int, ...] = (30, 60, 120, 240, 365)
WINDOW_TO_CANDIDATE_ID: dict[int, CandidateID] = {
    30: CandidateID.M3_W030,
    60: CandidateID.M3_W060,
    120: CandidateID.M3_W120,
    240: CandidateID.M3_W240,
    365: CandidateID.M3_W365,
}
M3_CANDIDATE_IDS: tuple[CandidateID, ...] = (
    CandidateID.M3_W030,
    CandidateID.M3_W060,
    CandidateID.M3_W120,
    CandidateID.M3_W240,
    CandidateID.M3_W365,
)


class MetricEvaluationError(ModelValidationError):
    """Base error for all contract and numerical failures during metric evaluation."""

    stage: FailureStage = FailureStage.METRIC_EVALUATION
    exit_status: FailureExitStatus = FailureExitStatus.NEEDS_MODEL_REVISION

    def __init__(
        self,
        message: str,
        error_type: str = "MetricEvaluationError",
        exit_status: FailureExitStatus = FailureExitStatus.NEEDS_MODEL_REVISION,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.exit_status = exit_status

    def as_protocol_failure(self) -> ProtocolFailure:
        return ProtocolFailure(
            stage=self.stage,
            error_type=self.error_type,
            exit_status=self.exit_status,
        )


@dataclass(frozen=True)
class DailyForecastMetrics:
    """Immutable single-day forecast evaluation metrics."""

    poisson_deviance: float
    mae: float
    rmse: float


@dataclass(frozen=True)
class StageForecastMetrics:
    """Immutable multi-day stage/block forecast evaluation metrics."""

    poisson_deviance: float
    mae: float
    rmse: float
    date_count: int


BlockForecastMetrics = StageForecastMetrics


@dataclass(frozen=True)
class CandidateEvaluation:
    """Immutable evaluation record for an M3 candidate window."""

    window: int
    candidate_id: CandidateID
    dev_metrics: StageForecastMetrics


@dataclass(frozen=True)
class ForecastGateResult:
    """Immutable inspection record for the three-part VAL forecast qualification gate."""

    val_primary_pass: bool
    val_secondary_pass: bool
    forecast_bootstrap_pass: bool
    forecast_gate_pass: bool


def create_b0_forecast() -> np.ndarray:
    """Construct the frozen uniform baseline B0 forecast vector.

    mu_B0[n] = 0.27 for all n in {0..99}, sum = 27.0.
    """
    return np.full(N_OUTCOMES, B0_VALUE, dtype=np.float64)


def candidate_id_for_window(window: int) -> CandidateID:
    """Map integer history window W to its closed CandidateID enum."""
    if window not in WINDOW_TO_CANDIDATE_ID:
        raise MetricEvaluationError(f"Invalid M3 candidate window: {window}")
    return WINDOW_TO_CANDIDATE_ID[window]


def window_for_candidate_id(candidate_id: CandidateID | str) -> int:
    """Map candidate ID (enum or string) to its integer history window W."""
    try:
        cid = CandidateID(candidate_id) if isinstance(candidate_id, str) else candidate_id
    except ValueError as err:
        raise MetricEvaluationError(f"Invalid M3 candidate ID: {candidate_id}") from err

    for w, c in WINDOW_TO_CANDIDATE_ID.items():
        if c == cid:
            return w
    raise MetricEvaluationError(f"Candidate ID {candidate_id} is not an M3 window candidate")


def validate_forecast_inputs(
    y: np.ndarray | Sequence[int | float],
    mu: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate daily observed counts y and forecast mu against the frozen contract.

    Contract:
      - shape(y) = (100,), shape(mu) = (100,)
      - y[n] >= 0, finite
      - sum(y) == 27
      - mu[n] > 0, finite
      - abs(sum(mu) - 27.0) <= 1e-10
    """
    y_arr = np.asarray(y)
    mu_arr = np.asarray(mu, dtype=np.float64)

    if y_arr.shape != (N_OUTCOMES,):
        raise MetricEvaluationError(
            f"Observed counts y must have shape ({N_OUTCOMES},), got {y_arr.shape}",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if mu_arr.shape != (N_OUTCOMES,):
        raise MetricEvaluationError(
            f"Expected counts mu must have shape ({N_OUTCOMES},), got {mu_arr.shape}",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )

    if not np.all(np.isfinite(y_arr)):
        raise MetricEvaluationError(
            "Observed counts y must be finite",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if np.any(y_arr < 0):
        raise MetricEvaluationError(
            "Observed counts must be non-negative",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if abs(float(np.sum(y_arr)) - EXPECTED_FORECAST_SUM) > FORECAST_SUM_TOLERANCE:
        raise MetricEvaluationError(
            f"Observed count sum must be 27, got {float(np.sum(y_arr))}",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )

    if not np.all(np.isfinite(mu_arr)):
        raise MetricEvaluationError(
            "Expected counts mu must be finite",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
    if np.any(mu_arr <= 0.0):
        raise MetricEvaluationError(
            "Expected counts mu must be strictly positive",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
    if abs(float(np.sum(mu_arr)) - EXPECTED_FORECAST_SUM) > FORECAST_SUM_TOLERANCE:
        raise MetricEvaluationError(
            f"Expected count sum must equal 27, got {float(np.sum(mu_arr))}",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )

    return y_arr, mu_arr


def compute_cell_poisson_deviance(
    y: int | float | np.ndarray,
    mu: float | np.ndarray,
) -> float | np.ndarray:
    """Compute exact cell Poisson deviance without epsilon.

    Formula:
      if y > 0: 2 * (y * ln(y / mu) - (y - mu))
      if y == 0: 2 * mu
    """
    if isinstance(y, (int, float, np.integer, np.floating)) and isinstance(mu, (int, float, np.floating)):
        y_val = float(y)
        mu_val = float(mu)
        if mu_val <= 0.0 or not math.isfinite(mu_val):
            raise MetricEvaluationError("mu must be strictly positive and finite")
        if y_val == 0.0:
            return 2.0 * mu_val
        if y_val > 0.0:
            return 2.0 * (y_val * math.log(y_val / mu_val) - (y_val - mu_val))
        raise MetricEvaluationError("Observed counts must be non-negative")

    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    res = np.empty_like(mu_arr, dtype=np.float64)

    zero_mask = (y_arr == 0.0)
    res[zero_mask] = 2.0 * mu_arr[zero_mask]

    pos_mask = (y_arr > 0.0)
    y_pos = y_arr[pos_mask]
    mu_pos = mu_arr[pos_mask]
    res[pos_mask] = 2.0 * (y_pos * np.log(y_pos / mu_pos) - (y_pos - mu_pos))

    return res


def compute_daily_poisson_deviance(
    y: np.ndarray | Sequence[int | float],
    mu: np.ndarray | Sequence[float],
) -> float:
    """Compute unrounded daily mean Poisson deviance over 100 outcomes."""
    y_arr, mu_arr = validate_forecast_inputs(y, mu)
    cell_pd = compute_cell_poisson_deviance(y_arr, mu_arr)
    return float(np.mean(cell_pd))


def compute_daily_mae(
    y: np.ndarray | Sequence[int | float],
    mu: np.ndarray | Sequence[float],
) -> float:
    """Compute unrounded daily MAE over 100 outcomes."""
    y_arr, mu_arr = validate_forecast_inputs(y, mu)
    return float(np.mean(np.abs(y_arr - mu_arr)))


def compute_daily_rmse(
    y: np.ndarray | Sequence[int | float],
    mu: np.ndarray | Sequence[float],
) -> float:
    """Compute unrounded daily RMSE over 100 outcomes."""
    y_arr, mu_arr = validate_forecast_inputs(y, mu)
    return math.sqrt(float(np.mean((y_arr - mu_arr) ** 2)))


def compute_daily_metrics(
    y: np.ndarray | Sequence[int | float],
    mu: np.ndarray | Sequence[float],
) -> DailyForecastMetrics:
    """Compute and return all three unrounded daily forecast metrics."""
    y_arr, mu_arr = validate_forecast_inputs(y, mu)
    cell_pd = compute_cell_poisson_deviance(y_arr, mu_arr)
    pd_val = float(np.mean(cell_pd))
    diff = y_arr - mu_arr
    mae_val = float(np.mean(np.abs(diff)))
    rmse_val = math.sqrt(float(np.mean(diff ** 2)))
    return DailyForecastMetrics(
        poisson_deviance=pd_val,
        mae=mae_val,
        rmse=rmse_val,
    )


def compute_stage_metrics(
    daily_metrics: Sequence[DailyForecastMetrics],
) -> StageForecastMetrics:
    """Aggregate a sequence of daily metrics across a stage or stability block.

    Formulas:
      PD_stage = mean_t(PD_t)
      MAE_stage = mean_t(MAE_t)
      RMSE_stage = sqrt(mean_t(RMSE_t^2))   (pooled, not mean_t(RMSE_t))
    """
    if not daily_metrics:
        raise MetricEvaluationError("Empty daily metrics sequence cannot be aggregated")

    date_count = len(daily_metrics)
    pd_stage = float(np.mean([m.poisson_deviance for m in daily_metrics]))
    mae_stage = float(np.mean([m.mae for m in daily_metrics]))
    rmse_stage = math.sqrt(float(np.mean([m.rmse ** 2 for m in daily_metrics])))

    return StageForecastMetrics(
        poisson_deviance=pd_stage,
        mae=mae_stage,
        rmse=rmse_stage,
        date_count=date_count,
    )


def compute_block_metrics(
    daily_metrics: Sequence[DailyForecastMetrics],
) -> BlockForecastMetrics:
    """Aggregate daily metrics across a contiguous stability block using exact stage rules."""
    return compute_stage_metrics(daily_metrics)


def select_dev_winner(
    candidate_metrics: dict[int, StageForecastMetrics] | Sequence[CandidateEvaluation],
) -> CandidateEvaluation:
    """Select the winning M3 candidate window using DEV stage metrics only.

    Selection rule:
      Lexicographic ascending tuple: (PD_DEV, MAE_DEV, RMSE_DEV, W)
      Lower is better on every component. W is the deterministic tie-breaker.
      Exact unrounded binary64 comparison with zero tolerance.
      B0 is never eligible as the M3 winner.
    """
    if isinstance(candidate_metrics, dict):
        evaluations: list[CandidateEvaluation] = []
        for w in CANDIDATE_WINDOWS:
            if w not in candidate_metrics:
                raise MetricEvaluationError(
                    f"Candidate universe must contain exactly {CANDIDATE_WINDOWS}, missing window {w}"
                )
            evaluations.append(
                CandidateEvaluation(
                    window=w,
                    candidate_id=candidate_id_for_window(w),
                    dev_metrics=candidate_metrics[w],
                )
            )
    else:
        evaluations = list(candidate_metrics)
        windows = {e.window for e in evaluations}
        if windows != set(CANDIDATE_WINDOWS):
            raise MetricEvaluationError(
                f"Candidate universe must contain exactly windows {CANDIDATE_WINDOWS}, got {windows}"
            )
        for e in evaluations:
            if e.candidate_id == CandidateID.B0_UNIFORM:
                raise MetricEvaluationError("B0 baseline is not eligible as an M3 candidate winner")

    winner = min(
        evaluations,
        key=lambda e: (
            e.dev_metrics.poisson_deviance,
            e.dev_metrics.mae,
            e.dev_metrics.rmse,
            e.window,
        ),
    )
    return winner


def evaluate_val_primary(pd_val_m3: float, pd_val_b0: float) -> bool:
    """Primary forecast validation predicate: PD_VAL_M3 < PD_VAL_B0 (strict)."""
    return float(pd_val_m3) < float(pd_val_b0)


def evaluate_val_secondary(
    mae_val_m3: float,
    mae_val_b0: float,
    rmse_val_m3: float,
    rmse_val_b0: float,
) -> bool:
    """Secondary forecast validation predicate: MAE_M3 <= MAE_B0 or RMSE_M3 <= RMSE_B0."""
    return (float(mae_val_m3) <= float(mae_val_b0)) or (float(rmse_val_m3) <= float(rmse_val_b0))


def evaluate_forecast_bootstrap(forecast_bootstrap_lower_bound: float) -> bool:
    """Forecast bootstrap predicate: forecast_bootstrap_lower_bound > 0.0 (strict)."""
    return float(forecast_bootstrap_lower_bound) > 0.0


def evaluate_forecast_gate(
    pd_val_m3: float,
    pd_val_b0: float,
    mae_val_m3: float,
    mae_val_b0: float,
    rmse_val_m3: float,
    rmse_val_b0: float,
    forecast_bootstrap_lower_bound: float,
) -> ForecastGateResult:
    """Evaluate the complete three-part forecast qualification gate (U-013).

    Gate passes iff:
      1. PD_VAL_M3 < PD_VAL_B0 (strict)
      2. (MAE_VAL_M3 <= MAE_VAL_B0) or (RMSE_VAL_M3 <= RMSE_VAL_B0)
      3. forecast_bootstrap_lower_bound > 0.0 (strict)
    """
    primary = evaluate_val_primary(pd_val_m3, pd_val_b0)
    secondary = evaluate_val_secondary(mae_val_m3, mae_val_b0, rmse_val_m3, rmse_val_b0)
    bootstrap = evaluate_forecast_bootstrap(forecast_bootstrap_lower_bound)
    gate = primary and secondary and bootstrap
    return ForecastGateResult(
        val_primary_pass=primary,
        val_secondary_pass=secondary,
        forecast_bootstrap_pass=bootstrap,
        forecast_gate_pass=gate,
    )
