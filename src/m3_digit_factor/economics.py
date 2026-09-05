"""Exact economic evaluation protocol, Top-K selection, and recommendation for XPIS v3 M3 (Slice M3-09).

This module implements:
1. Exact descending Top-K outcome selection with smaller-outcome tie breaking.
2. Multiplicity-preserving daily economic PnL calculation (in thousand VND).
3. Exact full-stability mean aggregation and 6-block positive count determination.
4. Single-call integration with the reviewed M3-08 economic bootstrap.
5. Per-K qualification across K in {1, 3, 5, 10}, qualified_top_k construction,
   aggregate economic signal adjudication, and single-K recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bootstrap import run_economic_bootstrap
from .contracts import FailureExitStatus, FailureStage, ProtocolFailure
from .model import FORECAST_SUM_TOLERANCE, ModelValidationError


ECONOMIC_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
COST_PER_SELECTED_NUMBER: float = 27.0
PAYOUT_PER_HIT_OCCURRENCE: float = 99.0
STABILITY_BLOCK_COUNT: int = 6
MINIMUM_POSITIVE_BLOCK_COUNT: int = 5
N_OUTCOMES: int = 100
EXPECTED_FORECAST_SUM: float = 27.0


class EconomicEvaluationError(ModelValidationError):
    """Base error for all contract and numerical failures during economic evaluation."""

    stage: FailureStage = FailureStage.ECONOMIC_EVALUATION
    exit_status: FailureExitStatus = FailureExitStatus.NEEDS_DATA_REVISION

    def __init__(
        self,
        message: str,
        error_type: str = "EconomicEvaluationError",
        exit_status: FailureExitStatus = FailureExitStatus.NEEDS_DATA_REVISION,
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
class PerKEconomicEvaluation:
    """Immutable evaluation record for a single portfolio size K."""

    k: int
    mean_economic_delta: float
    block_mean_economic_delta: tuple[float, ...]
    positive_block_count: int
    bootstrap_lower_bound: float
    qualifies: bool


@dataclass(frozen=True)
class EconomicEvaluationResult:
    """Immutable inspection record of the complete STABILITY economic evaluation."""

    per_k: dict[int, PerKEconomicEvaluation]
    economic_signal: bool
    qualified_top_k: list[int]
    recommended_top_k: list[int]
    bootstrap_executed: bool
    economic_delta: np.ndarray | None


def compute_outcome_pnl(hits: int) -> float:
    """Compute PnL for one selected outcome in thousand VND.

    PnL = 99 * hits - 27.
    """
    return PAYOUT_PER_HIT_OCCURRENCE * float(hits) - COST_PER_SELECTED_NUMBER


def rank_outcomes_for_date(mu_t: np.ndarray | Sequence[float]) -> np.ndarray:
    """Rank outcomes n in {0..99} by descending mu_t, breaking ties with smaller n.

    Equivalent to ascending sort on (-mu_t[n], n).
    """
    mu_arr = np.asarray(mu_t, dtype=np.float64)
    if mu_arr.shape != (N_OUTCOMES,):
        raise EconomicEvaluationError(
            f"mu_t must have shape ({N_OUTCOMES},), got {mu_arr.shape}",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
    # lexsort accepts (secondary_key, primary_key)
    return np.lexsort((np.arange(N_OUTCOMES, dtype=np.int64), -mu_arr))


def compute_daily_economic_delta(
    y_t: np.ndarray | Sequence[int | float],
    mu_t: np.ndarray | Sequence[float],
    k: int,
) -> float:
    """Compute actual daily net PnL (in thousand VND) for portfolio size K.

    economic_delta[t, K] = sum over selected K numbers of (99 * y[t, n] - 27)
                         = 99 * sum(y_selected) - 27 * K
    """
    if k not in ECONOMIC_K_VALUES:
        raise EconomicEvaluationError(f"k must be in {ECONOMIC_K_VALUES}, got {k}")

    y_arr = np.asarray(y_t)
    ranked = rank_outcomes_for_date(mu_t)
    selected = ranked[:k]
    hits = float(np.sum(y_arr[selected]))
    return PAYOUT_PER_HIT_OCCURRENCE * hits - COST_PER_SELECTED_NUMBER * float(k)


def validate_stability_arrays(
    y: np.ndarray | Sequence[Sequence[int | float]],
    mu: np.ndarray | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate STABILITY outcome and forecast matrices against the authoritative contract."""
    y_arr = np.asarray(y)
    mu_arr = np.asarray(mu, dtype=np.float64)

    if y_arr.ndim != 2 or y_arr.shape[1] != N_OUTCOMES:
        raise EconomicEvaluationError(
            f"Observed counts y must have shape (N, {N_OUTCOMES}), got {y_arr.shape}",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if mu_arr.ndim != 2 or mu_arr.shape[1] != N_OUTCOMES:
        raise EconomicEvaluationError(
            f"Expected counts mu must have shape (N, {N_OUTCOMES}), got {mu_arr.shape}",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
    if y_arr.shape[0] == 0:
        raise EconomicEvaluationError(
            "STABILITY cohorts cannot be empty",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if y_arr.shape[0] != mu_arr.shape[0]:
        raise EconomicEvaluationError(
            f"Row counts of y and mu must match, got {y_arr.shape[0]} and {mu_arr.shape[0]}",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )

    if not np.all(np.isfinite(y_arr)):
        raise EconomicEvaluationError(
            "Observed counts y must be finite",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    if np.any(y_arr < 0):
        raise EconomicEvaluationError(
            "Observed counts y must be non-negative",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    row_sums_y = np.sum(y_arr, axis=1)
    if np.any(np.abs(row_sums_y - EXPECTED_FORECAST_SUM) > FORECAST_SUM_TOLERANCE):
        raise EconomicEvaluationError(
            "Observed count row sum must be 27 for all target dates",
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )

    if not np.all(np.isfinite(mu_arr)):
        raise EconomicEvaluationError(
            "Expected counts mu must be finite",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
    if np.any(mu_arr <= 0.0):
        raise EconomicEvaluationError(
            "Expected counts mu must be strictly positive",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
    row_sums_mu = np.sum(mu_arr, axis=1)
    if np.any(np.abs(row_sums_mu - EXPECTED_FORECAST_SUM) > FORECAST_SUM_TOLERANCE):
        raise EconomicEvaluationError(
            "Expected count row sum must equal 27 for all target dates",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )

    return y_arr, mu_arr


def compute_economic_delta_matrix(
    y: np.ndarray | Sequence[Sequence[int | float]],
    mu: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute the full daily economic delta matrix of shape (N_stability, 4).

    Columns correspond to K in (1, 3, 5, 10).
    """
    y_arr, mu_arr = validate_stability_arrays(y, mu)
    n_days = y_arr.shape[0]
    delta_matrix = np.empty((n_days, len(ECONOMIC_K_VALUES)), dtype=np.float64)

    for t in range(n_days):
        ranked = rank_outcomes_for_date(mu_arr[t])
        for k_idx, k in enumerate(ECONOMIC_K_VALUES):
            selected = ranked[:k]
            hits = float(np.sum(y_arr[t, selected]))
            delta_matrix[t, k_idx] = PAYOUT_PER_HIT_OCCURRENCE * hits - COST_PER_SELECTED_NUMBER * float(k)

    return delta_matrix


def compute_full_stability_mean(daily_delta: np.ndarray) -> float:
    """Compute the arithmetic mean across all stability target dates (ILR-01).

    STRICTLY FORBIDDEN: unweighted mean of block means.
    """
    return float(np.mean(daily_delta))


def compute_stability_block_means(
    economic_delta: np.ndarray,
    stability_blocks: Sequence[Sequence[int]],
) -> np.ndarray:
    """Compute block means for each of the 6 stability blocks across all columns.

    Returns array of shape (6, n_columns).
    """
    if len(stability_blocks) != STABILITY_BLOCK_COUNT:
        raise EconomicEvaluationError(
            f"Stability blocks must contain exactly {STABILITY_BLOCK_COUNT} blocks, got {len(stability_blocks)}"
        )

    n_cols = economic_delta.shape[1]
    block_means = np.empty((STABILITY_BLOCK_COUNT, n_cols), dtype=np.float64)

    for b, block in enumerate(stability_blocks):
        block_indices = list(block)
        if len(block_indices) == 0:
            raise EconomicEvaluationError(f"Stability block {b} cannot be empty")
        block_means[b] = np.mean(economic_delta[block_indices], axis=0)

    return block_means


def select_recommended_k(
    per_k: dict[int, PerKEconomicEvaluation],
    qualified_top_k: list[int],
) -> list[int]:
    """Select the single recommended K from qualified_top_k.

    Ranking rule:
      argmax_{K in qualified_top_k} (bootstrap_lower_bound[K], mean_economic_delta[K], -K)

    Invariants:
      recommended_top_k is a subset of qualified_top_k.
      len(recommended_top_k) <= 1.
      If qualified_top_k is empty, returns [].
    """
    if not qualified_top_k:
        return []

    chosen_k = max(
        qualified_top_k,
        key=lambda k: (
            per_k[k].bootstrap_lower_bound,
            per_k[k].mean_economic_delta,
            -k,
        ),
    )
    return [chosen_k]


def evaluate_economic_protocol(
    y_stability: np.ndarray | Sequence[Sequence[int | float]],
    mu_stability: np.ndarray | Sequence[Sequence[float]],
    stability_blocks: Sequence[Sequence[int]],
    forecast_signal: bool = True,
) -> EconomicEvaluationResult:
    """Execute the complete STABILITY economic evaluation.

    If forecast_signal is False:
      Short-circuits immediately.
      No ranking or bootstrap executed.
      economic_signal = False, qualified_top_k = [], recommended_top_k = [].
      per_k = {}.

    If forecast_signal is True:
      - Computes (N, 4) daily economic delta matrix.
      - Invokes run_economic_bootstrap exactly ONCE on the (N, 4) matrix.
      - Aggregates full-stability means and 6-block means.
      - Evaluates per-K qualification: mean > 0, positive_blocks >= 5, bootstrap_lower > 0.
      - Constructs qualified_top_k in [1, 3, 5, 10] order.
      - Evaluates economic_signal = (len(qualified_top_k) > 0).
      - Selects recommended_top_k.
    """
    if not forecast_signal:
        return EconomicEvaluationResult(
            per_k={},
            economic_signal=False,
            qualified_top_k=[],
            recommended_top_k=[],
            bootstrap_executed=False,
            economic_delta=None,
        )

    delta_matrix = compute_economic_delta_matrix(y_stability, mu_stability)

    # Invoke economic bootstrap ONCE on the full (N, 4) matrix
    bs_result = run_economic_bootstrap(delta_matrix)

    # Compute 6-block means
    block_means = compute_stability_block_means(delta_matrix, stability_blocks)

    per_k: dict[int, PerKEconomicEvaluation] = {}

    for k_idx, k in enumerate(ECONOMIC_K_VALUES):
        full_mean = compute_full_stability_mean(delta_matrix[:, k_idx])
        b_means = tuple(float(x) for x in block_means[:, k_idx])
        pos_blocks = int(np.sum(block_means[:, k_idx] > 0.0))
        bs_lb = bs_result.lower_bounds[k]

        qualifies = bool(
            (full_mean > 0.0)
            and (pos_blocks >= MINIMUM_POSITIVE_BLOCK_COUNT)
            and (bs_lb > 0.0)
        )

        per_k[k] = PerKEconomicEvaluation(
            k=k,
            mean_economic_delta=full_mean,
            block_mean_economic_delta=b_means,
            positive_block_count=pos_blocks,
            bootstrap_lower_bound=bs_lb,
            qualifies=qualifies,
        )

    qualified_top_k = [k for k in ECONOMIC_K_VALUES if per_k[k].qualifies]
    economic_signal = bool(len(qualified_top_k) > 0)
    recommended_top_k = select_recommended_k(per_k, qualified_top_k)

    return EconomicEvaluationResult(
        per_k=per_k,
        economic_signal=economic_signal,
        qualified_top_k=qualified_top_k,
        recommended_top_k=recommended_top_k,
        bootstrap_executed=True,
        economic_delta=delta_matrix,
    )
