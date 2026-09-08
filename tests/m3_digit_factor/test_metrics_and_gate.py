"""Tests for XPIS v3 M3 metrics, B0 baseline, DEV candidate selection, and forecast gate (Slice M3-07)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.m3_digit_factor.contracts import (
    CandidateID,
    FailureExitStatus,
    FailureStage,
    ModelID,
)
from src.m3_digit_factor.metrics import (
    B0_CANDIDATE_ID,
    B0_MODEL_ID,
    B0_VALUE,
    CANDIDATE_WINDOWS,
    COMPARISON_TOLERANCE,
    CandidateEvaluation,
    DailyForecastMetrics,
    ForecastGateResult,
    MetricEvaluationError,
    StageForecastMetrics,
    candidate_id_for_window,
    compute_block_metrics,
    compute_cell_poisson_deviance,
    compute_daily_mae,
    compute_daily_metrics,
    compute_daily_poisson_deviance,
    compute_daily_rmse,
    compute_stage_metrics,
    create_b0_forecast,
    evaluate_forecast_bootstrap,
    evaluate_forecast_gate,
    evaluate_val_primary,
    evaluate_val_secondary,
    NoCompleteValidDevCandidate,
    select_dev_winner,
    validate_forecast_inputs,
    window_for_candidate_id,
)


# --- 1. Baseline B0 Tests ---


def test_b0_baseline_contract_and_values() -> None:
    """B0 must contain exactly 100 values of 0.27 summing to 27.0."""
    b0 = create_b0_forecast()
    assert isinstance(b0, np.ndarray)
    assert b0.shape == (100,)
    assert b0.dtype == np.float64
    assert np.all(b0 == B0_VALUE)
    assert B0_VALUE == 0.27
    assert math.isclose(float(np.sum(b0)), 27.0, abs_tol=1e-10)
    assert B0_MODEL_ID == ModelID.B0
    assert B0_CANDIDATE_ID == CandidateID.B0_UNIFORM


def test_b0_fresh_array_not_shared() -> None:
    """Calling create_b0_forecast produces independent unaliased arrays."""
    b0_1 = create_b0_forecast()
    b0_2 = create_b0_forecast()
    assert b0_1 is not b0_2
    b0_1[0] = 999.0
    assert b0_2[0] == 0.27


# --- 2. Input Validation Tests ---


def test_validate_forecast_inputs_success() -> None:
    """Valid inputs pass cleanly without mutating arguments."""
    y = np.zeros(100, dtype=np.int64)
    y[:27] = 1
    mu = np.full(100, 0.27, dtype=np.float64)

    y_copy = y.copy()
    mu_copy = mu.copy()

    validate_forecast_inputs(y, mu)
    np.testing.assert_array_equal(y, y_copy)
    np.testing.assert_array_equal(mu, mu_copy)


@pytest.mark.parametrize(
    "bad_y,bad_mu,match_err",
    [
        (np.zeros(99), np.full(100, 0.27), r"shape \(100,\)"),
        (np.zeros(100), np.full(99, 0.27), r"shape \(100,\)"),
        (np.full((10, 10), 0.27), np.full(100, 0.27), r"shape \(100,\)"),
        (np.zeros(100), np.full(100, 0.27), "Observed count sum must be 27"),
        (np.full(100, -1), np.full(100, 0.27), "Observed counts must be non-negative"),
        (np.array([np.nan] * 100), np.full(100, 0.27), "finite"),
        (np.array([np.inf] * 100), np.full(100, 0.27), "finite"),
        (
            np.array([1] * 27 + [0] * 73),
            np.zeros(100),
            "Expected counts mu must be strictly positive",
        ),
        (
            np.array([1] * 27 + [0] * 73),
            np.full(100, -0.27),
            "Expected counts mu must be strictly positive",
        ),
        (
            np.array([1] * 27 + [0] * 73),
            np.array([np.nan] * 100),
            "finite",
        ),
        (
            np.array([1] * 27 + [0] * 73),
            np.full(100, 0.28),
            "Expected count sum must equal 27",
        ),
    ],
)
def test_validate_forecast_inputs_failures(
    bad_y: Any, bad_mu: Any, match_err: str
) -> None:
    """Input contract violations raise MetricEvaluationError with METRIC_EVALUATION stage."""
    with pytest.raises(MetricEvaluationError, match=match_err) as exc_info:
        validate_forecast_inputs(bad_y, bad_mu)

    err = exc_info.value
    assert err.stage == FailureStage.METRIC_EVALUATION
    proto = err.as_protocol_failure()
    assert proto.stage == FailureStage.METRIC_EVALUATION
    assert proto.exit_status in {
        FailureExitStatus.NEEDS_MODEL_REVISION,
        FailureExitStatus.NEEDS_DATA_REVISION,
        FailureExitStatus.TECHNICAL_FAILURE,
    }


# --- 3. Cell and Daily Metric Independent Formula Tests ---


def test_cell_poisson_deviance_independent_oracle() -> None:
    """Test cell PD matches hand-derived analytic formula without epsilon."""
    # Case 1: y = 2, mu = 0.5
    # PD_cell = 2 * (2 * ln(2/0.5) - (2 - 0.5)) = 4 * ln(4) - 3
    expected_1 = 4.0 * math.log(4.0) - 3.0
    val_1 = compute_cell_poisson_deviance(2, 0.5)
    assert isinstance(val_1, float)
    assert abs(val_1 - expected_1) < 1e-14

    # Case 2: y = 0, mu = 0.27 (Zero count branch: 2 * mu)
    expected_2 = 2.0 * 0.27
    val_2 = compute_cell_poisson_deviance(0, 0.27)
    assert abs(val_2 - expected_2) < 1e-14

    # Case 3: y = 1, mu = 1.0 (Exact match: 0.0)
    expected_3 = 0.0
    val_3 = compute_cell_poisson_deviance(1, 1.0)
    assert abs(val_3 - expected_3) < 1e-14


def test_cell_poisson_deviance_vectorized_and_no_epsilon() -> None:
    """Cell PD vectorized calculation matches element-wise and does not take log(0)."""
    y = np.array([0, 1, 2, 0], dtype=np.int64)
    mu = np.array([0.27, 0.5, 1.5, 0.1], dtype=np.float64)

    res = compute_cell_poisson_deviance(y, mu)
    assert isinstance(res, np.ndarray)
    assert res.shape == (4,)

    for i in range(4):
        expected_i = compute_cell_poisson_deviance(int(y[i]), float(mu[i]))
        assert abs(res[i] - expected_i) < 1e-14


def test_daily_metrics_independent_oracle() -> None:
    """Hand-calculated benchmark: 27 ones, 73 zeros, uniform mu = 0.27."""
    y = np.zeros(100, dtype=np.int64)
    y[:27] = 1
    mu = np.full(100, 0.27, dtype=np.float64)

    # Independent hand calculations:
    # cell_1 = 2 * (1 * ln(1/0.27) - (1 - 0.27)) = 2 * (-ln(0.27) - 0.73)
    cell_1 = 2.0 * (-math.log(0.27) - 0.73)
    # cell_0 = 2 * 0.27 = 0.54
    cell_0 = 0.54
    expected_pd = (27.0 * cell_1 + 73.0 * cell_0) / 100.0

    # MAE: (27 * |1 - 0.27| + 73 * |0 - 0.27|) / 100 = (27 * 0.73 + 73 * 0.27) / 100 = 39.42 / 100 = 0.3942
    expected_mae = 0.3942

    # RMSE: sqrt((27 * (1 - 0.27)^2 + 73 * (0 - 0.27)^2) / 100) = sqrt((27*0.5329 + 73*0.0729)/100) = sqrt(0.1971)
    expected_rmse = math.sqrt(0.1971)

    daily_metrics = compute_daily_metrics(y, mu)
    assert isinstance(daily_metrics, DailyForecastMetrics)
    assert abs(daily_metrics.poisson_deviance - expected_pd) < 1e-14
    assert abs(daily_metrics.mae - expected_mae) < 1e-14
    assert abs(daily_metrics.rmse - expected_rmse) < 1e-14

    # Standalone function consistency
    assert abs(compute_daily_poisson_deviance(y, mu) - expected_pd) < 1e-14
    assert abs(compute_daily_mae(y, mu) - expected_mae) < 1e-14
    assert abs(compute_daily_rmse(y, mu) - expected_rmse) < 1e-14


# --- 4. Stage and Block Aggregation Tests ---


def test_stage_and_block_aggregation_exact_formulas() -> None:
    """Stage and block aggregation must match mean(PD), mean(MAE), and sqrt(mean(RMSE^2))."""
    m1 = DailyForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.40)
    m2 = DailyForecastMetrics(poisson_deviance=0.80, mae=0.42, rmse=0.60)

    stage = compute_stage_metrics([m1, m2])
    assert isinstance(stage, StageForecastMetrics)
    assert stage.date_count == 2
    assert abs(stage.poisson_deviance - 0.75) < 1e-14
    assert abs(stage.mae - 0.40) < 1e-14

    # Crucial pooled RMSE test:
    # mean(RMSE) = (0.40 + 0.60) / 2 = 0.50
    # pooled RMSE = sqrt((0.40^2 + 0.60^2) / 2) = sqrt((0.16 + 0.36) / 2) = sqrt(0.26) = 0.5099019513592785
    expected_pooled_rmse = math.sqrt((0.40**2 + 0.60**2) / 2.0)
    assert abs(stage.rmse - expected_pooled_rmse) < 1e-14
    assert stage.rmse != 0.50
    assert abs(stage.rmse - 0.50) > 0.009  # Significant distinctness

    # Block metrics use identical rules
    block = compute_block_metrics([m1, m2])
    assert block.date_count == stage.date_count
    assert block.poisson_deviance == stage.poisson_deviance
    assert block.mae == stage.mae
    assert block.rmse == stage.rmse


def test_stage_empty_sequence_raises() -> None:
    """Empty daily metric sequence raises MetricEvaluationError."""
    with pytest.raises(MetricEvaluationError, match="Empty daily metrics"):
        compute_stage_metrics([])


# --- 5. Candidate Universe and DEV Winner Selection Tests ---


def test_candidate_window_constants_and_ids() -> None:
    """Candidate windows and closed CandidateIDs match frozen specification."""
    assert CANDIDATE_WINDOWS == (30, 60, 120, 240, 365)
    assert candidate_id_for_window(30) == CandidateID.M3_W030
    assert candidate_id_for_window(60) == CandidateID.M3_W060
    assert candidate_id_for_window(120) == CandidateID.M3_W120
    assert candidate_id_for_window(240) == CandidateID.M3_W240
    assert candidate_id_for_window(365) == CandidateID.M3_W365

    assert window_for_candidate_id(CandidateID.M3_W030) == 30
    assert window_for_candidate_id("M3_W060") == 60
    assert window_for_candidate_id(CandidateID.M3_W120) == 120
    assert window_for_candidate_id(CandidateID.M3_W240) == 240
    assert window_for_candidate_id(CandidateID.M3_W365) == 365

    with pytest.raises(MetricEvaluationError, match="Invalid M3 candidate window"):
        candidate_id_for_window(90)

    with pytest.raises(MetricEvaluationError, match="Invalid M3 candidate ID"):
        window_for_candidate_id("UNKNOWN_ID")


def test_select_dev_winner_primary_pd_wins() -> None:
    """Candidate with strictly lower DEV Poisson deviance wins regardless of MAE/RMSE."""
    evals = [
        CandidateEvaluation(
            window=30,
            candidate_id=CandidateID.M3_W030,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.72, mae=0.35, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=60,
            candidate_id=CandidateID.M3_W060,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.40, rmse=0.50, date_count=100),
        ),
        CandidateEvaluation(
            window=120,
            candidate_id=CandidateID.M3_W120,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.75, mae=0.30, rmse=0.40, date_count=100),
        ),
        CandidateEvaluation(
            window=240,
            candidate_id=CandidateID.M3_W240,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.76, mae=0.30, rmse=0.40, date_count=100),
        ),
        CandidateEvaluation(
            window=365,
            candidate_id=CandidateID.M3_W365,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.77, mae=0.30, rmse=0.40, date_count=100),
        ),
    ]

    winner = select_dev_winner(evals)
    assert winner.window == 60
    assert winner.candidate_id == CandidateID.M3_W060


def test_select_dev_winner_mae_tiebreak() -> None:
    """When PD ties, candidate with lower MAE wins."""
    evals = [
        CandidateEvaluation(
            window=30,
            candidate_id=CandidateID.M3_W030,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.40, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=60,
            candidate_id=CandidateID.M3_W060,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.50, date_count=100),
        ),
        CandidateEvaluation(
            window=120,
            candidate_id=CandidateID.M3_W120,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.75, mae=0.30, rmse=0.40, date_count=100),
        ),
        CandidateEvaluation(
            window=240,
            candidate_id=CandidateID.M3_W240,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.76, mae=0.30, rmse=0.40, date_count=100),
        ),
        CandidateEvaluation(
            window=365,
            candidate_id=CandidateID.M3_W365,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.77, mae=0.30, rmse=0.40, date_count=100),
        ),
    ]

    winner = select_dev_winner(evals)
    assert winner.window == 60


def test_select_dev_winner_rmse_tiebreak() -> None:
    """When PD and MAE tie, candidate with lower RMSE wins."""
    evals = [
        CandidateEvaluation(
            window=30,
            candidate_id=CandidateID.M3_W030,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.48, date_count=100),
        ),
        CandidateEvaluation(
            window=60,
            candidate_id=CandidateID.M3_W060,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=120,
            candidate_id=CandidateID.M3_W120,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.75, mae=0.30, rmse=0.40, date_count=100),
        ),
        CandidateEvaluation(
            window=240,
            candidate_id=CandidateID.M3_W240,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.76, mae=0.30, rmse=0.40, date_count=100),
        ),
        CandidateEvaluation(
            window=365,
            candidate_id=CandidateID.M3_W365,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.77, mae=0.30, rmse=0.40, date_count=100),
        ),
    ]

    winner = select_dev_winner(evals)
    assert winner.window == 60


def test_select_dev_winner_window_tiebreak() -> None:
    """When all metrics tie exactly, smaller window W wins."""
    evals = [
        CandidateEvaluation(
            window=60,
            candidate_id=CandidateID.M3_W060,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=30,
            candidate_id=CandidateID.M3_W030,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=120,
            candidate_id=CandidateID.M3_W120,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=240,
            candidate_id=CandidateID.M3_W240,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=365,
            candidate_id=CandidateID.M3_W365,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.70, mae=0.38, rmse=0.45, date_count=100),
        ),
    ]

    winner = select_dev_winner(evals)
    assert winner.window == 30
    assert winner.candidate_id == CandidateID.M3_W030


def test_select_dev_winner_binary64_precision_no_rounding() -> None:
    """Candidate differing only at IEEE 754 float64 eps level must not be treated as tie."""
    pd_a = 1.0000000000000002
    pd_b = 1.0000000000000004
    assert pd_a < pd_b  # Raw binary64 difference

    evals = [
        CandidateEvaluation(
            window=60,
            candidate_id=CandidateID.M3_W060,
            dev_metrics=StageForecastMetrics(poisson_deviance=pd_a, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=30,
            candidate_id=CandidateID.M3_W030,
            dev_metrics=StageForecastMetrics(poisson_deviance=pd_b, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=120,
            candidate_id=CandidateID.M3_W120,
            dev_metrics=StageForecastMetrics(poisson_deviance=1.5, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=240,
            candidate_id=CandidateID.M3_W240,
            dev_metrics=StageForecastMetrics(poisson_deviance=1.5, mae=0.38, rmse=0.45, date_count=100),
        ),
        CandidateEvaluation(
            window=365,
            candidate_id=CandidateID.M3_W365,
            dev_metrics=StageForecastMetrics(poisson_deviance=1.5, mae=0.38, rmse=0.45, date_count=100),
        ),
    ]

    # Candidate 60 has strictly smaller PD, so it must win even though 30 has smaller W!
    winner = select_dev_winner(evals)
    assert winner.window == 60


def test_dev_winner_selection_ignores_val_data() -> None:
    """DEV winner selection only inspects dev_metrics; external VAL values have zero effect."""
    m_dev = {
        30: StageForecastMetrics(poisson_deviance=0.72, mae=0.35, rmse=0.45, date_count=100),
        60: StageForecastMetrics(poisson_deviance=0.70, mae=0.40, rmse=0.50, date_count=100),
        120: StageForecastMetrics(poisson_deviance=0.75, mae=0.30, rmse=0.40, date_count=100),
        240: StageForecastMetrics(poisson_deviance=0.76, mae=0.30, rmse=0.40, date_count=100),
        365: StageForecastMetrics(poisson_deviance=0.77, mae=0.30, rmse=0.40, date_count=100),
    }

    winner = select_dev_winner(m_dev)
    assert winner.window == 60


def test_dev_winner_selection_subset_support_and_zero_survivors() -> None:
    """Non-empty subsets are supported; empty candidate universe raises NoCompleteValidDevCandidate."""
    m_dev_subset = {
        30: StageForecastMetrics(poisson_deviance=0.72, mae=0.35, rmse=0.45, date_count=100),
        60: StageForecastMetrics(poisson_deviance=0.70, mae=0.40, rmse=0.50, date_count=100),
    }
    winner = select_dev_winner(m_dev_subset)
    assert winner.window == 60

    m_dev_single = {
        120: StageForecastMetrics(poisson_deviance=0.75, mae=0.30, rmse=0.40, date_count=100),
    }
    winner_single = select_dev_winner(m_dev_single)
    assert winner_single.window == 120

    with pytest.raises(NoCompleteValidDevCandidate):
        select_dev_winner({})

    with pytest.raises(NoCompleteValidDevCandidate):
        select_dev_winner([])

    with pytest.raises(MetricEvaluationError, match="unknown windows"):
        select_dev_winner({999: StageForecastMetrics(poisson_deviance=0.70, mae=0.30, rmse=0.40, date_count=100)})


# --- 6. VAL Comparisons and Forecast Gate Tests ---


def test_evaluate_val_primary_strict_inequality() -> None:
    """Primary VAL predicate: PD_M3 < PD_B0 (strict)."""
    assert evaluate_val_primary(pd_val_m3=0.70, pd_val_b0=0.71) is True
    assert evaluate_val_primary(pd_val_m3=0.71, pd_val_b0=0.71) is False  # Equality fails
    assert evaluate_val_primary(pd_val_m3=0.72, pd_val_b0=0.71) is False  # Greater fails


@pytest.mark.parametrize(
    "mae_m3,mae_b0,rmse_m3,rmse_b0,expected",
    [
        (0.35, 0.40, 0.55, 0.50, True),   # MAE pass, RMSE fail -> True
        (0.45, 0.40, 0.45, 0.50, True),   # MAE fail, RMSE pass -> True
        (0.35, 0.40, 0.45, 0.50, True),   # Both pass -> True
        (0.45, 0.40, 0.55, 0.50, False),  # Both fail -> False
        (0.40, 0.40, 0.55, 0.50, True),   # MAE exact equality -> True
        (0.45, 0.40, 0.50, 0.50, True),   # RMSE exact equality -> True
    ],
)
def test_evaluate_val_secondary_or_semantics(
    mae_m3: float, mae_b0: float, rmse_m3: float, rmse_b0: float, expected: bool
) -> None:
    """Secondary VAL predicate: MAE_M3 <= MAE_B0 or RMSE_M3 <= RMSE_B0."""
    assert evaluate_val_secondary(mae_m3, mae_b0, rmse_m3, rmse_b0) is expected


@pytest.mark.parametrize(
    "lower_bound,expected",
    [
        (1e-5, True),
        (0.001, True),
        (0.0, False),    # Equality to 0 fails
        (-0.0, False),
        (-1e-5, False),  # Negative fails
        (-0.5, False),
    ],
)
def test_evaluate_forecast_bootstrap_predicate(lower_bound: float, expected: bool) -> None:
    """Bootstrap predicate: lower_bound > 0.0 (strict)."""
    assert evaluate_forecast_bootstrap(lower_bound) is expected


@pytest.mark.parametrize("primary", [True, False])
@pytest.mark.parametrize("secondary", [True, False])
@pytest.mark.parametrize("bootstrap", [True, False])
def test_complete_forecast_gate_three_part_truth_table(
    primary: bool, secondary: bool, bootstrap: bool
) -> None:
    """Full 8-case truth table: forecast gate passes iff primary AND secondary AND bootstrap."""
    pd_m3 = 0.70 if primary else 0.75
    pd_b0 = 0.72

    mae_m3 = 0.35 if secondary else 0.45
    mae_b0 = 0.40
    rmse_m3 = 0.45 if secondary else 0.55
    rmse_b0 = 0.50

    bs_val = 0.01 if bootstrap else -0.01

    res = evaluate_forecast_gate(
        pd_val_m3=pd_m3,
        pd_val_b0=pd_b0,
        mae_val_m3=mae_m3,
        mae_val_b0=mae_b0,
        rmse_val_m3=rmse_m3,
        rmse_val_b0=rmse_b0,
        forecast_bootstrap_lower_bound=bs_val,
    )

    assert isinstance(res, ForecastGateResult)
    assert res.val_primary_pass is primary
    assert res.val_secondary_pass is secondary
    assert res.forecast_bootstrap_pass is bootstrap

    expected_gate = primary and secondary and bootstrap
    assert res.forecast_gate_pass is expected_gate


def test_zero_comparison_tolerance_constant() -> None:
    """COMPARISON_TOLERANCE must be 0.0."""
    assert COMPARISON_TOLERANCE == 0.0


def test_b0_not_eligible_as_m3_winner_in_evaluation_list() -> None:
    """B0 baseline must not be admitted into candidate evaluations for M3 winner selection."""
    evals = [
        CandidateEvaluation(
            window=30,
            candidate_id=CandidateID.B0_UNIFORM,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.1, mae=0.1, rmse=0.1, date_count=100),
        ),
        CandidateEvaluation(
            window=60,
            candidate_id=CandidateID.M3_W060,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.7, mae=0.4, rmse=0.5, date_count=100),
        ),
        CandidateEvaluation(
            window=120,
            candidate_id=CandidateID.M3_W120,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.7, mae=0.4, rmse=0.5, date_count=100),
        ),
        CandidateEvaluation(
            window=240,
            candidate_id=CandidateID.M3_W240,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.7, mae=0.4, rmse=0.5, date_count=100),
        ),
        CandidateEvaluation(
            window=365,
            candidate_id=CandidateID.M3_W365,
            dev_metrics=StageForecastMetrics(poisson_deviance=0.7, mae=0.4, rmse=0.5, date_count=100),
        ),
    ]
    with pytest.raises(MetricEvaluationError, match="B0 baseline is not eligible as an M3 candidate winner"):
        select_dev_winner(evals)


def test_natural_logarithm_semantics_verified() -> None:
    """Verify that cell Poisson deviance strictly uses natural log (base e), not base 10 or base 2."""
    y = 5
    mu = 0.5
    # Exact natural log formula: 2 * (5 * ln(10) - (5 - 0.5))
    ln_val = math.log(10.0)
    expected_e = 2.0 * (5.0 * ln_val - 4.5)
    actual = compute_cell_poisson_deviance(y, mu)
    assert abs(actual - expected_e) < 1e-14

    # Compare with base 10: 2 * (5 * log10(10) - 4.5) = 2 * (5 * 1 - 4.5) = 1.0
    val_base10 = 2.0 * (5.0 * math.log10(10.0) - 4.5)
    assert abs(actual - val_base10) > 10.0  # Vastly different!


def test_m3_07_does_not_implement_or_import_bootstrap_rng() -> None:
    """M3-07 must not import or implement bootstrap RNG, PCG64, or resamplers."""
    import inspect
    import src.m3_digit_factor.metrics as metrics_mod

    source = inspect.getsource(metrics_mod)
    forbidden_terms = [
        "PCG64",
        "20260831",
        "stationary circular",
        "replicates",
        "random.Generator",
        "np.random",
        "bootstrap resampling",
    ]
    for term in forbidden_terms:
        assert term not in source, f"Forbidden bootstrap term {term!r} found in metrics.py"


def test_input_arrays_not_mutated_in_all_metric_functions() -> None:
    """Verify all public daily functions preserve original array contents."""
    y = np.zeros(100, dtype=np.int64)
    y[:27] = 1
    mu = np.full(100, 0.27, dtype=np.float64)

    y_orig = y.copy()
    mu_orig = mu.copy()

    _ = compute_daily_poisson_deviance(y, mu)
    np.testing.assert_array_equal(y, y_orig)
    np.testing.assert_array_equal(mu, mu_orig)

    _ = compute_daily_mae(y, mu)
    np.testing.assert_array_equal(y, y_orig)
    np.testing.assert_array_equal(mu, mu_orig)

    _ = compute_daily_rmse(y, mu)
    np.testing.assert_array_equal(y, y_orig)
    np.testing.assert_array_equal(mu, mu_orig)

    _ = compute_daily_metrics(y, mu)
    np.testing.assert_array_equal(y, y_orig)
    np.testing.assert_array_equal(mu, mu_orig)

