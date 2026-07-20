import numpy as np
import pandas as pd

from src.evaluation.metrics import EvaluationMetrics


def test_break_even_expected_count_uses_count_payout_economics():
    metrics = EvaluationMetrics(cost_per_bet=27.0, payout_per_hit=99.0)
    assert np.isclose(metrics.break_even_expected_count, 27.0 / 99.0)
    assert np.isclose(metrics.expected_value_per_bet(27.0 / 99.0), 0.0)


def test_count_metrics_are_separate_from_binary_metrics():
    metrics = EvaluationMetrics(cost_per_bet=27.0, payout_per_hit=99.0)
    expected = np.array([[0.2, 0.4], [0.2, 0.4]])
    observed = np.array([[0, 1], [0, 0]])
    result = metrics.count_forecast_metrics(expected, observed)
    assert np.isclose(result["mean_expected_count"], 0.3)
    assert np.isclose(result["mean_observed_count"], 0.25)
    assert result["count_rmse"] > 0
    assert result["poisson_deviance"] > 0


def test_compute_full_can_report_binary_and_count_targets_together():
    metrics = EvaluationMetrics(cost_per_bet=27.0, payout_per_hit=99.0)
    results = pd.DataFrame([{"hit": True, "n_bets": 1, "n_hits": 2}])
    binary_probability = np.array([[0.5]])
    binary_label = np.array([[1]])
    expected_count = np.array([[0.4]])
    observed_count = np.array([[2]])
    summary = metrics.compute_full(
        results,
        proba_history=binary_probability,
        y_history=binary_label,
        expected_count_history=expected_count,
        count_history=observed_count,
    )
    assert "brier_score" in summary
    assert "count_rmse" in summary
    assert summary["total_hits"] == 2
