import numpy as np
import pandas as pd
import pytest

from src.count_v2.contracts import (
    ChronologyError,
    CountForecast,
    CountHistory,
    ModelContractError,
)
from src.count_v2.models import (
    CountModel,
    DirichletShrinkageMultinomialModel,
    EWMACountModel,
    RollingCountModel,
    UniformCountModel,
)


def history_from_draw_rows(
    dates: list[str],
    draw_rows: list[list[int]],
) -> CountHistory:
    if len(dates) != len(draw_rows):
        raise ValueError("test helper requires one draw row per date")
    if any(len(row) != 27 for row in draw_rows):
        raise ValueError("test helper requires exactly 27 draws per row")
    counts = np.vstack([
        np.bincount(np.asarray(row, dtype=np.int16), minlength=100)
        for row in draw_rows
    ]).astype(np.int16)
    return CountHistory(np.asarray(dates, dtype="<U10"), counts)


def test_uniform_model_produces_exact_point_forecast_and_zero_mean_se():
    history = history_from_draw_rows(
        ["2026-01-01", "2026-01-04"],
        [list(range(27)), list(range(27))],
    )
    model = UniformCountModel()
    assert isinstance(model, CountModel)
    assert model.model_identity == "B0_UNIFORM"

    forecast = model.predict_count(history, "2026-01-05")
    assert isinstance(forecast, CountForecast)
    assert forecast.target_date == "2026-01-05"
    assert forecast.history_start == "2026-01-01"
    assert forecast.history_end == "2026-01-04"
    assert forecast.model_identity == "B0_UNIFORM"
    assert forecast.expected_count.shape == (100,)
    assert forecast.expected_count.dtype == np.float64
    np.testing.assert_allclose(forecast.expected_count, np.full(100, 0.27))
    assert np.isclose(forecast.expected_count.sum(), 27.0)

    assert forecast.mean_standard_error is not None
    assert forecast.mean_standard_error.shape == (100,)
    np.testing.assert_allclose(forecast.mean_standard_error, np.zeros(100))


def test_uniform_model_rejects_chronology_violations():
    history = history_from_draw_rows(
        ["2026-01-01", "2026-01-04"],
        [list(range(27)), list(range(27))],
    )
    model = UniformCountModel()
    with pytest.raises(ChronologyError):
        model.predict_count(history, "2026-01-04")
    with pytest.raises(ChronologyError):
        model.predict_count(history, "2026-01-02")


def test_rolling_uses_only_trailing_observation_rows_not_calendar_days():
    history = history_from_draw_rows(
        ["2026-01-01", "2026-01-04", "2026-02-10", "2026-02-28"],
        [[1] * 27, [2] * 27, [3] * 27, [4] * 27],
    )
    model = RollingCountModel(window=2)
    assert isinstance(model, CountModel)
    assert model.model_identity == "B1_ROLLING_W2"

    forecast = model.predict_count(history, "2026-03-15")
    assert forecast.target_date == "2026-03-15"
    assert forecast.history_start == "2026-02-10"
    assert forecast.history_end == "2026-02-28"
    assert forecast.mean_standard_error is None

    expected = (history.counts[-2].astype(np.float64) + history.counts[-1].astype(np.float64)) / 2.0
    np.testing.assert_allclose(forecast.expected_count, expected)
    assert np.isclose(forecast.expected_count.sum(), 27.0)


def test_rolling_model_validates_window_and_history_length():
    with pytest.raises(ValueError, match="window"):
        RollingCountModel(window=0)
    with pytest.raises(ValueError, match="window"):
        RollingCountModel(window=-5)
    with pytest.raises(ValueError, match="window"):
        RollingCountModel(window=True)  # type: ignore

    history = history_from_draw_rows(
        ["2026-01-01"],
        [list(range(27))],
    )
    model = RollingCountModel(window=2)
    with pytest.raises(ChronologyError, match="fewer rows than window"):
        model.predict_count(history, "2026-01-02")


def test_ewma_uses_finite_normalized_row_weights_and_finite_neff():
    model = EWMACountModel(half_life=2.0)
    assert isinstance(model, CountModel)
    assert model.model_identity == "M1_EWMA_H2_DEVELOPMENT"
    assert model.EVIDENCE_CLASS == "DEVELOPMENT / EXPLORATORY"
    assert model.MEAN_SEMANTICS == "DEVELOPMENT_ONLY_POISSON_MEAN_APPROXIMATION"

    weights = model.normalized_weights(history_rows=3)
    expected = np.exp(-np.log(2.0) * np.array([2.0, 1.0, 0.0]) / 2.0)
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[2] > weights[1] > weights[0]  # latest has largest weight

    neff = model.effective_sample_size(3)
    assert np.isclose(neff, 1.0 / np.sum(expected**2))
    assert np.isfinite(neff)


def test_ewma_model_prediction_and_uncertainty():
    history = history_from_draw_rows(
        ["2026-01-01", "2026-01-04", "2026-02-10"],
        [[1] * 27, [2] * 27, [3] * 27],
    )
    model = EWMACountModel(half_life=2.0)
    forecast = model.predict_count(history, "2026-03-01")

    assert forecast.target_date == "2026-03-01"
    assert forecast.history_start == "2026-01-01"
    assert forecast.history_end == "2026-02-10"
    assert forecast.expected_count.shape == (100,)
    assert np.isclose(forecast.expected_count.sum(), 27.0)

    weights = model.normalized_weights(3)
    expected_calc = weights @ history.counts.astype(np.float64)
    np.testing.assert_allclose(forecast.expected_count, expected_calc)

    neff = model.effective_sample_size(3)
    assert forecast.mean_standard_error is not None
    np.testing.assert_allclose(forecast.mean_standard_error, np.sqrt(expected_calc / neff))
    assert forecast.predictive_distribution is None
    assert forecast.prediction_interval is None


def test_ewma_model_parameter_validation():
    with pytest.raises(ValueError, match="half_life"):
        EWMACountModel(half_life=0.0)
    with pytest.raises(ValueError, match="half_life"):
        EWMACountModel(half_life=-1.5)
    with pytest.raises(ValueError, match="half_life"):
        EWMACountModel(half_life=np.nan)
    with pytest.raises(ValueError, match="half_life"):
        EWMACountModel(half_life=np.inf)
    with pytest.raises(ValueError, match="half_life"):
        EWMACountModel(half_life=True)  # type: ignore

    model = EWMACountModel(half_life=5.0)
    with pytest.raises(ValueError, match="history_rows"):
        model.normalized_weights(0)
    with pytest.raises(ValueError, match="history_rows"):
        model.normalized_weights(-1)
    with pytest.raises(ValueError, match="history_rows"):
        model.normalized_weights(True)  # type: ignore


def test_dirichlet_shrinkage_matches_uniform_prior_posterior_mean():
    draw_rows = [
        [(27 * row_index + offset) % 100 for offset in range(27)]
        for row_index in range(100)
    ]
    dates = pd.date_range("2025-01-01", periods=100, freq="D").strftime("%Y-%m-%d").tolist()
    history = history_from_draw_rows(dates, draw_rows)
    np.testing.assert_array_equal(
        history.counts.sum(axis=0),
        np.full(100, 27, dtype=np.int64),
    )
    model = DirichletShrinkageMultinomialModel(window=100, prior_strength=10.0)
    assert isinstance(model, CountModel)
    assert model.model_identity == "M2_DIRICHLET_SHRINKAGE_MULTINOMIAL_W100_B10"

    probabilities = model.posterior_probabilities(history)
    np.testing.assert_allclose(probabilities, np.full(100, 0.01))
    assert probabilities.shape == (100,)
    assert np.isclose(probabilities.sum(), 1.0)

    forecast = model.predict_count(history, "2025-04-11")
    np.testing.assert_allclose(forecast.expected_count, np.full(100, 0.27))
    assert np.isclose(forecast.expected_count.sum(), 27.0)
    assert forecast.mean_standard_error is not None
    assert forecast.predictive_distribution is None
    assert forecast.prediction_interval is None


def test_dirichlet_shrinkage_posterior_mean_and_variance_on_skewed_data():
    history = history_from_draw_rows(
        ["2026-01-01", "2026-01-02"],
        [[7] * 27, [7] * 27],
    )
    model = DirichletShrinkageMultinomialModel(window=2, prior_strength=100.0)
    probs = model.posterior_probabilities(history)

    # beta = 100, W = 2 => alpha_total = 100 + 27*2 = 154
    # alpha_7 = 100/100 + 54 = 55 => p_7 = 55 / 154 = 0.357142857...
    # alpha_other = 100/100 + 0 = 1 => p_other = 1 / 154 = 0.0064935...
    alpha_total = 100.0 + 27.0 * 2.0
    expected_p7 = (1.0 + 54.0) / alpha_total
    expected_p_other = 1.0 / alpha_total

    assert np.isclose(probs[7], expected_p7)
    assert np.isclose(probs[0], expected_p_other)
    assert np.isclose(probs.sum(), 1.0)

    forecast = model.predict_count(history, "2026-01-03")
    np.testing.assert_allclose(forecast.expected_count, 27.0 * probs)
    assert np.isclose(forecast.expected_count.sum(), 27.0)

    # Variance: alpha_n * (alpha_total - alpha_n) / (alpha_total^2 * (alpha_total + 1))
    # SE = 27 * sqrt(var)
    alpha = np.full(100, 1.0)
    alpha[7] = 55.0
    var_p = (alpha * (alpha_total - alpha)) / (alpha_total**2 * (alpha_total + 1.0))
    expected_se = 27.0 * np.sqrt(var_p)
    np.testing.assert_allclose(forecast.mean_standard_error, expected_se)


def test_dirichlet_shrinkage_parameter_validation():
    with pytest.raises(ValueError, match="window"):
        DirichletShrinkageMultinomialModel(window=0, prior_strength=10.0)
    with pytest.raises(ValueError, match="window"):
        DirichletShrinkageMultinomialModel(window=-1, prior_strength=10.0)
    with pytest.raises(ValueError, match="window"):
        DirichletShrinkageMultinomialModel(window=True, prior_strength=10.0)  # type: ignore
    with pytest.raises(ValueError, match="prior_strength"):
        DirichletShrinkageMultinomialModel(window=10, prior_strength=0.0)
    with pytest.raises(ValueError, match="prior_strength"):
        DirichletShrinkageMultinomialModel(window=10, prior_strength=-5.0)
    with pytest.raises(ValueError, match="prior_strength"):
        DirichletShrinkageMultinomialModel(window=10, prior_strength=np.nan)
    with pytest.raises(ValueError, match="prior_strength"):
        DirichletShrinkageMultinomialModel(window=10, prior_strength=np.inf)
    with pytest.raises(ValueError, match="prior_strength"):
        DirichletShrinkageMultinomialModel(window=10, prior_strength=True)  # type: ignore
