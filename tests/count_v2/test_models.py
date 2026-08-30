import numpy as np
import pytest

from src.count_v2.contracts import (
    ChronologyError,
    CountForecast,
    CountHistory,
    ModelContractError,
)
from src.count_v2.models import (
    CountModel,
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
