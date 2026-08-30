from collections.abc import Mapping
import numpy as np
import pytest

from src.count_v2.contracts import (
    FORECAST_SUM_TOLERANCE,
    ChronologyError,
    CountContractError,
    CountForecast,
    CountHistory,
    CountOutcome,
    DatasetValidationError,
    EvaluationIntegrityError,
    ModelContractError,
    RawDrawBatch,
    canonical_date,
    require_target_after_history,
)


def make_count_row(draws: list[int]) -> np.ndarray:
    if len(draws) != 27:
        raise ValueError("test helper requires exactly 27 draws")
    row = np.zeros(100, dtype=np.int16)
    for value in draws:
        row[value] += 1
    return row


def test_exception_hierarchy():
    assert issubclass(DatasetValidationError, CountContractError)
    assert issubclass(ChronologyError, CountContractError)
    assert issubclass(ModelContractError, CountContractError)
    assert issubclass(EvaluationIntegrityError, CountContractError)
    assert issubclass(CountContractError, ValueError)


def test_canonical_date_valid_and_invalid():
    assert canonical_date("2026-01-05", field_name="test_date") == "2026-01-05"
    with pytest.raises(DatasetValidationError, match="test_date"):
        canonical_date("2026/01/05", field_name="test_date")
    with pytest.raises(DatasetValidationError, match="test_date"):
        canonical_date("2026-13-01", field_name="test_date")
    with pytest.raises(DatasetValidationError, match="test_date"):
        canonical_date("invalid-date", field_name="test_date")
    with pytest.raises(DatasetValidationError, match="test_date"):
        canonical_date(12345, field_name="test_date")


def test_raw_draw_batch_construction_and_serialization():
    dates = np.array(["2026-01-01", "2026-01-02"], dtype="<U10")
    draws = np.zeros((2, 27), dtype=np.int16)
    draws[0, :27] = list(range(27))
    draws[1, :27] = list(range(27, 54))

    batch = RawDrawBatch(dates=dates, draws=draws)
    assert batch.dates.shape == (2,)
    assert batch.draws.shape == (2, 27)
    assert batch.draws.dtype == np.int16
    assert not batch.draws.flags.writeable

    payload = batch.to_dict()
    assert isinstance(payload, dict)
    assert payload["dates"] == ["2026-01-01", "2026-01-02"]
    assert len(payload["draws"]) == 2

    restored = RawDrawBatch.from_dict(payload)
    np.testing.assert_array_equal(restored.dates, batch.dates)
    np.testing.assert_array_equal(restored.draws, batch.draws)


def test_raw_draw_batch_rejects_invalid_inputs():
    with pytest.raises(DatasetValidationError):
        RawDrawBatch(
            dates=np.array(["2026-01-02", "2026-01-01"], dtype="<U10"),  # non-monotonic
            draws=np.zeros((2, 27), dtype=np.int16),
        )
    with pytest.raises(DatasetValidationError):
        RawDrawBatch(
            dates=np.array(["2026-01-01", "2026-01-01"], dtype="<U10"),  # duplicate
            draws=np.zeros((2, 27), dtype=np.int16),
        )
    with pytest.raises(DatasetValidationError):
        RawDrawBatch(
            dates=np.array(["2026-01-01"], dtype="<U10"),
            draws=np.zeros((1, 26), dtype=np.int16),  # not 27 cols
        )
    with pytest.raises(DatasetValidationError):
        RawDrawBatch(
            dates=np.array(["2026-01-01"], dtype="<U10"),
            draws=np.full((1, 27), 100, dtype=np.int16),  # draw >= 100
        )
    with pytest.raises(DatasetValidationError):
        RawDrawBatch(
            dates=np.array(["2026-01-01"], dtype="<U10"),
            draws=np.full((1, 27), -1, dtype=np.int16),  # draw < 0
        )


def test_count_history_construction_and_serialization():
    dates = np.array(["2026-01-01", "2026-01-03"], dtype="<U10")
    counts = np.vstack([
        make_count_row(list(range(27))),
        make_count_row(list(range(1, 28))),
    ])
    history = CountHistory(dates=dates, counts=counts)
    assert history.dates.shape == (2,)
    assert history.counts.shape == (2, 100)
    assert history.counts.dtype == np.int16
    assert not history.counts.flags.writeable

    payload = history.to_dict()
    restored = CountHistory.from_dict(payload)
    np.testing.assert_array_equal(restored.dates, history.dates)
    np.testing.assert_array_equal(restored.counts, history.counts)


def test_count_history_rejects_invalid_inputs():
    with pytest.raises(DatasetValidationError):
        CountHistory(
            dates=np.array(["2026-01-01"], dtype="<U10"),
            counts=np.zeros((1, 100), dtype=np.int16),  # sum != 27
        )
    with pytest.raises(DatasetValidationError):
        CountHistory(
            dates=np.array(["2026-01-01"], dtype="<U10"),
            counts=np.zeros((1, 99), dtype=np.int16),  # shape != 100
        )


def test_count_outcome_construction_and_serialization():
    row = make_count_row(list(range(27)))
    outcome = CountOutcome(target_date="2026-01-05", observed_counts=row)
    assert outcome.target_date == "2026-01-05"
    assert outcome.observed_counts.shape == (100,)
    assert outcome.observed_counts.dtype == np.int16

    payload = outcome.to_dict()
    restored = CountOutcome.from_dict(payload)
    assert restored.target_date == outcome.target_date
    np.testing.assert_array_equal(restored.observed_counts, outcome.observed_counts)


def test_count_outcome_rejects_invalid_inputs():
    with pytest.raises(DatasetValidationError):
        CountOutcome(target_date="2026-01-05", observed_counts=np.zeros(100, dtype=np.int16))
    with pytest.raises(DatasetValidationError):
        CountOutcome(target_date="invalid", observed_counts=make_count_row(list(range(27))))


def test_calendar_gaps_are_valid_but_target_must_follow_every_history_date():
    history = CountHistory(
        dates=np.array(["2026-01-01", "2026-01-05"], dtype="<U10"),
        counts=np.vstack([
            make_count_row(list(range(27))),
            make_count_row(list(range(1, 28))),
        ]),
    )
    assert require_target_after_history(history, "2026-01-06") == "2026-01-06"
    with pytest.raises(ChronologyError, match="strictly after every history date"):
        require_target_after_history(history, "2026-01-05")
    with pytest.raises(ChronologyError, match="strictly after every history date"):
        require_target_after_history(history, "2026-01-02")


def test_forecast_contract_valid_construction_and_serialization():
    expected = np.full(100, 0.27, dtype=np.float64)
    se = np.full(100, 0.05, dtype=np.float64)
    lb = np.full(100, 0.17, dtype=np.float64)
    ub = np.full(100, 0.37, dtype=np.float64)
    pi = (np.zeros(100, dtype=np.float64), np.ones(100, dtype=np.float64))
    dist = {"family": "poisson"}

    forecast = CountForecast(
        target_date="2026-01-06",
        history_start="2026-01-01",
        history_end="2026-01-05",
        expected_count=expected,
        model_identity="M0_TEST",
        mean_standard_error=se,
        mean_lower_bound=lb,
        mean_upper_bound=ub,
        predictive_distribution=dist,
        prediction_interval=pi,
    )
    assert forecast.expected_count.shape == (100,)
    assert forecast.expected_count.dtype == np.float64
    assert not forecast.expected_count.flags.writeable

    payload = forecast.to_dict()
    restored = CountForecast.from_dict(payload)
    assert restored.target_date == "2026-01-06"
    assert restored.history_start == "2026-01-01"
    assert restored.history_end == "2026-01-05"
    assert restored.model_identity == "M0_TEST"
    np.testing.assert_allclose(restored.expected_count, forecast.expected_count)
    np.testing.assert_allclose(restored.mean_standard_error, forecast.mean_standard_error)
    np.testing.assert_allclose(restored.mean_lower_bound, forecast.mean_lower_bound)
    np.testing.assert_allclose(restored.mean_upper_bound, forecast.mean_upper_bound)
    assert restored.predictive_distribution == dist
    assert restored.prediction_interval is not None
    np.testing.assert_allclose(restored.prediction_interval[0], pi[0])
    np.testing.assert_allclose(restored.prediction_interval[1], pi[1])


def test_forecast_contract_rejects_conservation_failure():
    with pytest.raises(ModelContractError, match="MODEL_CONTRACT_FAILURE"):
        CountForecast(
            target_date="2026-01-06",
            history_start="2026-01-01",
            history_end="2026-01-05",
            expected_count=np.full(100, 0.26999, dtype=np.float64),
            model_identity="broken",
        )


@pytest.mark.parametrize(
    ("history_start", "history_end", "target_date"),
    [
        ("2026-01-05", "2026-01-01", "2026-01-06"),
        ("2026-01-01", "2026-01-06", "2026-01-06"),
        ("2026-01-01", "2026-01-07", "2026-01-06"),
    ],
)
def test_forecast_constructor_rejects_invalid_chronology(
    history_start: str,
    history_end: str,
    target_date: str,
):
    with pytest.raises(ChronologyError):
        CountForecast(
            target_date=target_date,
            history_start=history_start,
            history_end=history_end,
            expected_count=np.full(100, 0.27, dtype=np.float64),
            model_identity="chronology_probe",
        )


def test_forecast_from_dict_reuses_constructor_chronology_validation():
    payload = {
        "target_date": "2026-01-06",
        "history_start": "2026-01-01",
        "history_end": "2026-01-06",
        "expected_count": [0.27] * 100,
        "model_identity": "invalid_round_trip",
    }
    with pytest.raises(ChronologyError):
        CountForecast.from_dict(payload)


def test_forecast_rejects_negative_or_nan_expected_count():
    expected = np.full(100, 0.27, dtype=np.float64)
    expected[0] = -0.1
    expected[1] += 0.37
    with pytest.raises(ModelContractError):
        CountForecast(
            target_date="2026-01-06",
            history_start="2026-01-01",
            history_end="2026-01-05",
            expected_count=expected,
            model_identity="negative_probe",
        )

    expected_nan = np.full(100, 0.27, dtype=np.float64)
    expected_nan[0] = np.nan
    with pytest.raises(ModelContractError):
        CountForecast(
            target_date="2026-01-06",
            history_start="2026-01-01",
            history_end="2026-01-05",
            expected_count=expected_nan,
            model_identity="nan_probe",
        )
