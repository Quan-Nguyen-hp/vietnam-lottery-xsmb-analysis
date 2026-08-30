from collections.abc import Sequence
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from backtests import count_v2_research
from backtests.count_v2_research import (
    DevelopmentForecastSet,
    DevelopmentRunConfig,
    build_models,
    generate_walk_forward_forecasts,
    load_legacy_raw_draw_batch,
)
from src.count_v2.contracts import (
    CountForecast,
    CountOutcome,
    DatasetValidationError,
    EvaluationIntegrityError,
    RawDrawBatch,
)
from src.count_v2.dataset import count_matrix_from_raw

LEGACY_DRAW_COLUMNS = tuple(f"draw_{index:02d}" for index in range(27))


def make_legacy_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(date),
                **{
                    column: (row_index * 27 + draw_index) % 100
                    for draw_index, column in enumerate(LEGACY_DRAW_COLUMNS)
                },
            }
            for row_index, date in enumerate(dates)
        ]
    )


def make_adapter_batch(row_count: int = 6) -> RawDrawBatch:
    dates = pd.date_range("2026-01-01", periods=row_count, freq="2D").strftime("%Y-%m-%d")
    draws = np.asarray(
        [
            [(row_index * 27 + offset) % 100 for offset in range(27)]
            for row_index in range(row_count)
        ],
        dtype=np.int16,
    )
    return RawDrawBatch(np.asarray(dates, dtype="<U10"), draws)


def test_adapter_reads_df_and_exact_draw_columns_without_binary_s(monkeypatch):
    frame = make_legacy_frame(["2026-01-01", "2026-01-03"])
    frame.loc[0, list(LEGACY_DRAW_COLUMNS[:3])] = 7

    class FakeLoader:
        load_calls = 0

        def __init__(self, csv_path: Path | None = None):
            self.csv_path = csv_path

        def load(self):
            type(self).load_calls += 1
            return self

        @property
        def df(self):
            return frame

        def prize_cols(self):
            return list(LEGACY_DRAW_COLUMNS)

        @property
        def S(self):
            raise AssertionError("binary S must not be read")

    monkeypatch.setattr(count_v2_research, "DataLoader", FakeLoader)
    batch = count_v2_research.load_legacy_raw_draw_batch()
    assert FakeLoader.load_calls == 1
    assert batch.draws.shape == (len(frame), 27)
    assert np.count_nonzero(batch.draws[0] == 7) >= 3


def test_adapter_rejects_invalid_column_count_from_loader(monkeypatch):
    frame = make_legacy_frame(["2026-01-01"])

    class FakeLoader26:
        def load(self):
            return self

        @property
        def df(self):
            return frame

        def prize_cols(self):
            return list(LEGACY_DRAW_COLUMNS[:26])

    monkeypatch.setattr(count_v2_research, "DataLoader", FakeLoader26)
    with pytest.raises(DatasetValidationError, match="draw_columns"):
        count_v2_research.load_legacy_raw_draw_batch()


def test_build_models_preserves_exact_identity_order():
    config = DevelopmentRunConfig(
        test_rows=2,
        rolling_window=30,
        ewma_half_life=14.0,
        dirichlet_window=45,
        prior_strength=20.0,
        top_k=5,
    )
    models = build_models(config)
    assert len(models) == 4
    assert models[0].model_identity == "B0_UNIFORM"
    assert models[1].model_identity == "B1_ROLLING_W30"
    assert models[2].model_identity == "M1_EWMA_H14_DEVELOPMENT"
    assert models[3].model_identity == "M2_DIRICHLET_SHRINKAGE_MULTINOMIAL_W45_B20"


def test_walk_forward_forecasts_are_generated_from_strictly_prior_rows():
    batch = make_adapter_batch(row_count=6)
    config = DevelopmentRunConfig(
        test_rows=2,
        rolling_window=2,
        ewma_half_life=2.0,
        dirichlet_window=2,
        prior_strength=10.0,
        top_k=2,
    )
    result = generate_walk_forward_forecasts(batch, config)

    assert isinstance(result, DevelopmentForecastSet)
    assert len(result.target_dates) == 2
    assert result.observed_counts.shape == (2, 100)
    assert result.expected_counts.shape == (4, 2, 100)

    start = len(batch.dates) - config.test_rows
    counts = count_matrix_from_raw(batch)

    # Check B0
    b0_index = result.model_identities.index("B0_UNIFORM")
    np.testing.assert_allclose(result.expected_counts[b0_index, 0], np.full(100, 0.27))

    # Check B1 for target 0 (which is batch index 4) seeing trailing 2 rows: indices 2 and 3
    b1_index = result.model_identities.index("B1_ROLLING_W2")
    expected_b1_day0 = counts[:start][-2:].mean(axis=0)
    np.testing.assert_allclose(result.expected_counts[b1_index, 0], expected_b1_day0)

    # Check B1 for target 1 (which is batch index 5) seeing trailing 2 rows: indices 3 and 4
    expected_b1_day1 = counts[: start + 1][-2:].mean(axis=0)
    np.testing.assert_allclose(result.expected_counts[b1_index, 1], expected_b1_day1)


def test_walk_forward_rejects_insufficient_history():
    batch = make_adapter_batch(row_count=3)
    config = DevelopmentRunConfig(
        test_rows=2,
        rolling_window=5,  # requires 5 prior rows, but only 3 - 2 = 1 row available
        ewma_half_life=2.0,
        dirichlet_window=2,
        prior_strength=10.0,
        top_k=2,
    )
    with pytest.raises(ValueError, match="insufficient"):
        generate_walk_forward_forecasts(batch, config)
