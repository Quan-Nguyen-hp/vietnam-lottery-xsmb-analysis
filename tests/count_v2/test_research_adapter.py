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
    DevelopmentRunResult,
    build_models,
    evaluate_development_run,
    evaluate_forecast,
    generate_walk_forward_forecasts,
    load_legacy_raw_draw_batch,
    main,
    write_development_artifacts,
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


def make_development_config() -> DevelopmentRunConfig:
    return DevelopmentRunConfig(
        test_rows=2,
        rolling_window=2,
        ewma_half_life=2.0,
        dirichlet_window=2,
        prior_strength=10.0,
        top_k=2,
    )


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
    config = make_development_config()
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


def test_evaluate_forecast_requires_date_match_and_computes_metrics():
    forecast = CountForecast(
        target_date="2026-01-05",
        history_start="2026-01-01",
        history_end="2026-01-04",
        expected_count=np.full(100, 0.27, dtype=np.float64),
        model_identity="B0_UNIFORM",
    )
    # Outcome with 27 counts
    row = np.zeros(100, dtype=np.int16)
    row[:27] = 1
    outcome_mismatch = CountOutcome(
        target_date="2026-01-06",
        observed_counts=row,
    )
    outcome_match = CountOutcome(
        target_date="2026-01-05",
        observed_counts=row,
    )

    with pytest.raises(EvaluationIntegrityError, match="target dates must match"):
        evaluate_forecast(forecast, outcome_mismatch)

    metrics = evaluate_forecast(forecast, outcome_match)
    assert "count_mae" in metrics
    assert "count_rmse" in metrics
    assert "poisson_deviance" in metrics
    assert "count_calibration_error" in metrics
    assert "break_even_expected_count" in metrics


def test_evaluate_development_run_preserves_multiplicity_and_computes_summaries():
    batch = make_adapter_batch(row_count=6)
    config = make_development_config()
    forecast_set = generate_walk_forward_forecasts(batch, config)
    result = evaluate_development_run(forecast_set, config)

    assert isinstance(result, DevelopmentRunResult)
    assert "models" in result.summary
    models_summary = result.summary["models"]
    for model_id in result.forecast_set.model_identities:
        assert model_id in models_summary
        m_info = models_summary[model_id]
        for key in (
            "count_mae",
            "count_rmse",
            "poisson_deviance",
            "count_calibration_error",
            "break_even_expected_count",
            "top_k",
            "total_bets",
            "total_hits",
            "total_cost",
            "total_payout",
            "pnl",
            "roi",
        ):
            assert key in m_info
        assert m_info["top_k"] == config.top_k
        assert m_info["total_bets"] == config.test_rows * config.top_k
        assert m_info["total_cost"] == 27.0 * m_info["total_bets"]
        assert m_info["total_payout"] == 99.0 * m_info["total_hits"]
        assert np.isclose(m_info["pnl"], m_info["total_payout"] - m_info["total_cost"])


def test_artifacts_are_create_once_and_development_classified(tmp_path):
    config = make_development_config()
    forecast_set = generate_walk_forward_forecasts(make_adapter_batch(row_count=6), config)
    result = evaluate_development_run(forecast_set, config)
    run_dir = write_development_artifacts(
        result,
        config,
        run_id="DEV_TEST_001",
        artifact_root=tmp_path,
    )
    assert {path.name for path in run_dir.iterdir()} == {"summary.json", "forecasts.npz"}
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["evidence_class"] == "DEVELOPMENT / EXPLORATORY"
    assert payload["run_id"] == "DEV_TEST_001"
    assert payload["target_rows"] == 2

    npz = np.load(run_dir / "forecasts.npz")
    np.testing.assert_array_equal(npz["target_dates"], result.forecast_set.target_dates)
    np.testing.assert_array_equal(npz["observed_counts"], result.forecast_set.observed_counts)
    np.testing.assert_array_equal(npz["expected_counts"], result.forecast_set.expected_counts)

    with pytest.raises(FileExistsError):
        write_development_artifacts(result, config, run_id="DEV_TEST_001", artifact_root=tmp_path)


def test_main_cli_execution_with_custom_artifact_root(tmp_path, monkeypatch):
    frame = make_legacy_frame([f"2026-01-{i:02d}" for i in range(1, 10)])

    class FakeLoader:
        def load(self):
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

    argv = [
        "--test-rows",
        "2",
        "--rolling-window",
        "2",
        "--ewma-half-life",
        "2.0",
        "--dirichlet-window",
        "2",
        "--prior-strength",
        "10.0",
        "--top-k",
        "2",
        "--artifact-root",
        str(tmp_path),
        "--run-id",
        "CLI_RUN_001",
    ]
    exit_code = main(argv)
    assert exit_code == 0
    assert (tmp_path / "CLI_RUN_001" / "summary.json").exists()
    assert (tmp_path / "CLI_RUN_001" / "forecasts.npz").exists()
