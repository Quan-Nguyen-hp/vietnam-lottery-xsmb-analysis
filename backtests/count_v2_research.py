from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.count_v2.contracts import (
    CountForecast,
    CountHistory,
    CountOutcome,
    DatasetValidationError,
    EvaluationIntegrityError,
    RawDrawBatch,
)
from src.count_v2.dataset import count_matrix_from_raw, raw_draw_batch_from_frame
from src.count_v2.models import (
    CountModel,
    DirichletShrinkageMultinomialModel,
    EWMACountModel,
    RollingCountModel,
    UniformCountModel,
)
from src.data.loader import DataLoader

EVIDENCE_CLASS: str = "DEVELOPMENT / EXPLORATORY"
DEVELOPMENT_ARTIFACT_ROOT: Path = (
    ROOT_DIR / "research_artifacts" / "xpis_v2_count_first" / "development"
)


@dataclass(frozen=True)
class DevelopmentRunConfig:
    """Configuration for an exploratory count model research run."""

    test_rows: int
    rolling_window: int
    ewma_half_life: float
    dirichlet_window: int
    prior_strength: float
    top_k: int

    def __post_init__(self) -> None:
        for field_name, val in (
            ("test_rows", self.test_rows),
            ("rolling_window", self.rolling_window),
            ("dirichlet_window", self.dirichlet_window),
            ("top_k", self.top_k),
        ):
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        if self.top_k > 100:
            raise ValueError(f"top_k must be <= 100, got {self.top_k}")

        for field_name, val in (
            ("ewma_half_life", self.ewma_half_life),
            ("prior_strength", self.prior_strength),
        ):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or math.isnan(val)
                or math.isinf(val)
                or val <= 0
            ):
                raise ValueError(f"{field_name} must be a positive finite float")


@dataclass(frozen=True)
class DevelopmentForecastSet:
    """Matrix of walk-forward point forecasts and observed outcomes."""

    target_dates: np.ndarray
    observed_counts: np.ndarray
    model_identities: tuple[str, ...]
    expected_counts: np.ndarray


@dataclass(frozen=True)
class DevelopmentRunResult:
    """Complete results from an exploratory evaluation run."""

    forecast_set: DevelopmentForecastSet
    summary: dict[str, Any]


def load_legacy_raw_draw_batch(csv_path: Path | None = None) -> RawDrawBatch:
    """Ingest raw draws using DataLoader.df, discarding legacy binary matrix S."""
    loader = DataLoader(csv_path=csv_path) if csv_path is not None else DataLoader()
    loader.load()
    draw_columns = loader.prize_cols()
    return raw_draw_batch_from_frame(loader.df, draw_columns=draw_columns)


def build_models(config: DevelopmentRunConfig) -> tuple[CountModel, ...]:
    """Construct exactly the four canonical models in fixed order (B0, B1, M1, M2)."""
    return (
        UniformCountModel(),
        RollingCountModel(window=config.rolling_window),
        EWMACountModel(half_life=config.ewma_half_life),
        DirichletShrinkageMultinomialModel(
            window=config.dirichlet_window,
            prior_strength=config.prior_strength,
        ),
    )


def generate_walk_forward_forecasts(
    batch: RawDrawBatch,
    config: DevelopmentRunConfig,
) -> DevelopmentForecastSet:
    """Generate strictly prior-history walk-forward count forecasts for all models."""
    counts = count_matrix_from_raw(batch)
    total_rows = len(batch.dates)
    start = total_rows - config.test_rows
    minimum_history = max(1, config.rolling_window, config.dirichlet_window)
    if start < minimum_history:
        raise ValueError(
            f"test_rows ({config.test_rows}) leave insufficient prior draw rows ({start}) for configured windows ({minimum_history})"
        )

    models = build_models(config)
    expected_counts = np.empty((len(models), config.test_rows, 100), dtype=np.float64)

    for target_offset, target_index in enumerate(range(start, total_rows)):
        history = CountHistory(batch.dates[:target_index], counts[:target_index])
        target_date = str(batch.dates[target_index])
        for model_index, model in enumerate(models):
            forecast = model.predict_count(history, target_date)
            expected_counts[model_index, target_offset] = forecast.expected_count

    return DevelopmentForecastSet(
        target_dates=batch.dates[start:],
        observed_counts=counts[start:],
        model_identities=tuple(model.model_identity for model in models),
        expected_counts=expected_counts,
    )
