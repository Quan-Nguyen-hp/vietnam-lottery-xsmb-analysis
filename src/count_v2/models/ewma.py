from __future__ import annotations

import math
import numpy as np

from src.count_v2.contracts import (
    ChronologyError,
    CountForecast,
    CountHistory,
    require_target_after_history,
)


class EWMACountModel:
    """Model M1: Exponentially Weighted Moving Average (EWMA) Count Model."""

    EVIDENCE_CLASS: str = "DEVELOPMENT / EXPLORATORY"
    MEAN_SEMANTICS: str = "DEVELOPMENT_ONLY_POISSON_MEAN_APPROXIMATION"

    def __init__(self, half_life: float) -> None:
        if (
            isinstance(half_life, bool)
            or not isinstance(half_life, (int, float))
            or math.isnan(half_life)
            or math.isinf(half_life)
            or half_life <= 0
        ):
            raise ValueError("half_life must be a positive finite float number of draw rows")
        self.half_life = float(half_life)

    @property
    def model_identity(self) -> str:
        return f"M1_EWMA_H{self.half_life:g}_DEVELOPMENT"

    def normalized_weights(self, history_rows: int) -> np.ndarray:
        if (
            isinstance(history_rows, bool)
            or not isinstance(history_rows, int)
            or history_rows <= 0
        ):
            raise ValueError("history_rows must be a positive integer")
        ages = np.arange(history_rows - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-np.log(2.0) * ages / self.half_life)
        return weights / weights.sum()

    def effective_sample_size(self, history_rows: int) -> float:
        weights = self.normalized_weights(history_rows)
        return float(1.0 / np.sum(weights**2))

    def predict_count(self, history: CountHistory, target_date: str) -> CountForecast:
        target = require_target_after_history(history, target_date)
        history_rows = len(history.dates)
        if history_rows == 0:
            raise ChronologyError("History must contain at least one observation date")

        weights = self.normalized_weights(history_rows)
        expected = weights @ history.counts.astype(np.float64)
        neff = self.effective_sample_size(history_rows)
        mean_se = np.sqrt(expected / neff)

        return CountForecast(
            target_date=target,
            history_start=str(history.dates[0]),
            history_end=str(history.dates[-1]),
            expected_count=expected,
            model_identity=self.model_identity,
            mean_standard_error=mean_se,
        )
