from __future__ import annotations

import numpy as np

from src.count_v2.contracts import (
    ChronologyError,
    CountForecast,
    CountHistory,
    require_target_after_history,
)


class RollingCountModel:
    """Model B1: Rolling Count Baseline over W trailing historical draw rows."""

    def __init__(self, window: int) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer number of draw rows")
        self.window = window

    @property
    def model_identity(self) -> str:
        return f"B1_ROLLING_W{self.window}"

    def predict_count(self, history: CountHistory, target_date: str) -> CountForecast:
        target = require_target_after_history(history, target_date)
        if len(history.dates) < self.window:
            raise ChronologyError("rolling history has fewer rows than window")

        expected = history.counts[-self.window:].mean(axis=0, dtype=np.float64)
        return CountForecast(
            target_date=target,
            history_start=str(history.dates[-self.window]),
            history_end=str(history.dates[-1]),
            expected_count=expected,
            model_identity=self.model_identity,
            mean_standard_error=None,
        )
