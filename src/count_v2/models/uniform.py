from __future__ import annotations

import numpy as np

from src.count_v2.contracts import (
    ChronologyError,
    CountForecast,
    CountHistory,
    require_target_after_history,
)


class UniformCountModel:
    """Model B0: Uniform Count Baseline (27/100 = 0.27 everywhere)."""

    @property
    def model_identity(self) -> str:
        return "B0_UNIFORM"

    def predict_count(self, history: CountHistory, target_date: str) -> CountForecast:
        target = require_target_after_history(history, target_date)
        if len(history.dates) == 0:
            raise ChronologyError("History must contain at least one observation date")

        expected = np.full(100, 27.0 / 100.0, dtype=np.float64)
        se = np.zeros(100, dtype=np.float64)
        return CountForecast(
            target_date=target,
            history_start=str(history.dates[0]),
            history_end=str(history.dates[-1]),
            expected_count=expected,
            model_identity=self.model_identity,
            mean_standard_error=se,
        )
