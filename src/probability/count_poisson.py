"""Research-only model: chuyển kỳ vọng số nháy EWMA thành xác suất xuất hiện."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class CountEWMAPoissonPredictor(BaseProbabilityModel):
    """Ước lượng P(any hit) từ EWMA E[count], không dùng target-day data."""

    @property
    def name(self) -> str:
        return "count_ewma_poisson"

    @property
    def version(self) -> str:
        return "research-v1"

    def predict_proba(
        self,
        df_features: pd.DataFrame,
        df_history: pd.DataFrame | None = None,
        S_history: np.ndarray | None = None,
    ) -> np.ndarray:
        if df_history is None or len(df_history) == 0:
            return np.full(100, 0.27, dtype=float)
        prize_cols = [column for column in df_history.columns if column != "date"]
        draws = df_history[prize_cols].to_numpy(dtype=int)
        counts = np.stack([(draws == number).sum(axis=1) for number in range(100)], axis=1)
        window = min(540, len(counts))
        sample = counts[-window:].astype(float)
        ages = np.arange(window - 1, -1, -1, dtype=float)
        weights = np.exp(-np.log(2.0) * ages / 90.0)
        weights /= weights.sum()
        expected_count = weights @ sample
        return np.clip(1.0 - np.exp(-expected_count), 0.0, 1.0)
