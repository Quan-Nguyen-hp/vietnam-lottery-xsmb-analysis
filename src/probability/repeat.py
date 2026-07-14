"""
PROBABILITY MODEL LAYER — src/probability/repeat.py
Model 6: Loto Repeat Predictor (Lô Rơi)

Logic: P(số rơi lại ngày hôm nay | số đã ra hôm qua).
       Tính empirical repeat probability từ lịch sử.
Migrate từ src/methods/loto_repeat.py — vectorized, không dùng Python loop.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class RepeatPredictor(BaseProbabilityModel):

    @property
    def name(self) -> str:
        return "loto_repeat"

    @property
    def version(self) -> str:
        return "2.0"

    def predict_proba(
        self,
        df_features: pd.DataFrame,
        df_history: Optional[pd.DataFrame] = None,
        S_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Tính P(rơi lại | ra hôm qua) cho 100 số.
        Vectorized: dùng ma trận nhân thay vì Python loop.
        """
        if S_history is None or len(S_history) < 2:
            return np.full(100, 0.27)

        return self._compute_from_S(S_history)

    def _compute_from_S(self, S: np.ndarray) -> np.ndarray:
        N = S.shape[0]

        # yesterday = S[-1], but we need repeat probability for each number
        yesterday = S[-1].astype(float)  # (100,) binary

        # Vectorized: count how many times each number appeared AND repeated next day
        # repeat_counts[num] = sum_t( S[t, num] * S[t+1, num] )
        appear_counts = S[:-1].sum(axis=0).astype(float)   # (100,)
        repeat_counts = (S[:-1] * S[1:]).sum(axis=0).astype(float)  # (100,)

        # Laplace smoothed repeat probability for all numbers
        repeat_prob = (repeat_counts + 1) / (appear_counts + 6)  # (100,)

        # Only yesterday's active numbers get repeat probability boosted
        # Numbers not active yesterday get uniform prior
        proba = np.where(yesterday > 0, repeat_prob, 0.27 * 0.3)

        return np.clip(proba, 0.0, 1.0)
