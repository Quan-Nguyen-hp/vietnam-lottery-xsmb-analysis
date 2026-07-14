"""
PROBABILITY MODEL LAYER — src/probability/conditional.py
Model 2: Conditional Probability Predictor (Bạc Nhớ / Pattern Matching)

Logic: Tìm những ngày trong quá khứ có pattern "giống hôm qua nhất"
       → dự đoán ngày tiếp theo theo xác suất hậu nghiệm.
Migrate từ src/methods/conditional_prob.py với interface mới.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class ConditionalPredictor(BaseProbabilityModel):

    def __init__(self, min_shared: int = 7, top_n_days: int = 50):
        self._min_shared = min_shared
        self._top_n_days = top_n_days

    @property
    def name(self) -> str:
        return "conditional_probability"

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
        Pattern matching: tìm ngày hôm qua similar → dự đoán ngày kế.
        Cần S_history để tính similarity.
        """
        if S_history is None or len(S_history) < 100:
            return np.full(100, 0.27)

        return self._compute_from_S(S_history)

    def _compute_from_S(self, S: np.ndarray) -> np.ndarray:
        N = S.shape[0]
        # Similarity: số số chung giữa hôm qua và mỗi ngày trong quá khứ
        similarities = S[:-1] @ S[-1]  # (N-1,)

        matching = np.where(similarities >= self._min_shared)[0]
        if len(matching) < 15:
            matching = np.argsort(similarities)[-self._top_n_days:]

        next_days = matching + 1
        next_days = next_days[next_days < N]

        if len(next_days) == 0:
            return np.full(100, 0.27)

        # Tần suất xuất hiện trong ngày kế theo → xác suất có điều kiện
        freqs = S[next_days].sum(axis=0).astype(float)  # (100,)
        denom = len(next_days)

        proba = (freqs + 1) / (denom + 2)  # Laplace smoothing
        return proba
