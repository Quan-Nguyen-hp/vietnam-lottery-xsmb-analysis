"""
PROBABILITY MODEL LAYER — src/probability/poisson.py
Model 5: Poisson Estimator

Logic: Poisson CDF — P(số ra trong 1 ngày | lambda, delay hiện tại).
       P = 1 - exp(-λ × (delay + 1))
       λ = tần suất trung bình trong window dài hạn.
Migrate từ src/methods/poisson_estimator.py với interface mới.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class PoissonPredictor(BaseProbabilityModel):

    def __init__(self, window: int = 180):
        self._window = window

    @property
    def name(self) -> str:
        return "poisson_estimator"

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
        Dùng delay + freq_180d từ FeatureStore.
        Công thức: P = 1 - exp(-λ × (delay + 1))
        """
        if "delay" in df_features.columns and "freq_180d" in df_features.columns:
            delays = df_features["delay"].values.astype(float)
            lambdas = df_features["freq_180d"].values.astype(float)
            lambdas = np.where(lambdas > 0, lambdas, 0.01)
            proba = 1.0 - np.exp(-lambdas * (delays + 1))
            return np.clip(proba, 0.0, 1.0)

        if S_history is not None:
            return self._compute_from_S(S_history)

        return np.full(100, 0.27)

    def _compute_from_S(self, S: np.ndarray) -> np.ndarray:
        N = S.shape[0]
        W = min(N, self._window)
        lambdas = S[-W:].mean(axis=0).astype(float)
        lambdas = np.where(lambdas > 0, lambdas, 0.01)

        delays = np.zeros(100)
        for num in range(100):
            appeared = np.where(S[:, num] > 0)[0]
            delays[num] = N - 1 - appeared[-1] if len(appeared) > 0 else N

        proba = 1.0 - np.exp(-lambdas * (delays + 1))
        return np.clip(proba, 0.0, 1.0)
