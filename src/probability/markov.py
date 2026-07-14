"""
PROBABILITY MODEL LAYER — src/probability/markov.py
Model 3: Markov Chain Predictor

Logic: P(num hôm nay | num ra hôm qua) dựa trên transition matrix.
       Lấy trung bình vector transition của tất cả số ra hôm qua.
Migrate từ src/methods/markov_chain.py với interface mới + dùng features.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class MarkovPredictor(BaseProbabilityModel):

    @property
    def name(self) -> str:
        return "markov_chain"

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
        Ưu tiên dùng markov_order1 từ FeatureStore nếu có.
        Fallback: tính trực tiếp từ S_history.
        """
        if "markov_order1" in df_features.columns:
            proba = df_features["markov_order1"].values.astype(float)
            return np.clip(proba, 0.0, 1.0)

        if S_history is not None and len(S_history) >= 100:
            return self._compute_from_S(S_history)

        return np.full(100, 0.27)

    def _compute_from_S(self, S: np.ndarray) -> np.ndarray:
        N = S.shape[0]
        # Transition count matrix (100 × 100): T[i, j] = count(i→j)
        trans_counts = S[:-1].T @ S[1:]  # (100, 100)
        occurrences = S[:-1].sum(axis=0)  # (100,)
        occ_safe = np.where(occurrences > 0, occurrences, 1)

        # Normalize rows → transition probability matrix
        trans_probs = trans_counts / occ_safe[:, np.newaxis]  # (100, 100)
        trans_probs[occurrences == 0, :] = 0.0

        # Yesterday active numbers
        yesterday = np.where(S[-1] > 0)[0]
        if len(yesterday) == 0:
            return np.full(100, 0.27)

        # Average over all active numbers from yesterday
        scores = trans_probs[yesterday].mean(axis=0)  # (100,)

        # Smooth toward uniform
        scores = 0.85 * scores + 0.15 * 0.27
        return scores
