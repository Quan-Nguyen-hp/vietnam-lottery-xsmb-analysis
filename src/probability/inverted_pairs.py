"""
PROBABILITY MODEL LAYER — src/probability/inverted_pairs.py
Model 7: Inverted Pairs Predictor (Lô Lộn / Cặp Lộn)

Logic: Nếu số X ra hôm qua → P(số lộn X' ra hôm nay).
       Ví dụ: 12 ra hôm qua → 21 có xác suất cao hơn bình thường.
Migrate từ src/methods/inverted_pairs.py — vectorized hoàn toàn.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseProbabilityModel

# Precompute inversion map: inv_map[i] = số lộn của i
INV_MAP = np.array([(i % 10) * 10 + (i // 10) for i in range(100)])


class InvertedPairsPredictor(BaseProbabilityModel):

    @property
    def name(self) -> str:
        return "inverted_pairs"

    @property
    def version(self) -> str:
        return "2.0"

    def predict_proba(
        self,
        df_features: pd.DataFrame,
        df_history: Optional[pd.DataFrame] = None,
        S_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if S_history is None or len(S_history) < 2:
            return np.full(100, 0.27)

        return self._compute_from_S(S_history)

    def _compute_from_S(self, S: np.ndarray) -> np.ndarray:
        # Giới hạn 365 ngày gần nhất để tăng tốc — inverted pair pattern không cần dài hạn
        S = S[-365:] if len(S) > 365 else S

        # Vectorized co-occurrence: co_occur[i, j] = count(i at t, j at t+1)
        co_occur = S[:-1].T @ S[1:]   # (100, 100) — matrix multiply
        appear_counts = S[:-1].sum(axis=0).astype(float)  # (100,)

        # P(inv(j) tomorrow | j appeared today) for all j at once
        # conditional_prob[j] = co_occur[j, INV_MAP[j]] / appear_counts[j]
        numerators = co_occur[np.arange(100), INV_MAP].astype(float)  # (100,)
        denominators = appear_counts + 6  # Laplace
        conditional_prob = (numerators + 1) / denominators   # (100,)

        # Score[i] = conditional_prob[inv(i)] × I(inv(i) was active yesterday)
        yesterday = S[-1].astype(float)  # (100,) binary
        inv_yesterday = yesterday[INV_MAP]  # (100,) — was inv(i) active yesterday?
        inv_cond = conditional_prob[INV_MAP]  # (100,) — P(i | inv(i) appeared)

        # Apply score only when inv was active yesterday
        scores = np.where(inv_yesterday > 0, inv_cond, 0.27 * 0.2)
        return np.clip(scores, 0.0, 1.0)
