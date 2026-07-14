"""
PROBABILITY MODEL LAYER — src/probability/momentum.py
Model 4: Frequency Momentum Predictor

Logic: Số ra nhiều nhất trong window ngắn → đang trong "chuỗi nóng".
       Dùng multi-timeframe frequency từ FeatureStore.
Migrate từ src/methods/frequency_momentum.py + upgrade multi-scale.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class MomentumPredictor(BaseProbabilityModel):

    def __init__(self, window: int = 30):
        self._window = window

    @property
    def name(self) -> str:
        return "frequency_momentum"

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
        Dùng frequency features từ FeatureStore.
        Kết hợp freq ngắn hạn và momentum để phát hiện "chuỗi nóng".
        """
        if "freq_30d" in df_features.columns:
            freq30 = df_features["freq_30d"].values.astype(float)
            freq7 = df_features.get("freq_7d", pd.Series(freq30)).values.astype(float)
            freq90 = df_features.get("freq_90d", pd.Series(freq30)).values.astype(float)

            # Momentum: số đang nóng = freq ngắn hạn tăng so với trung hạn
            # Tích hợp 3 tín hiệu: freq gần, xu hướng, so với nền
            proba = 0.5 * freq30 + 0.3 * freq7 + 0.2 * freq90
            return np.clip(proba, 0.0, 1.0)

        if S_history is not None:
            W = min(len(S_history), self._window)
            freqs = S_history[-W:].mean(axis=0).astype(float)
            return freqs

        return np.full(100, 0.27)
