"""
PROBABILITY MODEL LAYER — src/probability/max_delay.py
Model 1: Max Delay Predictor (Lô Khan)

Logic: Số có current_delay / max_delay cao nhất → sắp nổ.
Xác suất tỷ lệ thuận với delay_ratio (đã normalize).
Migrate từ src/methods/max_delay.py với interface mới.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class MaxDelayPredictor(BaseProbabilityModel):

    @property
    def name(self) -> str:
        return "max_delay"

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
        Dùng delay features từ FeatureStore (delay + delay_percentile).
        Fallback: tính trực tiếp từ S_history nếu thiếu features.
        """
        if "delay" in df_features.columns and S_history is not None:
            return self._compute_from_S(S_history)

        if "delay_percentile" in df_features.columns:
            # delay_percentile = rank / 100, cao = delay lâu = sắp nổ
            return df_features["delay_percentile"].values.astype(float)

        return np.full(100, 0.27)

    def _compute_from_S(self, S: np.ndarray) -> np.ndarray:
        N = S.shape[0]
        ratios = np.zeros(100)
        for num in range(100):
            appeared = np.where(S[:, num] > 0)[0]
            if len(appeared) == 0:
                current_delay = N
                max_delay = N
            else:
                current_delay = N - 1 - appeared[-1]
                gaps = np.diff(appeared)
                max_delay = max(int(appeared[0]), int(gaps.max()) if len(gaps) > 0 else 0)
                max_delay = max(max_delay, 1)
            ratios[num] = current_delay / max_delay

        # Normalize sang [0, 1]
        rmin, rmax = ratios.min(), ratios.max()
        if rmax > rmin:
            return (ratios - rmin) / (rmax - rmin)
        return np.full(100, 0.5)
