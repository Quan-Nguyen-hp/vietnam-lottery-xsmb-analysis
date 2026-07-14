"""
PROBABILITY MODEL LAYER — src/probability/day_of_week.py
Model 8: Day of Week Predictor (Cầu Theo Thứ)

Logic: P(num | thứ ngày dự đoán) — tần suất lịch sử theo ngày trong tuần.
       Dùng time features từ FeatureStore nếu có.
Migrate từ src/methods/day_of_week.py với interface mới.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class DayOfWeekPredictor(BaseProbabilityModel):

    @property
    def name(self) -> str:
        return "day_of_week"

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
        Tính xác suất theo thứ trong tuần.
        Cần df_history để biết ngày của target, và S_history để đếm tần suất.
        """
        if df_history is None or S_history is None or len(df_history) < 28:
            return np.full(100, 0.27)

        # Xác định weekday của ngày target (ngày kế tiếp sau history)
        if "weekday" in df_features.columns:
            target_weekday = int(df_features["weekday"].iloc[0])
        else:
            last_date = pd.to_datetime(df_history["date"].iloc[-1])
            target_weekday = (last_date.weekday() + 1) % 7

        return self._compute_from_S(df_history, S_history, target_weekday)

    def _compute_from_S(
        self,
        df_history: pd.DataFrame,
        S: np.ndarray,
        target_weekday: int,
    ) -> np.ndarray:
        # Tìm các ngày trong history có cùng weekday
        weekdays = pd.to_datetime(df_history["date"]).dt.weekday.values
        matching = np.where(weekdays == target_weekday)[0]

        if len(matching) == 0:
            return np.full(100, 0.27)

        # Tần suất xuất hiện trên các ngày cùng thứ
        freqs = S[matching].sum(axis=0).astype(float)  # (100,)
        denom = len(matching)

        # Laplace smoothed probability
        proba = (freqs + 1) / (denom + 2)
        return proba
