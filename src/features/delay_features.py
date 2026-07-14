"""
FEATURE LAYER — src/features/delay_features.py
Trích xuất các đặc trưng từ thông tin Delay (Lô Khan).
Input: df_evidence với cột current_delay, historical_delays.
Output: delay, delay_sq, delay_zscore, delay_percentile, delay_rank,
        delay_momentum, delay_volatility, delay_ewma, delay_mean, delay_std
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor


class DelayFeatureExtractor(BaseFeatureExtractor):
    name = "delay_features"
    version = "1.0"

    def extract(self, df_evidence: pd.DataFrame) -> pd.DataFrame:
        results = []
        all_delays = df_evidence["current_delay"].values.astype(float)

        for _, row in df_evidence.iterrows():
            f = {}
            current_delay = float(row["current_delay"])
            hist_gaps: list = row["historical_delays"] if row["historical_delays"] else [current_delay]
            hist_arr = np.array(hist_gaps, dtype=float)

            # Raw delay
            f["delay"] = current_delay
            f["delay_sq"] = current_delay ** 2

            # Statistical moments of historical gaps
            f["delay_mean"] = float(np.mean(hist_arr))
            f["delay_std"] = float(np.std(hist_arr)) if len(hist_arr) > 1 else 1.0

            # Z-score: how unusual is the current delay vs its own history
            f["delay_zscore"] = (current_delay - f["delay_mean"]) / (f["delay_std"] + 1e-5)

            # Momentum: current delay vs average of last 3 gaps
            last_3 = hist_arr[-3:] if len(hist_arr) >= 3 else hist_arr
            f["delay_momentum"] = current_delay - float(np.mean(last_3))

            # Velocity: change in delay relative to previous gap
            f["delay_velocity"] = float(hist_arr[-1] - hist_arr[-2]) if len(hist_arr) >= 2 else 0.0

            # Acceleration: second derivative
            if len(hist_arr) >= 3:
                v1 = hist_arr[-1] - hist_arr[-2]
                v2 = hist_arr[-2] - hist_arr[-3]
                f["delay_acceleration"] = float(v1 - v2)
            else:
                f["delay_acceleration"] = 0.0

            # Volatility of delays (std of last 10 gaps)
            last_10 = hist_arr[-10:] if len(hist_arr) >= 10 else hist_arr
            f["delay_volatility"] = float(np.std(last_10)) if len(last_10) > 1 else 1.0

            # EWMA of historical gaps (alpha=0.2)
            ewma = float(hist_arr[0])
            for g in hist_arr[1:]:
                ewma = 0.2 * float(g) + 0.8 * ewma
            f["delay_ewma"] = ewma

            results.append(f)

        df_feat = pd.DataFrame(results, index=df_evidence.index)

        # Cross-number features (computed over all 100 numbers at once)
        delays_arr = df_feat["delay"].values
        ranks = np.argsort(np.argsort(delays_arr))
        df_feat["delay_rank"] = ranks
        df_feat["delay_percentile"] = ranks / 100.0

        return df_feat
