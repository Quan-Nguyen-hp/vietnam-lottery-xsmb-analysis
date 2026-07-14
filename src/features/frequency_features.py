"""
FEATURE LAYER — src/features/frequency_features.py
Trích xuất đặc trưng từ tần suất xuất hiện đa khung thời gian.
Input: df_evidence với count_3d, count_7d, ..., count_365d, count_all, history_days.
Output: freq_3d...freq_365d, freq_momentum_short, freq_momentum_long,
        freq_mean_30, freq_std_30, freq_skew_30, freq_kurt_30
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

WINDOWS = [3, 7, 14, 30, 60, 90, 120, 180, 365]


class FrequencyFeatureExtractor(BaseFeatureExtractor):
    name = "frequency_features"
    version = "1.0"

    def extract(self, df_evidence: pd.DataFrame) -> pd.DataFrame:
        results = []

        for _, row in df_evidence.iterrows():
            f = {}
            N = int(row["history_days"])

            # Frequency rate for each window
            for W in WINDOWS:
                count_col = f"count_{W}d"
                denom = min(N, W) if N > 0 else 1
                raw_count = float(row.get(count_col, 0))
                f[f"freq_{W}d"] = raw_count / denom if denom > 0 else 0.27

            # Frequency momentum: short-term vs medium-term
            f["freq_momentum_short"] = f["freq_7d"] - f["freq_30d"]
            f["freq_momentum_long"] = f["freq_30d"] - f["freq_90d"]
            f["freq_momentum_ultra"] = f["freq_3d"] - f["freq_14d"]

            # Trend ratio: how much faster/slower than long-term baseline
            long_base = f["freq_365d"] if f["freq_365d"] > 0 else 0.27
            f["freq_ratio_short"] = f["freq_7d"] / long_base
            f["freq_ratio_mid"] = f["freq_30d"] / long_base

            # Analytical moments of 30-day Bernoulli variable
            p = f["freq_30d"]
            f["freq_mean_30"] = p
            std = np.sqrt(p * (1 - p))
            f["freq_std_30"] = float(std)
            f["freq_skew_30"] = float((1 - 2 * p) / (std + 1e-5))
            f["freq_kurt_30"] = float((1 - 6 * p * (1 - p)) / (p * (1 - p) + 1e-5))

            # Overall frequency (all history)
            all_count = float(row.get("count_all", 0))
            f["freq_all"] = all_count / N if N > 0 else 0.27
            f["freq_above_expected"] = f["freq_all"] - 0.27  # deviation from uniform

            results.append(f)

        return pd.DataFrame(results, index=df_evidence.index)
