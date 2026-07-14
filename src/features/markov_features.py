"""
FEATURE LAYER — src/features/markov_features.py
Trích xuất đặc trưng từ chuỗi trạng thái Markov.
Input: df_evidence với state_yesterday, state_day_before, historical_delays.
Output: markov_order1, markov_order2, markov_entropy, markov_persistence,
        state_run_length (số ngày liên tiếp ở trạng thái hiện tại)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor


class MarkovFeatureExtractor(BaseFeatureExtractor):
    name = "markov_features"
    version = "1.0"

    def __init__(self, S_history: np.ndarray):
        """
        Args:
            S_history: Ma trận nhị phân (N, 100) từ DataLoader.
                       Cần thiết để tính transition probabilities chính xác.
        """
        self._S = S_history

    def extract(self, df_evidence: pd.DataFrame) -> pd.DataFrame:
        results = []
        N = self._S.shape[0] if self._S is not None else 0

        for _, row in df_evidence.iterrows():
            num = int(row["number"])
            f = {}

            if N > 2 and self._S is not None:
                x1 = self._S[:-1, num]
                y1 = self._S[1:, num]
                state_prev = int(row["state_yesterday"])

                # Order 1: P(appear | state_yesterday)
                mask = (x1 == state_prev)
                f["markov_order1"] = float((np.sum(y1[mask]) + 1) / (np.sum(mask) + 2))

                # Order 2: P(appear | state_yesterday, state_day_before)
                if N > 3:
                    state_prev2 = int(row["state_day_before"])
                    x2_1 = self._S[:-2, num]
                    x2_2 = self._S[1:-1, num]
                    y2 = self._S[2:, num]
                    mask2 = (x2_1 == state_prev2) & (x2_2 == state_prev)
                    f["markov_order2"] = float((np.sum(y2[mask2]) + 1) / (np.sum(mask2) + 2))
                else:
                    f["markov_order2"] = 0.27

                # Transition probabilities
                p01 = float((np.sum((x1 == 0) & (y1 == 1)) + 1) / (np.sum(x1 == 0) + 2))
                p11 = float((np.sum((x1 == 1) & (y1 == 1)) + 1) / (np.sum(x1 == 1) + 2))
                p00 = 1.0 - p01
                p10 = 1.0 - p11

                # Entropy of transition from each state
                h0 = -p00 * np.log2(p00 + 1e-9) - p01 * np.log2(p01 + 1e-9)
                h1 = -p10 * np.log2(p10 + 1e-9) - p11 * np.log2(p11 + 1e-9)
                p_stat = float(np.mean(self._S[:, num]))
                f["markov_entropy"] = float((1 - p_stat) * h0 + p_stat * h1)

                # Persistence: how much more likely to stay in current state
                # Positive = tends to cluster; Negative = alternates
                f["markov_persistence"] = float(p11 - p01)

                # Stationary probability (from transition matrix)
                # π = p01 / (1 - p11 + p01)
                denom = (1 - p11 + p01)
                f["markov_stationary"] = float(p01 / denom) if denom > 1e-5 else 0.27

            else:
                f["markov_order1"] = 0.27
                f["markov_order2"] = 0.27
                f["markov_entropy"] = 1.0
                f["markov_persistence"] = 0.0
                f["markov_stationary"] = 0.27

            # State run length: how many consecutive days in same state
            state_now = int(row["state_yesterday"])
            run = 1
            if N > 1 and self._S is not None:
                for day_back in range(2, min(N + 1, 30)):
                    if self._S[-day_back, num] == state_now:
                        run += 1
                    else:
                        break
            f["markov_run_length"] = run

            results.append(f)

        return pd.DataFrame(results, index=df_evidence.index)
