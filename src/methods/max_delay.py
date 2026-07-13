import pandas as pd
import numpy as np
from . import BasePredictor

class MaxDelayPredictor(BasePredictor):
    def __init__(self):
        super().__init__("Max Delay (Lô Khan)")

    def predict(self, history_df: pd.DataFrame, top_k: int = 5, S: np.ndarray = None) -> list[int]:
        if S is None:
            prize_cols = [c for c in history_df.columns if c != 'date']
            arr = history_df[prize_cols].values.astype(int)
            N = len(arr)
            S = np.zeros((N, 100), dtype=int)
            rows = np.repeat(np.arange(N), arr.shape[1])
            cols = arr.flatten()
            valid = (cols >= 0) & (cols < 100)
            S[rows[valid], cols[valid]] = 1
            
        N = len(S)
        if N < 100:
            return list(range(top_k))
            
        ratios = np.zeros(100)
        for num in range(100):
            appeared_indices = np.where(S[:, num] > 0)[0]
            if len(appeared_indices) == 0:
                current_delay = N
                max_delay = N
            else:
                current_delay = N - 1 - appeared_indices[-1]
                first_delay = appeared_indices[0]
                other_delays = np.diff(appeared_indices) - 1
                max_delay = max(first_delay, other_delays.max()) if len(other_delays) > 0 else first_delay
                
            if max_delay == 0:
                max_delay = 1
            ratios[num] = current_delay / max_delay
            
        sorted_nums = np.argsort(ratios)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
