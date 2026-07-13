import pandas as pd
import numpy as np
from . import BasePredictor

class FrequencyMomentumPredictor(BasePredictor):
    def __init__(self, window_size: int = 30):
        super().__init__(f"Tần suất Động lượng (Frequency Momentum - {window_size} ngày)")
        self.window_size = window_size

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
            
        # Sum last window_size rows of binary presence matrix S
        freqs = S[-min(self.window_size, N):].sum(axis=0)
        
        sorted_nums = np.argsort(freqs)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
