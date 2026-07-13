import pandas as pd
import numpy as np
from . import BasePredictor

class PoissonEstimatorPredictor(BasePredictor):
    def __init__(self, window_size: int = 180):
        super().__init__(f"Ước lượng Poisson (Poisson Estimator - {window_size} ngày)")
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
            
        # Lambda calculation: average presence per day in window
        lambdas = S[-min(self.window_size, N):].mean(axis=0)
        lambdas = np.where(lambdas > 0, lambdas, 0.01)
        
        # Current delay calculation
        current_delays = np.zeros(100)
        for num in range(100):
            appeared_indices = np.where(S[:, num] > 0)[0]
            if len(appeared_indices) == 0:
                current_delays[num] = N
            else:
                current_delays[num] = N - 1 - appeared_indices[-1]
                
        # Poisson CDF score
        scores = 1.0 - np.exp(-lambdas * (current_delays + 1))
        
        sorted_nums = np.argsort(scores)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
