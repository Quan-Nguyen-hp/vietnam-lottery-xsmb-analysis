import pandas as pd
import numpy as np
from . import BasePredictor

class ConditionalProbabilityPredictor(BasePredictor):
    def __init__(self, name: str = "Bạc Nhớ (Conditional Similarity)", min_shared: int = 7, top_n_days: int = 50):
        super().__init__(name)
        self.min_shared = min_shared
        self.top_n_days = top_n_days

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
            
        # Similarities via dot product with the last day
        similarities = S[:-1] @ S[-1]
        
        # Get matching days
        matching_indices = np.where(similarities >= self.min_shared)[0]
        if len(matching_indices) < 15:
            matching_indices = np.argsort(similarities)[-self.top_n_days:]
            
        next_day_indices = matching_indices + 1
        next_day_indices = next_day_indices[next_day_indices < N]
        
        if len(next_day_indices) == 0:
            return list(range(top_k))
            
        # Sum frequency of appearances on the next days
        freqs = S[next_day_indices].sum(axis=0)
        
        sorted_nums = np.argsort(freqs)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
