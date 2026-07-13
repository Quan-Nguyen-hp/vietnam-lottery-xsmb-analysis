import pandas as pd
import numpy as np
from . import BasePredictor

class MarkovChainPredictor(BasePredictor):
    def __init__(self):
        super().__init__("Xích Markov (Markov Chain)")

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
            
        # Transition counts: sum_{t} S[t, i] * S[t+1, j]
        trans_counts = S[:-1].T @ S[1:]
        
        # Total occurrences on all days except the last
        occurrences = S[:-1].sum(axis=0)
        
        # Transition probabilities calculation
        occurrences_clamped = np.where(occurrences > 0, occurrences, 1)
        trans_probs = trans_counts / occurrences_clamped[:, np.newaxis]
        trans_probs[occurrences == 0, :] = 0
        
        # Yesterday's active lotos
        yesterday_lotos = np.where(S[-1] > 0)[0]
        if len(yesterday_lotos) == 0:
            return list(range(top_k))
            
        # Mean transition vector for yesterday's active lotos
        scores = trans_probs[yesterday_lotos].mean(axis=0)
        
        sorted_nums = np.argsort(scores)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
