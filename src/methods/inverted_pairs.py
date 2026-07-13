import pandas as pd
import numpy as np
from . import BasePredictor

class InvertedPairsPredictor(BasePredictor):
    def __init__(self):
        super().__init__("Cặp lộn (Inverted Pairs)")

    def _get_inverted(self, num: int) -> int:
        """Lấy số lộn của num (ví dụ 12 -> 21, 55 -> 55, 03 -> 30)"""
        d1 = num // 10
        d2 = num % 10
        return d2 * 10 + d1

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
        if N < 2:
            return list(range(top_k))

        # 1. Xác định danh sách số đã về ngày hôm trước (dòng cuối cùng)
        yesterday_nums = np.where(S[-1] > 0)[0]

        # 2. Với mỗi số đã về hôm trước, tính xác suất số lộn của nó về hôm nay
        scores = np.zeros(100)
        
        for num in yesterday_nums:
            inv = self._get_inverted(num)
            
            # Tìm những ngày num xuất hiện ở quá khứ (trừ ngày cuối)
            appeared_days = np.where(S[:-1, num] > 0)[0]
            if len(appeared_days) == 0:
                scores[inv] = max(scores[inv], 0.15)
                continue
                
            # Đếm số lần inv xuất hiện vào ngày tiếp theo
            hits = 0
            for day in appeared_days:
                if S[day + 1, inv] > 0:
                    hits += 1
                    
            prob = (hits + 1) / (len(appeared_days) + 6)
            scores[inv] = max(scores[inv], prob)

        # 3. Sắp xếp và trả về top_k
        sorted_nums = np.argsort(scores)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
