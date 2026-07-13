import pandas as pd
import numpy as np
from . import BasePredictor

class LotoRepeatPredictor(BasePredictor):
    def __init__(self):
        super().__init__("Lô rơi (Loto Repeat)")

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

        # 2. Tính xác suất rơi lại cho từng số trong quá khứ
        scores = np.zeros(100)
        for num in yesterday_nums:
            # Các ngày số này xuất hiện (trừ ngày cuối vì chưa biết hôm nay có rơi hay không)
            appeared_days = np.where(S[:-1, num] > 0)[0]
            if len(appeared_days) == 0:
                scores[num] = 0.15  # xác suất mặc định nếu chưa từng xuất hiện trước đó
                continue

            # Số lần rơi lại vào ngày tiếp theo
            repeats = 0
            for day in appeared_days:
                if S[day + 1, num] > 0:
                    repeats += 1

            # Tính xác suất rơi lại có Laplace smoothing để tránh mẫu nhỏ
            scores[num] = (repeats + 1) / (len(appeared_days) + 6)

        # 3. Sắp xếp và trả về top_k
        sorted_nums = np.argsort(scores)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
