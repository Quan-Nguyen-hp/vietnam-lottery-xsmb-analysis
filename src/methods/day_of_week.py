import pandas as pd
import numpy as np
from . import BasePredictor

class DayOfWeekPredictor(BasePredictor):
    def __init__(self):
        super().__init__("Cầu ngày trong tuần (Day of Week)")

    def predict(self, history_df: pd.DataFrame, top_k: int = 5, S: np.ndarray = None) -> list[int]:
        if history_df is None or len(history_df) == 0:
            return list(range(top_k))

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
        if N < 28:  # Cần ít nhất 4 tuần lịch sử
            return list(range(top_k))

        # 1. Xác định thứ của ngày cần dự đoán (ngày tiếp theo sau dòng cuối của history_df)
        last_date = pd.to_datetime(history_df["date"].iloc[-1])
        target_weekday = (last_date.weekday() + 1) % 7

        # 2. Tìm tất cả các ngày trong quá khứ có cùng thứ này
        history_weekdays = pd.to_datetime(history_df["date"]).dt.weekday.values
        matching_indices = np.where(history_weekdays == target_weekday)[0]

        if len(matching_indices) == 0:
            return list(range(top_k))

        # 3. Tính tần suất xuất hiện của các số vào ngày thứ này
        freqs = S[matching_indices].sum(axis=0)

        # 4. Sắp xếp và trả về top_k
        sorted_nums = np.argsort(freqs)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
