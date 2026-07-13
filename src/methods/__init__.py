import pandas as pd
import numpy as np

class BasePredictor:
    def __init__(self, name: str):
        self.name = name

    def predict(self, history_df: pd.DataFrame, top_k: int = 5, S: np.ndarray = None) -> list[int]:
        """
        Dự đoán top_k số loto cho ngày tiếp theo dựa trên dữ liệu lịch sử đã qua (history_df).
        S: Ma trận nhị phân (N, 100) của lịch sử (nếu có, để tối ưu hóa thời gian tính toán).
        """
        raise NotImplementedError
