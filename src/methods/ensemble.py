import pandas as pd
import numpy as np
from . import BasePredictor

class EnsemblePredictor(BasePredictor):
    def __init__(self, predictors: list[BasePredictor], vote_top_n: int = 10,
                 weights: dict[str, float] | None = None):
        """
        predictors : Danh sách các predictor thành phần.
        vote_top_n : Mỗi predictor đề xuất top N số trước khi tính điểm.
        weights    : Dict {tên_predictor: trọng_số} — nếu None thì mặc định equal weight (1.0).
                     Trọng số cao hơn → phương pháp đó ảnh hưởng nhiều hơn lên kết quả cuối.
        """
        super().__init__("Đồng thuận Ensemble (Weighted Voting)")
        self.predictors = predictors
        self.vote_top_n = vote_top_n
        self.weights = weights or {}  # empty dict → equal weight 1.0 cho tất cả

    def predict(self, history_df: pd.DataFrame, top_k: int = 5, S: np.ndarray = None) -> list[int]:
        votes = {i: 0.0 for i in range(100)}

        for p in self.predictors:
            if p.name == self.name:
                continue
            try:
                # Lấy trọng số của predictor này (mặc định 1.0 nếu không khai báo)
                w = self.weights.get(p.name, 1.0)
                if w <= 0:
                    continue  # bỏ qua phương pháp có trọng số âm / bằng 0

                # Forward S để tái dùng ma trận nhị phân đã tính sẵn
                preds = p.predict(history_df, top_k=self.vote_top_n, S=S)

                # Điểm xếp hạng: hạng 1 = vote_top_n điểm, hạng cuối = 1 điểm
                for rank, num in enumerate(preds):
                    if 0 <= num < 100:
                        votes[num] += w * (self.vote_top_n - rank)

            except Exception:
                continue

        # Sắp xếp theo điểm bình chọn tổng hợp (giảm dần)
        sorted_nums = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:top_k]]
