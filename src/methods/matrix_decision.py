import json
import numpy as np
import pandas as pd
from pathlib import Path
from . import BasePredictor

from .max_delay import MaxDelayPredictor
from .conditional_prob import ConditionalProbabilityPredictor
from .markov_chain import MarkovChainPredictor
from .frequency_momentum import FrequencyMomentumPredictor
from .poisson_estimator import PoissonEstimatorPredictor
from .loto_repeat import LotoRepeatPredictor
from .inverted_pairs import InvertedPairsPredictor
from .day_of_week import DayOfWeekPredictor

class MatrixDecisionPredictor(BasePredictor):
    def __init__(self, rules_path: str = None):
        super().__init__("Ma trận Quyết định (Matrix Decision)")
        self.predictors = {
            "delay": MaxDelayPredictor(),
            "cond_prob": ConditionalProbabilityPredictor(),
            "markov": MarkovChainPredictor(),
            "momentum": FrequencyMomentumPredictor(window_size=30),
            "poisson": PoissonEstimatorPredictor(window_size=180),
            "repeat": LotoRepeatPredictor(),
            "pairs": InvertedPairsPredictor(),
            "day_of_week": DayOfWeekPredictor(),
        }
        
        # Thiết lập luật lọc mặc định (sẽ được ghi đè bằng file cấu hình tối ưu)
        self.rules = {
            "delay_min"     : 0.1,
            "delay_max"     : 0.9,
            "poisson_min"   : 0.2,
            "markov_min"    : 0.1,
            "momentum_min"  : 0.2,
            "repeat_min"    : 0.0,
            "pairs_min"     : 0.0,
            "cond_prob_min" : 0.0,
            "day_of_week_min": 0.0,
            
            # Trọng số sắp xếp mặc định
            "weight_poisson"  : 1.0,
            "weight_markov"   : 1.0,
            "weight_momentum" : 1.0,
            "weight_repeat"   : 1.0,
            "weight_pairs"    : 1.0,
            "weight_cond_prob": 1.0,
            "weight_day_of_week": 1.0,
        }
        
        if rules_path:
            self.load_rules(rules_path)

    def load_rules(self, rules_path: str) -> None:
        p = Path(rules_path)
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                    self.rules.update(data.get("rules", {}))
            except Exception as e:
                print(f"⚠️ Không thể load rules từ {rules_path}: {e}")

    def build_matrix(self, history_df: pd.DataFrame, S: np.ndarray = None) -> dict[int, dict[str, float]]:
        """
        Xây dựng ma trận đặc trưng 100xM.
        Trả về dict {num: {feature_name: normalized_score}}
        """
        # Sinh ma trận nhị phân chung để tối ưu tốc độ
        if S is None:
            prize_cols = [c for c in history_df.columns if c != 'date']
            arr = history_df[prize_cols].values.astype(int)
            N = len(arr)
            S = np.zeros((N, 100), dtype=int)
            rows = np.repeat(np.arange(N), arr.shape[1])
            cols = arr.flatten()
            valid = (cols >= 0) & (cols < 100)
            S[rows[valid], cols[valid]] = 1

        feature_matrix = {num: {} for num in range(100)}

        for key, pred in self.predictors.items():
            # Lấy toàn bộ 100 số theo thứ tự ưu tiên giảm dần
            preds_100 = pred.predict(history_df, top_k=100, S=S)
            
            # Chuẩn hóa điểm dựa trên xếp hạng (100 - rank) / 100
            for rank, num in enumerate(preds_100):
                if 0 <= num < 100:
                    score = (100 - rank) / 100.0
                    feature_matrix[num][key] = score

        return feature_matrix

    def predict(self, history_df: pd.DataFrame, top_k: int = 5, S: np.ndarray = None) -> list[int]:
        # 1. Sinh ma trận đặc trưng cho ngày hiện tại
        matrix = self.build_matrix(history_df, S=S)

        # 2. Định nghĩa hàm kiểm tra xem một số có đạt điều kiện lọc không
        def check_rules(num: int, r: dict) -> bool:
            f = matrix[num]
            # Kiểm tra khoảng trễ lô khan
            if not (r["delay_min"] <= f["delay"] <= r["delay_max"]):
                return False
            # Kiểm tra các ngưỡng tối thiểu khác
            if f["poisson"] < r["poisson_min"]: return False
            if f["markov"] < r["markov_min"]: return False
            if f["momentum"] < r["momentum_min"]: return False
            if f["repeat"] < r["repeat_min"]: return False
            if f["pairs"] < r["pairs_min"]: return False
            if f["cond_prob"] < r["cond_prob_min"]: return False
            if f["day_of_week"] < r["day_of_week_min"]: return False
            return True

        # 3. Lọc các số đạt chuẩn
        candidates = [num for num in range(100) if check_rules(num, self.rules)]

        # 4. Thuật toán lọc động (Dynamic Thresholding):
        # Nếu không đủ số ứng viên, nới lỏng dần các ngưỡng cho đến khi đủ top_k
        current_rules = dict(self.rules)
        iteration = 0
        max_iterations = 20

        while len(candidates) < top_k and iteration < max_iterations:
            iteration += 1
            # Nới lỏng 15% mỗi vòng lặp
            current_rules["delay_min"] = max(0.0, current_rules["delay_min"] - 0.05)
            current_rules["delay_max"] = min(1.0, current_rules["delay_max"] + 0.05)
            for k in ["poisson_min", "markov_min", "momentum_min", "repeat_min", "pairs_min", "cond_prob_min", "day_of_week_min"]:
                current_rules[k] = max(0.0, current_rules[k] - 0.05)
            
            candidates = [num for num in range(100) if check_rules(num, current_rules)]

        # 5. Sắp xếp các ứng viên dựa trên tổng điểm có trọng số
        def get_weighted_score(num: int) -> float:
            f = matrix[num]
            score = 0.0
            for k in ["poisson", "markov", "momentum", "repeat", "pairs", "cond_prob", "day_of_week"]:
                w = self.rules.get(f"weight_{k}", 1.0)
                score += f[k] * w
            return score

        sorted_candidates = sorted(candidates, key=get_weighted_score, reverse=True)
        
        # Nếu vẫn thiếu (rất hiếm), bù đắp bằng các số có điểm cao nhất
        if len(sorted_candidates) < top_k:
            all_sorted = sorted(list(range(100)), key=get_weighted_score, reverse=True)
            for num in all_sorted:
                if num not in sorted_candidates:
                    sorted_candidates.append(num)
                if len(sorted_candidates) >= top_k:
                    break

        return sorted_candidates[:top_k]
