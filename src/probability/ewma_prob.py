"""
PROBABILITY MODEL LAYER — src/probability/ewma_prob.py
Model 10: EWMA / Time Decay Probability

Khác với Frequency30 (mọi ngày trong 30 ngày có trọng số bằng nhau):
EWMA gán trọng số theo hàm mũ — ngày gần đây có giá trị cao hơn.

Công thức EWMA:
    P_ewma(num) = Σ(w_t × I(num xuất hiện ngày t)) / Σ(w_t)
    w_t = α × (1 - α)^(N - 1 - t)    [ngày gần nhất có w cao nhất]

Tham số:
    alpha: Tốc độ phân rã. Cao → phản ứng nhanh. Thấp → mượt hơn.
    Mặc định α = 0.06 (halflife ≈ 11 ngày)

So sánh với Frequency30:
    Frequency30:  w = [1,1,1,...,1] (flat)
    EWMA(α=0.06): w = [0.18,...,0.94,1.00] (decreasing về quá khứ)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class EWMAPredictor(BaseProbabilityModel):
    """
    Model 10: EWMA / Time Decay Probability.

    Tính xác suất có trọng số thời gian: ngày gần đây quan trọng hơn.
    Hỗ trợ nhiều alpha để tạo multi-scale EWMA features.
    """

    def __init__(
        self,
        alpha: float = 0.06,          # Tốc độ phân rã chính (halflife ≈ 11 ngày)
        window: int = 180,            # Chỉ nhìn lại tối đa 180 ngày
        multi_scale: bool = True,     # Kết hợp nhiều alpha
    ):
        self._alpha = alpha
        self._window = window
        self._multi_scale = multi_scale

        # Multi-scale alphas: nhanh, trung bình, chậm
        # halflife = 3d, 7d, 14d, 30d, 60d
        self._alphas = {
            "ewma_3d":  1 - np.exp(-np.log(2) / 3),    # α ≈ 0.206
            "ewma_7d":  1 - np.exp(-np.log(2) / 7),    # α ≈ 0.094
            "ewma_14d": 1 - np.exp(-np.log(2) / 14),   # α ≈ 0.048
            "ewma_30d": 1 - np.exp(-np.log(2) / 30),   # α ≈ 0.023
            "ewma_60d": 1 - np.exp(-np.log(2) / 60),   # α ≈ 0.011
        }

    @property
    def name(self) -> str:
        return "ewma_probability"

    @property
    def version(self) -> str:
        return "1.0"

    def predict_proba(
        self,
        df_features: pd.DataFrame,
        df_history: Optional[pd.DataFrame] = None,
        S_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Tính EWMA probability cho 100 số.

        Ưu tiên dùng S_history trực tiếp để tính EWMA chính xác.
        Fallback: dùng freq features từ df_features.
        """
        if S_history is not None:
            return self._compute_from_S(S_history)
        else:
            return self._compute_from_features(df_features)

    def _compute_from_S(self, S_history: np.ndarray) -> np.ndarray:
        """
        Tính EWMA trực tiếp từ ma trận nhị phân S.
        Vectorized: tính cả 100 số cùng lúc.
        """
        N = S_history.shape[0]
        W = min(N, self._window)
        S_slice = S_history[-W:]  # (W, 100)

        if self._multi_scale:
            return self._multi_scale_combine(S_slice)
        else:
            return self._single_ewma(S_slice, self._alpha)

    def _single_ewma(self, S_slice: np.ndarray, alpha: float) -> np.ndarray:
        """
        Single-scale EWMA cho tất cả 100 số.
        S_slice: (W, 100) với chiều 0 là thời gian (cũ → mới)
        """
        W = S_slice.shape[0]
        # Trọng số: ngày mới nhất (index W-1) có trọng số cao nhất
        # w_t = alpha * (1-alpha)^(W-1-t)
        t = np.arange(W)
        weights = alpha * (1 - alpha) ** (W - 1 - t)  # (W,)
        weight_sum = weights.sum()

        # Matrix multiply: (W,) @ (W, 100) = (100,)
        ewma_prob = weights @ S_slice / (weight_sum + 1e-10)  # (100,)
        return ewma_prob.astype(float)

    def _multi_scale_combine(self, S_slice: np.ndarray) -> np.ndarray:
        """
        Kết hợp nhiều EWMA scale bằng trọng số bình quân.
        Scale nhanh hơn nhận trọng số thấp hơn (nhiều noise hơn).
        """
        scale_weights = {
            "ewma_3d":  0.10,  # Rất nhạy, nhiều noise
            "ewma_7d":  0.20,
            "ewma_14d": 0.30,
            "ewma_30d": 0.25,
            "ewma_60d": 0.15,  # Chậm nhưng ổn định
        }

        combined = np.zeros(100)
        total_w = 0.0
        for scale_name, alpha in self._alphas.items():
            ewma = self._single_ewma(S_slice, alpha)
            w = scale_weights.get(scale_name, 0.2)
            combined += w * ewma
            total_w += w

        return (combined / total_w).astype(float)

    def _compute_from_features(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Fallback: tính EWMA estimate từ frequency features.
        Kém chính xác hơn nhưng không cần S_history.
        """
        n = len(df_features)
        # Xấp xỉ: EWMA ≈ weighted average của multi-timeframe frequencies
        weights = {"freq_7d": 0.35, "freq_14d": 0.30, "freq_30d": 0.20, "freq_60d": 0.15}
        result = np.zeros(n)
        total_w = 0.0
        for col, w in weights.items():
            if col in df_features.columns:
                result += w * df_features[col].values.astype(float)
                total_w += w
        if total_w > 0:
            result /= total_w
        else:
            result[:] = 0.27
        return result

    def compute_all_scales(self, S_history: np.ndarray) -> pd.DataFrame:
        """
        Tính tất cả 5 scale EWMA và trả DataFrame (100 rows × 5 cols).
        Hữu ích cho Feature Layer hoặc debugging.
        """
        N = S_history.shape[0]
        W = min(N, self._window)
        S_slice = S_history[-W:]
        records = {}
        for scale_name, alpha in self._alphas.items():
            records[scale_name] = self._single_ewma(S_slice, alpha)
        return pd.DataFrame(records)
