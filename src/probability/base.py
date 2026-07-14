"""
PROBABILITY MODEL LAYER — src/probability/base.py
Interface chuẩn cho tất cả Probability Models trong XPIS.
Mỗi model hoạt động độc lập và có thể backtest riêng.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseProbabilityModel(ABC):
    """
    Interface chuẩn cho Layer 4 — Probability Model Layer.

    Nguyên tắc:
    - Mỗi model nhận FeatureVector (100 rows) → trả xác suất (100,).
    - Hoạt động độc lập, không biết về các model khác.
    - Có thể backtest riêng lẻ.
    - Tương thích ngược với BasePredictor cũ.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên định danh của model."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version của model."""
        ...

    @abstractmethod
    def predict_proba(
        self,
        df_features: pd.DataFrame,
        df_history: Optional[pd.DataFrame] = None,
        S_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Dự báo xác suất xuất hiện cho 100 số.

        Args:
            df_features: FeatureVector DataFrame (100 rows × N_features cols)
            df_history: Lịch sử thô (optional, cho các model cần thêm context)
            S_history: Ma trận nhị phân (optional)

        Returns:
            np.ndarray shape (100,) — xác suất của từng số (00-99).
            Tổng không nhất thiết bằng 1.0 (mỗi số là xác suất độc lập).
        """
        ...

    def top_k(self, proba: np.ndarray, k: int = 10) -> list[int]:
        """Trả về K số có xác suất cao nhất."""
        return list(np.argsort(proba)[::-1][:k])

    def evaluate_on_day(
        self,
        proba: np.ndarray,
        actual_numbers: list[int],
        k: int = 10,
    ) -> dict:
        """
        Đánh giá kết quả dự báo cho một ngày.

        Returns:
            dict với keys: hit_in_top_k, top_k, proba_max, proba_min
        """
        top = self.top_k(proba, k)
        hits = [n for n in actual_numbers if n in top]
        return {
            "hit": len(hits) > 0,
            "hits": hits,
            "top_k": top,
            "n_hits": len(hits),
            "proba_max": float(np.max(proba)),
            "proba_min": float(np.min(proba)),
        }
