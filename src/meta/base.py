"""
META LEARNING LAYER — src/meta/base.py
Interface cho Meta Learner — nơi duy nhất tổng hợp các model thành phần.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseMetaLearner(ABC):
    """
    Interface chuẩn cho Layer 5 — Meta Learning Layer.

    Nguyên tắc:
    - Input: FeatureVector (100 rows × 250 cols) + predictions từ 10 models
    - Output: Probability (100,) + Confidence (100,) + Rank (100,)
    - Là nơi DUY NHẤT kết hợp các model thành phần.
    - Không tự tính feature, không đọc raw data.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> None:
        """Huấn luyện meta learner từ feature matrix và label."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Trả xác suất đã được học (calibrated).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray (n_samples,) — xác suất đã học
        """
        ...

    def predict_batch(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Tiện ích: predict từ FeatureVector DataFrame.
        Bỏ qua cột 'number' và 'date' trước khi predict.
        """
        meta_cols = [c for c in df_features.columns if c not in ("number", "date")]
        X = df_features[meta_cols].values.astype(np.float32)
        return self.predict_proba(X)

    def is_trained(self) -> bool:
        """Kiểm tra model đã được train chưa."""
        return False

    def save(self, path: str) -> None:
        """Lưu model ra file."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Tải model từ file."""
        raise NotImplementedError
