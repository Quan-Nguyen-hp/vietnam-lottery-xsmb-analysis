"""
FEATURE LAYER — src/features/base.py
Interface chuẩn cho tất cả FeatureExtractor trong XPIS.
Feature được derive từ Evidence — không bao giờ đọc raw data trực tiếp.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureExtractor(ABC):
    """
    Interface chuẩn cho Feature Extractor.

    Nguyên tắc:
    - Input: DataFrame evidence (100 rows, từ EvidenceStore)
    - Output: DataFrame feature columns (100 rows, các cột mới)
    - Không đọc raw data, không có side effects.
    - Mỗi Extractor chịu trách nhiệm một nhóm feature cụ thể.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên định danh của extractor (dùng để debug & logging)."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version của extractor (ví dụ: '1.0', '2.1')."""
        ...

    @abstractmethod
    def extract(self, df_evidence: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất features từ Evidence DataFrame.

        Args:
            df_evidence: DataFrame 100 rows từ EvidenceBuilder/Store.
                         Mỗi row là evidence của 1 số (00-99).

        Returns:
            DataFrame cùng index (100 rows), các cột là feature mới.
            Không bao gồm lại các cột từ df_evidence.
        """
        ...

    def feature_names(self, df_evidence: pd.DataFrame) -> list[str]:
        """Trả về danh sách tên feature mà extractor này tạo ra."""
        sample = self.extract(df_evidence)
        return list(sample.columns)
