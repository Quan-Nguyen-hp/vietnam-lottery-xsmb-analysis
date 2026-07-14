"""
DATA LAYER — src/data/loader.py
Chịu trách nhiệm duy nhất: đọc, validate và trả ra DataFrame sạch.
Không thực hiện bất kỳ phép tính thống kê hay dự báo nào.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class DataLoader:
    """
    Tải và chuẩn bị dữ liệu XSMB từ CSV.

    Nguyên tắc:
    - Chỉ đọc và trả dữ liệu sạch.
    - Không thực hiện tính toán thống kê.
    - Là điểm vào duy nhất cho raw data trong XPIS.
    """

    DEFAULT_CSV = Path(__file__).parent.parent.parent / "data" / "xsmb-2-digits.csv"

    def __init__(self, csv_path: Optional[Path] = None):
        self._csv_path = csv_path or self.DEFAULT_CSV
        self._df: Optional[pd.DataFrame] = None
        self._S: Optional[np.ndarray] = None

    def load(self) -> "DataLoader":
        """Tải dữ liệu từ CSV và sắp xếp theo ngày tăng dần."""
        df = pd.read_csv(self._csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self._df = df
        self._S = self._build_sparse_matrix(df)
        return self

    def _build_sparse_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Tạo ma trận nhị phân S[day, num] từ DataFrame."""
        prize_cols = [c for c in df.columns if c != "date"]
        N = len(df)
        S = np.zeros((N, 100), dtype=np.int8)
        arr = df[prize_cols].values.astype(int)
        rows = np.repeat(np.arange(N), arr.shape[1])
        cols = arr.flatten()
        valid = (cols >= 0) & (cols < 100)
        S[rows[valid], cols[valid]] = 1
        return S

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Chưa tải dữ liệu. Gọi .load() trước.")
        return self._df

    @property
    def S(self) -> np.ndarray:
        if self._S is None:
            raise RuntimeError("Chưa tải dữ liệu. Gọi .load() trước.")
        return self._S

    @property
    def total_days(self) -> int:
        return len(self.df)

    def slice_history(self, up_to_idx: int) -> tuple[pd.DataFrame, np.ndarray]:
        """Trả ra lịch sử trước ngày có index `up_to_idx` (exclusive)."""
        return self.df.iloc[:up_to_idx].reset_index(drop=True), self.S[:up_to_idx]

    def prize_cols(self) -> list[str]:
        return [c for c in self.df.columns if c != "date"]
