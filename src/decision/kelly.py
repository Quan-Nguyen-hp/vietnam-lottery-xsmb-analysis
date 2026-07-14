"""
DECISION INTELLIGENCE — src/decision/kelly.py
Kelly Criterion: tính tỷ lệ vốn tối ưu cho mỗi số.
Giúp tối đa hóa tăng trưởng dài hạn, tránh overbet.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class KellyCriterion:
    """
    Tính fraction vốn tối ưu theo Kelly Criterion.

    Công thức Kelly:
        f* = (p * b - q) / b
        f* = (p * b - (1 - p)) / b

    Trong đó:
        p  = xác suất thắng (từ Meta Learner, đã calibrated)
        b  = odds - 1 (ví dụ: lô bao 1 ăn 70 → b=69)
        q  = 1 - p

    Fractional Kelly:
        f = f* * fraction (fraction < 1 để giảm rủi ro)
    """

    def __init__(
        self,
        odds: float = 70.0,        # 1 ăn 70 (standard lô bao)
        kelly_fraction: float = 0.25,  # 1/4 Kelly để an toàn
        max_fraction: float = 0.10,    # Không bao giờ đặt > 10% vốn vào 1 số
        min_prob: float = 0.30,        # Không đặt nếu xác suất < 30%
    ):
        self._odds = odds
        self._b = odds - 1.0
        self._fraction = kelly_fraction
        self._max_fraction = max_fraction
        self._min_prob = min_prob

    def compute(
        self,
        proba: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Tính Kelly fraction cho 100 số.

        Args:
            proba:      xác suất calibrated (100,)
            confidence: confidence score (100,), optional
                        Nếu có → điều chỉnh odds ảo theo confidence

        Returns:
            np.ndarray (100,) — fraction vốn [0, max_fraction]
        """
        p = np.clip(proba, 1e-6, 1 - 1e-6)
        q = 1.0 - p

        # Nếu có confidence → dùng "effective odds"
        # Confidence thấp → giảm odds ảo → Kelly nhỏ lại
        if confidence is not None:
            effective_b = self._b * np.clip(confidence, 0.1, 1.0)
        else:
            effective_b = np.full(100, self._b)

        # Kelly formula
        kelly_raw = (p * effective_b - q) / (effective_b + 1e-8)

        # Fractional Kelly
        kelly_scaled = kelly_raw * self._fraction

        # Không đặt vào số có probability thấp
        kelly_scaled[p < self._min_prob] = 0.0

        # Không đặt nếu Kelly âm (expected value âm)
        kelly_scaled = np.maximum(kelly_scaled, 0.0)

        # Cap tại max_fraction
        kelly_scaled = np.minimum(kelly_scaled, self._max_fraction)

        return kelly_scaled

    def capital_allocation(
        self,
        kelly_fractions: np.ndarray,
        total_capital: float,
        top_k: int = 10,
    ) -> dict:
        """
        Tính số tiền cụ thể cần đặt cho từng số trong top-K.

        Returns:
            dict: {number: amount_to_bet}
        """
        top_indices = np.argsort(kelly_fractions)[::-1][:top_k]
        allocation = {}
        for idx in top_indices:
            if kelly_fractions[idx] > 0:
                allocation[int(idx)] = round(kelly_fractions[idx] * total_capital, 2)
        return allocation
