"""
META LEARNING LAYER — src/meta/regime_detector.py
Bộ phát hiện trạng thái quay thưởng thị trường (Regime-Switching Detector).
Phân tích lịch sử gần nhất để xác định thị trường đang thuộc chế độ nào:
- REPEAT: Tỷ lệ lô rơi lại cao
- KHAN: Tỷ lệ bùng nổ lô khan lâu ngày cao
- BALANCED: Tần suất phân bổ cân bằng / Bạc nhớ
"""
from __future__ import annotations

from enum import Enum
import numpy as np


class MarketRegime(str, Enum):
    REPEAT = "REPEAT"        # Chế độ Lô Rơi (nổ lại từ ngày trước)
    KHAN = "KHAN"            # Chế độ Bùng nổ Lô Khan (khan >= 15 ngày nổ)
    BALANCED = "BALANCED"    # Chế độ Phân bổ Cân bằng / Bạc nhớ


class RegimeDetector:
    """Xác định trạng thái thị trường XSMB dựa trên ma trận nhị phân S_history."""

    def __init__(self, repeat_threshold: float = 0.28, khan_threshold: float = 0.08):
        """
        Args:
            repeat_threshold: Ngưỡng tỷ lệ lô rơi để xếp vào REPEAT regime (mặc định 0.28).
            khan_threshold: Ngưỡng tỷ lệ nổ lô khan để xếp vào KHAN regime (mặc định 0.08).
        """
        self.repeat_threshold = repeat_threshold
        self.khan_threshold = khan_threshold

    def detect(self, S_history: np.ndarray, window_days: int = 30) -> tuple[MarketRegime, dict[str, float]]:
        """
        Phân tích cửa sổ window_days ngày gần nhất trong S_history.

        Args:
            S_history: Ma trận nhị phân (N, 100).
            window_days: Số ngày gần nhất để phân tích (mặc định 30).

        Returns:
            Tuple của (MarketRegime, dictionary các chỉ số đo lường).
        """
        if S_history is None or len(S_history) < 2:
            return MarketRegime.BALANCED, {"repeat_rate": 0.27, "khan_rate": 0.10}

        window = min(window_days, len(S_history) - 1)
        sample_S = S_history[-(window + 1):]

        # 1. Tính Tỷ lệ Lô Rơi (Repeat Rate) trong window_days ngày
        # Repeat = số nổ hôm nay mà hôm qua cũng nổ
        repeats = (sample_S[:-1] * sample_S[1:]).sum(axis=1)  # (window,)
        active_counts = sample_S[1:].sum(axis=1)             # (window,)
        repeat_rate = float(np.mean(np.divide(repeats, np.maximum(active_counts, 1))))

        # 2. Tính Tỷ lệ Lô Khan Nổ (Khan Breakout Rate)
        # Tính khoảng trễ delay khan tại từng ngày
        khan_hits = 0
        total_hits = 0

        for i in range(1, len(sample_S)):
            past = sample_S[:i]
            # Tính delay của 100 số tại ngày i-1
            delays = np.zeros(100, dtype=int)
            for num in range(100):
                appeared = np.where(past[:, num] == 1)[0]
                if len(appeared) > 0:
                    delays[num] = i - 1 - appeared[-1]
                else:
                    delays[num] = i

            actives_today = np.where(sample_S[i] == 1)[0]
            if len(actives_today) > 0:
                khan_count = np.sum(delays[actives_today] >= 15)
                khan_hits += khan_count
                total_hits += len(actives_today)

        khan_rate = float(khan_hits / total_hits) if total_hits > 0 else 0.10

        # Phân loại Regime
        if repeat_rate >= self.repeat_threshold:
            regime = MarketRegime.REPEAT
        elif khan_rate >= self.khan_threshold:
            regime = MarketRegime.KHAN
        else:
            regime = MarketRegime.BALANCED

        metrics = {
            "repeat_rate": round(repeat_rate, 4),
            "khan_rate": round(khan_rate, 4),
            "window_days": window,
        }
        return regime, metrics
