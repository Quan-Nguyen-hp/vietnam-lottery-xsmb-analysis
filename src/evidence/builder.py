"""
EVIDENCE LAYER — src/evidence/builder.py
Xây dựng EvidenceObject cho 100 số từ lịch sử tại một thời điểm.
Không tính điểm, không dự báo — chỉ tổng hợp raw observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import EvidenceObject
from .store import EvidenceStore


class EvidenceBuilder:
    """
    Xây dựng Evidence snapshot cho 100 số tại ngày target_date.

    Nguyên tắc:
    - Chỉ dùng dữ liệu lịch sử TRƯỚC target_date (không data leakage).
    - Output là raw observation, không phải computed feature.
    - Hỗ trợ build trực tiếp hoặc build+lưu qua EvidenceStore.
    """

    def __init__(self, store: Optional[EvidenceStore] = None):
        self._store = store or EvidenceStore()

    def build_all(
        self,
        df_history: pd.DataFrame,
        S_history: np.ndarray,
        target_date: datetime,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Xây dựng Evidence DataFrame (100 rows) cho target_date.

        Args:
            df_history: DataFrame lịch sử TRƯỚC target_date
            S_history: Ma trận nhị phân (N, 100) tương ứng
            target_date: Ngày cần tạo evidence
            save: Có lưu vào EvidenceStore không

        Returns:
            DataFrame 100 rows, mỗi row là evidence của 1 số
        """
        date_str = target_date.strftime("%Y-%m-%d")
        N = len(df_history)

        # Kiểm tra cache
        if self._store.exists(date_str):
            cached = self._store.load(date_str)
            if cached is not None:
                return cached

        rows = []
        for num in range(100):
            ev = self._build_one(num, df_history, S_history, target_date, N)
            rows.append(ev)

        df_ev = pd.DataFrame(rows)

        if save:
            self._store.save(date_str, df_ev)

        return df_ev

    def _build_one(
        self,
        num: int,
        df_history: pd.DataFrame,
        S_history: np.ndarray,
        target_date: datetime,
        N: int,
    ) -> dict:
        """Xây dựng evidence dict cho một số."""
        date_str = target_date.strftime("%Y-%m-%d")

        # --- Delays ---
        appeared_days = np.where(S_history[:, num] == 1)[0]
        if len(appeared_days) == 0:
            current_delay = N + 1
            historical_delays = []
        else:
            gaps = [int(appeared_days[0] + 1)]
            if len(appeared_days) > 1:
                gaps += np.diff(appeared_days).astype(int).tolist()
            current_delay = int(N - appeared_days[-1])
            historical_delays = gaps

        # --- Frequency counts ---
        def count_in_window(w: int) -> int:
            return int(S_history[-w:, num].sum()) if N >= w else int(S_history[:, num].sum())

        # --- Markov states ---
        state_yesterday = int(S_history[-1, num]) if N >= 1 else 0
        state_day_before = int(S_history[-2, num]) if N >= 2 else 0

        # --- Number properties ---
        head = num // 10
        tail = num % 10
        is_twin = (head == tail)
        inv_head, inv_tail = tail, head
        inverted = inv_head * 10 + inv_tail
        mir_head = (head + 5) % 10
        mir_tail = (tail + 5) % 10
        mirror = mir_head * 10 + mir_tail

        # --- Yesterday context ---
        yesterday_actives: list[int] = []
        yesterday_head_cam: list[int] = []
        yesterday_tail_cam: list[int] = []

        if N >= 1:
            yesterday_actives = np.where(S_history[-1] == 1)[0].tolist()
            appeared_heads = set(np.where(S_history[-1] == 1)[0] // 10)
            appeared_tails = set(np.where(S_history[-1] == 1)[0] % 10)
            yesterday_head_cam = [d for d in range(10) if d not in appeared_heads]
            yesterday_tail_cam = [d for d in range(10) if d not in appeared_tails]

        # --- Time context ---
        weekday = target_date.weekday()
        day_of_month = target_date.day
        month = target_date.month
        is_weekend = weekday in (5, 6)

        return {
            "number": num,
            "date": date_str,
            "history_days": N,
            # Delay
            "current_delay": current_delay,
            "historical_delays": historical_delays,
            # Frequency
            "count_3d": count_in_window(3),
            "count_7d": count_in_window(7),
            "count_14d": count_in_window(14),
            "count_30d": count_in_window(30),
            "count_60d": count_in_window(60),
            "count_90d": count_in_window(90),
            "count_120d": count_in_window(120),
            "count_180d": count_in_window(180),
            "count_365d": count_in_window(365),
            "count_all": int(S_history[:, num].sum()),
            # Markov
            "state_yesterday": state_yesterday,
            "state_day_before": state_day_before,
            # Number properties
            "head": head,
            "tail": tail,
            "is_twin": is_twin,
            "inverted": inverted,
            "mirror": mirror,
            # Context
            "yesterday_actives": yesterday_actives,
            "yesterday_head_cam": yesterday_head_cam,
            "yesterday_tail_cam": yesterday_tail_cam,
            # Time
            "weekday": weekday,
            "day_of_month": day_of_month,
            "month": month,
            "is_weekend": is_weekend,
        }
