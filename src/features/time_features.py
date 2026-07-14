"""
FEATURE LAYER — src/features/time_features.py
Trích xuất đặc trưng thời gian.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

# Tết Nguyên Đán thường rơi vào tháng 1-2
TET_MONTHS = {1, 2}


class TimeFeatureExtractor(BaseFeatureExtractor):
    name = "time_features"
    version = "1.0"

    def extract(self, df_evidence: pd.DataFrame) -> pd.DataFrame:
        results = []

        for _, row in df_evidence.iterrows():
            f = {}

            weekday = int(row["weekday"])    # 0=Mon...6=Sun
            day = int(row["day_of_month"])
            month = int(row["month"])

            f["weekday"] = weekday
            f["day_of_month"] = day
            f["month"] = month
            f["is_weekend"] = int(row["is_weekend"])

            # Week of month (1-5)
            f["week_of_month"] = (day - 1) // 7 + 1

            # Quarter (1-4)
            f["quarter"] = (month - 1) // 3 + 1

            # Cyclical encoding (captures circular nature of weekday/month)
            f["weekday_sin"] = float(np.sin(2 * np.pi * weekday / 7))
            f["weekday_cos"] = float(np.cos(2 * np.pi * weekday / 7))
            f["month_sin"] = float(np.sin(2 * np.pi * month / 12))
            f["month_cos"] = float(np.cos(2 * np.pi * month / 12))
            f["day_sin"] = float(np.sin(2 * np.pi * day / 31))
            f["day_cos"] = float(np.cos(2 * np.pi * day / 31))

            # Holiday flags
            f["is_near_tet"] = int(month in TET_MONTHS)
            f["is_start_of_month"] = int(day <= 5)
            f["is_end_of_month"] = int(day >= 25)
            f["is_monday"] = int(weekday == 0)
            f["is_friday"] = int(weekday == 4)

            results.append(f)

        return pd.DataFrame(results, index=df_evidence.index)
