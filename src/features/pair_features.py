"""
FEATURE LAYER — src/features/pair_features.py
Trích xuất đặc trưng từ cặp số (Mirror, Inverted/Lộn, Twin, Distance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor


class PairFeatureExtractor(BaseFeatureExtractor):
    name = "pair_features"
    version = "1.0"

    def extract(self, df_evidence: pd.DataFrame) -> pd.DataFrame:
        results = []

        for _, row in df_evidence.iterrows():
            num = int(row["number"])
            f = {}

            head = int(row["head"])
            tail = int(row["tail"])
            inverted = int(row["inverted"])
            mirror = int(row["mirror"])

            # Basic structural properties
            f["is_twin"] = int(row["is_twin"])
            f["head_digit"] = head
            f["tail_digit"] = tail
            f["digit_sum"] = head + tail
            f["digit_diff"] = abs(head - tail)
            f["digit_product"] = head * tail

            # Distance metrics
            f["dist_to_inverted"] = abs(num - inverted)
            f["dist_to_mirror"] = abs(num - mirror)
            f["dist_to_100"] = 100 - num  # distance to end of range

            # Number in same family
            f["same_head_count"] = 10  # always 10 numbers share the same head
            f["same_tail_count"] = 10  # always 10 numbers share the same tail

            # Parity features
            f["is_even"] = int(num % 2 == 0)
            f["head_even"] = int(head % 2 == 0)
            f["tail_even"] = int(tail % 2 == 0)

            # Position in head group
            f["position_in_head"] = tail  # position within X0-X9

            results.append(f)

        return pd.DataFrame(results, index=df_evidence.index)
