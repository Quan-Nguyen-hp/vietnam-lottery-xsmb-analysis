"""
FEATURE LAYER — src/features/bayesian_features.py
Trích xuất đặc trưng xác suất có điều kiện (Bayesian / Bạc Nhớ).
Input: df_evidence với yesterday_actives, yesterday_head_cam, yesterday_tail_cam.
Output: cond_prob_yesterday, cond_prob_head_cam, cond_prob_tail_cam,
        cond_prob_inverted, cond_prob_mirror, bayesian_posterior
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor


class BayesianFeatureExtractor(BaseFeatureExtractor):
    name = "bayesian_features"
    version = "1.0"

    def __init__(self, S_history: np.ndarray):
        """
        Args:
            S_history: Ma trận nhị phân (N, 100) để tính co-occurrence.
        """
        self._S = S_history
        self._precompute()

    def _precompute(self) -> None:
        """Pre-compute transition and co-occurrence matrices."""
        S = self._S
        N = S.shape[0] if S is not None else 0

        if N > 1:
            # Co-occurrence: M[a, b] = P(b today | a yesterday)
            co_occur = S[:-1].T @ S[1:]   # (100, 100)
            freq_yest = S[:-1].sum(axis=0)  # (100,)
            self._M = (co_occur + 1) / (freq_yest[:, None] + 2)

            # Head câm co-occurrence: C_head[d, b] = P(b | head d câm yesterday)
            S_reshaped = S.reshape(N, 10, 10)
            head_cam = (S_reshaped.sum(axis=2) == 0).astype(int)  # (N, 10)
            tail_cam = (S_reshaped.sum(axis=1) == 0).astype(int)  # (N, 10)

            hc_co = head_cam[:-1].T @ S[1:]   # (10, 100)
            hc_cnt = head_cam[:-1].sum(axis=0)  # (10,)
            self._C_head = (hc_co + 1) / (hc_cnt[:, None] + 2)

            tc_co = tail_cam[:-1].T @ S[1:]   # (10, 100)
            tc_cnt = tail_cam[:-1].sum(axis=0)  # (10,)
            self._C_tail = (tc_co + 1) / (tc_cnt[:, None] + 2)
        else:
            self._M = np.ones((100, 100)) * 0.27
            self._C_head = np.ones((10, 100)) * 0.27
            self._C_tail = np.ones((10, 100)) * 0.27

    def extract(self, df_evidence: pd.DataFrame) -> pd.DataFrame:
        results = []

        for _, row in df_evidence.iterrows():
            num = int(row["number"])
            f = {}

            # Base frequency (prior)
            p_prior = float(row.get("freq_30d", 0.27)) if "freq_30d" in row.index else 0.27

            # P(num | yesterday_actives) — Bạc nhớ
            yesterday_actives = row.get("yesterday_actives", [])
            if yesterday_actives and self._S is not None:
                active_arr = np.zeros(100, dtype=np.int8)
                for a in yesterday_actives:
                    if 0 <= a < 100:
                        active_arr[a] = 1
                f["cond_prob_yesterday"] = float(
                    (active_arr @ self._M[:, num]) / (active_arr.sum() + 1e-5)
                )
            else:
                f["cond_prob_yesterday"] = p_prior

            # P(num | yesterday head câm)
            head_cam = row.get("yesterday_head_cam", [])
            if head_cam:
                f["cond_prob_head_cam"] = float(np.mean(self._C_head[head_cam, num]))
            else:
                f["cond_prob_head_cam"] = p_prior

            # P(num | yesterday tail câm)
            tail_cam = row.get("yesterday_tail_cam", [])
            if tail_cam:
                f["cond_prob_tail_cam"] = float(np.mean(self._C_tail[tail_cam, num]))
            else:
                f["cond_prob_tail_cam"] = p_prior

            # P(num | inverted appeared yesterday)
            inverted = int(row["inverted"])
            inv_in_yesterday = inverted in (yesterday_actives or [])
            f["inverted_appeared_yesterday"] = int(inv_in_yesterday)
            f["cond_prob_inverted"] = float(self._M[inverted, num]) if inv_in_yesterday else p_prior

            # P(num | mirror appeared yesterday)
            mirror = int(row["mirror"])
            mir_in_yesterday = mirror in (yesterday_actives or [])
            f["mirror_appeared_yesterday"] = int(mir_in_yesterday)
            f["cond_prob_mirror"] = float(self._M[mirror, num]) if mir_in_yesterday else p_prior

            # Bayesian posterior: combine prior + conditional evidence
            # Simple geometric mean of likelihoods
            cond_probs = [
                f["cond_prob_yesterday"],
                f["cond_prob_head_cam"],
                f["cond_prob_tail_cam"],
            ]
            f["bayesian_posterior"] = float(np.mean(cond_probs))

            results.append(f)

        return pd.DataFrame(results, index=df_evidence.index)
