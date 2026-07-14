"""
PROBABILITY MODEL LAYER — src/probability/bayesian.py
Model 9: Bayesian Predictor

Khác với Conditional Probability (xét 1 điều kiện):
Bayesian kết hợp NHIỀU bằng chứng cùng lúc thông qua Bayesian updating.

Công thức:
    P(num | E1, E2, E3) ∝ P(E1|num) × P(E2|num) × P(E3|num) × P(num)

Bằng chứng được kết hợp:
    E1 = Hôm qua số nào ra (co-occurrence)
    E2 = Đầu câm hôm qua (head_cam)
    E3 = Đuôi câm hôm qua (tail_cam)
    E4 = Lô lộn (inverted) có ra không
    E5 = Số gương (mirror) có ra không
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseProbabilityModel


class BayesianPredictor(BaseProbabilityModel):
    """
    Model 9: Bayesian Predictor — Kết hợp đa bằng chứng.

    Sử dụng các features đã được tính sẵn trong FeatureStore
    (cond_prob_yesterday, cond_prob_head_cam, cond_prob_tail_cam, ...).
    Áp dụng Naive Bayes log-sum để kết hợp các likelihoods.

    Không cần train — tính giải tích hoàn toàn.
    """

    @property
    def name(self) -> str:
        return "bayesian_predictor"

    @property
    def version(self) -> str:
        return "1.0"

    def predict_proba(
        self,
        df_features: pd.DataFrame,
        df_history: Optional[pd.DataFrame] = None,
        S_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Kết hợp các conditional probability features theo Naive Bayes.

        Input: FeatureVector (100 rows) từ FeatureStore
        - Dùng: cond_prob_yesterday, cond_prob_head_cam, cond_prob_tail_cam,
                 bayesian_posterior, inverted_appeared_yesterday, mirror_appeared_yesterday
        Output: np.ndarray (100,) — xác suất kết hợp
        """
        n = len(df_features)

        # Lấy prior — dùng freq_30d nếu có, ngược lại uniform
        prior = df_features.get("freq_30d", pd.Series(np.full(n, 0.27))).values.astype(float)
        prior = np.clip(prior, 1e-6, 1 - 1e-6)

        # Log prior
        log_prob = np.log(prior)

        # --- Evidence 1: Conditional prob từ yesterday actives ---
        if "cond_prob_yesterday" in df_features.columns:
            cond_yest = df_features["cond_prob_yesterday"].values.astype(float)
            cond_yest = np.clip(cond_yest, 1e-6, 1 - 1e-6)
            # Log-likelihood ratio: log P(E|occur) - log P(E|uniform)
            log_prob += np.log(cond_yest) - np.log(0.27)

        # --- Evidence 2: Head câm ---
        if "cond_prob_head_cam" in df_features.columns:
            cond_head = df_features["cond_prob_head_cam"].values.astype(float)
            cond_head = np.clip(cond_head, 1e-6, 1 - 1e-6)
            # Trọng số nhẹ hơn evidence 1 (head câm kém đặc thù hơn)
            log_prob += 0.5 * (np.log(cond_head) - np.log(0.27))

        # --- Evidence 3: Tail câm ---
        if "cond_prob_tail_cam" in df_features.columns:
            cond_tail = df_features["cond_prob_tail_cam"].values.astype(float)
            cond_tail = np.clip(cond_tail, 1e-6, 1 - 1e-6)
            log_prob += 0.5 * (np.log(cond_tail) - np.log(0.27))

        # --- Evidence 4: Bonus nếu lô lộn xuất hiện hôm qua ---
        if "inverted_appeared_yesterday" in df_features.columns:
            inv_flag = df_features["inverted_appeared_yesterday"].values.astype(float)
            # Nếu lô lộn của số này ra hôm qua → boost nhẹ
            log_prob += 0.3 * inv_flag

        # --- Evidence 5: Bonus nếu số gương xuất hiện hôm qua ---
        if "mirror_appeared_yesterday" in df_features.columns:
            mir_flag = df_features["mirror_appeared_yesterday"].values.astype(float)
            log_prob += 0.2 * mir_flag

        # --- Evidence 6: Markov Order 1 ---
        if "markov_order1" in df_features.columns:
            markov = df_features["markov_order1"].values.astype(float)
            markov = np.clip(markov, 1e-6, 1 - 1e-6)
            log_prob += 0.7 * (np.log(markov) - np.log(0.27))

        # Convert log-prob → probability (softmax normalization không bắt buộc,
        # nhưng rescale để range [0,1] dễ compare)
        log_prob -= log_prob.max()  # numerical stability
        raw_prob = np.exp(log_prob)

        # Rescale: mỗi số vẫn là xác suất độc lập, không normalize về sum=1
        # Target range: [0, 1] với median ≈ 0.27
        median_raw = np.median(raw_prob)
        if median_raw > 0:
            scaled = raw_prob / median_raw * 0.27
        else:
            scaled = raw_prob

        return np.clip(scaled, 0.0, 1.0)
