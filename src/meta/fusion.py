"""
META FUSION LAYER — src/meta/fusion.py
Triển khai bộ tổng hợp xác suất (Meta Fusion) dựa trên đánh giá chất lượng dự báo thực tế.

Công thức tính trọng số động:
    Quality_Score = 0.35 * (1.0 - Brier) + 0.35 * exp(-LogLoss) + 0.20 * Precision@10 + 0.10 * max(0.0, ROI)
    Trọng số = Quality_Score / Sum(Quality_Scores)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class MetaFusion:
    """
    Tổng hợp xác suất đa mô hình động dựa trên chất lượng dự báo thực tế.
    """

    def __init__(self):
        self._weights: dict[str, float] = {}

    def compute_dynamic_weights(
        self,
        historical_predictions: dict[str, np.ndarray],  # {model_name: matrix (n_days, 100)}
        historical_labels: np.ndarray,                  # matrix (n_days, 100)
    ) -> dict[str, float]:
        """
        Tính toán trọng số động cho từng mô hình từ kết quả lịch sử.
        """
        n_days = len(historical_labels)
        if n_days == 0:
            return {}

        scores = {}
        for name, preds in historical_predictions.items():
            # 1. Brier Score (lower is better, range [0, 1])
            brier = float(np.mean((preds - historical_labels) ** 2))
            
            # 2. Log Loss (lower is better, range [0, inf])
            p_clipped = np.clip(preds, 1e-7, 1.0 - 1e-7)
            logloss = float(-np.mean(historical_labels * np.log(p_clipped) + (1.0 - historical_labels) * np.log(1.0 - p_clipped)))
            
            # 3. Precision @ 10
            precisions = []
            for t in range(n_days):
                top10 = np.argsort(preds[t])[::-1][:10]
                actuals = np.where(historical_labels[t] > 0)[0]
                hits = len([n for n in top10 if n in actuals])
                precisions.append(hits / 10.0)
            precision_10 = float(np.mean(precisions))
            
            # 4. ROI giả lập (odds=3.666 cho Lô)
            total_bets = n_days * 10
            total_hits = 0
            for t in range(n_days):
                top10 = np.argsort(preds[t])[::-1][:10]
                actual_counts = historical_labels[t]
                total_hits += sum(actual_counts[n] for n in top10)
            
            # cost = bets * 27, payout = hits * 99
            cost = total_bets * 27.0
            payout = total_hits * 99.0
            roi = float((payout - cost) / cost) if cost > 0 else 0.0

            # Tính điểm chất lượng
            brier_term = 1.0 - brier
            logloss_term = float(np.exp(-logloss))
            roi_term = max(0.0, roi)
            
            quality_score = 0.35 * brier_term + 0.35 * logloss_term + 0.20 * precision_10 + 0.10 * roi_term
            scores[name] = max(quality_score, 1e-5)

        # Chuẩn hóa trọng số
        total_score = sum(scores.values())
        self._weights = {k: v / total_score for k, v in scores.items()}
        return self._weights

    def fuse(self, model_probas: dict[str, np.ndarray]) -> np.ndarray:
        """
        Tổng hợp xác suất đầu ra sử dụng trọng số động.
        """
        if not model_probas:
            return np.full(100, 0.27)

        # Fallback về uniform weights nếu chưa tính được trọng số động
        if not self._weights:
            names = list(model_probas.keys())
            self._weights = {name: 1.0 / len(names) for name in names}

        fused = np.zeros(100)
        total_w = 0.0

        for name, proba in model_probas.items():
            w = self._weights.get(name, 0.0)
            fused += w * proba
            total_w += w

        if total_w > 0:
            fused /= total_w
        else:
            fused = np.mean(list(model_probas.values()), axis=0)

        return np.clip(fused, 0.0, 1.0)

    @property
    def weights(self) -> dict[str, float]:
        return self._weights
