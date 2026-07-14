"""
DECISION INTELLIGENCE — src/decision/confidence.py
Tính mức độ đồng thuận giữa các model thành phần.
Confidence cao = các model nhất trí. Confidence thấp = các model xung đột.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ConfidenceEngine:
    """
    Tính Confidence score từ sự đồng thuận giữa nhiều model.

    Ý nghĩa:
    - Confidence = 95%: tất cả models đều đồng ý → đáng tin
    - Confidence = 30%: models xung đột nhau → skip
    - Không phải "model chính xác bao nhiêu %",
      mà là "các models đồng nhất bao nhiêu %"
    """

    def compute(
        self,
        model_probas: dict[str, np.ndarray],
        meta_proba: np.ndarray,
    ) -> np.ndarray:
        """
        Tính confidence cho 100 số.

        Args:
            model_probas: {model_name: proba_array(100,)} từ 10 models Layer 4
            meta_proba: proba_array(100,) từ Meta Learner

        Returns:
            np.ndarray (100,) confidence score, range [0, 1]
        """
        if not model_probas:
            return np.ones(100) * 0.5

        # Stack tất cả model predictions
        stack = np.vstack(list(model_probas.values()))  # (n_models, 100)

        # Phương sai giữa các models (thấp = đồng thuận = confidence cao)
        variance = np.var(stack, axis=0)  # (100,)
        max_variance = np.max(variance) + 1e-8

        # Confidence = 1 - normalized_variance
        confidence_raw = 1.0 - (variance / max_variance)

        # Bonus confidence khi meta_proba đồng ý với majority vote
        median_proba = np.median(stack, axis=0)
        agreement = 1.0 - np.abs(meta_proba - median_proba)

        # Kết hợp: 70% variance-based + 30% agreement-based
        confidence = 0.7 * confidence_raw + 0.3 * agreement

        return np.clip(confidence, 0.0, 1.0)

    def agreement_matrix(
        self, model_probas: dict[str, np.ndarray], top_k: int = 10
    ) -> pd.DataFrame:
        """
        Tạo ma trận đồng thuận: mỗi model vote top-K → đếm số model đồng ý.
        Hữu ích cho debugging và reporting.
        """
        votes = np.zeros(100, dtype=int)
        for name, proba in model_probas.items():
            top = np.argsort(proba)[::-1][:top_k]
            votes[top] += 1

        return pd.DataFrame({
            "number": range(100),
            "vote_count": votes,
            "vote_ratio": votes / max(len(model_probas), 1),
        })
