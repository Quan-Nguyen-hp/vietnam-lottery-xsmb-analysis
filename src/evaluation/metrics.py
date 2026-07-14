"""
EVALUATION LAYER — src/evaluation/metrics.py
Thư viện tính toán các chỉ số đánh giá (Evaluation Metrics) nâng cao trong XPIS v1.1.
Tích hợp: ROI, WinRate, Brier Score, Log Loss, AUC-ROC, ECE, Precision@K, Recall@K.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

try:
    from sklearn.metrics import roc_auc_score
    _sklearn_available = True
except ImportError:
    _sklearn_available = False


class EvaluationMetrics:
    """
    Tính toán các chỉ số kiểm định chất lượng dự báo và lợi nhuận tài chính.
    """

    def __init__(self, odds: float = 3.666, cost_per_bet: float = 27.0):
        self._odds = odds
        self._cost = cost_per_bet

    def roi(self, results: pd.DataFrame) -> float:
        """Tính ROI thực tế."""
        total_bets = results["n_bets"].sum()
        total_hits = results["n_hits"].sum()
        total_cost = total_bets * self._cost
        total_win = total_hits * self._odds * self._cost
        if total_cost == 0:
            return 0.0
        return float((total_win - total_cost) / total_cost)

    def brier_score(self, proba_all: np.ndarray, y_all: np.ndarray) -> float:
        """Độ lệch bình phương trung bình (Brier Score) — càng thấp càng tốt."""
        return float(np.mean((proba_all - y_all) ** 2))

    def log_loss(self, proba_all: np.ndarray, y_all: np.ndarray) -> float:
        """Hàm mất mát entropy chéo (Log Loss) — càng thấp càng tốt."""
        p = np.clip(proba_all, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y_all * np.log(p) + (1.0 - y_all) * np.log(1.0 - p)))

    def auc_roc(self, proba_all: np.ndarray, y_all: np.ndarray) -> float:
        """Chỉ số diện tích dưới đường cong ROC (AUC-ROC) — cao là tốt."""
        if not _sklearn_available:
            return 0.5
        try:
            # Nếu tất cả nhãn đều giống nhau (chỉ 0 hoặc chỉ 1), AUC không xác định được
            if len(np.unique(y_all)) < 2:
                return 0.5
            return float(roc_auc_score(y_all, proba_all))
        except Exception:
            return 0.5

    def ece_score(self, proba_all: np.ndarray, y_all: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error (Sai số hiệu chuẩn kỳ vọng) — càng thấp càng tốt."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_all)
        if n == 0:
            return 0.0
            
        for i in range(n_bins):
            mask = (proba_all >= bins[i]) & (proba_all < bins[i + 1])
            bin_size = np.sum(mask)
            if bin_size == 0:
                continue
            acc = float(y_all[mask].mean())
            conf = float(proba_all[mask].mean())
            ece += (bin_size / n) * abs(acc - conf)
        return float(ece)

    def precision_at_k(self, proba_matrix: np.ndarray, y_matrix: np.ndarray, k: int = 10) -> float:
        """Độ chính xác xếp hạng nhóm K (Precision @ K)."""
        n_days = len(proba_matrix)
        if n_days == 0:
            return 0.0

        precisions = []
        for t in range(n_days):
            top_k = np.argsort(proba_matrix[t])[::-1][:k]
            actual_positives = np.where(y_matrix[t] > 0)[0]
            hits = len([n for n in top_k if n in actual_positives])
            precisions.append(hits / float(k))
        return float(np.mean(precisions))

    def recall_at_k(self, proba_matrix: np.ndarray, y_matrix: np.ndarray, k: int = 10) -> float:
        """Tỷ lệ bắt trúng nhóm K (Recall @ K) so với tổng số dương tính thực tế."""
        n_days = len(proba_matrix)
        if n_days == 0:
            return 0.0

        recalls = []
        for t in range(n_days):
            top_k = np.argsort(proba_matrix[t])[::-1][:k]
            actual_positives = np.where(y_matrix[t] > 0)[0]
            if len(actual_positives) == 0:
                continue
            hits = len([n for n in top_k if n in actual_positives])
            recalls.append(hits / float(len(actual_positives)))
        return float(np.mean(recalls)) if recalls else 0.0

    def max_drawdown(self, cumulative_pnl: np.ndarray) -> float:
        """Độ sụt giảm tài khoản lớn nhất từ đỉnh (MDD)."""
        if len(cumulative_pnl) == 0:
            return 0.0
        peak = np.maximum.accumulate(cumulative_pnl)
        # Tránh chia cho 0
        peak_safe = np.where(np.abs(peak) > 0, peak, 1.0)
        drawdowns = (cumulative_pnl - peak) / np.abs(peak_safe)
        return float(np.min(drawdowns))

    def compute_full(
        self,
        results: pd.DataFrame,
        proba_history: Optional[np.ndarray] = None,   # matrix (n_days, 100)
        y_history: Optional[np.ndarray] = None,       # matrix (n_days, 100)
    ) -> dict:
        """Tổng hợp đầy đủ các chỉ số đánh giá."""
        n_days = len(results)
        hit_rate = float(results["hit"].mean()) if n_days > 0 else 0.0

        metrics = {
            "n_days": n_days,
            "hit_rate": hit_rate,
            "roi": self.roi(results),
            "total_bets": int(results["n_bets"].sum()),
            "total_hits": int(results["n_hits"].sum()),
        }

        # Nếu có ma trận xác suất và nhãn lịch sử đầy đủ
        if proba_history is not None and y_history is not None:
            p_flat = proba_history.flatten()
            y_flat = y_history.flatten()
            metrics["brier_score"] = self.brier_score(p_flat, y_flat)
            metrics["log_loss"] = self.log_loss(p_flat, y_flat)
            metrics["auc_roc"] = self.auc_roc(p_flat, y_flat)
            metrics["ece"] = self.ece_score(p_flat, y_flat)
            metrics["precision_10"] = self.precision_at_k(proba_history, y_history, k=10)
            metrics["recall_10"] = self.recall_at_k(proba_history, y_history, k=10)

        # Tính chuỗi thắng liên tiếp lớn nhất
        hits = results["hit"].values if "hit" in results.columns else []
        max_streak = curr_streak = 0
        for h in hits:
            if h:
                curr_streak += 1
                max_streak = max(max_streak, curr_streak)
            else:
                curr_streak = 0
        metrics["max_win_streak"] = max_streak

        return metrics
