"""
META LEARNING LAYER — src/meta/calibration.py
Hiệu chỉnh xác suất đầu ra để khớp với tần suất thực tế.
Hỗ trợ Platt Scaling (Logistic) và Isotonic Regression.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve
    _sklearn_available = True
except ImportError:
    _sklearn_available = False


class ProbabilityCalibrator:
    """
    Hiệu chỉnh xác suất từ Meta Learner.

    Ví dụ:
        Model raw: 0.71  →  Calibrated: 0.66
        Tức là khi model nói 71%, thực tế chỉ trúng 66% lần.

    Phương pháp:
    - 'platt': Logistic regression (tốt cho output lệch về 0 hoặc 1)
    - 'isotonic': Isotonic regression (linh hoạt hơn, cần nhiều data hơn)
    """

    def __init__(self, method: Literal["platt", "isotonic"] = "isotonic"):
        if not _sklearn_available:
            raise ImportError("scikit-learn cần thiết cho Calibration. pip install scikit-learn")
        self._method = method
        self._calibrator = None
        self._is_trained = False

    def fit(self, proba_raw: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """
        Huấn luyện calibrator.

        Args:
            proba_raw: Xác suất thô từ model (shape: N,)
            y_true: Label thực tế 0/1 (shape: N,)
        """
        proba_raw = np.clip(proba_raw, 1e-6, 1 - 1e-6)

        if self._method == "platt":
            self._calibrator = LogisticRegression(C=1.0)
            self._calibrator.fit(proba_raw.reshape(-1, 1), y_true)
        else:
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(proba_raw, y_true)

        self._is_trained = True
        return self

    def calibrate(self, proba_raw: np.ndarray) -> np.ndarray:
        """
        Áp dụng calibration lên xác suất thô.

        Returns:
            np.ndarray — xác suất đã được hiệu chỉnh
        """
        if not self._is_trained:
            return proba_raw  # passthrough nếu chưa train

        proba_raw = np.clip(proba_raw, 1e-6, 1 - 1e-6)

        if self._method == "platt":
            return self._calibrator.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
        else:
            return self._calibrator.predict(proba_raw)

    def reliability_diagram(
        self, proba_raw: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> dict:
        """
        Tính dữ liệu để vẽ Reliability Diagram.

        Returns:
            dict với 'fraction_of_positives' và 'mean_predicted_value'
        """
        frac, mean_pred = calibration_curve(y_true, proba_raw, n_bins=n_bins)
        return {
            "fraction_of_positives": frac.tolist(),
            "mean_predicted_value": mean_pred.tolist(),
        }

    def ece_score(
        self, proba_raw: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> float:
        """Expected Calibration Error — càng nhỏ càng tốt."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for i in range(n_bins):
            mask = (proba_raw >= bins[i]) & (proba_raw < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = proba_raw[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)
        return float(ece)
