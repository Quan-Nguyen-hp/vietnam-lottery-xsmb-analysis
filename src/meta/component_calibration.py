"""Nested calibration riêng cho từng component trước MetaFusion."""
from __future__ import annotations

import numpy as np

from .calibration import ProbabilityCalibrator


def calibration_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Composite score thống nhất với production: Brier + LogLoss + ECE."""
    p = np.clip(np.asarray(probs, dtype=float).reshape(-1), 1e-7, 1.0 - 1e-7)
    y = np.asarray(labels, dtype=float).reshape(-1)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= lower) & (p < upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(y[mask].mean() - p[mask].mean()))
    return 0.5 * brier + 0.3 * logloss + 0.2 * ece


class ComponentCalibrationManager:
    """Fit Platt/Isotonic riêng, chọn winner trên split selection độc lập."""

    def __init__(self):
        self._calibrators: dict[str, ProbabilityCalibrator | None] = {}
        self.methods: dict[str, str] = {}
        self.selection_scores: dict[str, float] = {}

    def fit(
        self,
        raw_fit: dict[str, np.ndarray],
        fit_labels: np.ndarray,
        raw_selection: dict[str, np.ndarray],
        selection_labels: np.ndarray,
    ) -> "ComponentCalibrationManager":
        y_fit = np.asarray(fit_labels).reshape(-1)
        y_sel = np.asarray(selection_labels).reshape(-1)
        for name, fit_values in raw_fit.items():
            fit_flat = np.asarray(fit_values).reshape(-1)
            selection_flat = np.asarray(raw_selection[name]).reshape(-1)
            candidates: list[tuple[str, ProbabilityCalibrator | None, np.ndarray]] = [
                ("identity", None, selection_flat)
            ]
            for method in ("platt", "isotonic"):
                calibrator = ProbabilityCalibrator(method=method).fit(fit_flat, y_fit)
                candidates.append((method, calibrator, calibrator.calibrate(selection_flat)))
            winner_method, winner, winner_probs = min(
                candidates,
                key=lambda item: calibration_score(item[2], y_sel),
            )
            self._calibrators[name] = winner
            self.methods[name] = winner_method
            self.selection_scores[name] = calibration_score(winner_probs, y_sel)
        return self

    def calibrate(self, name: str, raw_probabilities: np.ndarray) -> np.ndarray:
        calibrator = self._calibrators.get(name)
        raw = np.asarray(raw_probabilities, dtype=float)
        if calibrator is None:
            return raw
        original_shape = raw.shape
        return calibrator.calibrate(raw.reshape(-1)).reshape(original_shape)

    def calibrate_dict(self, raw_probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: self.calibrate(name, values) for name, values in raw_probabilities.items()}
