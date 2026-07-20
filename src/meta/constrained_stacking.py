"""Constrained stacking research challenger cho MetaFusion."""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


class ConstrainedStacking:
    """Tối ưu trọng số simplex, regularize về uniform để chống overfit."""

    def __init__(self, regularization: float = 0.10, max_iter: int = 300):
        self.regularization = regularization
        self.max_iter = max_iter
        self.weights: dict[str, float] = {}
        self.objective_value: float | None = None

    @staticmethod
    def _objective(weights: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> float:
        p = np.clip(np.tensordot(X, weights, axes=([2], [0])), 1e-6, 1.0 - 1e-6)
        logloss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        brier = np.mean((p - y) ** 2)
        uniform = np.full_like(weights, 1.0 / len(weights))
        return float(0.5 * logloss + 0.5 * brier + reg * np.sum((weights - uniform) ** 2))

    def fit(
        self,
        model_predictions: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> "ConstrainedStacking":
        if not model_predictions:
            raise ValueError("model_predictions không được rỗng")
        names = list(model_predictions)
        arrays = [np.asarray(model_predictions[name], dtype=float) for name in names]
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays):
            raise ValueError("Các model predictions phải cùng shape")
        y = np.asarray(labels, dtype=float)
        if y.shape != shape:
            raise ValueError("labels phải cùng shape với model predictions")

        X = np.stack(arrays, axis=2)
        n_models = len(names)
        initial = np.full(n_models, 1.0 / n_models)
        if minimize is None:
            solution = initial
            objective_value = self._objective(solution, X, y, self.regularization)
        else:
            result = minimize(
                self._objective,
                initial,
                args=(X, y, self.regularization),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n_models,
                constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
                options={"maxiter": self.max_iter, "ftol": 1e-8},
            )
            solution = result.x if result.success else initial
            objective_value = self._objective(solution, X, y, self.regularization)

        solution = np.clip(solution, 0.0, 1.0)
        solution /= solution.sum()
        self.weights = {name: float(weight) for name, weight in zip(names, solution)}
        self.objective_value = float(objective_value)
        return self

    def fuse(self, model_predictions: dict[str, np.ndarray]) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("Cần fit trước khi fuse")
        return sum(
            self.weights[name] * np.asarray(values, dtype=float)
            for name, values in model_predictions.items()
            if name in self.weights
        )
