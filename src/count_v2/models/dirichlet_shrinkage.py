from __future__ import annotations

import math
import numpy as np

from src.count_v2.contracts import (
    FORECAST_SUM_TOLERANCE,
    ChronologyError,
    CountForecast,
    CountHistory,
    ModelContractError,
    require_target_after_history,
)


class DirichletShrinkageMultinomialModel:
    """Model M2: Dirichlet-Shrinkage Multinomial Model with symmetric uniform prior."""

    def __init__(self, window: int, prior_strength: float) -> None:
        if (
            isinstance(window, bool)
            or not isinstance(window, int)
            or window <= 0
        ):
            raise ValueError("window must be a positive integer number of draw rows")

        if (
            isinstance(prior_strength, bool)
            or not isinstance(prior_strength, (int, float))
            or math.isnan(prior_strength)
            or math.isinf(prior_strength)
            or prior_strength <= 0
        ):
            raise ValueError("prior_strength must be a positive finite float")

        self.window = int(window)
        self.prior_strength = float(prior_strength)

    @property
    def model_identity(self) -> str:
        return f"M2_DIRICHLET_SHRINKAGE_MULTINOMIAL_W{self.window}_B{self.prior_strength:g}"

    def posterior_probabilities(self, history: CountHistory) -> np.ndarray:
        if len(history.dates) < self.window:
            raise ChronologyError("Dirichlet history has fewer rows than window")

        accumulated = history.counts[-self.window:].sum(axis=0, dtype=np.float64)
        alpha = self.prior_strength / 100.0 + accumulated
        alpha_total = self.prior_strength + 27.0 * float(self.window)
        probabilities = alpha / alpha_total

        if abs(float(probabilities.sum()) - 1.0) > FORECAST_SUM_TOLERANCE:
            raise ModelContractError("MODEL_CONTRACT_FAILURE: posterior probabilities must sum to 1")

        return probabilities

    def predict_count(self, history: CountHistory, target_date: str) -> CountForecast:
        target = require_target_after_history(history, target_date)
        if len(history.dates) < self.window:
            raise ChronologyError("Dirichlet history has fewer rows than window")

        accumulated = history.counts[-self.window:].sum(axis=0, dtype=np.float64)
        alpha = self.prior_strength / 100.0 + accumulated
        alpha_total = self.prior_strength + 27.0 * float(self.window)
        probabilities = alpha / alpha_total

        if abs(float(probabilities.sum()) - 1.0) > FORECAST_SUM_TOLERANCE:
            raise ModelContractError("MODEL_CONTRACT_FAILURE: posterior probabilities must sum to 1")

        expected = 27.0 * probabilities

        # Dirichlet posterior variance on simplex probability p_n:
        # Var(p_n) = alpha_n * (alpha_total - alpha_n) / (alpha_total^2 * (alpha_total + 1))
        # SE(mu_n) = 27.0 * sqrt(Var(p_n))
        var_p = (alpha * (alpha_total - alpha)) / (alpha_total**2 * (alpha_total + 1.0))
        mean_se = 27.0 * np.sqrt(var_p)

        return CountForecast(
            target_date=target,
            history_start=str(history.dates[-self.window]),
            history_end=str(history.dates[-1]),
            expected_count=expected,
            model_identity=self.model_identity,
            mean_standard_error=mean_se,
        )
