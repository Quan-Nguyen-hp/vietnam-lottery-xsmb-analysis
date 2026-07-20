"""Shadow challenger dự báo E[số nháy] theo EWMA, không tác động production bets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CountExpectationForecast:
    expected_counts: np.ndarray
    lower_bounds: np.ndarray
    effective_sample_size: float
    picks: list[int]


class EWMAHitCountChallenger:
    """EWMA half-life 90 + cổng lower confidence bound, research-only."""

    name = "ewma_hl90_lcb"
    version = "research-v1"

    def __init__(
        self,
        half_life: float = 90.0,
        lookback_days: int = 540,
        z_score: float = 1.96,
        cost_per_bet: float = 27.0,
        payout_per_hit: float = 99.0,
        top_k: int = 4,
    ):
        self.half_life = half_life
        self.lookback_days = lookback_days
        self.z_score = z_score
        self.cost_per_bet = cost_per_bet
        self.payout_per_hit = payout_per_hit
        self.top_k = top_k

    @property
    def break_even_expected_count(self) -> float:
        return self.cost_per_bet / self.payout_per_hit

    @staticmethod
    def build_count_matrix(df_history: pd.DataFrame, prize_cols: list[str]) -> np.ndarray:
        draws = df_history[prize_cols].to_numpy(dtype=int)
        counts = np.zeros((len(draws), 100), dtype=np.int8)
        for number in range(100):
            counts[:, number] = np.sum(draws == number, axis=1)
        return counts

    def predict_from_counts(self, count_history: np.ndarray) -> CountExpectationForecast:
        counts = np.asarray(count_history)
        if counts.ndim != 2 or counts.shape[1] != 100:
            raise ValueError("count_history phải có shape (n_days, 100)")
        if len(counts) == 0:
            raise ValueError("Cần ít nhất một ngày lịch sử")

        window = min(self.lookback_days, len(counts))
        sample = counts[-window:].astype(float)
        ages = np.arange(window - 1, -1, -1, dtype=float)
        weights = np.exp(-np.log(2.0) * ages / self.half_life)
        weights /= weights.sum()
        expected_counts = weights @ sample
        effective_n = float(1.0 / np.sum(weights**2))
        standard_error = np.sqrt(np.clip(expected_counts, 1e-9, None) / effective_n)
        lower_bounds = np.maximum(0.0, expected_counts - self.z_score * standard_error)

        eligible = np.where(lower_bounds > self.break_even_expected_count)[0]
        ranked = eligible[np.argsort(lower_bounds[eligible])[::-1]]
        picks = [int(number) for number in ranked[: self.top_k]]
        return CountExpectationForecast(expected_counts, lower_bounds, effective_n, picks)

    def predict(self, df_history: pd.DataFrame, prize_cols: list[str]) -> CountExpectationForecast:
        return self.predict_from_counts(self.build_count_matrix(df_history, prize_cols))

    def to_shadow_dict(self, forecast: CountExpectationForecast) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "status": "RESEARCH_ONLY",
            "target": "expected_hit_count",
            "half_life_days": self.half_life,
            "lookback_days": self.lookback_days,
            "effective_sample_size": round(forecast.effective_sample_size, 2),
            "break_even_expected_count": self.break_even_expected_count,
            "expected_counts": [float(value) for value in forecast.expected_counts],
            "lower_bounds_95": [float(value) for value in forecast.lower_bounds],
            "picks": [
                {
                    "number": number,
                    "expected_count": float(forecast.expected_counts[number]),
                    "lower_bound_95": float(forecast.lower_bounds[number]),
                    "expected_value_k": float(
                        self.payout_per_hit * forecast.expected_counts[number] - self.cost_per_bet
                    ),
                    "decision": "PAPER_TRADE",
                }
                for number in forecast.picks
            ],
        }
