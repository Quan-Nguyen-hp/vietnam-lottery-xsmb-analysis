from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.count_v2.contracts import CountForecast, CountHistory
from src.count_v2.models.ewma import EWMACountModel
from src.count_v2.models.rolling import RollingCountModel
from src.count_v2.models.uniform import UniformCountModel


@runtime_checkable
class CountModel(Protocol):
    """Protocol for all XPIS v2 count models."""

    @property
    def model_identity(self) -> str:
        ...

    def predict_count(self, history: CountHistory, target_date: str) -> CountForecast:
        ...


__all__ = [
    "CountModel",
    "EWMACountModel",
    "RollingCountModel",
    "UniformCountModel",
]
