from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import datetime
import numbers
from typing import Any

import numpy as np
import pandas as pd

FORECAST_SUM_TOLERANCE: float = 1e-6


class CountContractError(ValueError):
    """Base exception for XPIS v2 count contract violations."""


class DatasetValidationError(CountContractError):
    """Raised when dataset ingestion or transformation validation fails."""


class ChronologyError(CountContractError):
    """Raised when chronological precedence or date integrity is violated."""


class ModelContractError(CountContractError):
    """Raised when model forecast output violates mathematical invariants."""


class EvaluationIntegrityError(CountContractError):
    """Raised when evaluation alignment or date matching fails."""


def canonical_date(value: object, *, field_name: str) -> str:
    """Validate that a date value is a canonical YYYY-MM-DD string."""
    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str):
        raise DatasetValidationError(
            f"Expected string for '{field_name}', got {type(value).__name__}: {value!r}"
        )
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        raise DatasetValidationError(
            f"Invalid canonical date format for '{field_name}' (expected YYYY-MM-DD): {value!r}"
        )
    try:
        parsed = datetime.date.fromisoformat(value)
    except ValueError as err:
        raise DatasetValidationError(
            f"Invalid calendar date for '{field_name}': {value!r}"
        ) from err
    return parsed.isoformat()


def require_target_after_history(history: CountHistory, target_date: str) -> str:
    """Ensure target_date is canonical and strictly after every date in history."""
    canonical = canonical_date(target_date, field_name="target_date")
    if len(history.dates) > 0 and np.any(history.dates >= canonical):
        raise ChronologyError("target_date must be strictly after every history date")
    return canonical


def _validate_dates_vector(dates: Any, field_name: str) -> np.ndarray:
    if isinstance(dates, (str, bytes, Mapping)):
        raise DatasetValidationError(
            f"Expected array or sequence for '{field_name}', got {type(dates).__name__}"
        )

    if isinstance(dates, np.ndarray):
        if dates.ndim != 1:
            raise DatasetValidationError(
                f"'{field_name}' must be a 1D array, got shape {dates.shape}"
            )
        raw_list = dates.tolist()
    elif isinstance(dates, Sequence):
        raw_list = list(dates)
    else:
        raise DatasetValidationError(
            f"Expected array or sequence for '{field_name}', got {type(dates).__name__}"
        )

    canonical_list = [
        canonical_date(d, field_name=f"{field_name}[{i}]")
        for i, d in enumerate(raw_list)
    ]
    arr = np.asarray(canonical_list, dtype="<U10")

    if arr.ndim != 1:
        raise DatasetValidationError(
            f"'{field_name}' must be a 1D array, got shape {arr.shape}"
        )

    if len(arr) > 1:
        if np.any(arr[:-1] >= arr[1:]):
            raise DatasetValidationError(
                f"'{field_name}' must be strictly ascending with no duplicates"
            )

    arr.setflags(write=False)
    return arr


def _validated_int_array(
    data: Any,
    field_name: str,
    *,
    expected_dim: int,
    expected_last_dim: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> np.ndarray:
    if isinstance(data, (str, bytes, Mapping)):
        raise DatasetValidationError(
            f"Expected array or sequence for '{field_name}', got {type(data).__name__}"
        )

    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (list, tuple, Sequence)):
        arr = np.asarray(data)
    else:
        raise DatasetValidationError(
            f"Expected array or sequence for '{field_name}', got {type(data).__name__}"
        )

    if arr.ndim != expected_dim:
        raise DatasetValidationError(
            f"'{field_name}' must have {expected_dim} dimensions, got shape {arr.shape}"
        )

    if expected_dim == 1 and arr.shape[0] != expected_last_dim:
        raise DatasetValidationError(
            f"'{field_name}' must have shape ({expected_last_dim},), got {arr.shape}"
        )
    elif expected_dim == 2 and arr.shape[1] != expected_last_dim:
        raise DatasetValidationError(
            f"'{field_name}' must have shape (N, {expected_last_dim}), got {arr.shape}"
        )

    # 1. Reject boolean dtype
    if arr.dtype == bool or np.issubdtype(arr.dtype, np.bool_):
        raise DatasetValidationError(
            f"'{field_name}' must contain integer values, got boolean"
        )

    # 2. Check object arrays for booleans, nulls, non-integers
    if arr.dtype == object:
        for x in arr.flat:
            if isinstance(x, bool) or not isinstance(x, (numbers.Integral, numbers.Real)) or pd.isna(x):
                raise DatasetValidationError(
                    f"'{field_name}' must contain integer values, got {type(x).__name__}"
                )
            if isinstance(x, numbers.Real) and not isinstance(x, numbers.Integral):
                if not np.isfinite(x) or x != int(x):
                    raise DatasetValidationError(
                        f"'{field_name}' must contain integer values, got non-integral value {x}"
                    )
        arr = arr.astype(np.float64)

    # 3. Check numeric array null / nonfinite / integrality
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise DatasetValidationError(
                f"'{field_name}' contains NaN or infinite values"
            )
        if not np.all(arr == np.floor(arr)):
            raise DatasetValidationError(
                f"'{field_name}' must contain integer values, got floating point with fraction"
            )
    elif not np.issubdtype(arr.dtype, np.integer):
        raise DatasetValidationError(
            f"'{field_name}' must contain integer values, got dtype {arr.dtype}"
        )

    # 4. Range validation BEFORE casting / narrowing
    if min_val is not None and np.any(arr < min_val):
        raise DatasetValidationError(
            f"'{field_name}' contains values < {min_val}"
        )
    if max_val is not None and np.any(arr > max_val):
        raise DatasetValidationError(
            f"'{field_name}' contains values > {max_val}"
        )

    # 5. Safe narrowing cast
    res = arr.astype(np.int16, copy=True)
    res.setflags(write=False)
    return res


def _validated_float_vector(
    vec: Any,
    field_name: str,
    *,
    allow_negative: bool = False,
) -> np.ndarray:
    if isinstance(vec, (list, tuple)):
        vec = np.asarray(vec, dtype=np.float64)
    elif isinstance(vec, np.ndarray):
        vec = vec.astype(np.float64)
    else:
        raise ModelContractError(
            f"Expected array or list for '{field_name}', got {type(vec).__name__}"
        )

    if vec.ndim != 1 or vec.shape[0] != 100:
        raise ModelContractError(
            f"'{field_name}' must have shape (100,), got {vec.shape}"
        )

    if not np.all(np.isfinite(vec)):
        raise ModelContractError(
            f"'{field_name}' contains NaN or infinite values"
        )

    if not allow_negative and np.any(vec < 0.0):
        raise ModelContractError(
            f"'{field_name}' must be non-negative"
        )

    res = vec.copy()
    res.setflags(write=False)
    return res


@dataclass(frozen=True)
class RawDrawBatch:
    """Raw draw observations ingested from lottery records."""

    dates: np.ndarray
    draws: np.ndarray

    def __post_init__(self) -> None:
        validated_dates = _validate_dates_vector(self.dates, "dates")
        validated_draws = _validated_int_array(
            self.draws,
            "draws",
            expected_dim=2,
            expected_last_dim=27,
            min_val=0,
            max_val=99,
        )
        if len(validated_dates) != validated_draws.shape[0]:
            raise DatasetValidationError(
                f"Mismatch: dates length ({len(validated_dates)}) != draws rows ({validated_draws.shape[0]})"
            )
        object.__setattr__(self, "dates", validated_dates)
        object.__setattr__(self, "draws", validated_draws)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dates": [str(d) for d in self.dates],
            "draws": [list(map(int, row)) for row in self.draws],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RawDrawBatch:
        return cls(
            dates=payload["dates"],
            draws=payload["draws"],
        )


@dataclass(frozen=True)
class CountHistory:
    """Historical aggregate count matrix strictly preceding target evaluation."""

    dates: np.ndarray
    counts: np.ndarray

    def __post_init__(self) -> None:
        validated_dates = _validate_dates_vector(self.dates, "dates")
        validated_counts = _validated_int_array(
            self.counts,
            "counts",
            expected_dim=2,
            expected_last_dim=100,
            min_val=0,
            max_val=27,
        )
        if len(validated_dates) != validated_counts.shape[0]:
            raise DatasetValidationError(
                f"Mismatch: dates length ({len(validated_dates)}) != counts rows ({validated_counts.shape[0]})"
            )
        if len(validated_dates) > 0 and not np.all(validated_counts.sum(axis=1) == 27):
            raise DatasetValidationError(
                "All rows in CountHistory counts must sum exactly to 27"
            )
        object.__setattr__(self, "dates", validated_dates)
        object.__setattr__(self, "counts", validated_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dates": [str(d) for d in self.dates],
            "counts": [list(map(int, row)) for row in self.counts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CountHistory:
        return cls(
            dates=payload["dates"],
            counts=payload["counts"],
        )


@dataclass(frozen=True)
class CountOutcome:
    """Observed count outcome for a single target date."""

    target_date: str
    observed_counts: np.ndarray

    def __post_init__(self) -> None:
        target = canonical_date(self.target_date, field_name="target_date")
        validated_counts = _validated_int_array(
            self.observed_counts,
            "observed_counts",
            expected_dim=1,
            expected_last_dim=100,
            min_val=0,
            max_val=27,
        )
        if int(validated_counts.sum()) != 27:
            raise DatasetValidationError(
                f"CountOutcome observed_counts must sum exactly to 27, got {validated_counts.sum()}"
            )
        object.__setattr__(self, "target_date", target)
        object.__setattr__(self, "observed_counts", validated_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_date": self.target_date,
            "observed_counts": list(map(int, self.observed_counts)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CountOutcome:
        return cls(
            target_date=payload["target_date"],
            observed_counts=payload["observed_counts"],
        )


@dataclass(frozen=True)
class CountForecast:
    """Point and distributional forecast emitted by a count model."""

    target_date: str
    history_start: str
    history_end: str
    expected_count: np.ndarray
    model_identity: str
    mean_standard_error: np.ndarray | None = None
    mean_lower_bound: np.ndarray | None = None
    mean_upper_bound: np.ndarray | None = None
    predictive_distribution: dict[str, Any] | None = None
    prediction_interval: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        target = canonical_date(self.target_date, field_name="target_date")
        history_start = canonical_date(self.history_start, field_name="history_start")
        history_end = canonical_date(self.history_end, field_name="history_end")

        if history_start > history_end or history_end >= target:
            raise ChronologyError(
                f"forecast requires history_start <= history_end < target_date (got {history_start} <= {history_end} < {target})"
            )

        expected = _validated_float_vector(self.expected_count, "expected_count")
        if abs(float(expected.sum()) - 27.0) > FORECAST_SUM_TOLERANCE:
            raise ModelContractError(
                "MODEL_CONTRACT_FAILURE: expected_count must sum to 27 within 1e-6"
            )

        object.__setattr__(self, "target_date", target)
        object.__setattr__(self, "history_start", history_start)
        object.__setattr__(self, "history_end", history_end)
        object.__setattr__(self, "expected_count", expected)

        if self.mean_standard_error is not None:
            se = _validated_float_vector(self.mean_standard_error, "mean_standard_error")
            object.__setattr__(self, "mean_standard_error", se)

        if self.mean_lower_bound is not None:
            lb = _validated_float_vector(self.mean_lower_bound, "mean_lower_bound", allow_negative=True)
            object.__setattr__(self, "mean_lower_bound", lb)

        if self.mean_upper_bound is not None:
            ub = _validated_float_vector(self.mean_upper_bound, "mean_upper_bound", allow_negative=True)
            object.__setattr__(self, "mean_upper_bound", ub)

        if self.prediction_interval is not None:
            if not isinstance(self.prediction_interval, (tuple, list)) or len(self.prediction_interval) != 2:
                raise ModelContractError("prediction_interval must be a 2-tuple of (lower, upper)")
            pi_lower = _validated_float_vector(self.prediction_interval[0], "prediction_interval[0]")
            pi_upper = _validated_float_vector(self.prediction_interval[1], "prediction_interval[1]")
            object.__setattr__(self, "prediction_interval", (pi_lower, pi_upper))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "target_date": self.target_date,
            "history_start": self.history_start,
            "history_end": self.history_end,
            "expected_count": [float(x) for x in self.expected_count],
            "model_identity": self.model_identity,
        }
        if self.mean_standard_error is not None:
            result["mean_standard_error"] = [float(x) for x in self.mean_standard_error]
        if self.mean_lower_bound is not None:
            result["mean_lower_bound"] = [float(x) for x in self.mean_lower_bound]
        if self.mean_upper_bound is not None:
            result["mean_upper_bound"] = [float(x) for x in self.mean_upper_bound]
        if self.predictive_distribution is not None:
            result["predictive_distribution"] = self.predictive_distribution
        if self.prediction_interval is not None:
            result["prediction_interval"] = (
                [float(x) for x in self.prediction_interval[0]],
                [float(x) for x in self.prediction_interval[1]],
            )
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CountForecast:
        data = dict(payload)
        expected_count = np.asarray(data["expected_count"], dtype=np.float64)
        mean_se = (
            np.asarray(data["mean_standard_error"], dtype=np.float64)
            if "mean_standard_error" in data and data["mean_standard_error"] is not None
            else None
        )
        mean_lb = (
            np.asarray(data["mean_lower_bound"], dtype=np.float64)
            if "mean_lower_bound" in data and data["mean_lower_bound"] is not None
            else None
        )
        mean_ub = (
            np.asarray(data["mean_upper_bound"], dtype=np.float64)
            if "mean_upper_bound" in data and data["mean_upper_bound"] is not None
            else None
        )
        pi_data = data.get("prediction_interval")
        prediction_interval = (
            (
                np.asarray(pi_data[0], dtype=np.float64),
                np.asarray(pi_data[1], dtype=np.float64),
            )
            if pi_data is not None
            else None
        )

        return cls(
            target_date=str(data["target_date"]),
            history_start=str(data["history_start"]),
            history_end=str(data["history_end"]),
            expected_count=expected_count,
            model_identity=str(data["model_identity"]),
            mean_standard_error=mean_se,
            mean_lower_bound=mean_lb,
            mean_upper_bound=mean_ub,
            predictive_distribution=data.get("predictive_distribution"),
            prediction_interval=prediction_interval,
        )
