from __future__ import annotations

from collections.abc import Sequence
import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.count_v2.contracts import (
    ChronologyError,
    CountHistory,
    CountOutcome,
    DatasetValidationError,
    RawDrawBatch,
    canonical_date,
)


def raw_draw_batch_from_frame(
    frame: pd.DataFrame,
    *,
    draw_columns: Sequence[str],
    date_column: str = "date",
) -> RawDrawBatch:
    """Convert a pandas DataFrame with raw draw columns into a validated RawDrawBatch."""
    if not isinstance(frame, pd.DataFrame):
        raise DatasetValidationError(
            f"Expected pandas DataFrame, got {type(frame).__name__}"
        )

    if (
        not isinstance(draw_columns, Sequence)
        or len(draw_columns) != 27
        or len(set(draw_columns)) != 27
    ):
        raise DatasetValidationError(
            f"draw_columns must contain exactly 27 unique column names, got {len(draw_columns)}"
        )

    if date_column not in frame.columns:
        raise DatasetValidationError(
            f"Missing required date column '{date_column}' in frame"
        )

    missing_draws = [col for col in draw_columns if col not in frame.columns]
    if missing_draws:
        raise DatasetValidationError(
            f"Missing draw columns in frame: {missing_draws}"
        )

    # Date normalization
    raw_dates = frame[date_column].tolist()
    date_strings: list[str] = []
    for i, val in enumerate(raw_dates):
        if pd.isna(val):
            raise DatasetValidationError(
                f"null date found at row {i} in column '{date_column}'"
            )
        if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
            date_str = val.strftime("%Y-%m-%d")
        else:
            date_str = str(val)
        date_strings.append(canonical_date(date_str, field_name=f"{date_column}[{i}]"))

    dates_arr = np.asarray(date_strings, dtype="<U10")

    # Draw extraction & validation
    draws_subset = frame[list(draw_columns)]

    # Check for boolean columns
    if any(dtype == bool or dtype == "boolean" for dtype in draws_subset.dtypes):
        raise DatasetValidationError("boolean draw values detected in frame")

    draws_vals = draws_subset.to_numpy()

    if pd.isna(draws_vals).any():
        raise DatasetValidationError("null draw detected in frame")

    if not np.issubdtype(draws_vals.dtype, np.integer):
        if np.issubdtype(draws_vals.dtype, np.floating):
            if not np.all(np.isfinite(draws_vals)) or not np.all(draws_vals == np.floor(draws_vals)):
                raise DatasetValidationError("non-integer draw value detected in frame")
        elif not np.issubdtype(draws_vals.dtype, np.number):
            # Check if strings or objects can be parsed to integers without fraction
            try:
                draws_vals = draws_vals.astype(np.float64)
                if not np.all(np.isfinite(draws_vals)) or not np.all(draws_vals == np.floor(draws_vals)):
                    raise DatasetValidationError("non-integer draw value detected in frame")
            except (ValueError, TypeError) as err:
                raise DatasetValidationError(
                    f"non-integer draw value detected in frame: {err}"
                ) from err

    draws_arr = draws_vals.astype(np.int16)

    if np.any(draws_arr < 0) or np.any(draws_arr > 99):
        raise DatasetValidationError("draw numbers must be in the range [0, 99]")

    return RawDrawBatch(dates=dates_arr, draws=draws_arr)


def count_matrix_from_raw(batch: RawDrawBatch) -> np.ndarray:
    """Transform raw draw observations into canonical count matrix C of shape (N, 100)."""
    if not isinstance(batch, RawDrawBatch):
        raise DatasetValidationError(
            f"Expected RawDrawBatch, got {type(batch).__name__}"
        )

    n_rows = len(batch.dates)
    counts = np.zeros((n_rows, 100), dtype=np.int16)
    if n_rows > 0:
        rows = np.repeat(np.arange(n_rows), batch.draws.shape[1])
        np.add.at(counts, (rows, batch.draws.reshape(-1)), 1)
        if not np.all(counts.sum(axis=1) == 27):
            raise DatasetValidationError("count rows must sum exactly to 27")

    counts.setflags(write=False)
    return counts


def count_history_before(batch: RawDrawBatch, *, target_date: str) -> CountHistory:
    """Extract strictly preceding count history before target_date."""
    target = canonical_date(target_date, field_name="target_date")
    mask = batch.dates < target
    if not np.any(mask):
        raise ChronologyError("history must contain at least one row before target_date")

    counts = count_matrix_from_raw(batch)
    return CountHistory(dates=batch.dates[mask], counts=counts[mask])


def count_outcome_for_date(batch: RawDrawBatch, *, target_date: str) -> CountOutcome:
    """Extract observed count outcome for target_date."""
    target = canonical_date(target_date, field_name="target_date")
    indices = np.where(batch.dates == target)[0]
    if len(indices) == 0:
        raise DatasetValidationError(f"target date {target} not found in batch")
    if len(indices) > 1:
        raise DatasetValidationError(f"multiple entries for target date {target} in batch")

    counts = count_matrix_from_raw(batch)[indices[0]]
    return CountOutcome(target_date=target, observed_counts=counts)
