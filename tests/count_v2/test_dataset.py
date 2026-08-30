from collections.abc import Sequence
import numpy as np
import pandas as pd
import pytest

from src.count_v2.contracts import (
    ChronologyError,
    CountHistory,
    CountOutcome,
    DatasetValidationError,
    RawDrawBatch,
)
from src.count_v2.dataset import (
    count_history_before,
    count_matrix_from_raw,
    count_outcome_for_date,
    raw_draw_batch_from_frame,
)

DRAW_COLUMNS = tuple(f"draw_{index:02d}" for index in range(27))


def make_frame(
    dates: list[str],
    rows: list[list[int]] | None = None,
) -> pd.DataFrame:
    draw_rows = rows if rows is not None else [list(range(27)) for _ in dates]
    if len(draw_rows) != len(dates):
        raise ValueError("test helper requires one draw row per date")
    if any(len(row) != 27 for row in draw_rows):
        raise ValueError("test helper requires exactly 27 draws per row")
    return pd.DataFrame(
        [
            {"date": date, **dict(zip(DRAW_COLUMNS, row, strict=True))}
            for date, row in zip(dates, draw_rows, strict=True)
        ]
    )


def test_repeated_draw_is_valid_and_preserves_multiplicity():
    frame = make_frame(["2026-01-01"], [[7, 7, 7] + list(range(10, 34))])
    batch = raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS)
    counts = count_matrix_from_raw(batch)
    assert counts.shape == (1, 100)
    assert counts.dtype == np.int16
    assert counts[0, 7] == 3
    assert counts[0].sum() == 27


def test_invalid_middle_row_is_rejected_without_row_dropping():
    frame = make_frame(["2026-01-01", "2026-01-02", "2026-01-03"])
    frame.loc[1, DRAW_COLUMNS[8]] = np.nan
    with pytest.raises(DatasetValidationError, match="null draw"):
        raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS)


def test_draw_column_count_validation():
    frame = make_frame(["2026-01-01"])
    # 26 columns
    with pytest.raises(DatasetValidationError, match="draw_columns"):
        raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS[:26])
    # 28 columns
    extra_cols = DRAW_COLUMNS + ("draw_27",)
    frame["draw_27"] = 0
    with pytest.raises(DatasetValidationError, match="draw_columns"):
        raw_draw_batch_from_frame(frame, draw_columns=extra_cols)


def test_missing_date_or_draw_column_rejected():
    frame = make_frame(["2026-01-01"])
    with pytest.raises(DatasetValidationError, match="Missing required date column"):
        raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS, date_column="missing_date")

    cols = list(DRAW_COLUMNS)
    cols[0] = "nonexistent_col"
    with pytest.raises(DatasetValidationError, match="Missing draw columns"):
        raw_draw_batch_from_frame(frame, draw_columns=cols)


def test_non_integer_draw_values_rejected():
    frame = make_frame(["2026-01-01"])
    frame_float = frame.copy()
    frame_float[DRAW_COLUMNS[0]] = 7.5
    with pytest.raises(DatasetValidationError, match="integer"):
        raw_draw_batch_from_frame(frame_float, draw_columns=DRAW_COLUMNS)

    frame_bool = make_frame(["2026-01-01"])
    frame_bool[DRAW_COLUMNS[0]] = True
    with pytest.raises(DatasetValidationError, match="boolean|integer"):
        raw_draw_batch_from_frame(frame_bool, draw_columns=DRAW_COLUMNS)


def test_out_of_range_draw_values_rejected():
    frame_neg = make_frame(["2026-01-01"])
    frame_neg.loc[0, DRAW_COLUMNS[0]] = -1
    with pytest.raises(DatasetValidationError):
        raw_draw_batch_from_frame(frame_neg, draw_columns=DRAW_COLUMNS)

    frame_high = make_frame(["2026-01-01"])
    frame_high.loc[0, DRAW_COLUMNS[0]] = 100
    with pytest.raises(DatasetValidationError):
        raw_draw_batch_from_frame(frame_high, draw_columns=DRAW_COLUMNS)


def test_raw_draw_batch_from_frame_rejects_overflow_values():
    frame = make_frame(["2026-01-01"])
    frame.loc[0, DRAW_COLUMNS[0]] = 65536
    with pytest.raises(DatasetValidationError):
        raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS)

    frame_float = make_frame(["2026-01-01"])
    frame_float.loc[0, DRAW_COLUMNS[0]] = 65536.0
    with pytest.raises(DatasetValidationError):
        raw_draw_batch_from_frame(frame_float, draw_columns=DRAW_COLUMNS)



def test_date_chronology_and_duplicates_rejected():
    frame_desc = make_frame(["2026-01-02", "2026-01-01"])
    with pytest.raises(DatasetValidationError):
        raw_draw_batch_from_frame(frame_desc, draw_columns=DRAW_COLUMNS)

    frame_dup = make_frame(["2026-01-01", "2026-01-01"])
    with pytest.raises(DatasetValidationError):
        raw_draw_batch_from_frame(frame_dup, draw_columns=DRAW_COLUMNS)


def test_timestamp_dates_converted_correctly():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                **dict(zip(DRAW_COLUMNS, range(27), strict=True)),
            },
            {
                "date": pd.Timestamp("2026-01-03"),
                **dict(zip(DRAW_COLUMNS, range(1, 28), strict=True)),
            },
        ]
    )
    batch = raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS)
    assert list(batch.dates) == ["2026-01-01", "2026-01-03"]


def test_count_history_before_uses_all_prior_rows_with_calendar_gaps():
    frame = make_frame(["2026-01-01", "2026-01-04", "2026-01-10"])
    batch = raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS)

    history = count_history_before(batch, target_date="2026-01-10")
    assert list(history.dates) == ["2026-01-01", "2026-01-04"]
    assert history.counts.shape == (2, 100)

    # If no prior rows
    with pytest.raises(ChronologyError, match="must contain at least one row"):
        count_history_before(batch, target_date="2026-01-01")


def test_count_outcome_for_date_extracts_exact_row():
    frame = make_frame(["2026-01-01", "2026-01-04", "2026-01-10"])
    batch = raw_draw_batch_from_frame(frame, draw_columns=DRAW_COLUMNS)

    outcome = count_outcome_for_date(batch, target_date="2026-01-04")
    assert outcome.target_date == "2026-01-04"
    assert outcome.observed_counts.shape == (100,)
    assert outcome.observed_counts.sum() == 27

    with pytest.raises(DatasetValidationError, match="not found"):
        count_outcome_for_date(batch, target_date="2026-01-05")
