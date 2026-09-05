"""Strict ingestion of the canonical M3 observed-draw CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray

from .authority import CANONICAL_DATA_PATH, collect_repository_identity
from .contracts import FailureExitStatus, FailureStage


CANONICAL_HEADER = (
    'date',
    'special',
    'prize1',
    'prize2_1',
    'prize2_2',
    'prize3_1',
    'prize3_2',
    'prize3_3',
    'prize3_4',
    'prize3_5',
    'prize3_6',
    'prize4_1',
    'prize4_2',
    'prize4_3',
    'prize4_4',
    'prize5_1',
    'prize5_2',
    'prize5_3',
    'prize5_4',
    'prize5_5',
    'prize5_6',
    'prize6_1',
    'prize6_2',
    'prize6_3',
    'prize7_1',
    'prize7_2',
    'prize7_3',
    'prize7_4',
)

_DRAW_TOKEN = re.compile(r'[0-9]{1,2}\Z')
_DATE_TOKEN = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\Z')


class DatasetValidationError(ValueError):
    """Fail-closed M3 raw-dataset validation failure."""

    failure_stage = FailureStage.DATA_VALIDATION
    exit_status = FailureExitStatus.NEEDS_DATA_REVISION

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True)
class CanonicalRawDataset:
    """Immutable recorded-date sequence and multiplicity-preserving counts."""

    recorded_dates: tuple[date, ...]
    count_matrix: NDArray[np.int16]

    def __post_init__(self) -> None:
        if self.count_matrix.dtype != np.int16 or self.count_matrix.ndim != 2:
            raise DatasetValidationError('COUNT_MATRIX_INVALID')
        if self.count_matrix.shape != (len(self.recorded_dates), 100):
            raise DatasetValidationError('COUNT_MATRIX_SHAPE_INVALID')
        if np.any(self.count_matrix < 0) or not np.all(self.count_matrix.sum(axis=1) == 27):
            raise DatasetValidationError('COUNT_MATRIX_INVALID')
        self.count_matrix.setflags(write=False)

    @property
    def row_count(self) -> int:
        return len(self.recorded_dates)


def _parse_date(raw_value: str) -> date:
    if _DATE_TOKEN.fullmatch(raw_value) is None:
        raise DatasetValidationError('DATE_INVALID')
    try:
        return date.fromisoformat(raw_value)
    except ValueError as error:
        raise DatasetValidationError('DATE_INVALID') from error


def _parse_draw(raw_value: str) -> int:
    if _DRAW_TOKEN.fullmatch(raw_value) is None:
        raise DatasetValidationError('DRAW_VALUE_INVALID')
    value = int(raw_value)
    if not 0 <= value <= 99:
        raise DatasetValidationError('DRAW_VALUE_INVALID')
    return value


def load_canonical_dataset(repository_root: Path) -> CanonicalRawDataset:
    """Load only the authority-locked raw CSV without repairing its rows."""
    root = Path(repository_root)
    collect_repository_identity(root)
    csv_path = root / CANONICAL_DATA_PATH

    try:
        with csv_path.open('r', encoding='utf-8', newline='') as handle:
            rows = csv.reader(handle)
            try:
                header = tuple(next(rows))
            except StopIteration as error:
                raise DatasetValidationError('HEADER_INVALID') from error
            if header != CANONICAL_HEADER:
                raise DatasetValidationError('HEADER_INVALID')

            recorded_dates: list[date] = []
            draw_rows: list[list[int]] = []
            previous_date: date | None = None
            for raw_row in rows:
                if len(raw_row) != len(CANONICAL_HEADER):
                    raise DatasetValidationError('ROW_WIDTH_INVALID')
                recorded_date = _parse_date(raw_row[0])
                if previous_date is not None and recorded_date <= previous_date:
                    raise DatasetValidationError('DATE_NOT_INCREASING')
                draws = [_parse_draw(raw_value) for raw_value in raw_row[1:]]
                if len(draws) != 27:
                    raise DatasetValidationError('ROW_WIDTH_INVALID')
                recorded_dates.append(recorded_date)
                draw_rows.append(draws)
                previous_date = recorded_date
    except FileNotFoundError as error:
        raise DatasetValidationError('CANONICAL_DATA_PATH_ABSENT') from error
    except csv.Error as error:
        raise DatasetValidationError('CSV_MALFORMED') from error

    counts = np.zeros((len(draw_rows), 100), dtype=np.int16)
    for row_index, draws in enumerate(draw_rows):
        np.add.at(counts[row_index], draws, 1)
    return CanonicalRawDataset(tuple(recorded_dates), counts)
