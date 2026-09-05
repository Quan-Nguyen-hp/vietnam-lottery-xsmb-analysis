"""Behavioral contract tests for strict M3 canonical CSV ingestion."""

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.m3_digit_factor.authority import AuthorityValidationError
from src.m3_digit_factor.dataset import (
    CANONICAL_HEADER,
    DatasetValidationError,
    load_canonical_dataset,
)


EXPECTED_HEADER = (
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


def valid_draws() -> list[int]:
    return list(range(27))


def write_canonical_csv(
    root: Path,
    rows: list[list[object]],
    *,
    header: tuple[str, ...] = EXPECTED_HEADER,
) -> None:
    target = root / 'data' / 'xsmb-2-digits.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def canonical_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.m3_digit_factor import dataset

    monkeypatch.setattr(dataset, 'collect_repository_identity', lambda root: object())
    return tmp_path


def test_exact_independently_pinned_canonical_header_and_chronological_rows_are_accepted(
    canonical_root: Path,
):
    write_canonical_csv(
        canonical_root,
        [
            ['2026-01-01', *valid_draws()],
            ['2026-01-03', *valid_draws()],
        ],
    )

    dataset = load_canonical_dataset(canonical_root)

    assert CANONICAL_HEADER == EXPECTED_HEADER
    assert dataset.recorded_dates == (date(2026, 1, 1), date(2026, 1, 3))
    assert dataset.row_count == 2


@pytest.mark.parametrize(
    'header',
    [
        EXPECTED_HEADER[:2] + (EXPECTED_HEADER[3], EXPECTED_HEADER[2]) + EXPECTED_HEADER[4:],
        EXPECTED_HEADER[:-1],
        EXPECTED_HEADER + ('unapproved_column',),
    ],
)
def test_closed_header_rejects_reordered_missing_or_extra_fields(canonical_root: Path, header: tuple[str, ...]):
    write_canonical_csv(canonical_root, [['2026-01-01', *valid_draws()]], header=header)

    with pytest.raises(DatasetValidationError, match='HEADER_INVALID'):
        load_canonical_dataset(canonical_root)


@pytest.mark.parametrize(
    'rows, error_code',
    [
        ([['2026-01-01', *valid_draws()], ['2026-01-01', *valid_draws()]], 'DATE_NOT_INCREASING'),
        ([['2026-01-02', *valid_draws()], ['2026-01-01', *valid_draws()]], 'DATE_NOT_INCREASING'),
        ([['2026-01-01', *valid_draws()], ['not-a-date', *valid_draws()]], 'DATE_INVALID'),
    ],
)
def test_recorded_date_validation_rejects_duplicates_reverse_order_and_malformed_dates(
    canonical_root: Path,
    rows: list[list[object]],
    error_code: str,
):
    write_canonical_csv(canonical_root, rows)

    with pytest.raises(DatasetValidationError, match=error_code):
        load_canonical_dataset(canonical_root)


@pytest.mark.parametrize(
    'invalid_value, error_code',
    [
        ('', 'DRAW_VALUE_INVALID'),
        ('null', 'DRAW_VALUE_INVALID'),
        ('fourty-two', 'DRAW_VALUE_INVALID'),
        ('42.0', 'DRAW_VALUE_INVALID'),
        ('-1', 'DRAW_VALUE_INVALID'),
        ('100', 'DRAW_VALUE_INVALID'),
    ],
)
def test_present_malformed_draw_values_fail_closed(
    canonical_root: Path,
    invalid_value: str,
    error_code: str,
):
    draws: list[object] = valid_draws()
    draws[0] = invalid_value
    write_canonical_csv(canonical_root, [['2026-01-01', *draws]])

    with pytest.raises(DatasetValidationError, match=error_code):
        load_canonical_dataset(canonical_root)


@pytest.mark.parametrize('draw_count', [26, 28])
def test_rows_with_not_exactly_27_outcomes_are_not_partially_accepted(
    canonical_root: Path,
    draw_count: int,
):
    write_canonical_csv(canonical_root, [['2026-01-01', *range(draw_count)]])

    with pytest.raises(DatasetValidationError, match='ROW_WIDTH_INVALID'):
        load_canonical_dataset(canonical_root)


def test_boundaries_calendar_gaps_and_draw_multiplicity_are_preserved(canonical_root: Path):
    repeated = [42, 42, 42, *range(24)]
    boundary_draws = [0, 99, *range(1, 26)]
    write_canonical_csv(
        canonical_root,
        [
            ['2026-01-01', *repeated],
            ['2026-01-04', *boundary_draws],
        ],
    )

    dataset = load_canonical_dataset(canonical_root)

    assert dataset.count_matrix.shape == (2, 100)
    assert dataset.count_matrix.dtype == np.int16
    assert dataset.count_matrix[0, 42] == 3
    assert dataset.count_matrix[1, 0] == 1
    assert dataset.count_matrix[1, 99] == 1
    assert np.array_equal(dataset.count_matrix.sum(axis=1), np.array([27, 27]))
    assert np.all(dataset.count_matrix >= 0)
    assert dataset.count_matrix.flags.writeable is False


def test_invalid_present_row_is_not_dropped_or_imputed(canonical_root: Path):
    draws: list[object] = valid_draws()
    draws[-1] = ''
    write_canonical_csv(
        canonical_root,
        [
            ['2026-01-01', *valid_draws()],
            ['2026-01-02', *draws],
        ],
    )

    with pytest.raises(DatasetValidationError, match='DRAW_VALUE_INVALID'):
        load_canonical_dataset(canonical_root)


def test_dataset_result_contains_no_m3_03_split_state(canonical_root: Path):
    write_canonical_csv(canonical_root, [['2026-01-01', *valid_draws()]])

    dataset = load_canonical_dataset(canonical_root)

    assert not hasattr(dataset, 'eligible_targets')
    assert not hasattr(dataset, 'dev_indices')
    assert not hasattr(dataset, 'val_indices')
    assert not hasattr(dataset, 'stability_indices')


def test_repository_or_canonical_data_identity_failure_is_not_bypassed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from src.m3_digit_factor import dataset

    write_canonical_csv(tmp_path, [['2026-01-01', *valid_draws()]])
    monkeypatch.setattr(
        dataset,
        'collect_repository_identity',
        lambda root: (_ for _ in ()).throw(AuthorityValidationError('identity mismatch')),
    )

    with pytest.raises(AuthorityValidationError, match='identity mismatch'):
        load_canonical_dataset(tmp_path)
