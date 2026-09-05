"""Contract tests for M3 observed-event eligibility and chronological splits."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.m3_digit_factor.dataset import CanonicalRawDataset, load_canonical_dataset
from src.m3_digit_factor.splits import (
    InsufficientDevelopmentSample,
    build_chronological_splits,
)


def make_dataset(row_count: int, *, calendar_stride: int = 1) -> CanonicalRawDataset:
    dates = tuple(date(2000, 1, 1) + timedelta(days=index * calendar_stride) for index in range(row_count))
    counts = np.zeros((row_count, 100), dtype=np.int16)
    counts[:, 0] = 27
    return CanonicalRawDataset(dates, counts)


def test_first_eligible_target_is_dataset_row_365_not_eligible_ordinal_zero():
    splits = build_chronological_splits(make_dataset(605))

    assert splits.eligible_target_indices[0] == 365
    assert splits.eligible_target_indices[-1] == 604


def test_eligibility_counts_recorded_events_even_when_dates_have_calendar_gaps():
    splits = build_chronological_splits(make_dataset(605, calendar_stride=3))

    assert splits.N == 240
    assert splits.eligible_target_indices == tuple(range(365, 605))


def test_n_239_fails_with_the_frozen_insufficient_development_sample_semantic():
    with pytest.raises(InsufficientDevelopmentSample, match='InsufficientDevelopmentSample'):
        build_chronological_splits(make_dataset(604))


def test_n_240_passes_with_the_minimum_chronological_partition_and_six_equal_blocks():
    splits = build_chronological_splits(make_dataset(605))

    assert splits.N == 240
    assert splits.N_dev == 120
    assert splits.N_val == 60
    assert splits.N_stability == 60
    assert [len(block) for block in splits.stability_blocks] == [10, 10, 10, 10, 10, 10]


@pytest.mark.parametrize('eligible_count', [240, 241, 242, 243, 244, 245, 251, 1001])
def test_floor_formulas_and_contiguous_partition_hold_for_eligible_counts(eligible_count: int):
    splits = build_chronological_splits(make_dataset(365 + eligible_count))
    combined = splits.dev_indices + splits.val_indices + splits.stability_indices

    assert splits.N_dev == eligible_count // 2
    assert splits.N_val == eligible_count // 4
    assert splits.N_stability == eligible_count - splits.N_dev - splits.N_val
    assert len(splits.dev_indices) == splits.N_dev
    assert len(splits.val_indices) == splits.N_val
    assert len(splits.stability_indices) == splits.N_stability
    assert combined == splits.eligible_target_indices
    assert tuple(sorted(combined)) == combined
    assert len(set(combined)) == eligible_count


@pytest.mark.parametrize('eligible_count', [240, 241, 245, 246, 247, 248, 251, 1001])
def test_six_stability_blocks_follow_explicit_q_r_distribution(eligible_count: int):
    splits = build_chronological_splits(make_dataset(365 + eligible_count))
    q, r = divmod(splits.N_stability, 6)
    block_sizes = [len(block) for block in splits.stability_blocks]

    assert len(splits.stability_blocks) == 6
    assert block_sizes == [q + 1] * r + [q] * (6 - r)
    assert max(block_sizes) - min(block_sizes) <= 1
    assert tuple(index for block in splits.stability_blocks for index in block) == splits.stability_indices
    assert all(tuple(sorted(block)) == block for block in splits.stability_blocks)


def test_current_canonical_dataset_split_evidence_is_preserved():
    splits = build_chronological_splits(load_canonical_dataset(Path('.')))

    assert splits.N == 7177
    assert splits.N_dev == 3588
    assert splits.N_val == 1794
    assert splits.N_stability == 1795
    assert [len(block) for block in splits.stability_blocks] == [300, 299, 299, 299, 299, 299]
