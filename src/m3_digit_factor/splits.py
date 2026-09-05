"""Observed-event eligibility and deterministic chronological M3 splits."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import CanonicalRawDataset, DatasetValidationError


PRIOR_RECORDED_EVENTS_REQUIRED = 365
MINIMUM_ELIGIBLE_TARGETS = 240
STABILITY_BLOCK_COUNT = 6


class InsufficientDevelopmentSample(DatasetValidationError):
    """Raised when the frozen minimum eligible development sample is unavailable."""

    def __init__(self) -> None:
        super().__init__('InsufficientDevelopmentSample')


@dataclass(frozen=True)
class ChronologicalSplits:
    """Immutable dataset-row-index partitions of eligible recorded draw events."""

    eligible_target_indices: tuple[int, ...]
    dev_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    stability_indices: tuple[int, ...]
    stability_blocks: tuple[tuple[int, ...], ...]
    N: int
    N_dev: int
    N_val: int
    N_stability: int


def build_chronological_splits(dataset: CanonicalRawDataset) -> ChronologicalSplits:
    """Derive only the frozen observed-event eligibility and chronological splits."""
    if not isinstance(dataset, CanonicalRawDataset):
        raise TypeError('dataset must be a CanonicalRawDataset')

    eligible_target_indices = tuple(range(PRIOR_RECORDED_EVENTS_REQUIRED, dataset.row_count))
    eligible_count = len(eligible_target_indices)
    if eligible_count < MINIMUM_ELIGIBLE_TARGETS:
        raise InsufficientDevelopmentSample()

    dev_count = eligible_count // 2
    val_count = eligible_count // 4
    stability_count = eligible_count - dev_count - val_count
    dev_indices = eligible_target_indices[:dev_count]
    val_indices = eligible_target_indices[dev_count : dev_count + val_count]
    stability_indices = eligible_target_indices[dev_count + val_count :]

    block_size, larger_block_count = divmod(stability_count, STABILITY_BLOCK_COUNT)
    block_sizes = (block_size + 1,) * larger_block_count + (block_size,) * (
        STABILITY_BLOCK_COUNT - larger_block_count
    )
    stability_blocks: list[tuple[int, ...]] = []
    block_start = 0
    for size in block_sizes:
        stability_blocks.append(stability_indices[block_start : block_start + size])
        block_start += size

    return ChronologicalSplits(
        eligible_target_indices=eligible_target_indices,
        dev_indices=dev_indices,
        val_indices=val_indices,
        stability_indices=stability_indices,
        stability_blocks=tuple(stability_blocks),
        N=eligible_count,
        N_dev=dev_count,
        N_val=val_count,
        N_stability=stability_count,
    )
