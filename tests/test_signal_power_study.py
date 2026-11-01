"""Hồi quy rò rỉ thời gian cho backtests/signal_power_study.py (dữ liệu tổng hợp, chạy nhanh)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backtests'))

import signal_power_study as sps  # noqa: E402


def _synthetic(n: int = 600) -> tuple[pd.DatetimeIndex, np.ndarray]:
    rng = np.random.default_rng(1)
    dates = pd.date_range('2010-01-01', periods=n, freq='D')
    c = np.zeros((n, 100))
    for t in range(n):
        np.add.at(c[t], rng.integers(0, 100, size=27), 1.0)
    return dates, c


def test_signals_use_only_past_rows() -> None:
    dates, c = _synthetic()
    sig = sps.build_signals(dates, c)
    assert sps.poison_check(dates, c, sig, n_checks=3)


def test_signal_row_zero_is_blind() -> None:
    dates, c = _synthetic(200)
    sig = sps.build_signals(dates, c)
    for name, v in sig.items():
        if name.startswith(('freq', 'ewma', 'last1', 'inverted', 'dow')):
            assert np.allclose(v[0], 0.0), name


def test_topk_excess_is_zero_for_constant_scores() -> None:
    _, c = _synthetic(50)
    ex = sps.topk_excess(np.zeros_like(c), c, 2)
    assert np.allclose(ex, 0.0)


def test_load_counts_never_reaches_holdout() -> None:
    dates, c = sps.load_counts()
    assert dates[-1] < pd.Timestamp(sps.holdout_start())
    assert np.allclose(c.sum(axis=1), 27)
