"""Kiểm thử Frozen Prospective Challenger V1: đúng công thức, không rò rỉ, không đọc holdout, gate một lần."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backtests'))

import frozen_challenger_v1 as fc  # noqa: E402
import head_signal_followup as hsf  # noqa: E402
import signal_power_study as sps  # noqa: E402


def _counts(n: int = 500, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.zeros((n, 100))
    for t in range(n):
        np.add.at(c[t], rng.integers(0, 100, size=27), 1.0)
    return c


def _z(a: np.ndarray) -> np.ndarray:
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else np.zeros_like(a)


def test_rule_matches_explicit_formula() -> None:
    f = np.random.default_rng(0).integers(0, 12, size=100).astype(float)
    head = np.array([f[(n // 10) * 10:(n // 10) * 10 + 10].sum() for n in range(100)])
    tail = np.array([f[n % 10::10].sum() for n in range(100)])
    expect = np.round(_z(head) + _z(tail) + _z(f), 12)
    assert np.allclose(fc.rule_scores(f[None])[0], expect, atol=1e-12)


def test_batch_scores_equal_single_day_forecast_and_ignore_future() -> None:
    c = _counts()
    sc = fc.scores_all(c)
    for t in (90, 91, 250, 499):
        assert np.allclose(sc[t], fc.forecast_scores(c[:t]))
        poisoned = c.copy()
        poisoned[t:] = 99.0
        assert np.allclose(fc.scores_all(poisoned)[t], sc[t])


def test_equals_exploratory_h2_definition() -> None:
    c = _counts()
    dates = sps.pd.date_range('2012-01-01', periods=len(c), freq='D')
    sig = sps.build_signals(dates, c)
    h = hsf.digit_freq(sig['freq_90'], 'head')
    t = hsf.digit_freq(sig['freq_90'], 'tail')
    h2 = sps._cross_z(h) + sps._cross_z(t) + sps._cross_z(sig['freq_90'])
    assert np.allclose(fc.scores_all(c)[fc.MIN_HISTORY:], h2[fc.MIN_HISTORY:], atol=1e-9)


def test_pick_tie_break_and_k() -> None:
    assert fc.pick_top(np.zeros(100)) == [0, 1]
    s = np.zeros(100)
    s[[7, 3]] = 1.0
    assert fc.pick_top(s) == [3, 7]


def test_forecast_requires_history() -> None:
    with pytest.raises(ValueError):
        fc.forecast_scores(_counts(10))


def test_gate_truth_table() -> None:
    gate = {'ic_floor': 0.002}
    base = {'ic_lower_975': 0.001, 'ic_mean': 0.004, 'top2_excess_mean': 0.003}
    assert fc.gate_decision(base, gate)['PASS']
    assert not fc.gate_decision({**base, 'ic_lower_975': -0.0001}, gate)['PASS']
    assert not fc.gate_decision({**base, 'ic_mean': 0.0019}, gate)['PASS']
    assert not fc.gate_decision({**base, 'top2_excess_mean': -0.001}, gate)['PASS']


def test_sample_size_formulas_are_consistent() -> None:
    n = fc.required_n(0.1, 0.004)
    assert fc.mde(0.1, int(np.ceil(n))) == pytest.approx(0.004, rel=1e-3)
    assert fc.required_n(0.1, 0.002) == pytest.approx(4 * n)


def test_read_prefix_never_parses_holdout_values(tmp_path, monkeypatch) -> None:
    header = 'date,' + ','.join(f'p{i}' for i in range(27))
    ok = ','.join(['5'] * 27)
    csv = tmp_path / 'x.csv'
    csv.write_text(f'{header}\n2026-07-19,{ok}\n2026-07-20,{ok}\n2026-07-21,GARBAGE,!!\n2026-07-22,\n', encoding='utf-8')
    monkeypatch.setattr(fc, 'CSV', csv)
    lines, dates, n_after = fc.read_prefix('2026-07-21')
    assert dates == ['2026-07-19', '2026-07-20'] and n_after == 2
    assert all('GARBAGE' not in ln for ln in lines)


def test_invalid_rows_are_ignored_or_fail_closed() -> None:
    header = 'date,' + ','.join(f'p{i}' for i in range(27))
    good = '2026-01-01,' + ','.join(['1'] * 27)
    bad = '2026-01-02,' + ','.join(['1'] * 26)
    with pytest.raises(ValueError):
        fc.counts_from_lines([header, good, bad], strict=True)
    dates, c = fc.counts_from_lines([header, good, bad], strict=False)
    assert dates == ['2026-01-01'] and c.sum() == 27


def test_frozen_artifacts_verify_and_holdout_gate_is_closed() -> None:
    assert fc.verify() == []
    with pytest.raises(SystemExit):
        fc.cmd_score_holdout(None)  # type: ignore[arg-type]
    assert not fc.RESULT_PATH.exists()
