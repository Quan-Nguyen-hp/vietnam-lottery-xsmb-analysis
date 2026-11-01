"""Kiểm thử Pre-registration R2: S/E tách biệt, E so với break-even, và chỉ dùng dữ liệu < T khi dự báo T."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backtests'))

import frozen_challenger_v1 as v1  # noqa: E402
import frozen_challenger_v1_prereg_r2 as r2  # noqa: E402

HEADER = 'date,' + ','.join(f'p{i}' for i in range(27))


def _lines(n: int = 330, seed: int = 5, bad_every: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    out = [HEADER]
    d0 = 738000  # ordinal
    from datetime import date
    for i in range(n):
        vals = rng.integers(0, 100, size=27)
        row = ','.join(str(int(v)) for v in vals)
        if bad_every and i % bad_every == bad_every - 1:
            row = ','.join(str(int(v)) for v in vals[:26])  # dòng không hợp lệ (26 giá trị)
        out.append(f'{date.fromordinal(d0 + i).isoformat()},{row}')
    return out


def _res(ic_lo: float, ex_mean: float, ex_lo: float) -> dict:
    return {'ic_lower_975': ic_lo, 'top2_excess_mean': ex_mean, 'top2_excess_lower_975': ex_lo}


def test_decision_separates_science_from_economics() -> None:
    # Trường hợp mà gate G1-G3 cũ sẽ PASS nhưng ROI kỳ vọng âm: excess dương nhưng dưới break-even.
    d = r2.decide(_res(0.001, 0.0015, -0.0100))
    assert d['S_scientific_confirmed'] and not d['E_economic_qualified']
    assert d['outcome'] == 'SCIENTIFIC_CONFIRMED_NOT_ECONOMICALLY_QUALIFIED'
    assert 'NO production' in d['allowed_next_step']


def test_economic_needs_lower_bound_above_break_even_not_zero() -> None:
    assert not r2.decide(_res(0.001, 0.02, r2.BE - 1e-6))['E_economic_qualified']
    assert not r2.decide(_res(0.001, 0.02, 0.0))['E_economic_qualified']
    assert r2.decide(_res(0.001, 0.03, r2.BE + 1e-6))['outcome'] == 'SCIENTIFIC_CONFIRMED_ECONOMIC_QUALIFIED'


def test_economic_requires_scientific_confirmation() -> None:
    d = r2.decide(_res(-0.0001, 0.05, 0.04))
    assert not d['S_scientific_confirmed'] and not d['E_economic_qualified'] and d['outcome'] == 'NOT_CONFIRMED'


def test_power_formulas_match_independent_closed_form() -> None:
    # Cỡ mẫu và power nhất quán hai chiều
    sd, delta = 0.1, 0.004
    n = ((r2.Z_A + r2.Z_B) * sd / delta) ** 2
    assert r2.power_s(delta, int(np.ceil(n)), sd) == pytest.approx(0.80, abs=0.002)
    assert r2.power_s(0.0, 1000, sd) == pytest.approx(r2.ALPHA, abs=1e-9)  # type I = alpha
    assert r2.power_e(r2.BE, 1000, 0.375) == pytest.approx(r2.ALPHA, abs=1e-9)  # E tại biên = alpha
    n_e = r2.n_for_e(0.02, 0.375)
    assert r2.power_e(0.02, int(np.ceil(n_e)), 0.375) == pytest.approx(0.80, abs=0.002)
    assert 0.375 / np.sqrt(2561) == pytest.approx(0.00741, abs=2e-5)  # SE excess tại N_target


def test_forecast_uses_only_rows_strictly_before_target() -> None:
    lines = _lines()
    dates = [ln.split(',')[0] for ln in lines[1:]]
    for t in (dates[100], dates[200], dates[-1]):
        base = r2.forecast_from_lines(lines, t)
        poisoned = lines[:1] + [ln if ln.split(',')[0] < t else f"{ln.split(',')[0]},{','.join(['7'] * 27)}"
                                for ln in lines[1:]]  # đổi kết quả của T và tương lai
        assert r2.forecast_from_lines(poisoned, t) == base
        truncated = [ln for ln in lines if ln.split(',')[0] < t or ln == HEADER]
        assert r2.forecast_from_lines(truncated, t) == base  # bỏ hẳn T và tương lai


def test_score_window_equals_prospective_forecast_each_day() -> None:
    lines = _lines(bad_every=40)  # có dòng không hợp lệ rải rác
    hold = lines[200].split(',')[0]
    dates, scores, counts = r2.score_window(lines, hold, 60)
    assert len(dates) == 60 and counts.shape == (60, 100)
    for i in (0, 1, 30, 59):
        assert v1.pick_top(scores[i]) == r2.forecast_from_lines(lines, dates[i])


def test_future_validity_cannot_change_earlier_scores() -> None:
    lines = _lines()
    hold = lines[200].split(',')[0]
    dates, base, _ = r2.score_window(lines, hold, 20)
    broken = lines[:226] + [f"{ln.split(',')[0]},1,2,3" for ln in lines[226:]]  # làm hỏng dòng tương lai
    _, after, _ = r2.score_window(broken[:226] + lines[226:], hold, 20)
    assert np.allclose(base, after)
    d2, s2, _ = r2.score_window(broken, hold, 20)  # cửa sổ 20 đầu tiên không đổi dù dòng sau đó hỏng
    assert d2 == dates and np.allclose(s2, base)


def test_r2_artifacts_verify_rule_unchanged_and_gate_closed() -> None:
    assert r2.verify() == []
    assert v1.verify() == []
    with pytest.raises(SystemExit):
        r2.cmd_score_holdout(None)  # type: ignore[arg-type]
    assert not r2.RESULT_PATH.exists()
