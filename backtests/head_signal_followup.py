"""Follow-up cố định cho tín hiệu duy nhất có IC ổn định ở cả 3 giai đoạn: tần suất theo CHỮ SỐ ĐẦU/ĐUÔI (90 draw).

Câu hỏi (chốt trước khi chạy):
1. IC gộp (và Bonferroni trên toàn bộ họ 22 + biến thể cửa sổ) có vượt ngưỡng không?
2. Có bền theo cửa sổ (30..365) và theo từng năm không?
3. Có khai thác được về kinh tế không? Break-even cần top-K excess > 27/99 - 0.27 = 0.00273 nháy/số.
   Hai quy tắc chọn duy nhất được thử (không tìm thêm): H1 = head_freq_90 lexicographic rồi freq_90,
   H2 = z(head_freq_90) + z(tail_freq_90) + z(freq_90).
Lưu ý: tín hiệu được nhìn thấy trên cả 3 giai đoạn nên kết quả này mang tính thăm dò, KHÔNG phải xác nhận độc lập.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from signal_power_study import (
    PERIODS,
    _cross_z,
    block_boot_ci,
    build_signals,
    load_counts,
    period_mask,
    spearman_rows,
    topk_excess,
)

RESULTS = Path(__file__).resolve().parent / 'results'
BREAK_EVEN = 27 / 99 - 0.27


def digit_freq(freq: np.ndarray, which: str) -> np.ndarray:
    grp = (np.arange(100) // 10) if which == 'head' else (np.arange(100) % 10)
    per = np.stack([freq[:, grp == g].sum(axis=1) for g in range(10)], axis=1)
    return per[:, grp]


def ic_stats(sc: np.ndarray, c: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    ic = spearman_rows(sc[mask], c[mask])
    return float(ic.mean()), float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))), int(mask.sum())


def main() -> None:
    dates, c = load_counts()
    sig = build_signals(dates, c)
    allm = period_mask(dates, '2007-01-01', '2099-01-01')
    out: dict = {'windows': {}, 'yearly_head90': {}, 'rules': {}}
    lines = ['# Head/tail digit-frequency follow-up (exploratory, pre-holdout)', '']

    # 1-2. cửa sổ
    n_family = 22 + 2 * 5  # họ gốc + biến thể cửa sổ head/tail
    thr = norm.ppf(1 - 0.025 / n_family)
    lines += [f'Bonferroni |t| cho {n_family} tín hiệu: {thr:.2f}', '', '| signal | IC gộp | t gộp | IC DEV | IC VAL | IC FINAL |', '|---|---:|---:|---:|---:|---:|']
    cs = np.vstack([np.zeros((1, 100)), np.cumsum(c, axis=0)])
    idx = np.arange(len(c))
    cand = {}
    for which in ('head', 'tail'):
        for w in (30, 60, 90, 180, 365):
            f = cs[idx] - cs[np.maximum(0, idx - w)]
            cand[f'{which}_freq_{w}'] = digit_freq(f, which)
    for name, sc in cand.items():
        ic, t, _ = ic_stats(sc, c, allm)
        per = [ic_stats(sc, c, period_mask(dates, lo, hi))[0] for lo, hi in PERIODS.values()]
        out['windows'][name] = {'ic': ic, 't': t, 'periods': per}
        lines.append(f'| {name} | {ic:+.4f} | {t:+.2f} | {per[0]:+.4f} | {per[1]:+.4f} | {per[2]:+.4f} |')
    lines.append('')

    # theo năm
    yrs = np.asarray(dates.year)
    pos, tot = 0, 0
    lines += ['## head_freq_90 theo năm', '', '| năm | n | IC |', '|---|---:|---:|']
    for y in range(2007, 2027):
        m = allm & (yrs == y)
        if m.sum() < 100:
            continue
        ic, _, n = ic_stats(cand['head_freq_90'], c, m)
        out['yearly_head90'][y] = ic
        pos += ic > 0
        tot += 1
        lines.append(f'| {y} | {n} | {ic:+.4f} |')
    from scipy.stats import binomtest

    lines += ['', f'Năm IC>0: {pos}/{tot}; binomial p (một phía) = {binomtest(int(pos), tot, 0.5, alternative="greater").pvalue:.4f}', '']

    # 3. kinh tế
    z = lambda a: _cross_z(a)  # noqa: E731
    h90, t90, f90 = cand['head_freq_90'], cand['tail_freq_90'], sig['freq_90']
    rules = {
        'baseline_freq_90': f90,
        'H1_head90_then_freq90': h90 * 1e6 + f90,
        'H2_z_head+tail+freq90': z(h90) + z(t90) + z(f90),
    }
    lines += ['## Quy tắc chọn (excess = nháy/pick vượt TB cả 100 số; break-even = +%.5f)' % BREAK_EVEN, '',
              '| rule | K | period | excess [CI95] | ROI flat ước lượng |', '|---|---:|---|---|---:|']
    for rname, sc in rules.items():
        for k in (2, 5):
            for pname, (lo, hi) in PERIODS.items():
                m = period_mask(dates, lo, hi)
                ex = topk_excess(sc[m], c[m], k)
                em, elo, ehi = block_boot_ci(ex)
                roi = (0.27 + em) * 99 / 27 - 1
                out['rules'][f'{rname}|k{k}|{pname}'] = {'excess': em, 'lo': elo, 'hi': ehi, 'roi': roi}
                lines.append(f'| {rname} | {k} | {pname} | {em:+.4f} [{elo:+.4f}, {ehi:+.4f}] | {roi:+.2%} |')
    md = '\n'.join(lines)
    (RESULTS / 'head_signal_followup.md').write_text(md, encoding='utf-8')
    (RESULTS / 'head_signal_followup.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
