"""High-power, leakage-safe signal-existence study for XSMB (2-digit loto).

Hỏi một câu duy nhất: có tín hiệu nào từ lịch sử (< T) dự báo được số nháy của ngày T tốt hơn ngẫu nhiên không?

Giao thức (chốt TRƯỚC khi xem kết quả):
- Dữ liệu cắt cứng trước ngày bắt đầu locked holdout (đọc từ evaluation_policy.json). Holdout không bao giờ được chạm.
- Mọi tín hiệu tại draw t chỉ dùng C[:t] (kiểm tra bằng --poison-check).
- Ba giai đoạn chronological cố định: DEV 2007-2015 | VAL 2016-2020 | FINAL 2021 -> trước holdout.
  Hyperparameter/ứng viên chỉ được CHỌN trên DEV (+VAL để xác nhận); FINAL chỉ để báo cáo.
- Metric theo đúng kinh tế sản phẩm (trả 99k / nháy, chi 27k / số): số nháy của số được chọn.
  * IC: Spearman(score, count) mỗi ngày trên 100 số, trung bình theo ngày (công suất cao nhất).
  * Excess hits@K: nháy trung bình của top-K trừ nháy trung bình của cả 100 số cùng ngày
    (kỳ vọng ngẫu nhiên chính xác, không nhiễu realization). Break-even excess = 27/99 - 27/100 = 0.0027...
    nên ROI flat ~ 99 * excess / 27 + (99*0.27-27)/27.
- Bất định: moving-block bootstrap (block 7) seed 42; hiệu chỉnh đa so sánh Bonferroni theo số tín hiệu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data' / 'xsmb-2-digits.csv'
POLICY = ROOT / 'predictions' / 'evaluation_policy.json'
RESULTS = ROOT / 'backtests' / 'results'

WARMUP = 400
PERIODS = {
    'DEV 2007-2015': ('2007-01-01', '2015-12-31'),
    'VAL 2016-2020': ('2016-01-01', '2020-12-31'),
    'FINAL 2021-pre-holdout': ('2021-01-01', '2099-01-01'),
}
SEED = 42


def holdout_start() -> str:
    return json.loads(POLICY.read_text(encoding='utf-8'))['locked_holdout']['start_date']


def load_counts(cut_date: str | None = None) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Trả (dates, C) với C[t, n] = số lần số n xuất hiện ngày t (0..27), chỉ các ngày < cut_date."""
    df = pd.read_csv(CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    cut = pd.Timestamp(cut_date or holdout_start())
    df = df[df['date'] < cut].reset_index(drop=True)
    vals = df.drop(columns=['date']).to_numpy(dtype=np.int64)
    c = np.zeros((len(df), 100), dtype=np.float64)
    for j in range(vals.shape[1]):
        np.add.at(c, (np.arange(len(df)), vals[:, j]), 1.0)
    return pd.DatetimeIndex(df['date']), c


def _rev(n: np.ndarray) -> np.ndarray:
    return (n % 10) * 10 + n // 10


def build_signals(dates: pd.DatetimeIndex, c: np.ndarray) -> dict[str, np.ndarray]:
    """Mỗi signal là mảng (N, 100); hàng t CHỈ phụ thuộc c[:t]."""
    n_days = len(c)
    cs = np.vstack([np.zeros((1, 100)), np.cumsum(c, axis=0)])  # cs[t] = sum c[:t]
    sig: dict[str, np.ndarray] = {}
    idx = np.arange(n_days)

    for w in (7, 14, 30, 60, 90, 180, 365):
        lo = np.maximum(0, idx - w)
        sig[f'freq_{w}'] = cs[idx] - cs[lo]
    sig['freq_all'] = cs[idx].copy()

    for hl in (15, 45, 90, 180, 365):
        a = 0.5 ** (1.0 / hl)
        out = np.zeros((n_days, 100))
        e = np.zeros(100)
        for t in range(n_days):
            out[t] = e  # trạng thái trước khi thấy c[t]
            e = a * e + (1 - a) * c[t]
        sig[f'ewma_hl{hl}'] = out

    # delay: số draw kể từ lần cuối xuất hiện (chưa từng xuất hiện -> t+1)
    delay = np.zeros((n_days, 100))
    last = np.full(100, -1, dtype=np.int64)
    for t in range(n_days):
        delay[t] = t - last
        last = np.where(c[t] > 0, t, last)
    sig['delay'] = delay
    mean_gap = np.maximum(cs[idx], 1.0)  # tổng nháy tới t; gap trung bình ~ t / count
    sig['delay_over_meangap'] = delay * mean_gap / np.maximum(idx[:, None], 1)

    prev1 = np.vstack([np.zeros((1, 100)), c[:-1]])
    prev2 = np.vstack([np.zeros((2, 100)), c[:-2]])
    sig['last1'] = prev1
    sig['last1_or_2'] = prev1 + prev2
    rev = _rev(np.arange(100))
    sig['inverted_last1'] = prev1[:, rev]

    # tần suất cùng thứ tự trong tuần (104 draw cùng weekday gần nhất)
    dow = np.asarray(dates.dayofweek)
    dow_freq = np.zeros((n_days, 100))
    for d in range(7):
        pos = np.flatnonzero(dow == d)
        for k, t in enumerate(pos):
            lo = max(0, k - 104)
            if k > lo:
                dow_freq[t] = c[pos[lo:k]].sum(axis=0)
    sig['dow_freq_104'] = dow_freq

    # phân phối chữ số: tần suất đầu/đuôi gần đây (90 draw), chuẩn hóa theo số lần
    heads, tails = np.arange(100) // 10, np.arange(100) % 10
    f90 = sig['freq_90']
    head_f = np.stack([f90[:, heads == h].sum(axis=1) for h in range(10)], axis=1)
    tail_f = np.stack([f90[:, tails == h].sum(axis=1) for h in range(10)], axis=1)
    sig['head_freq_90'] = head_f[:, heads]
    sig['tail_freq_90'] = tail_f[:, tails]
    return sig


def spearman_rows(score: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Spearman ngang hàng; hàng hằng số -> 0."""
    rs = np.apply_along_axis(rankdata, 1, score)
    rt = np.apply_along_axis(rankdata, 1, target)
    rs -= rs.mean(axis=1, keepdims=True)
    rt -= rt.mean(axis=1, keepdims=True)
    den = np.sqrt((rs**2).sum(axis=1) * (rt**2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        ic = (rs * rt).sum(axis=1) / den
    return np.nan_to_num(ic, nan=0.0)


def topk_excess(score: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Nháy TB của top-k trừ nháy TB cả 100 số (cùng ngày).

    Tie ở biên được chia đều theo kỳ vọng (không random) để không tạo nhiễu realization.
    """
    out = np.empty(len(score))
    for i in range(len(score)):
        s = score[i]
        order = np.argsort(-s, kind='stable')
        sel = np.zeros(100)
        remaining, j = float(k), 0
        while remaining > 1e-9 and j < 100:
            grp = np.flatnonzero(s == s[order[j]])
            take = min(remaining, float(len(grp)))
            sel[grp] = take / len(grp)
            remaining -= take
            j += len(grp)
        out[i] = (sel * target[i]).sum() / k - target[i].mean()
    return out

def block_boot_ci(x: np.ndarray, block: int = 7, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """Mean, CI95 thấp/cao bằng circular moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def period_mask(dates: pd.DatetimeIndex, lo: str, hi: str) -> np.ndarray:
    m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
    m &= np.arange(len(dates)) >= WARMUP
    return np.asarray(m)


def _cross_z(a: np.ndarray) -> np.ndarray:
    """Chuẩn hóa chéo theo từng ngày (không dùng thông tin giữa các ngày)."""
    mu = a.mean(axis=1, keepdims=True)
    sd = a.std(axis=1, keepdims=True)
    return (a - mu) / np.where(sd > 0, sd, 1.0)


def ridge_walk_forward(
    sig: dict[str, np.ndarray],
    c: np.ndarray,
    names: list[str],
    refit_every: int = 250,
    train_days: int = 1500,
    alpha: float = 1e4,
) -> np.ndarray:
    """Kết hợp tuyến tính có học (ridge) – cận trên của tín hiệu 'có thể học' từ chính các feature này.

    Tại draw t dùng mô hình fit trên các draw < t_refit <= t với nhãn đã biết (target của draw s < t_refit).
    """
    n_days = len(c)
    feats = np.stack([_cross_z(sig[k]) for k in names], axis=2)  # (N,100,F)
    target = c - c.mean(axis=1, keepdims=True)
    pred = np.zeros((n_days, 100))
    coef = None
    for t in range(WARMUP, n_days):
        if coef is None or (t - WARMUP) % refit_every == 0:
            lo = max(WARMUP // 2, t - train_days)
            x = feats[lo:t].reshape(-1, feats.shape[2])
            y = target[lo:t].reshape(-1)
            xtx = x.T @ x + alpha * np.eye(x.shape[1])
            coef = np.linalg.solve(xtx, x.T @ y)
        pred[t] = feats[t] @ coef
    return pred


def evaluate(dates, c, scores: dict[str, np.ndarray], ks=(2, 10)) -> dict:
    res: dict = {}
    for name, sc in scores.items():
        res[name] = {}
        for pname, (lo, hi) in PERIODS.items():
            m = period_mask(dates, lo, hi)
            ic = spearman_rows(sc[m], c[m])
            ic_mean, ic_lo, ic_hi = block_boot_ci(ic)
            row = {'n_draws': int(m.sum()), 'ic': ic_mean, 'ic_lo': ic_lo, 'ic_hi': ic_hi,
                   'ic_t': float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))}
            for k in ks:
                ex = topk_excess(sc[m], c[m], k)
                em, elo, ehi = block_boot_ci(ex)
                row[f'top{k}_excess'] = em
                row[f'top{k}_lo'] = elo
                row[f'top{k}_hi'] = ehi
            res[name][pname] = row
    return res


def poison_check(dates, c, sig, n_checks: int = 5) -> bool:
    """Đổi toàn bộ c[t:] thành rác rồi dựng lại signal: hàng t phải không đổi."""
    rng = np.random.default_rng(0)
    ok = True
    for t in rng.integers(WARMUP, len(c) - 1, size=n_checks):
        c2 = c.copy()
        c2[t:] = rng.integers(0, 3, size=c2[t:].shape).astype(float)
        sig2 = build_signals(dates, c2)
        for k, v in sig.items():
            if not np.allclose(v[t], sig2[k][t]):
                print(f'LEAK in {k} at t={t}')
                ok = False
    return ok


FREQ_FAMILY = ['freq_7', 'freq_14', 'freq_30', 'freq_60', 'freq_90', 'freq_180', 'freq_365', 'freq_all',
               'ewma_hl15', 'ewma_hl45', 'ewma_hl90', 'ewma_hl180', 'ewma_hl365']


def topk_weights(score: np.ndarray, k: int) -> np.ndarray:
    """Ma trận trọng số chọn top-k (N,100), tie ở biên chia đều. Chỉ phụ thuộc score nên tái dùng được khi hoán vị nhãn."""
    w = np.zeros(score.shape)
    for i in range(len(score)):
        s = score[i]
        order = np.argsort(-s, kind='stable')
        remaining, j = float(k), 0
        while remaining > 1e-9 and j < 100:
            grp = np.flatnonzero(s == s[order[j]])
            take = min(remaining, float(len(grp)))
            w[i, grp] = take / len(grp)
            remaining -= take
            j += len(grp)
    return w


def shift_null(dates, c, sig, k: int = 2, n_shift: int = 3000, lo: str = '2007-01-01', hi: str = '2099-01-01') -> dict:
    """Null hoán vị theo dịch vòng của nhãn: giữ nguyên cấu trúc tự tương quan của tín hiệu và nhãn,
    chỉ phá sự căn chỉnh thời gian giữa chúng. Thống kê: top-k excess gộp toàn kỳ; và MAX trên cả họ tần suất
    (hiệu chỉnh đa so sánh đúng nghĩa)."""
    m = period_mask(dates, lo, hi)
    idx = np.flatnonzero(m)
    n = len(idx)
    cm = c[idx]
    centered = (cm - cm.mean(axis=1, keepdims=True))
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(60, n - 60, size=n_shift)  # tránh dịch quá nhỏ (vẫn tự tương quan với chính nó)
    obs = {}
    stats_null = np.zeros((n_shift, len(FREQ_FAMILY)))
    for j, name in enumerate(FREQ_FAMILY):
        w = topk_weights(sig[name][idx], k)
        obs[name] = float((w * centered).sum() / (k * n))
        for si, s in enumerate(shifts):
            stats_null[si, j] = (w * np.roll(centered, s, axis=0)).sum() / (k * n)
    sd = stats_null.std(axis=0)
    out = {'k': k, 'n_draws': int(n), 'n_shift': n_shift, 'signals': {}}
    z_obs = np.array([obs[nm] for nm in FREQ_FAMILY]) / sd
    z_null_max = (stats_null / sd).max(axis=1)
    for j, name in enumerate(FREQ_FAMILY):
        p_one = float((stats_null[:, j] >= obs[name]).mean())
        out['signals'][name] = {'excess': obs[name], 'null_sd': float(sd[j]), 'z': float(z_obs[j]), 'p_one_sided': p_one,
                                'p_max_adjusted': float((z_null_max >= z_obs[j]).mean())}
    return out


def to_markdown(res: dict, n_signals: int, hold: str, n_total: int) -> str:
    from scipy.stats import norm

    thr = float(norm.ppf(1 - 0.025 / n_signals))
    lines = [
        '# Signal power study (chronological, pre-holdout)',
        '',
        f'Dữ liệu: {n_total} draw trước holdout {hold}; warm-up {WARMUP}; tín hiệu tại t chỉ dùng c[:t].',
        f'Bonferroni |t| ngưỡng cho {n_signals} tín hiệu: {thr:.2f}. IC = Spearman(score, count) trung bình theo ngày.',
        'excess = nháy TB/pick của top-K trừ nháy TB/pick của cả 100 số cùng ngày; ROI flat ≈ 99·excess/27 + base.',
        '',
    ]
    for pname in PERIODS:
        lines += [f'## {pname}', '', '| signal | n | IC | IC t | top2 excess [CI95] | top10 excess [CI95] |',
                  '|---|---:|---:|---:|---|---|']
        rows = sorted(res.items(), key=lambda kv: -abs(kv[1][pname]['ic_t']))
        for name, d in rows:
            r = d[pname]
            lines.append(
                f"| {name} | {r['n_draws']} | {r['ic']:+.4f} | {r['ic_t']:+.2f} | "
                f"{r['top2_excess']:+.4f} [{r['top2_lo']:+.4f}, {r['top2_hi']:+.4f}] | "
                f"{r['top10_excess']:+.4f} [{r['top10_lo']:+.4f}, {r['top10_hi']:+.4f}] |"
            )
        lines.append('')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--poison-check', action='store_true')
    ap.add_argument('--shift-null', action='store_true')
    ap.add_argument('--out', default=str(RESULTS / 'signal_power_study'))
    args = ap.parse_args()

    dates, c = load_counts()
    hold = holdout_start()
    print(f'draws={len(c)} first={dates[0].date()} last={dates[-1].date()} holdout_start={hold}')
    assert dates[-1] < pd.Timestamp(hold)
    sig = build_signals(dates, c)
    if args.poison_check:
        small = slice(0, 1200)
        ok = poison_check(dates[small], c[small], build_signals(dates[small], c[small]))
        print('poison_check', 'PASS' if ok else 'FAIL')
        if not ok:
            raise SystemExit(1)

    if args.shift_null:
        out = {}
        for lo, hi, tag in (('2007-01-01', '2099-01-01', 'ALL'), ('2007-01-01', '2015-12-31', 'DEV'), ('2016-01-01', '2020-12-31', 'VAL'), ('2021-01-01', '2099-01-01', 'FINAL')):
            r = shift_null(dates, c, sig, k=2, lo=lo, hi=hi)
            out[tag] = r
            print(tag, r['n_draws'])
            for nm, v in r['signals'].items():
                print(f"  {nm:11s} excess={v['excess']:+.4f} z={v['z']:+.2f} p={v['p_one_sided']:.3f} p_maxadj={v['p_max_adjusted']:.3f}")
        Path(args.out + '_shift_null.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
        return

    scores = dict(sig)
    base_names = ['freq_30', 'freq_90', 'freq_365', 'ewma_hl90', 'delay_over_meangap', 'last1', 'dow_freq_104',
                  'inverted_last1', 'head_freq_90', 'tail_freq_90']
    scores['RIDGE_combo'] = ridge_walk_forward(sig, c, base_names)
    scores['RANDOM_noise'] = np.random.default_rng(SEED).normal(size=c.shape)  # kiểm chứng calibration của test
    res = evaluate(dates, c, scores)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    md = to_markdown(res, n_signals=len(scores) - 1, hold=hold, n_total=len(c))
    out.with_suffix('.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
