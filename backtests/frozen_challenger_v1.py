"""Frozen Prospective Challenger V1 (XSMB) — rule đóng băng, chấm holdout đúng MỘT lần.

Rule (không tham số tự do, không tinh chỉnh):
    f[n]  = số nháy của số n trong 90 draw hợp lệ liền trước T
    H[h]  = tổng f[n] trên 10 số cùng chữ số đầu h;  L[l] = tổng f[n] trên 10 số cùng chữ số đuôi l
    score[n] = z(H[n // 10]) + z(L[n % 10]) + z(f[n])      (z = chuẩn hóa chéo 100 số trong ngày, sd tổng thể)
    picks    = top-2 theo score, đồng điểm -> số nhỏ hơn trước (score làm tròn 12 chữ số)

Quy ước giữ nguyên tinh thần dự án: chỉ dùng lịch sử có ngày < T; không LLM chọn số; không đụng policy production.

Lệnh:
    freeze         Đóng băng spec + manifest (từ chối ghi đè; đổi rule = phiên bản mới).
    verify         Kiểm tra hash code/spec/dữ liệu trước holdout khớp manifest.
    selftest       Chạy toàn bộ đường đo trên cửa sổ TRƯỚC holdout (không chạm holdout).
    forecast       In top-2 cho một ngày (chỉ dùng dòng có ngày < T).
    score-holdout  Chấm holdout một lần, chỉ khi đủ N_target draw hợp lệ; ghi holdout_result.json bằng mode 'x'.

Giai đoạn đóng băng KHÔNG đọc giá trị kết quả của các dòng thuộc holdout: file CSV được stream theo dòng,
với dòng ngày >= holdout chỉ phân tích cột ngày.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

VERSION = 'frozen-challenger-v1.0.0'
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data' / 'xsmb-2-digits.csv'
POLICY = ROOT / 'predictions' / 'evaluation_policy.json'
OUT_DIR = ROOT / 'backtests' / 'results' / 'frozen_challenger_v1'
SPEC_PATH = OUT_DIR / 'spec.json'
MANIFEST_PATH = OUT_DIR / 'manifest.json'
RESULT_PATH = OUT_DIR / 'holdout_result.json'

WINDOW = 90
TOP_K = 2
N_NUM = 100
PRIZES = 27
MIN_HISTORY = 90
WARMUP = 400  # chỉ dùng cho thống kê discovery/selftest, khớp signal_power_study
BREAK_EVEN_EXCESS = 27 / 99 - PRIZES / N_NUM  # excess nháy/số tối thiểu để flat ROI >= 0
MWU_EXCESS2 = 0.0050  # minimum worthwhile uplift của excess top-2 (nháy/số), ~2x break-even
ALPHA = 0.025  # một phía
POWER = 0.80
INVALID_INFLATION = 1.05  # dự phòng draw thiếu/không đủ 27 giá trị
BOOT_BLOCK = 7
BOOT_N = 10000
SEED = 42
PERIODS = {
    'DEV 2007-2015': ('2007-01-01', '2015-12-31'),
    'VAL 2016-2020': ('2016-01-01', '2020-12-31'),
    'FINAL 2021-pre-holdout': ('2021-01-01', '2099-12-31'),
}


# ----------------------------------------------------------------------------- dữ liệu


def holdout_start() -> str:
    return json.loads(POLICY.read_text(encoding='utf-8'))['locked_holdout']['start_date']


def _parse_line(line: str) -> tuple[str, list[int] | None]:
    parts = line.strip().split(',')
    values: list[int] | None
    try:
        values = [int(x) for x in parts[1:]]
        if len(values) != PRIZES or any(v < 0 or v > 99 for v in values):
            values = None
    except ValueError:
        values = None
    return parts[0], values


def read_prefix(cutoff: str) -> tuple[list[str], list[str], int]:
    """Stream CSV: trả (header+dòng ngày < cutoff, dates, số dòng ngày >= cutoff).

    Dòng ngày >= cutoff chỉ bị đọc phần ngày (cột đầu), không parse giá trị.
    """
    lines: list[str] = []
    dates: list[str] = []
    n_after = 0
    last = ''
    with CSV.open(encoding='utf-8') as fh:
        header = fh.readline().rstrip('\r\n')
        lines.append(header)
        for raw in fh:
            if not raw.strip():
                continue
            d = raw.split(',', 1)[0].strip()
            date.fromisoformat(d)
            if d <= last:
                raise ValueError(f'CSV không tăng dần theo ngày tại {d}')
            last = d
            if d < cutoff:
                lines.append(raw.rstrip('\r\n'))
                dates.append(d)
            else:
                n_after += 1
    return lines, dates, n_after


def counts_from_lines(lines: list[str], *, strict: bool) -> tuple[list[str], np.ndarray]:
    """Dòng -> (dates, C[t, n] = số lần n xuất hiện). Dòng không hợp lệ: strict -> lỗi, ngược lại bỏ qua."""
    out_dates: list[str] = []
    rows: list[list[int]] = []
    for line in lines[1:]:
        d, vals = _parse_line(line)
        if vals is None:
            if strict:
                raise ValueError(f'Dòng không hợp lệ {d}')
            continue
        out_dates.append(d)
        rows.append(vals)
    c = np.zeros((len(rows), N_NUM))
    for i, vals in enumerate(rows):
        np.add.at(c[i], vals, 1.0)
    return out_dates, c


def load_all_lines() -> list[str]:
    with CSV.open(encoding='utf-8') as fh:
        return [ln.rstrip('\r\n') for ln in fh if ln.strip()]


# ----------------------------------------------------------------------------- rule


def _z(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=1, keepdims=True)
    sd = a.std(axis=1, keepdims=True)
    return np.where(sd > 0, (a - mu) / np.where(sd > 0, sd, 1.0), 0.0)


def rule_scores(f: np.ndarray) -> np.ndarray:
    """f: (m, 100) số nháy trong 90 draw trước -> score (m, 100)."""
    m = len(f)
    grid = f.reshape(m, 10, 10)  # [ngày, đầu, đuôi]
    head = np.repeat(grid.sum(axis=2), 10, axis=1)  # index 10h+l -> H[h]
    tail = np.tile(grid.sum(axis=1), (1, 10))  # index 10h+l -> L[l]
    return np.round(_z(head) + _z(tail) + _z(f), 12)


def scores_all(c: np.ndarray) -> np.ndarray:
    """Score cho mọi hàng t, chỉ dùng c[max(0,t-90):t]. Hàng t < MIN_HISTORY không có nghĩa."""
    n = len(c)
    cs = np.vstack([np.zeros((1, N_NUM)), np.cumsum(c, axis=0)])
    idx = np.arange(n)
    return rule_scores(cs[idx] - cs[np.maximum(0, idx - WINDOW)])


def forecast_scores(history: np.ndarray) -> np.ndarray:
    """Score cho ngày kế tiếp từ ma trận đếm lịch sử (T-1 trở về trước)."""
    if len(history) < MIN_HISTORY:
        raise ValueError(f'Cần >= {MIN_HISTORY} draw lịch sử, có {len(history)}')
    return rule_scores(history[-WINDOW:].sum(axis=0)[None, :])[0]


def pick_top(score: np.ndarray, k: int = TOP_K) -> list[int]:
    return [int(x) for x in np.lexsort((np.arange(N_NUM), -score))[:k]]


# ----------------------------------------------------------------------------- metric


def daily_metrics(score: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trả (IC Spearman mỗi ngày, excess top-2 mỗi ngày = nháy TB của 2 pick trừ nháy TB cả 100 số)."""
    rs = rankdata(score, axis=1)
    rc = rankdata(counts, axis=1)
    rs = rs - rs.mean(axis=1, keepdims=True)
    rc = rc - rc.mean(axis=1, keepdims=True)
    den = np.sqrt((rs**2).sum(axis=1) * (rc**2).sum(axis=1))
    ic = np.where(den > 0, (rs * rc).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)
    picks = np.array([pick_top(s) for s in score])
    ex = np.take_along_axis(counts, picks, axis=1).mean(axis=1) - counts.mean(axis=1)
    return ic, ex


def block_bootstrap_lower(x: np.ndarray, seed: int = SEED) -> float:
    """Cận dưới một phía 97.5% của trung bình (circular moving-block bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = math.ceil(n / BOOT_BLOCK)
    starts = rng.integers(0, n, size=(BOOT_N, n_blocks))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]).reshape(BOOT_N, -1)[:, :n] % n
    return float(np.percentile(x[idx].mean(axis=1), 100 * ALPHA))


def evaluate(score: np.ndarray, counts: np.ndarray) -> dict:
    ic, ex = daily_metrics(score, counts)
    n = len(ic)
    return {
        'n_draws': int(n),
        'ic_mean': float(ic.mean()),
        'ic_sd': float(ic.std(ddof=1)),
        'ic_t': float(ic.mean() / (ic.std(ddof=1) / math.sqrt(n))),
        'ic_lower_975': block_bootstrap_lower(ic),
        'top2_excess_mean': float(ex.mean()),
        'top2_excess_sd': float(ex.std(ddof=1)),
        'top2_excess_lower_975': block_bootstrap_lower(ex),
        'top2_roi_flat_equiv': float((PRIZES / N_NUM + ex.mean()) * 99 / 27 - 1),
    }


def gate_decision(res: dict, gate: dict) -> dict:
    """Confirmatory gate — tất cả phải đúng. Hàm thuần để kiểm thử."""
    g1 = res['ic_lower_975'] > 0
    g2 = res['ic_mean'] >= gate['ic_floor']
    g3 = res['top2_excess_mean'] > 0
    return {'G1_ic_lower_975_gt_0': bool(g1), 'G2_ic_mean_ge_floor': bool(g2),
            'G3_top2_excess_gt_0': bool(g3), 'PASS': bool(g1 and g2 and g3)}


def required_n(sigma: float, delta: float, alpha: float = ALPHA, power: float = POWER) -> float:
    return ((norm.ppf(1 - alpha) + norm.ppf(power)) * sigma / delta) ** 2


def mde(sigma: float, n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    return (norm.ppf(1 - alpha) + norm.ppf(power)) * sigma / math.sqrt(n)


def joint_power(n: int, delta_ic: float, kappa: float, sd_ic: float, sd_ex: float, rho: float, ic_floor: float) -> float:
    """Mô phỏng chuẩn xấp xỉ: P(G1 & G2 & G3) nếu IC thật = delta_ic, excess thật = kappa*delta_ic."""
    rng = np.random.default_rng(SEED)
    cov = np.array([[sd_ic**2, rho * sd_ic * sd_ex], [rho * sd_ic * sd_ex, sd_ex**2]]) / n
    s = rng.multivariate_normal([delta_ic, kappa * delta_ic], cov, size=200_000)
    g1 = s[:, 0] > norm.ppf(1 - ALPHA) * sd_ic / math.sqrt(n)
    return float((g1 & (s[:, 0] >= ic_floor) & (s[:, 1] > 0)).mean())


# ----------------------------------------------------------------------------- hash / spec


def _canon(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_sha256() -> str:
    return sha256_bytes(Path(__file__).read_bytes().replace(b'\r\n', b'\n'))


def data_prefix_sha256(lines: list[str]) -> str:
    return sha256_bytes('\n'.join(lines).encode('utf-8'))


def spec_sha256(spec: dict) -> str:
    return sha256_bytes(_canon(spec))


def _period_stats(dates: list[str], ic: np.ndarray, ex: np.ndarray, offset: int) -> dict:
    out = {}
    for name, (lo, hi) in PERIODS.items():
        sel = np.array([lo <= dates[offset + i] <= hi for i in range(len(ic))])
        out[name] = {'n': int(sel.sum()), 'ic_mean': float(ic[sel].mean()), 'top2_excess_mean': float(ex[sel].mean())}
    return out


def build_spec(lines: list[str], dates: list[str], c: np.ndarray, hold: str) -> dict:
    sc = scores_all(c)
    lo = WARMUP
    ic, ex = daily_metrics(sc[lo:], c[lo:])
    sd_ic, sd_ex = float(ic.std(ddof=1)), float(ex.std(ddof=1))
    rho = float(np.corrcoef(ic, ex)[0, 1])
    kappa = float(np.cov(ex, ic)[0, 1] / ic.var(ddof=1))  # excess top-2 / IC (hồi quy trong discovery)
    delta_ic = MWU_EXCESS2 / kappa
    ic_floor = BREAK_EVEN_EXCESS / kappa
    n0 = required_n(sd_ic, delta_ic)
    n_target = math.ceil(n0 * INVALID_INFLATION)
    return {
        'id': 'XSMB-FROZEN-CHALLENGER-V1',
        'version': VERSION,
        'frozen_on': date.today().isoformat(),
        'status': 'FROZEN_PROSPECTIVE_NOT_PRODUCTION',
        'rule': {
            'formula': 'score[n] = z(H[n//10]) + z(L[n%10]) + z(f[n]); picks = top-2 by score',
            'f': 'count of number n over the 90 valid draws immediately before target date T (rows with date < T)',
            'H': 'sum of f over the 10 numbers sharing the same tens digit (head)',
            'L': 'sum of f over the 10 numbers sharing the same units digit (tail)',
            'z': 'per-day cross-sectional standardisation over 100 numbers, population sd; sd==0 -> zeros',
            'score_rounding_decimals': 12,
            'tie_break': 'lower number first',
            'window_draws': WINDOW, 'top_k': TOP_K, 'min_history_draws': MIN_HISTORY,
            'invalid_rows': 'rows without exactly 27 values in 0..99 are ignored entirely (not history, not evaluated)',
            'free_parameters_tuned_after_freeze': 0,
        },
        'data': {
            'csv': 'data/xsmb-2-digits.csv',
            'prefix_cutoff_exclusive': hold,
            'prefix_rows': len(dates),
            'prefix_last_date': dates[-1],
            'prefix_sha256': data_prefix_sha256(lines),
        },
        'primary_metric': {
            'name': 'mean daily Spearman IC between score and per-number draw count (100 numbers/day)',
            'why': 'full-ranking accuracy; highest statistical power for one draw/day',
            'aggregation': 'mean over evaluated draws; one-sided 97.5% lower bound by circular block bootstrap',
            'bootstrap': {'block': BOOT_BLOCK, 'resamples': BOOT_N, 'seed': SEED},
        },
        'secondary_metric': {
            'name': 'top-2 excess hits per pick = mean count of 2 picks - mean count over 100 numbers (same day)',
            'break_even_excess': BREAK_EVEN_EXCESS,
            'note': 'reported with CI; underpowered by design; used only for direction (G3) and for the paper-trade stage',
        },
        'minimum_worthwhile_uplift': {
            'top2_excess_hits_per_pick': MWU_EXCESS2,
            'roi_flat_equiv': (PRIZES / N_NUM + MWU_EXCESS2) * 99 / 27 - 1,
            'ic_equiv': delta_ic,
            'rationale': 'about 1.8x break-even excess; smaller uplift is not worth any production/paper-trade change',
        },
        'discovery_stats_in_sample_not_evidence': {
            'window': f'draws {WARMUP}..{len(dates) - 1} of the pre-holdout prefix',
            'sd_ic_daily': sd_ic, 'sd_top2_excess_daily': sd_ex, 'corr_ic_excess': rho,
            'kappa_excess_per_ic': kappa,
            'ic_mean': float(ic.mean()), 'top2_excess_mean': float(ex.mean()),
            'periods': _period_stats(dates, ic, ex, lo),
        },
        'sample_size': {
            'alpha_one_sided': ALPHA, 'power': POWER, 'delta_ic': delta_ic, 'sigma_ic': sd_ic,
            'n_required_raw': n0, 'invalid_inflation': INVALID_INFLATION, 'n_target_valid_draws': n_target,
            'approx_years_at_one_draw_per_day': n_target / 365.25,
            'earliest_possible_score_date_approx': date.fromordinal(date.fromisoformat(hold).toordinal() + n_target).isoformat(),
            'joint_power_all_gates': joint_power(n_target, delta_ic, kappa, sd_ic, sd_ex, rho, ic_floor),
            'mde_ic_by_n': {str(n): mde(sd_ic, n) for n in (180, 365, 730, 1460, n_target)},
            'mde_top2_excess_by_n': {str(n): mde(sd_ex, n) for n in (180, 365, 730, 1460, n_target)},
        },
        'gate': {
            'evaluation_window': 'first N_target valid draws with date >= holdout start; single look; no interim scoring',
            'holdout_start': hold,
            'G1': 'one-sided 97.5% block-bootstrap lower bound of mean IC > 0',
            'G2': 'mean IC >= ic_floor (break-even IC = break_even_excess / kappa)',
            'G3': 'mean top-2 excess > 0',
            'ic_floor': ic_floor,
            'after_pass': 'only then consider a separate paper-trade comparison against the production policy; production unchanged by this spec',
            'after_fail': 'ABANDON challenger; no retuning on the holdout; a new rule needs a new version and new data',
        },
        'forbidden': [
            'reading or scoring holdout outcomes before N_target valid draws exist',
            'changing rule, window, K, tie-break, metric, MWU or gate after freeze (new version required)',
            'selecting among variants using holdout data',
            'editing predictions/evaluation_policy.json or prediction_log.json from this module',
        ],
    }


def cmd_freeze(_: argparse.Namespace) -> None:
    if MANIFEST_PATH.exists():
        raise SystemExit('Manifest đã tồn tại: không ghi đè. Muốn đổi rule phải tạo phiên bản mới.')
    hold = holdout_start()
    lines, dates, n_after = read_prefix(hold)
    d, c = counts_from_lines(lines, strict=True)
    assert d == dates and (c.sum(axis=1) == PRIZES).all()
    spec = build_spec(lines, dates, c, hold)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False), encoding='utf-8')
    manifest = {'version': VERSION, 'spec_sha256': spec_sha256(spec), 'code_sha256': code_sha256(),
                'data_prefix_sha256': spec['data']['prefix_sha256'], 'frozen_on': spec['frozen_on']}
    with MANIFEST_PATH.open('x', encoding='utf-8') as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True))
    print(f'Frozen {VERSION}. Prefix rows={len(dates)}; dòng holdout hiện có (chỉ đếm ngày, chưa đọc giá trị): {n_after}')
    print(json.dumps({k: spec['sample_size'][k] for k in ('n_required_raw', 'n_target_valid_draws', 'approx_years_at_one_draw_per_day', 'joint_power_all_gates')}, indent=2))
    print(json.dumps(manifest, indent=2))


def verify() -> list[str]:
    errs: list[str] = []
    if not (SPEC_PATH.exists() and MANIFEST_PATH.exists()):
        return ['thiếu spec/manifest']
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    man = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    if spec_sha256(spec) != man['spec_sha256']:
        errs.append('spec hash lệch')
    if code_sha256() != man['code_sha256']:
        errs.append('code hash lệch')
    lines, dates, _ = read_prefix(spec['data']['prefix_cutoff_exclusive'])
    if data_prefix_sha256(lines) != man['data_prefix_sha256']:
        errs.append('dữ liệu trước holdout đã bị thay đổi')
    if len(dates) != spec['data']['prefix_rows']:
        errs.append('số dòng prefix lệch')
    return errs


def cmd_verify(_: argparse.Namespace) -> None:
    errs = verify()
    print('VERIFY', 'OK' if not errs else 'FAIL: ' + '; '.join(errs))
    raise SystemExit(1 if errs else 0)


def cmd_selftest(_: argparse.Namespace) -> None:
    hold = holdout_start()
    lines, dates, _ = read_prefix(hold)
    _, c = counts_from_lines(lines, strict=True)
    sc = scores_all(c)
    sel = slice(len(c) - 1000, len(c))
    res = evaluate(sc[sel], c[sel])
    print('selftest (1000 draw cuối TRƯỚC holdout, in-sample, không phải bằng chứng):')
    print(json.dumps(res, indent=2))


def cmd_forecast(args: argparse.Namespace) -> None:
    t = args.date
    date.fromisoformat(t)
    lines = [ln for ln in load_all_lines() if ln.split(',', 1)[0].strip() < t or ln.startswith('date')]
    _, c = counts_from_lines(lines, strict=False)
    print(json.dumps({'date': t, 'version': VERSION, 'picks': pick_top(forecast_scores(c))}))


def cmd_score_holdout(_: argparse.Namespace) -> None:
    errs = verify()
    if errs:
        raise SystemExit('Từ chối chấm: ' + '; '.join(errs))
    if RESULT_PATH.exists():
        raise SystemExit('Holdout đã được chấm một lần. Không chấm lại.')
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    hold, n_target = spec['gate']['holdout_start'], spec['sample_size']['n_target_valid_draws']
    _, _, n_after = read_prefix(hold)  # chỉ đếm theo cột ngày
    if n_after < n_target:
        raise SystemExit(f'Chưa mở gate: {n_after}/{n_target} dòng sau holdout. Không đọc/chấm kết quả.')
    dates, c = counts_from_lines(load_all_lines(), strict=False)
    first = next(i for i, d in enumerate(dates) if d >= hold)
    if len(dates) - first < n_target:
        raise SystemExit(f'Chưa đủ draw hợp lệ: {len(dates) - first}/{n_target}. Không chấm.')
    sel = slice(first, first + n_target)
    res = evaluate(scores_all(c)[sel], c[sel])
    out = {'version': VERSION, 'spec_sha256': spec_sha256(spec), 'code_sha256': code_sha256(),
           'window': {'start': dates[first], 'end': dates[first + n_target - 1], 'n': n_target},
           'result': res, 'gate': gate_decision(res, spec['gate'])}
    with RESULT_PATH.open('x', encoding='utf-8') as fh:
        fh.write(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('freeze', cmd_freeze), ('verify', cmd_verify), ('selftest', cmd_selftest),
                     ('score-holdout', cmd_score_holdout)):
        sub.add_parser(name).set_defaults(fn=fn)
    fp = sub.add_parser('forecast')
    fp.add_argument('--date', required=True)
    fp.set_defaults(fn=cmd_forecast)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
