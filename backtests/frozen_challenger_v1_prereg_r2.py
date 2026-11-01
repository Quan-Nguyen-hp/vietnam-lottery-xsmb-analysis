"""Frozen Challenger V1 — Pre-registration REVISION R2 (corrective).

KHÔNG đổi prediction rule. Module này import nguyên rule/metric đọc dữ liệu từ `frozen_challenger_v1.py`
(hash code v1 được ghim trong manifest R2 cùng fingerprint của toàn bộ picks trên dữ liệu trước holdout),
và thay thế phần *gate xác nhận + cỡ mẫu* bằng hai kết luận tách biệt:

  S — Scientific confirmation : IC dương có ý nghĩa thống kê trên holdout độc lập.
  E — Economic qualification  : excess top-2 vượt break-even (ROI flat > 0) với bằng chứng thống kê,
                                tức cận dưới một phía 97.5% của excess > break-even.  Yêu cầu S đã đạt.

S KHÔNG tự động dẫn tới E. 'NOT economically qualified' không có nghĩa 'đã chứng minh lỗ'.

Lệnh: power-audit | freeze | verify | forecast --date | score-holdout
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
from scipy.stats import norm, t as student_t

import frozen_challenger_v1 as v1

REVISION = 'frozen-challenger-v1-prereg-r2'
OUT_DIR = v1.OUT_DIR / 'prereg_r2'
SPEC_PATH = OUT_DIR / 'spec.json'
MANIFEST_PATH = OUT_DIR / 'manifest.json'
AUDIT_PATH = OUT_DIR / 'power_audit.json'
RESULT_PATH = v1.RESULT_PATH  # dùng chung với v1: holdout chỉ được chấm MỘT lần cho mọi revision

BE = v1.BREAK_EVEN_EXCESS
ALPHA = v1.ALPHA
POWER = v1.POWER
Z_A = float(norm.ppf(1 - ALPHA))
Z_B = float(norm.ppf(POWER))


# ----------------------------------------------------------------------------- thống kê


def lower_bound(x: np.ndarray, n_boot: int = v1.BOOT_N, seed: int = v1.SEED) -> float:
    """Cận dưới một phía 97.5% của trung bình (circular moving-block bootstrap, giống v1)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = math.ceil(n / v1.BOOT_BLOCK)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(v1.BOOT_BLOCK)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    return float(np.percentile(x[idx].mean(axis=1), 100 * ALPHA))


def power_s(true_ic: float, n: int, sd_ic: float) -> float:
    """Power của kiểm định S (một phía) nếu IC thật = true_ic."""
    return float(norm.cdf(true_ic / (sd_ic / math.sqrt(n)) - Z_A))


def power_e(true_ex: float, n: int, sd_ex: float) -> float:
    """Power của E1 (cận dưới > break-even) nếu excess top-2 thật = true_ex."""
    return float(norm.cdf((true_ex - BE) / (sd_ex / math.sqrt(n)) - Z_A))


def n_for_e(true_ex: float, sd_ex: float, power: float = POWER) -> float:
    return ((Z_A + float(norm.ppf(power))) * sd_ex / (true_ex - BE)) ** 2


def decide(res: dict) -> dict:
    """Hai kết luận tách biệt. E chỉ xét khi S đạt; E dùng cận dưới so với BREAK-EVEN, không phải 0."""
    s = bool(res['ic_lower_975'] > 0)
    e = bool(s and res['top2_excess_lower_975'] > BE)
    outcome = ('SCIENTIFIC_CONFIRMED_ECONOMIC_QUALIFIED' if e else
               'SCIENTIFIC_CONFIRMED_NOT_ECONOMICALLY_QUALIFIED' if s else 'NOT_CONFIRMED')
    nxt = {
        'SCIENTIFIC_CONFIRMED_ECONOMIC_QUALIFIED': 'Control Plane may consider a separate paper-trade comparison; production unchanged by this spec',
        'SCIENTIFIC_CONFIRMED_NOT_ECONOMICALLY_QUALIFIED': 'retain evidence only; NO production or paper-trade change; may motivate independent replication',
        'NOT_CONFIRMED': 'abandon challenger; no retuning on holdout',
    }[outcome]
    return {'S_scientific_confirmed': s, 'E_economic_qualified': e, 'outcome': outcome, 'allowed_next_step': nxt}


def evaluate(score: np.ndarray, counts: np.ndarray) -> dict:
    ic, ex = v1.daily_metrics(score, counts)
    n = len(ic)
    lo_ex = lower_bound(ex)
    roi = lambda e: float((v1.PRIZES / v1.N_NUM + e) * 99 / 27 - 1)  # noqa: E731
    return {
        'n_draws': int(n),
        'ic_mean': float(ic.mean()), 'ic_sd': float(ic.std(ddof=1)),
        'ic_t': float(ic.mean() / (ic.std(ddof=1) / math.sqrt(n))),
        'ic_lower_975': lower_bound(ic),
        'top2_excess_mean': float(ex.mean()), 'top2_excess_sd': float(ex.std(ddof=1)),
        'top2_excess_lower_975': lo_ex,
        'break_even_excess': BE,
        'top2_excess_minus_break_even': float(ex.mean() - BE),
        'roi_flat_equiv_mean': roi(float(ex.mean())), 'roi_flat_equiv_lower_975': roi(lo_ex),
    }


# ----------------------------------------------------------------------------- dữ liệu / forecast


def forecast_from_lines(lines: list[str], t: str) -> list[int]:
    """Top-2 cho ngày t chỉ từ các dòng có ngày < t (target t và tương lai bị loại BẰNG LỌC NGÀY trước khi parse)."""
    keep = [lines[0]] + [ln for ln in lines[1:] if ln.split(',', 1)[0].strip() < t]
    _, c = v1.counts_from_lines(keep, strict=False)
    return v1.pick_top(v1.forecast_scores(c))


def score_window(lines: list[str], hold: str, n_target: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(dates, scores, counts) cho n_target draw hợp lệ đầu tiên có ngày >= hold. Score hàng t chỉ dùng c[:t]."""
    dates, c = v1.counts_from_lines(lines, strict=False)
    first = next((i for i, d in enumerate(dates) if d >= hold), None)
    if first is None or len(dates) - first < n_target:
        raise ValueError('Chưa đủ draw hợp lệ')
    sel = slice(first, first + n_target)
    return dates[sel], v1.scores_all(c)[sel], c[sel]


# ----------------------------------------------------------------------------- hash


def _canon(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_sha256() -> str:
    return sha(Path(__file__).read_bytes().replace(b'\r\n', b'\n'))


def rule_fingerprint(c: np.ndarray) -> str:
    """Hash toàn bộ picks của rule trên mọi ngày t >= MIN_HISTORY của dữ liệu trước holdout."""
    sc = v1.scores_all(c)
    picks = [v1.pick_top(sc[t]) for t in range(v1.MIN_HISTORY, len(c))]
    return sha(json.dumps(picks).encode())


# ----------------------------------------------------------------------------- power audit độc lập


def _simulate(ic: np.ndarray, ex: np.ndarray, true_ic: float, true_ex: float, n: int, n_sims: int,
              n_boot: int, floor: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ic0, ex0 = ic - ic.mean(), ex - ex.mean()
    s = e1 = eco = old = 0
    for _ in range(n_sims):
        idx = rng.integers(0, len(ic0), size=n)
        a, b = ic0[idx] + true_ic, ex0[idx] + true_ex
        bs = int(rng.integers(1 << 31))
        sci = lower_bound(a, n_boot, bs) > 0
        e_ok = lower_bound(b, n_boot, bs + 1) > BE
        s += sci
        e1 += e_ok
        eco += sci and e_ok
        old += sci and a.mean() >= floor and b.mean() > 0  # gate G1-G3 của v1 (đã thay thế)
    return {'S': s / n_sims, 'E1_only': e1 / n_sims, 'S_and_E': eco / n_sims, 'v1_G1G2G3': old / n_sims}


def cmd_power_audit(args: argparse.Namespace) -> None:
    spec1 = json.loads(v1.SPEC_PATH.read_text(encoding='utf-8'))
    d, n_t = spec1['discovery_stats_in_sample_not_evidence'], spec1['sample_size']['n_target_valid_draws']
    sd_ic, sd_ex, kappa = d['sd_ic_daily'], d['sd_top2_excess_daily'], d['kappa_excess_per_ic']
    delta, floor = spec1['sample_size']['delta_ic'], spec1['gate']['ic_floor']
    lines, dates, _ = v1.read_prefix(spec1['data']['prefix_cutoff_exclusive'])
    _, c = v1.counts_from_lines(lines, strict=True)
    sc = v1.scores_all(c)
    ic, ex = v1.daily_metrics(sc[v1.WARMUP:], c[v1.WARMUP:])

    out: dict = {'n_target': n_t, 'sd_ic': sd_ic, 'sd_ex': sd_ex, 'kappa': kappa, 'delta_ic': delta}
    # 1) SE độc lập
    out['se'] = {'ic_at_n_target': sd_ic / math.sqrt(n_t), 'excess_at_n_target': sd_ex / math.sqrt(n_t),
                 'excess_at_700_wrong_claim': sd_ex / math.sqrt(700)}
    # 2) Cỡ mẫu: công thức chuẩn vs t-Student (df = n-1) giải bằng vòng lặp
    n_norm = ((Z_A + Z_B) * sd_ic / delta) ** 2
    n_t_dist = next(n for n in range(10, 100000) if
                    1 - student_t.cdf(student_t.ppf(1 - ALPHA, n - 1) - delta / (sd_ic / math.sqrt(n)), n - 1) >= POWER)
    out['n_required'] = {'normal_formula': n_norm, 't_distribution_search': n_t_dist,
                         'with_5pct_inflation': math.ceil(n_norm * v1.INVALID_INFLATION)}
    # 3) Power S: closed form tại các IC thật
    true_ics = [0.002, 0.003, 0.004, 0.0045, 0.00531, delta, 0.0075]
    out['power_S_closed_form_at_n_target'] = {f'{x:.5f}': power_s(x, n_t, sd_ic) for x in true_ics}
    # 4) Power E: closed form + N cần thiết
    true_exs = [0.005, 0.0066, 0.01, 0.02, 0.0235, 0.03]
    out['power_E_closed_form_at_n_target'] = {f'{x:.4f}': power_e(x, n_t, sd_ex) for x in true_exs}
    out['n_needed_for_E_80pct'] = {f'{x:.4f}': {'draws': n_for_e(x, sd_ex), 'years': n_for_e(x, sd_ex) / 365.25}
                                   for x in (0.005, 0.0066, 0.01, 0.02)}
    out['min_true_excess_for_E_80pct_at_n_target'] = BE + (Z_A + Z_B) * sd_ex / math.sqrt(n_t)
    # 5) Monte Carlo với gate thật (bootstrap) trên phân bố ngày thực nghiệm trước holdout
    scen = {'null (IC=0, ex=0)': (0.0, 0.0), 'MWU (ic=delta, ex=0.005)': (delta, delta * kappa),
            'shrunk 0.7x': (0.7 * delta, 0.7 * delta * kappa), 'shrunk 0.5x': (0.5 * delta, 0.5 * delta * kappa),
            'break-even boundary (ex=BE)': (BE / kappa, BE),
            'large (ex=min for E 80%)': (out['min_true_excess_for_E_80pct_at_n_target'] / kappa,
                                         out['min_true_excess_for_E_80pct_at_n_target'])}
    out['monte_carlo'] = {'n_sims': args.sims, 'n_boot': args.boot, 'results': {}}
    for i, (name, (ti, te)) in enumerate(scen.items()):
        out['monte_carlo']['results'][name] = {'true_ic': ti, 'true_ex': te,
                                              **_simulate(ic, ex, ti, te, n_t, args.sims, args.boot, floor, 1000 + i)}
    # 6) Kiểm tra nhất quán tự động
    mc = out['monte_carlo']['results']
    cf_s = power_s(delta, n_t, sd_ic)
    checks = {
        'n_normal_vs_t_within_1pct': abs(n_norm - n_t_dist) / n_norm < 0.01,
        'S_power_closed_form_ge_80pct': cf_s >= 0.80,
        'S_power_MC_matches_closed_form_within_0.07': abs(mc['MWU (ic=delta, ex=0.005)']['S'] - cf_s) < 0.07,
        'S_type_I_error_le_0.05': mc['null (IC=0, ex=0)']['S'] <= 0.05,
        'E1_type_I_error_at_break_even_le_0.05': mc['break-even boundary (ex=BE)']['E1_only'] <= 0.05,
        'v1_joint_power_MC_matches_63.5pct_within_0.08': abs(mc['MWU (ic=delta, ex=0.005)']['v1_G1G2G3'] - 0.635) < 0.08,
        'E_power_at_MWU_is_low_le_0.15': power_e(0.005, n_t, sd_ex) <= 0.15,
    }
    out['checks'] = {k: bool(v) for k, v in checks.items()}
    out['all_checks_pass'] = all(out['checks'].values())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(out, indent=2, sort_keys=True))
    if not out['all_checks_pass']:
        raise SystemExit(1)


# ----------------------------------------------------------------------------- freeze / verify


def build_spec(spec1: dict, audit: dict) -> dict:
    n_t = spec1['sample_size']['n_target_valid_draws']
    mc = audit['monte_carlo']['results']
    return {
        'id': 'XSMB-FROZEN-CHALLENGER-V1', 'revision': REVISION, 'frozen_on': date.today().isoformat(),
        'supersedes': {'revision': 'frozen-challenger-v1.0.0 (gate G1-G3)', 'reason': [
            'G3 chỉ yêu cầu excess > 0 trong khi break-even là +0.00273: có thể PASS mà ROI kỳ vọng âm',
            'joint power G1-G3 = 63.5%, không phải 80% danh nghĩa; cỡ mẫu 2.561 không được mô tả là 80% cho cả gate']},
        'status': 'FROZEN_PROSPECTIVE_NOT_PRODUCTION',
        'prediction_rule': {'unchanged': True, 'v1_spec_sha256': v1.spec_sha256(spec1),
                            'v1_code_sha256': v1.code_sha256(), 'rule': spec1['rule']},
        'data': spec1['data'],
        'metrics': {'primary': spec1['primary_metric'], 'secondary': spec1['secondary_metric'],
                    'minimum_worthwhile_uplift': spec1['minimum_worthwhile_uplift']},
        'sample_size': {
            'n_target_valid_draws': n_t, 'approx_years': n_t / 365.25,
            'earliest_possible_score_date_approx': spec1['sample_size']['earliest_possible_score_date_approx'],
            'sd_ic_daily': audit['sd_ic'], 'sd_top2_excess_daily': audit['sd_ex'],
            'se_ic_at_n_target': audit['se']['ic_at_n_target'], 'se_top2_excess_at_n_target': audit['se']['excess_at_n_target'],
            'alpha_one_sided': ALPHA, 'nominal_power_target': POWER,
            'power_S_at_MWU_closed_form': audit['power_S_closed_form_at_n_target'][f"{audit['delta_ic']:.5f}"],
            'power_S_at_MWU_monte_carlo': mc['MWU (ic=delta, ex=0.005)']['S'],
            'power_S_at_0.7x_MWU_monte_carlo': mc['shrunk 0.7x']['S'],
            'power_S_at_0.5x_MWU_monte_carlo': mc['shrunk 0.5x']['S'],
            'power_E_at_MWU_closed_form': power_e(0.005, n_t, audit['sd_ex']),
            'min_true_excess_for_E_80pct_at_n_target': audit['min_true_excess_for_E_80pct_at_n_target'],
            'n_needed_for_E_80pct_at_MWU_draws': audit['n_needed_for_E_80pct']['0.0050']['draws'],
            'superseded_v1_joint_power_G1G2G3': 0.63515,
            'note': 'Power 80% chỉ áp dụng cho S (khoảng 82% tại N_target, nếu IC thật = IC của MWU). '
                    'E gần như không thể đạt trừ khi hiệu ứng thật lớn gấp nhiều lần discovery.',
            'power_audit_sha256': sha(AUDIT_PATH.read_bytes()),
        },
        'confirmatory_protocol': {
            'evaluation_window': 'N_target draw hợp lệ đầu tiên có ngày >= holdout start; một lần nhìn; không chấm giữa chừng',
            'holdout_start': spec1['gate']['holdout_start'],
            'S_scientific_confirmation': 'one-sided 97.5% block-bootstrap lower bound of mean IC > 0 (block 7, 10000, seed 42)',
            'E_economic_qualification': 'S passed AND one-sided 97.5% block-bootstrap lower bound of mean top-2 excess > break_even_excess '
                                        f'({BE:.6f}); NOT merely > 0',
            'E_informational_only': ['ROI flat equivalent of mean and of lower bound', 'excess minus break-even'],
            'outcomes': {
                'NOT_CONFIRMED': 'abandon; no retuning on holdout',
                'SCIENTIFIC_CONFIRMED_NOT_ECONOMICALLY_QUALIFIED': 'retain evidence; no production/paper-trade change; this is the expected outcome if the effect is real but small',
                'SCIENTIFIC_CONFIRMED_ECONOMIC_QUALIFIED': 'Control Plane may consider a separate paper-trade comparison; production unchanged by this spec'},
            'not_qualified_means_not_disproved': True,
            'one_shot': 'score-holdout writes holdout_result.json with mode x; shared with v1 so only one revision can ever score',
        },
        'holdout_usage_rules': {
            'allowed': 'outcomes of draws strictly before target date T (date < T) as history for the forecast of T',
            'forbidden': ['outcome of T', 'any later date', 'scoring or peeking before N_target valid draws exist'],
            'enforcement': ['forecast_from_lines filters rows by date < T before parsing',
                            'scores_all row t uses c[max(0,t-90):t] only (poison test)',
                            'validity of a future row never changes the score of T (window = 90 valid rows before T)',
                            'row validity is judged only from format (27 values in 0..99), never from outcome'],
        },
        'forbidden': spec1['forbidden'] + ['using S alone as evidence of profitability'],
        'limitations': [
            'discovery IC/excess are in-sample and selected after seeing all 3 periods (winner curse): true power below nominal',
            'rule IC in VAL 2016-2020 was only 0.0013 vs 0.0064/0.0071 elsewhere: effect not uniform over time',
            'one lottery, one draw/day: E needs hundreds of years at plausible effects; E can only pass for a very large effect',
            'scipy is used but not declared directly in pyproject.toml (transitive via scikit-learn)'],
    }


def cmd_freeze(_: argparse.Namespace) -> None:
    if MANIFEST_PATH.exists():
        raise SystemExit('Manifest R2 đã tồn tại: không ghi đè. Muốn đổi phải tạo revision mới.')
    errs = v1.verify()
    if errs:
        raise SystemExit('v1 không còn nguyên vẹn: ' + '; '.join(errs))
    if not AUDIT_PATH.exists():
        raise SystemExit('Chạy power-audit trước.')
    audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
    if not audit['all_checks_pass']:
        raise SystemExit('power-audit chưa đạt.')
    spec1 = json.loads(v1.SPEC_PATH.read_text(encoding='utf-8'))
    lines, _, _ = v1.read_prefix(spec1['data']['prefix_cutoff_exclusive'])
    _, c = v1.counts_from_lines(lines, strict=True)
    spec = build_spec(spec1, audit)
    SPEC_PATH.write_text(json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False), encoding='utf-8')
    man1 = json.loads(v1.MANIFEST_PATH.read_text(encoding='utf-8'))
    manifest = {
        'revision': REVISION, 'frozen_on': spec['frozen_on'],
        'spec_sha256': sha(_canon(spec)), 'code_sha256_r2': code_sha256(),
        'v1_manifest': man1, 'v1_code_sha256': v1.code_sha256(), 'v1_spec_sha256': v1.spec_sha256(spec1),
        'rule_fingerprint_prefix_picks_sha256': rule_fingerprint(c),
        'data_prefix_sha256': spec['data']['prefix_sha256'], 'power_audit_sha256': sha(AUDIT_PATH.read_bytes()),
    }
    with MANIFEST_PATH.open('x', encoding='utf-8') as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


def verify() -> list[str]:
    errs: list[str] = []
    if not (SPEC_PATH.exists() and MANIFEST_PATH.exists() and AUDIT_PATH.exists()):
        return ['thiếu spec/manifest/audit R2']
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    man = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    errs += [f'v1: {e}' for e in v1.verify()]
    if sha(_canon(spec)) != man['spec_sha256']:
        errs.append('spec R2 hash lệch')
    if code_sha256() != man['code_sha256_r2']:
        errs.append('code R2 hash lệch')
    if sha(AUDIT_PATH.read_bytes()) != man['power_audit_sha256']:
        errs.append('power audit hash lệch')
    if v1.code_sha256() != man['v1_code_sha256'] or man['v1_manifest']['code_sha256'] != man['v1_code_sha256']:
        errs.append('code rule v1 đã đổi')
    spec1 = json.loads(v1.SPEC_PATH.read_text(encoding='utf-8'))
    lines, _, _ = v1.read_prefix(spec1['data']['prefix_cutoff_exclusive'])
    _, c = v1.counts_from_lines(lines, strict=True)
    if rule_fingerprint(c) != man['rule_fingerprint_prefix_picks_sha256']:
        errs.append('fingerprint picks của rule đã đổi')
    return errs


def cmd_verify(_: argparse.Namespace) -> None:
    errs = verify()
    print('VERIFY R2', 'OK' if not errs else 'FAIL: ' + '; '.join(errs))
    raise SystemExit(1 if errs else 0)


def cmd_forecast(args: argparse.Namespace) -> None:
    date.fromisoformat(args.date)
    print(json.dumps({'date': args.date, 'revision': REVISION, 'picks': forecast_from_lines(v1.load_all_lines(), args.date)}))


def cmd_score_holdout(_: argparse.Namespace) -> None:
    errs = verify()
    if errs:
        raise SystemExit('Từ chối chấm: ' + '; '.join(errs))
    if RESULT_PATH.exists():
        raise SystemExit('Holdout đã được chấm một lần (bởi revision nào cũng vậy). Không chấm lại.')
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    hold, n_t = spec['confirmatory_protocol']['holdout_start'], spec['sample_size']['n_target_valid_draws']
    _, _, n_after = v1.read_prefix(hold)  # chỉ đếm theo cột ngày
    if n_after < n_t:
        raise SystemExit(f'Chưa mở gate: {n_after}/{n_t} dòng sau holdout. Không đọc/chấm kết quả.')
    try:
        dates, scores, counts = score_window(v1.load_all_lines(), hold, n_t)
    except ValueError as exc:
        raise SystemExit(f'{exc}. Không chấm.') from exc
    res = evaluate(scores, counts)
    out = {'revision': REVISION, 'spec_sha256': sha(_canon(spec)), 'code_sha256_r2': code_sha256(),
           'window': {'start': dates[0], 'end': dates[-1], 'n': n_t}, 'result': res, 'decision': decide(res)}
    with RESULT_PATH.open('x', encoding='utf-8') as fh:
        fh.write(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    pa = sub.add_parser('power-audit')
    pa.add_argument('--sims', type=int, default=500)
    pa.add_argument('--boot', type=int, default=500)
    pa.set_defaults(fn=cmd_power_audit)
    for name, fn in (('freeze', cmd_freeze), ('verify', cmd_verify), ('score-holdout', cmd_score_holdout)):
        sub.add_parser(name).set_defaults(fn=fn)
    fp = sub.add_parser('forecast')
    fp.add_argument('--date', required=True)
    fp.set_defaults(fn=cmd_forecast)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
