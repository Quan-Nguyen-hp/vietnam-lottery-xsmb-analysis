"""Reproduce the fixed-window stability diagnostic from the accuracy audit.

Run from any directory with --output pointing to a fresh artifact directory.
This is retrospective research; it does not alter production configuration.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backtests'))
from chronological_accuracy import check_window, paired_interval, score_picks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads((ROOT / 'predictions/evaluation_policy.json').read_text(encoding='utf-8'))
    check_window('2009-01-01', '2025-12-31', policy)
    csv = ROOT / 'data/xsmb-2-digits.csv'
    df = pd.read_csv(csv).sort_values('date').reset_index(drop=True)
    df = df[df.date < policy['locked_holdout']['start_date']].reset_index(drop=True)
    counts = np.zeros((len(df), 100), dtype=np.int64)
    for idx, nums in enumerate(df.drop(columns='date').to_numpy(dtype=int)):
        np.add.at(counts[idx], nums, 1)
    binary = counts > 0
    prefix = np.vstack([np.zeros(100), np.cumsum(binary, axis=0)])
    rows = []
    for idx in np.flatnonzero((df.date >= '2009-01-01') & (df.date <= '2025-12-31')):
        signals = {'frequency_all': prefix[idx] / idx}
        for window in (30, 90, 365):
            signals[f'frequency_{window}'] = (prefix[idx] - prefix[max(0, idx-window)]) / min(idx, window)
        sample = counts[max(0, idx-365):idx]
        weights = np.exp2(-np.arange(len(sample)-1, -1, -1) / 90)
        signals['count_ewma90'] = weights @ sample / weights.sum()
        for name, signal in signals.items():
            picks = np.argsort(signal)[::-1][:2].tolist()
            rows.append({'date': df.iloc[idx].date, 'method': name, **score_picks(counts[idx], picks),
                         'random_expected_unique_hits': 2 * binary[idx].mean(),
                         'numbers': ','.join(f'{number:02d}' for number in picks)})
    frame = pd.DataFrame(rows)
    metrics = {}
    for name, group in frame.groupby('method'):
        periods = []
        for start, end in (('2009-01-01', '2014-12-31'), ('2015-01-01', '2020-12-31'),
                           ('2021-01-01', '2025-12-31')):
            block = group[(group.date >= start) & (group.date <= end)]
            delta = (block.unique_hits - block.random_expected_unique_hits).to_numpy() / 2
            periods.append({'start': start, 'end': end, 'draws': len(block),
                            'unique_hits': int(block.unique_hits.sum()),
                            'precision': float(block.unique_hits.mean() / 2),
                            'mean_count_hits': float(block.count_hits.mean()),
                            'roi': float(block.pnl_k.sum() / (54 * len(block))),
                            'precision_lift_vs_uniform_expectation': float(delta.mean()),
                            'lift_ci95': paired_interval(delta)})
        metrics[name] = periods
    summary = {'classification': 'retrospective_fixed_signal_stability_diagnostic_not_untouched_holdout',
               'seed': 42, 'periods': metrics, 'dataset_sha256': hashlib.sha256(csv.read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / 'daily_scores.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
