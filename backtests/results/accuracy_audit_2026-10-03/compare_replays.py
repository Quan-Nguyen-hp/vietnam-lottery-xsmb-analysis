"""Reproduce the paired comparison of completed, frozen production replays."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backtests'))
from chronological_accuracy import paired_interval  # noqa: E402


def roi_interval(group):
    """Seven-draw circular block resampling of total PnL / total exposure."""
    size = len(group)
    rng = np.random.default_rng(42)
    width = min(7, size)
    starts = rng.integers(0, size, size=(5000, int(np.ceil(size / width))))
    indices = ((starts[..., None] + np.arange(width)) % size).reshape(5000, -1)[:, :size]
    costs = 27 * group.n_picks.to_numpy()[indices].sum(axis=1)
    pnl = group.pnl_k.to_numpy()[indices].sum(axis=1)
    return np.quantile(pnl[costs > 0] / costs[costs > 0], [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--corrected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summaries = {label: json.loads((path / 'summary.json').read_text(encoding='utf-8'))
                 for label, path in (('original', args.original), ('corrected', args.corrected))}
    for name in ('start', 'end', 'dataset_sha256', 'policy_sha256', 'model_registry_sha256'):
        if summaries['original'][name] != summaries['corrected'][name]:
            raise ValueError(f'Replays disagree on {name}')
    frames = {label: pd.read_csv(path / 'daily_scores.csv')
              for label, path in (('original', args.original), ('corrected', args.corrected))}
    top2 = {label: frame[frame.method == 'production_top2'].set_index('date')
            for label, frame in frames.items()}
    if not top2['original'].index.equals(top2['corrected'].index):
        raise ValueError('Paired comparison requires identical chronological dates')
    if not all((frame.n_picks == 2).all() for frame in top2.values()):
        raise ValueError('Forced ranking must have exactly two picks per draw')
    delta = (top2['corrected'].unique_hits - top2['original'].unique_hits).to_numpy() / 2
    count_delta = (top2['corrected'].count_hits - top2['original'].count_hits).to_numpy()
    result = {
        'classification': 'retrospective_pre_holdout_comparison_not_independent_final_validation',
        'start': summaries['original']['start'], 'end': summaries['original']['end'],
        'draws': len(delta), 'bootstrap': {'block_draws': 7, 'resamples': 5000, 'seed': 42},
        'precision_delta': float(delta.mean()), 'precision_delta_ci95': paired_interval(delta),
        'precision_relative_delta': float(delta.mean() / top2['original'].unique_hits.mean() * 2),
        'mean_count_hits_delta': float(count_delta.mean()), 'count_hits_delta_ci95': paired_interval(count_delta),
        'leave_one_draw_out_precision_delta_range': [
            float((delta.sum() - delta.max()) / (len(delta)-1)),
            float((delta.sum() - delta.min()) / (len(delta)-1))],
        'periods': [], 'policy': {},
    }
    for positions in np.array_split(np.arange(len(delta)), 3):
        part = delta[positions]
        result['periods'].append({'start': top2['original'].index[positions[0]],
                                  'end': top2['original'].index[positions[-1]],
                                  'draws': len(part), 'precision_delta': float(part.mean()),
                                  'precision_delta_ci95': paired_interval(part)})
    for label, frame in frames.items():
        policy = frame[frame.method == 'production_policy']
        result['policy'][label] = {**summaries[label]['metrics']['production_policy'],
                                   'roi_ci95': roi_interval(policy)}
    history = pd.read_csv(args.original / 'history.csv')
    horizon = pd.to_datetime(history.date).diff().dt.days
    history['horizon_days'] = horizon
    evaluated = history[history.date.isin(top2['original'].index)].copy()
    counts = np.zeros((len(evaluated), 100), dtype=int)
    for idx, nums in enumerate(evaluated.drop(columns=['date', 'horizon_days']).to_numpy(dtype=int)):
        np.add.at(counts[idx], nums, 1)
    expected_precision = (counts > 0).mean(axis=1)
    result['uniform_expected_precision'] = float(expected_precision.mean())
    for label, frame in top2.items():
        lift = frame.unique_hits.to_numpy() / 2 - expected_precision
        result[f'{label}_lift_vs_uniform_expectation'] = float(lift.mean())
        result[f'{label}_lift_ci95'] = paired_interval(lift)
    result['horizons'] = []
    for days, group in evaluated.groupby('horizon_days'):
        row = {'horizon_days': int(days), 'draws': len(group)}
        for label, frame in top2.items():
            subset = frame.loc[group.date]
            row[label] = {'precision': float(subset.unique_hits.mean()/2),
                          'mean_count_hits': float(subset.count_hits.mean())}
        result['horizons'].append(row)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
