"""Replay the real daily pipeline before the locked holdout, in isolated caches.

Example: uv run python backtests/chronological_accuracy.py --start 2026-01-22
--end 2026-07-20 --output .cache/accuracy-original --label original
All challenger rankings are research diagnostics; production policy is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from functools import lru_cache
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def check_window(start: str, end: str, policy: dict) -> None:
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError('start must precede end')
    if pd.Timestamp(end) >= pd.Timestamp(policy['locked_holdout']['start_date']):
        raise ValueError('Research must end before the locked holdout')


def score_picks(counts: np.ndarray, picks: list[int]) -> dict:
    unique_hits = int(np.count_nonzero(counts[picks]))
    hits = int(counts[picks].sum())
    return {
        'n_picks': len(picks), 'unique_hits': unique_hits, 'count_hits': hits,
        'precision': unique_hits / len(picks) if picks else None,
        'recall': unique_hits / max(1, np.count_nonzero(counts)),
        'any_hit': int(hits > 0), 'pnl_k': 99 * hits - 27 * len(picks),
    }


def paired_interval(delta: np.ndarray, seed: int = 42, resamples: int = 5000) -> list[float]:
    """Paired moving-block bootstrap (seven draws) for a daily mean difference."""
    delta = np.asarray(delta, dtype=float)
    if not len(delta):
        raise ValueError('No evaluation draws')
    rng = np.random.default_rng(seed)
    width = min(7, len(delta))
    starts = rng.integers(0, len(delta), size=(resamples, int(np.ceil(len(delta) / width))))
    indices = ((starts[..., None] + np.arange(width)) % len(delta)).reshape(resamples, -1)
    samples = delta[indices[:, :len(delta)]].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def score_entry(entry: dict, counts: np.ndarray, binary: np.ndarray, idx: int, rng) -> list[dict]:
    """Score fixed, pre-holdout diagnostics using only the prefix before idx."""
    target = entry['pipeline_metadata']['date']
    proba = np.array(entry['ensemble_proba'])
    components = {k: np.asarray(v) for k, v in entry['model_probas'].items()}
    weights = entry['pipeline_metadata']['dynamic_weights']
    signals = {'production': proba, 'uniform_fusion': np.mean(list(components.values()), axis=0),
               'frequency_all': binary[:idx].mean(axis=0)}
    signals.update({f'frequency_{w}': binary[max(0, idx-w):idx].mean(axis=0) for w in (30, 90, 365)})
    sample = counts[max(0, idx-365):idx]
    ew_weights = np.exp2(-np.arange(len(sample)-1, -1, -1) / 90)
    signals['count_ewma90'] = ew_weights @ sample / ew_weights.sum()
    # Ablation cố định: loại score min-max/median rescale, không fit thêm model.
    for removed in ('max_delay', 'bayesian_predictor', 'loto_repeat'):
        if removed not in components:
            continue
        keep = [name for name in components if name != removed]
        denom = sum(weights.get(name, 0) for name in keep)
        if denom > 0:
            signals[f'without_{removed}'] = sum(weights.get(name, 0)*components[name] for name in keep) / denom
    n_policy = len(entry['bets'])
    picks = {'production_policy': [bet['number'] for bet in entry['bets']],
             'random_top2': rng.choice(100, size=2, replace=False).tolist()}
    for name, signal in signals.items():
        ranked = np.argsort(signal)[::-1]
        picks[f'{name}_top2'] = ranked[:2].tolist()
        if name != 'production':
            picks[f'{name}_matched'] = ranked[:n_policy].tolist()
    rows = []
    for name, selected in picks.items():
        row = {'date': target, 'method': name, 'numbers': ','.join(f'{n:02d}' for n in selected)}
        row.update(score_picks(counts[idx], selected))
        row['brier'] = float(np.mean((proba - binary[idx]) ** 2)) if name.startswith('production_') else None
        rows.append(row)
    for shrink in (0.0, 0.5):
        p = shrink * proba + (1-shrink) * (1 - 0.99**27)
        selected = picks['random_top2'] if shrink == 0 else np.argsort(p)[::-1][:2].tolist()
        rows.append({'date': target, 'method': f'shrink_{shrink}_top2',
                     'numbers': ','.join(f'{n:02d}' for n in selected),
                     **score_picks(counts[idx], selected), 'brier': float(np.mean((p - binary[idx]) ** 2))})
    return rows


def summarize(scores: pd.DataFrame) -> dict:
    result = {}
    for name, group in scores.groupby('method', sort=True):
        n, picks = len(group), int(group.n_picks.sum())
        row = {
            'draws': n, 'picks': picks, 'active_draws': int((group.n_picks > 0).sum()),
            'unique_hits': int(group.unique_hits.sum()), 'count_hits': int(group.count_hits.sum()),
            'precision': float(group.unique_hits.sum() / picks) if picks else None,
            'mean_hits': float(group.count_hits.mean()), 'recall': float(group.recall.mean()),
            'draw_hit_rate': float(group.any_hit.mean()),
            'roi': float(group.pnl_k.sum() / (27 * picks)) if picks else None,
            'hit_distribution': {str(k): int(v) for k, v in group.count_hits.value_counts().sort_index().items()},
            'periods': [],
        }
        for positions in np.array_split(np.arange(len(group)), min(3, len(group))):
            block = group.iloc[positions]
            count = int(block.n_picks.sum())
            row['periods'].append({
                'start': block.iloc[0].date, 'end': block.iloc[-1].date, 'draws': len(block),
                'picks': count, 'precision': float(block.unique_hits.sum() / count) if count else None,
                'mean_hits': float(block.count_hits.mean()),
            })
        if group.brier.notna().all():
            row['brier'] = float(group.brier.mean())
        result[name] = row
    champion = scores[scores.method == 'production_top2'].set_index('date')
    for name, group in scores.groupby('method'):
        if name == 'production_top2' or not (group.n_picks == 2).all():
            continue
        group = group.set_index('date').loc[champion.index]
        delta = (group.unique_hits - champion.unique_hits).to_numpy() / 2
        result[name]['precision_delta_vs_production_top2'] = float(delta.mean())
        result[name]['precision_delta_ci95'] = paired_interval(delta)
    return result


def resume_entries(path: Path, expected_dates: list[str]) -> list[dict]:
    entries = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    dates = [entry['pipeline_metadata']['date'] for entry in entries]
    if dates != expected_dates[:len(dates)]:
        raise ValueError('Saved forecasts must be a complete chronological prefix')
    return entries


def replay(args) -> dict:
    source = args.source_root.resolve()
    output = args.output.resolve()
    source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [source / 'daily_predict.py', *sorted((source / 'src').rglob('*.py'))]}
    policy = json.loads((source / 'predictions/evaluation_policy.json').read_text(encoding='utf-8'))
    configuration_hashes = {name: hashlib.sha256((source / 'predictions' / name).read_bytes()).hexdigest()
                            for name in ('evaluation_policy.json', 'model_registry.json')}
    check_window(args.start, args.end, policy)
    if (output / 'summary.json').exists() or ((output / 'forecasts.jsonl').exists() and not args.resume):
        raise ValueError('Use a fresh output directory; evidence is create-once')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'source_manifest_original.json'
    if args.resume and json.loads(manifest_path.read_text(encoding='utf-8')) != source_hashes:
        raise ValueError('Resume requires the same frozen source implementation')
    if not args.resume:
        manifest_path.write_text(json.dumps(source_hashes, indent=2), encoding='utf-8')
    config_path = output / 'configuration_manifest.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != configuration_hashes:
        raise ValueError('Resume requires the same model and policy configuration')
    config_path.write_text(json.dumps(configuration_hashes, indent=2), encoding='utf-8')
    raw_csv = source / 'data/xsmb-2-digits.csv'
    df = pd.read_csv(raw_csv)
    df['date'] = pd.to_datetime(df.date)
    df = df[df.date <= pd.Timestamp(args.end)].sort_values('date').reset_index(drop=True)
    prize_cols = [c for c in df.columns if c != 'date']
    values = df[prize_cols].to_numpy(dtype=float)
    if len(prize_cols) != 27 or df.date.duplicated().any() or not np.isfinite(values).all():
        raise ValueError('Invalid 27-prize draw data')
    if not np.equal(values, np.floor(values)).all() or ((values < 0) | (values > 99)).any():
        raise ValueError('Invalid two-digit prize values')
    indices = df.index[(df.date >= pd.Timestamp(args.start)) & (df.date <= pd.Timestamp(args.end))]
    if not len(indices) or indices[0] < 455 + args.warmup_days:
        raise ValueError('Insufficient history for exact production replay')
    truncated = output / 'history.csv'
    if args.resume:
        pd.testing.assert_frame_equal(pd.read_csv(truncated, parse_dates=['date']), df)
    else:
        df.to_csv(truncated, index=False)
    counts = np.zeros((len(df), 100), dtype=np.int64)
    for idx, nums in enumerate(values.astype(int)):
        np.add.at(counts[idx], nums, 1)
    binary = counts > 0
    sys.path.insert(0, str(source))
    pipeline = importlib.import_module('daily_predict')
    evidence_store = pipeline.EvidenceStore(output / 'evidence')
    # LRU chỉ giữ bản đọc của snapshot bất biến; không cache kết quả fit/predict.
    evidence_store.load = lru_cache(maxsize=480)(evidence_store.load)
    feature_store = pipeline.FeatureStore(store_root=output / 'features', evidence_store=evidence_store)
    risk_module = importlib.import_module('decision.risk_filters')
    prior, scores = [], []
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    first_idx = indices[0] - args.warmup_days
    if args.resume:
        expected_dates = df.iloc[first_idx:indices[-1]+1].date.dt.strftime('%Y-%m-%d').tolist()
        prior = resume_entries(output / 'forecasts.jsonl', expected_dates)
        for entry in prior:
            idx = int(df.index[df.date == pd.Timestamp(entry['pipeline_metadata']['date'])][0])
            if idx >= indices[0]:
                scores.extend(score_entry(entry, counts, binary, idx, rng))
        first_idx += len(prior)
    git_commit = pipeline.get_git_commit()
    with ExitStack() as stack:
        stack.enter_context(patch.object(pipeline, 'DATA_CSV', truncated))
        stack.enter_context(patch.object(pipeline, 'EvidenceStore', lambda: evidence_store))
        stack.enter_context(patch.object(pipeline, 'FeatureStore', lambda **kwargs: feature_store))
        stack.enter_context(patch.object(risk_module, 'MI_EDGES_PATH', output / 'mi_edges.json'))
        stack.enter_context(patch.object(pipeline, 'get_git_commit', lambda: git_commit))
        stack.enter_context(threadpool_limits(limits=1))
        forecasts = stack.enter_context((output / 'forecasts.jsonl').open('a' if args.resume else 'w', encoding='utf-8'))
        diagnostics = stack.enter_context((output / 'pipeline.log').open('a' if args.resume else 'w', encoding='utf-8'))
        for idx in range(first_idx, indices[-1] + 1):
            target = df.iloc[idx].date.date()
            with redirect_stdout(diagnostics):
                entry = pipeline.run_shared_prediction_pipeline(target, top_k=2, prior_entries=prior)
            prior.append(entry)
            forecasts.write(json.dumps(entry, ensure_ascii=False) + '\n')
            forecasts.flush()
            if idx < indices[0]:
                continue
            scores.extend(score_entry(entry, counts, binary, idx, rng))
            elapsed = time.perf_counter()-t0
            print(f'{args.label} {target}: {idx-indices[0]+1}/{len(indices)} draws, {elapsed:.1f}s', flush=True)
    frame = pd.DataFrame(scores)
    frame.to_csv(output / 'daily_scores.csv', index=False)
    summary = {
        'label': args.label, 'start': args.start, 'end': args.end, 'warmup_draws': args.warmup_days,
        'evaluation_class': 'historical_chronological_replay_pre_holdout_not_independent_final_validation',
        'dataset_sha256': hashlib.sha256(truncated.read_bytes()).hexdigest(),
        'policy_sha256': hashlib.sha256((source / 'predictions/evaluation_policy.json').read_bytes()).hexdigest(),
        'model_registry_sha256': hashlib.sha256((source / 'predictions/model_registry.json').read_bytes()).hexdigest(),
        'source_sha256': source_hashes,
        'elapsed_seconds': time.perf_counter()-t0, 'metrics': summarize(frame),
        'environment': prior[-1]['pipeline_metadata']['experiment']['environment'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def rescore(output: Path, start: str, end: str, label: str, policy_path: Path) -> dict:
    """Recompute metrics from saved real forecasts; never regenerate predictions."""
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    check_window(start, end, policy)
    df = pd.read_csv(output / 'history.csv')
    counts = np.zeros((len(df), 100), dtype=np.int64)
    for idx, nums in enumerate(df.drop(columns='date').to_numpy(dtype=int)):
        np.add.at(counts[idx], nums, 1)
    date_index = {str(d): i for i, d in enumerate(df.date)}
    expected = df.date[(df.date >= start) & (df.date <= end)].tolist()
    scores, dates, entries = [], [], []
    rng = np.random.default_rng(42)
    for line in (output / 'forecasts.jsonl').read_text(encoding='utf-8').splitlines():
        entry = json.loads(line)
        target = entry['pipeline_metadata']['date']
        if target < start:
            continue
        if target > end:
            raise ValueError('Forecast outside the approved window')
        dates.append(target)
        entries.append(entry)
        scores.extend(score_entry(entry, counts, counts > 0, date_index[target], rng))
    if dates != expected:
        raise ValueError('Forecasts must cover each evaluation draw once in chronological order')
    frame = pd.DataFrame(scores)
    frame.to_csv(output / 'daily_scores.csv', index=False)
    summary = {
        'label': label, 'start': start, 'end': end, 'draws': len(dates),
        'evaluation_class': 'historical_chronological_replay_pre_holdout_not_independent_final_validation',
        'dataset_sha256': hashlib.sha256((output / 'history.csv').read_bytes()).hexdigest(),
        'forecast_sha256': hashlib.sha256((output / 'forecasts.jsonl').read_bytes()).hexdigest(),
        'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        'model_registry_sha256': json.loads((output / 'configuration_manifest.json').read_text(encoding='utf-8'))['model_registry.json']
                                if (output / 'configuration_manifest.json').exists() else
                                hashlib.sha256((output / 'source/predictions/model_registry.json').read_bytes()).hexdigest(),
        'source_sha256': json.loads((output / 'source_manifest_original.json').read_text(encoding='utf-8')),
        'environment': entries[-1]['pipeline_metadata']['experiment']['environment'],
        'metrics': summarize(frame),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--warmup-days', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='current')
    parser.add_argument('--score-only', action='store_true', help='Score saved forecasts after a completed replay')
    parser.add_argument('--resume', action='store_true', help='Resume a prefix using exactly the same source and dataset')
    args = parser.parse_args()
    if args.warmup_days < 0:
        parser.error('warmup-days cannot be negative')
    if args.score_only:
        rescore(args.output.resolve(), args.start, args.end, args.label,
                args.source_root / 'predictions/evaluation_policy.json')
    else:
        replay(args)


if __name__ == '__main__':
    main()
