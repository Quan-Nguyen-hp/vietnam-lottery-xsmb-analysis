import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / 'backtests' / 'chronological_accuracy.py'
    spec = importlib.util.spec_from_file_location('accuracy_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scores_distinguish_binary_hits_and_repeated_hits():
    runner = load_runner()
    counts = np.zeros(100, dtype=int)
    counts[12], counts[42] = 3, 1
    score = runner.score_picks(counts, [12, 99])
    assert score['count_hits'] == 3
    assert score['unique_hits'] == 1
    assert score['precision'] == 0.5
    assert score['recall'] == 0.5
    assert score['pnl_k'] == 243
    assert runner.score_picks(counts, [])['precision'] is None


def test_experiments_refuse_locked_holdout():
    runner = load_runner()
    policy = {'locked_holdout': {'start_date': '2026-07-21'}}
    runner.check_window('2026-01-01', '2026-07-20', policy)
    with pytest.raises(ValueError, match='holdout'):
        runner.check_window('2026-01-01', '2026-07-21', policy)


def test_paired_interval_preserves_draws_and_is_reproducible():
    runner = load_runner()
    values = np.ones(30) * 0.25
    first = runner.paired_interval(values)
    assert first == runner.paired_interval(values)
    assert first == pytest.approx([0.25, 0.25])


def test_summary_preserves_chronological_periods():
    runner = load_runner()
    rows = []
    for day in range(1, 10):
        rows.append({'date': f'2020-01-{day:02d}', 'method': 'production_top2',
                     'n_picks': 2, 'unique_hits': 1, 'count_hits': 2, 'recall': 0.1,
                     'any_hit': 1, 'pnl_k': 144, 'brier': 0.2})
    summary = runner.summarize(pd.DataFrame(rows))['production_top2']
    assert summary['picks'] == 18
    assert summary['precision'] == 0.5
    assert [p['draws'] for p in summary['periods']] == [3, 3, 3]
    assert summary['periods'][1]['start'] == '2020-01-04'


def test_frequency_challenger_never_reads_target_counts():
    runner = load_runner()
    counts = np.zeros((2, 100), dtype=int)
    counts[0, 12], counts[0, 42] = 14, 13
    counts[1, 99] = 27
    entry = {'pipeline_metadata': {'date': '2020-01-02', 'dynamic_weights': {'component': 1}},
             'ensemble_proba': [0.24] * 100, 'model_probas': {'component': [0.24] * 100}, 'bets': []}
    rows = runner.score_entry(entry, counts, counts > 0, 1, np.random.default_rng(42))
    historical = next(r for r in rows if r['method'] == 'frequency_all_top2')
    assert set(historical['numbers'].split(',')) == {'12', '42'}
    assert historical['count_hits'] == 0


def test_resume_requires_complete_chronological_prefix(tmp_path):
    runner = load_runner()
    path = tmp_path / 'forecasts.jsonl'
    path.write_text('{"pipeline_metadata": {"date": "2020-01-01"}}\n', encoding='utf-8')
    assert len(runner.resume_entries(path, ['2020-01-01', '2020-01-02'])) == 1
    with pytest.raises(ValueError, match='prefix'):
        runner.resume_entries(path, ['2020-01-02', '2020-01-01'])
