from datetime import date, datetime
import json

import numpy as np
import pandas as pd
import pytest

import daily_predict
from src.decision import risk_filters
from src.evidence.builder import EvidenceBuilder
from src.evidence.store import EvidenceStore
from src.features.bayesian_features import BayesianFeatureExtractor
from src.features.feature_store import FeatureStore
from src.probability.inverted_pairs import InvertedPairsPredictor
from src.probability.markov import MarkovPredictor


def history(n=400):
    df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=n), 'special': np.zeros(n, dtype=int)})
    s = np.zeros((n, 100), dtype=np.int8)
    s[:, 0] = 1
    return df, s


def test_bayesian_transition_counts_do_not_overflow():
    _, s = history()
    model = BayesianFeatureExtractor(s)
    assert model._M[0, 0] == pytest.approx(400 / 401)
    assert np.all(model._M >= 0)


def test_markov_fallback_counts_do_not_overflow():
    _, s = history()
    p = MarkovPredictor().predict_proba(pd.DataFrame(index=range(100)), S_history=s)
    assert p[0] == pytest.approx(0.8905)
    assert np.all(p >= 0)


def test_inverted_pair_counts_do_not_overflow():
    _, s = history()
    s[:, 12] = s[:, 21] = 1
    p = InvertedPairsPredictor().predict_proba(pd.DataFrame(index=range(100)), S_history=s)
    assert p[12] == pytest.approx(365 / 370)


def test_mutual_information_matches_identical_and_independent_series(tmp_path, monkeypatch):
    _, s = history(1200)
    s[:, 0] = s[:, 1] = np.arange(1200) % 2
    s[:, 2] = (np.arange(1200) // 2) % 2
    monkeypatch.setattr(risk_filters, 'MI_EDGES_PATH', tmp_path / 'edges.json')
    filters = risk_filters.RiskFilters()
    filters.build_empirical_correlation(s)
    assert filters.get_correlation(0, 1) == pytest.approx(1, abs=1e-8)
    assert filters.get_correlation(0, 2) == pytest.approx(0, abs=1e-8)


def test_parquet_roundtrip_keeps_all_features(tmp_path):
    df, s = history(20)
    target = datetime(2020, 1, 21)
    store = EvidenceStore(tmp_path / 'evidence')
    ev = EvidenceBuilder(store).build_all(df, s, target)
    first = FeatureStore(store_root=tmp_path / 'first').build(ev, '2020-01-21', S=s)
    loaded = EvidenceBuilder(store).build_all(df, s, target)
    second = FeatureStore(store_root=tmp_path / 'second').build(loaded, '2020-01-21', S=s)
    pd.testing.assert_frame_equal(first, second)


def test_evidence_cache_cannot_replace_changed_history(tmp_path):
    df, s = history(20)
    target = datetime(2020, 1, 21)
    store = EvidenceStore(tmp_path / 'evidence')
    builder = EvidenceBuilder(store)
    original = builder.build_all(df, s, target)
    frozen = store.evidence_path('2020-01-21').read_bytes()
    s[:, 0], s[:, 1] = 0, 1
    df['special'] = 1
    revised = builder.build_all(df, s, target)
    assert revised.loc[0, 'count_all'] == 0
    assert revised.loc[1, 'count_all'] == 20
    assert original.loc[0, 'count_all'] == 20
    assert store.evidence_path('2020-01-21').read_bytes() == frozen


def test_evidence_rejects_target_or_future_observations(tmp_path):
    df, s = history(20)
    with pytest.raises(ValueError, match='history'):
        EvidenceBuilder(EvidenceStore(tmp_path)).build_all(df, s, datetime(2020, 1, 20))


def test_feature_cache_cannot_replace_changed_history(tmp_path):
    df, s = history(20)
    builder = EvidenceBuilder(EvidenceStore(tmp_path / 'evidence'))
    fs = FeatureStore(store_root=tmp_path / 'features')
    target = datetime(2020, 1, 21)
    first = fs.build(builder.build_all(df, s, target, save=False), '2020-01-21', S=s)
    s[:, 0], s[:, 1] = 0, 1
    df['special'] = 1
    second = fs.build(builder.build_all(df, s, target, save=False), '2020-01-21', S=s)
    assert first.loc[0, 'freq_all'] == 1
    assert second.loc[0, 'freq_all'] == 0
    assert second.loc[1, 'freq_all'] == 1


def test_feature_builder_rejects_future_matrix(tmp_path):
    df, s = history(20)
    ev = EvidenceBuilder(EvidenceStore(tmp_path / 'ev')).build_all(df, s, datetime(2020, 1, 21), save=False)
    with pytest.raises(ValueError, match='history'):
        FeatureStore(store_root=tmp_path / 'features').build(ev, '2020-01-21', S=np.vstack([s, s[-1]]))


def test_no_cache_flag_bypasses_parquet(tmp_path):
    df, s = history(20)
    ev = EvidenceBuilder(EvidenceStore(tmp_path / 'ev')).build_all(df, s, datetime(2020, 1, 21), save=False)
    fs = FeatureStore(store_root=tmp_path / 'features')
    fs.build(ev, '2020-01-21', S=s)
    fs.snapshot_path('2020-01-21').write_bytes(b'not a parquet file')
    rebuilt = fs.build(ev, '2020-01-21', S=s, use_cache=False, save_parquet=False)
    assert rebuilt.loc[0, 'freq_all'] == 1


def test_feature_cache_checks_matrix_content_not_only_row_count(tmp_path):
    df, s = history(20)
    ev = EvidenceBuilder(EvidenceStore(tmp_path / 'ev')).build_all(df, s, datetime(2020, 1, 21), save=False)
    fs = FeatureStore(store_root=tmp_path / 'features')
    first = fs.build(ev, '2020-01-21', S=s)
    changed = s.copy()
    changed[:, 1] = 1
    second = fs.build(ev, '2020-01-21', S=changed)
    assert first.loc[1, 'markov_order1'] == pytest.approx(1 / 21)
    assert second.loc[1, 'markov_order1'] == pytest.approx(0.5)


def test_feature_failure_is_not_silently_dropped(tmp_path):
    df, s = history(20)
    ev = EvidenceBuilder(EvidenceStore(tmp_path / 'ev')).build_all(df, s, datetime(2020, 1, 21), save=False)
    with pytest.raises(RuntimeError, match='delay_features'):
        FeatureStore(store_root=tmp_path / 'features').build(ev.drop(columns=['historical_delays']), '2020-01-21', S=s)


def test_metadata_declares_dynamic_extractors_before_first_build(tmp_path):
    fs = FeatureStore(store_root=tmp_path)
    metadata = json.loads((tmp_path / fs.version / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['extractor_versions']['markov_features'] == '1.0'
    assert metadata['extractor_versions']['bayesian_features'] == '1.2'


def test_prior_weights_use_only_past_dates_in_chronological_order():
    entries = [
        {'pipeline_metadata': {'date': '2020-01-03', 'dynamic_weights': {'x': 0.3}}},
        {'pipeline_metadata': {'date': '2020-01-06', 'dynamic_weights': {'x': 0.6}}},
        {'pipeline_metadata': {'date': '2020-01-04', 'dynamic_weights': {'x': 0.4}}},
        {'pipeline_metadata': {'date': '2020-01-03', 'dynamic_weights': {'x': 0.8}}},
        {'pipeline_metadata': {'date': '2020-01-05', 'dynamic_weights': {'x': 0.5}}},
        {'pipeline_metadata': {'dynamic_weights': {'x': 1.0}}},
    ]
    selected = daily_predict.chronological_prior_entries(entries, date(2020, 1, 5))
    assert [e['pipeline_metadata']['date'] for e in selected] == ['2020-01-03', '2020-01-04']
    assert selected[0]['pipeline_metadata']['dynamic_weights']['x'] == 0.8
