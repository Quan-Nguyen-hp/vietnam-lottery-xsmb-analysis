"""Mandatory verification tests for XPIS v3 M3 P3 candidate-level fail-closed qualification.

Implements exact 25 required tests covering:
1. W30 candidate-local failure does not stop W60-W365
2. initialization failure disqualifies only candidate
3. fit failure disqualifies only candidate
4. ForecastContractViolation disqualifies only candidate
5. all partial rows are discarded
6. one surviving candidate wins
7. multiple survivors use unchanged tuple
8. zero survivors publish NoCompleteValidDevCandidate / METRIC_EVALUATION / NEEDS_MODEL_REVISION
9. zero-survivor failure artifact contains exact 6-key candidate qualification evidence
10. ordinary failure artifact remains exact legacy 5-key shape
11. OptimizerExecutionError remains whole-run TECHNICAL_FAILURE
12. generic MetricEvaluationError remains whole-run
13. candidate evaluation ordering cannot alter surviving set/winner
14. candidate_qualification always serialized frozen W order
15. scientific CSVs contain only COMPLETE_VALID M3 candidates
16. each survivor has exactly N_dev DEV daily rows
17. disqualified candidates have exactly zero scientific rows
18. selected candidate belongs to complete_valid_candidates
19. VAL failure produces no fallback
20. STABILITY failure produces no fallback
21. authority has exactly 74 keys
22. authority provenance remains unchanged
23. protocol fingerprint reflects new authority
24. success inventory remains exact 9
25. failure inventory remains exact 1
"""

from __future__ import annotations

from datetime import date, timedelta
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.m3_digit_factor.artifacts import (
    SUCCESS_ARTIFACT_COUNT,
    SUCCESS_ARTIFACT_NAMES,
)
from src.m3_digit_factor.authority import (
    AUTHORITY_KEYS,
    CANONICAL_SPEC_RELATIVE_PATH,
    FROZEN_LITERALS,
    RepositoryIdentity,
    build_authority,
    protocol_fingerprint,
    validate_authority,
)
from src.m3_digit_factor.bootstrap import (
    ForecastBootstrapResult,
)
from src.m3_digit_factor.contracts import (
    CandidateID,
    CandidateStatus,
    DevelopmentExitStatus,
    FailureExitStatus,
    FailureStage,
)
from src.m3_digit_factor.model import (
    ForecastContractViolation,
)
from src.m3_digit_factor.dataset import CanonicalRawDataset
from src.m3_digit_factor.economics import (
    EconomicEvaluationResult,
    PerKEconomicEvaluation,
)
from src.m3_digit_factor.failure import (
    FAILURE_ARTIFACT_NAME,
    validate_failure_artifact,
)
from src.m3_digit_factor.fitting import (
    FittedModelResult,
    NonFiniteModelFit,
    OptimizerExecutionError,
    OptimizerNonConvergence,
)
from src.m3_digit_factor.initialization import (
    SVDLeadingSubspaceAmbiguity,
    ZeroDigitMarginal,
)
from src.m3_digit_factor.metrics import (
    CANDIDATE_WINDOWS,
    MetricEvaluationError,
    StageForecastMetrics,
    select_dev_winner,
)
from src.m3_digit_factor.runner import (
    _run_development_with_dependencies,
)
from src.m3_digit_factor.validation import (
    validate_success_artifacts,
)


def _make_synthetic_dataset(row_count: int = 605) -> CanonicalRawDataset:
    dates = tuple(date(2010, 1, 1) + timedelta(days=i) for i in range(row_count))
    counts = np.zeros((row_count, 100), dtype=np.int16)
    counts[:, :27] = 1
    return CanonicalRawDataset(dates, counts)


def _make_git_runner(status: str = '', repo_name: str = 'Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis', data_path: str = 'data/xsmb-2-digits.csv') -> Any:
    git_values = {
        ('status', '--porcelain'): status,
        ('config', '--get', 'remote.origin.url'): f'git@github.com:{repo_name}.git',
        ('rev-parse', 'HEAD'): 'a' * 40,
        ('rev-parse', 'HEAD^{tree}'): 'b' * 40,
        ('ls-files', '--error-unmatch', 'data/xsmb-2-digits.csv'): data_path,
        ('rev-parse', 'HEAD:data/xsmb-2-digits.csv'): 'c' * 40,
        ('hash-object', 'data/xsmb-2-digits.csv'): 'c' * 40,
    }

    def runner(*args: str) -> str:
        if args in git_values:
            return git_values[args]
        raise ValueError(f'Unhandled git command in mock: {args}')

    return runner


def _mock_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
    mu = np.full(100, 0.27, dtype=np.float64)
    if W == 60:
        mu[:27] = 0.28
        mu[27:] = (27.0 - 27 * 0.28) / 73.0
    return FittedModelResult(
        a=np.zeros(10),
        b=np.zeros(10),
        u=np.zeros(10),
        v=np.zeros(10),
        gamma=0.0,
        theta=np.zeros(41),
        mu=mu,
        objective_value=0.5,
        iterations=5,
    )


def _pass_bootstrapper(d: Any) -> ForecastBootstrapResult:
    return ForecastBootstrapResult(
        bootstrap_lower_bound=0.01,
        bootstrap_replications=2000,
        replicate_means=np.array([0.01]),
    )


def _mock_economic_eval(y: Any, mu: Any, blocks: Any, forecast_signal: bool = True) -> EconomicEvaluationResult:
    per_k = {
        1: PerKEconomicEvaluation(1, 10.0, (10.0,) * 6, 6, 2.0, True),
        3: PerKEconomicEvaluation(3, 15.0, (15.0,) * 6, 6, 3.0, True),
        5: PerKEconomicEvaluation(5, 12.0, (12.0,) * 6, 6, 2.5, True),
        10: PerKEconomicEvaluation(10, -5.0, (-5.0,) * 6, 0, -10.0, False),
    }
    return EconomicEvaluationResult(
        per_k=per_k,
        economic_signal=True,
        qualified_top_k=[1, 3, 5],
        recommended_top_k=[3],
        bootstrap_executed=True,
        economic_delta=np.ones((60, 4)),
    )


def _make_valid_authority() -> dict[str, Any]:
    ident = RepositoryIdentity(
        repository='Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis',
        data_path='data/xsmb-2-digits.csv',
        source_commit='b' * 40,
        source_tree='c' * 40,
        data_commit='d' * 40,
        data_blob='e' * 40,
        data_sha256='f' * 64,
    )
    return build_authority(ident)


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    data_dir = repo / 'data'
    data_dir.mkdir()
    (data_dir / 'xsmb-2-digits.csv').write_bytes(b'dummy-canonical-data')
    return repo


@pytest.fixture
def real_spec_blob() -> bytes:
    import subprocess
    real_spec_path = Path(CANONICAL_SPEC_RELATIVE_PATH)
    if not real_spec_path.exists():
        real_spec_path = Path(__file__).resolve().parent.parent.parent / CANONICAL_SPEC_RELATIVE_PATH
    return subprocess.check_output(
        ['git', 'cat-file', 'blob', f'HEAD:{CANONICAL_SPEC_RELATIVE_PATH}'],
        cwd=real_spec_path.parent.parent.parent,
    )


# ---------------------------------------------------------------------------
# Test 1: W30 candidate-local failure does not stop W60-W365
# ---------------------------------------------------------------------------
def test_01_w30_candidate_local_failure_does_not_stop_w60_w365(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_w30_fails(counts: Any, W: int | None = None) -> FittedModelResult:
        if W == 30:
            raise SVDLeadingSubspaceAmbiguity('Leading singular value gap below threshold for W30')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_w30_fail_w60_w365_survive',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_w30_fails,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'COMPLETED'
    assert result.exit_status == DevelopmentExitStatus.ECONOMIC_SIGNAL_FOUND

    adj = json.loads((result.output_dir / 'development_adjudication.json').read_text(encoding='utf-8'))
    assert adj['complete_valid_candidates'] == ['M3_W060', 'M3_W120', 'M3_W240', 'M3_W365']
    w30_qual = next(q for q in adj['candidate_qualification'] if q['W'] == 30)
    assert w30_qual['status'] == CandidateStatus.DISQUALIFIED_MODEL_INITIALIZATION.value
    assert w30_qual['error_type'] == 'SVDLeadingSubspaceAmbiguity'
    assert w30_qual['first_failed_target_index'] is not None


# ---------------------------------------------------------------------------
# Test 2: initialization failure disqualifies only candidate
# ---------------------------------------------------------------------------
def test_02_initialization_failure_disqualifies_only_candidate(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_init_fail(counts: Any, W: int | None = None) -> FittedModelResult:
        if W == 60:
            raise ZeroDigitMarginal('Zero marginal observed')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_init_fail_single',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_init_fail,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'COMPLETED'
    adj = json.loads((result.output_dir / 'development_adjudication.json').read_text(encoding='utf-8'))
    assert 'M3_W060' not in adj['complete_valid_candidates']
    assert set(adj['complete_valid_candidates']) == {'M3_W030', 'M3_W120', 'M3_W240', 'M3_W365'}
    q60 = next(q for q in adj['candidate_qualification'] if q['W'] == 60)
    assert q60['status'] == CandidateStatus.DISQUALIFIED_MODEL_INITIALIZATION.value


# ---------------------------------------------------------------------------
# Test 3: fit failure disqualifies only candidate
# ---------------------------------------------------------------------------
def test_03_fit_failure_disqualifies_only_candidate(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_fit_fail(counts: Any, W: int | None = None) -> FittedModelResult:
        if W == 120:
            raise OptimizerNonConvergence('SLSQP failed to converge')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_fit_fail_single',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_fit_fail,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'COMPLETED'
    adj = json.loads((result.output_dir / 'development_adjudication.json').read_text(encoding='utf-8'))
    assert 'M3_W120' not in adj['complete_valid_candidates']
    q120 = next(q for q in adj['candidate_qualification'] if q['W'] == 120)
    assert q120['status'] == CandidateStatus.DISQUALIFIED_MODEL_FIT.value
    assert q120['error_type'] == 'OptimizerNonConvergence'


# ---------------------------------------------------------------------------
# Test 4: ForecastContractViolation disqualifies only candidate
# ---------------------------------------------------------------------------
def test_04_forecast_contract_violation_disqualifies_only_candidate(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_fc_fail(counts: Any, W: int | None = None) -> FittedModelResult:
        if W == 240:
            raise ForecastContractViolation('Post-fit forecast contract failed: non-finite')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_fc_fail_single',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_fc_fail,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'COMPLETED'
    adj = json.loads((result.output_dir / 'development_adjudication.json').read_text(encoding='utf-8'))
    assert 'M3_W240' not in adj['complete_valid_candidates']
    q240 = next(q for q in adj['candidate_qualification'] if q['W'] == 240)
    assert q240['status'] == CandidateStatus.DISQUALIFIED_FORECAST_CONTRACT.value
    assert q240['error_type'] == 'ForecastContractViolation'


# ---------------------------------------------------------------------------
# Test 5: all partial rows are discarded
# ---------------------------------------------------------------------------
def test_05_all_partial_rows_are_discarded(repo_root: Path, real_spec_blob: bytes) -> None:
    call_counts: dict[int, int] = {}

    def fitter_mid_fail(counts: Any, W: int | None = None) -> FittedModelResult:
        w_val = W or 0
        call_counts[w_val] = call_counts.get(w_val, 0) + 1
        if w_val == 30 and call_counts[w_val] > 10:
            raise OptimizerNonConvergence('Failed halfway through DEV targets')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_discard_partial',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_mid_fail,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'COMPLETED'
    with gzip.open(result.output_dir / 'daily_forecast_scores.csv.gz', 'rt') as f:
        daily_df = pd.read_csv(f)
    assert 'M3_W030' not in daily_df['candidate_id'].values

    metrics_df = pd.read_csv(result.output_dir / 'forecast_metrics.csv')
    assert 'M3_W030' not in metrics_df['candidate_id'].values


# ---------------------------------------------------------------------------
# Test 6: one surviving candidate wins
# ---------------------------------------------------------------------------
def test_06_one_surviving_candidate_wins() -> None:
    candidate_metrics = {
        365: StageForecastMetrics(poisson_deviance=0.8, mae=0.08, rmse=0.15, date_count=100)
    }
    winner = select_dev_winner(candidate_metrics)
    assert winner.window == 365
    assert winner.candidate_id == CandidateID.M3_W365


# ---------------------------------------------------------------------------
# Test 7: multiple survivors use unchanged tuple
# ---------------------------------------------------------------------------
def test_07_multiple_survivors_use_unchanged_tuple() -> None:
    candidate_metrics = {
        60: StageForecastMetrics(poisson_deviance=0.85, mae=0.08, rmse=0.15, date_count=100),
        120: StageForecastMetrics(poisson_deviance=0.82, mae=0.09, rmse=0.16, date_count=100),
    }
    winner = select_dev_winner(candidate_metrics)
    assert winner.window == 120
    assert winner.candidate_id == CandidateID.M3_W120


# ---------------------------------------------------------------------------
# Test 8: zero survivors publish NoCompleteValidDevCandidate / METRIC_EVALUATION / NEEDS_MODEL_REVISION
# ---------------------------------------------------------------------------
def test_08_zero_survivors_publish_no_complete_valid_dev_candidate(repo_root: Path, real_spec_blob: bytes) -> None:
    def all_failing_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
        raise OptimizerNonConvergence('All candidates fail')

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_zero_survivor_result',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=all_failing_fitter,
    )
    assert result.status == 'FAILED'
    assert result.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION

    fail_data = json.loads((result.output_dir / FAILURE_ARTIFACT_NAME).read_text(encoding='utf-8'))
    assert fail_data['failed_stage'] == FailureStage.METRIC_EVALUATION.value
    assert fail_data['error_type'] == 'NoCompleteValidDevCandidate'
    assert fail_data['development_exit_status'] == FailureExitStatus.NEEDS_MODEL_REVISION.value


# ---------------------------------------------------------------------------
# Test 9: zero-survivor failure artifact contains exact 6-key candidate qualification evidence
# ---------------------------------------------------------------------------
def test_09_zero_survivor_failure_artifact_contains_exact_6_key_evidence(repo_root: Path, real_spec_blob: bytes) -> None:
    def all_failing_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
        raise NonFiniteModelFit('Non finite fit for all')

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_zero_survivor_6key',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=all_failing_fitter,
    )
    fail_path = result.output_dir / FAILURE_ARTIFACT_NAME
    fail_data = validate_failure_artifact(fail_path.read_bytes())
    assert set(fail_data.keys()) == {
        'status',
        'failed_stage',
        'error_type',
        'development_exit_status',
        'candidate_qualification',
        'protocol_snapshot',
    }
    assert len(fail_data['candidate_qualification']) == 5
    for entry in fail_data['candidate_qualification']:
        assert entry['status'] == CandidateStatus.DISQUALIFIED_MODEL_FIT.value


# ---------------------------------------------------------------------------
# Test 10: ordinary failure artifact remains exact legacy 5-key shape
# ---------------------------------------------------------------------------
def test_10_ordinary_failure_artifact_remains_exact_legacy_5_key_shape(repo_root: Path, real_spec_blob: bytes) -> None:
    git_runner = _make_git_runner(data_path='data/wrong_path.csv')
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_ordinary_5key',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=_mock_fitter,
    )
    assert result.status == 'FAILED'
    fail_data = json.loads((result.output_dir / FAILURE_ARTIFACT_NAME).read_text(encoding='utf-8'))
    assert set(fail_data.keys()) == {
        'status',
        'failed_stage',
        'error_type',
        'development_exit_status',
        'protocol_snapshot',
    }
    assert 'candidate_qualification' not in fail_data


# ---------------------------------------------------------------------------
# Test 11: OptimizerExecutionError remains whole-run TECHNICAL_FAILURE
# ---------------------------------------------------------------------------
def test_11_optimizer_execution_error_remains_whole_run_technical_failure(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_tech_fail(counts: Any, W: int | None = None) -> FittedModelResult:
        if W == 30:
            raise OptimizerExecutionError('SciPy internal segfault or unexpected runtime error')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_tech_fail_whole_run',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_tech_fail,
    )
    assert result.status == 'FAILED'
    assert result.exit_status == FailureExitStatus.TECHNICAL_FAILURE
    fail_data = json.loads((result.output_dir / FAILURE_ARTIFACT_NAME).read_text(encoding='utf-8'))
    assert fail_data['failed_stage'] == FailureStage.MODEL_FIT.value
    assert fail_data['error_type'] == 'OptimizerExecutionError'
    assert 'candidate_qualification' not in fail_data


# ---------------------------------------------------------------------------
# Test 12: generic MetricEvaluationError remains whole-run
# ---------------------------------------------------------------------------
def test_12_generic_metric_evaluation_error_remains_whole_run() -> None:
    with pytest.raises(MetricEvaluationError, match='contain unknown windows'):
        select_dev_winner({999: StageForecastMetrics(poisson_deviance=0.8, mae=0.08, rmse=0.15, date_count=100)})


# ---------------------------------------------------------------------------
# Test 13: candidate evaluation ordering cannot alter surviving set/winner
# ---------------------------------------------------------------------------
def test_13_candidate_evaluation_ordering_cannot_alter_surviving_set_winner() -> None:
    eval_1 = {
        60: StageForecastMetrics(poisson_deviance=0.85, mae=0.08, rmse=0.15, date_count=100),
        120: StageForecastMetrics(poisson_deviance=0.80, mae=0.09, rmse=0.16, date_count=100),
        240: StageForecastMetrics(poisson_deviance=0.82, mae=0.07, rmse=0.14, date_count=100),
    }
    eval_2 = {
        240: StageForecastMetrics(poisson_deviance=0.82, mae=0.07, rmse=0.14, date_count=100),
        60: StageForecastMetrics(poisson_deviance=0.85, mae=0.08, rmse=0.15, date_count=100),
        120: StageForecastMetrics(poisson_deviance=0.80, mae=0.09, rmse=0.16, date_count=100),
    }
    win1 = select_dev_winner(eval_1)
    win2 = select_dev_winner(eval_2)
    assert win1.window == win2.window == 120
    assert win1.candidate_id == win2.candidate_id == CandidateID.M3_W120


# ---------------------------------------------------------------------------
# Test 14: candidate_qualification always serialized frozen W order
# ---------------------------------------------------------------------------
def test_14_candidate_qualification_always_serialized_frozen_w_order(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_w120_fails(counts: Any, W: int | None = None) -> FittedModelResult:
        if W == 120:
            raise OptimizerNonConvergence('W120 fail')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_frozen_w_order',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_w120_fails,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'COMPLETED'
    adj = json.loads((result.output_dir / 'development_adjudication.json').read_text(encoding='utf-8'))
    windows = [q['W'] for q in adj['candidate_qualification']]
    assert windows == list(CANDIDATE_WINDOWS)


# ---------------------------------------------------------------------------
# Test 15: scientific CSVs contain only COMPLETE_VALID M3 candidates
# ---------------------------------------------------------------------------
def test_15_scientific_csvs_contain_only_complete_valid_m3_candidates(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_w30_w240_fail(counts: Any, W: int | None = None) -> FittedModelResult:
        if W in (30, 240):
            raise OptimizerNonConvergence(f'W{W} fail')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_csv_complete_valid_only',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_w30_w240_fail,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'COMPLETED'

    metrics_df = pd.read_csv(result.output_dir / 'forecast_metrics.csv')
    dev_cands = set(metrics_df[metrics_df['stage'] == 'DEV']['candidate_id'])
    assert dev_cands == {'B0_UNIFORM', 'M3_W060', 'M3_W120', 'M3_W365'}
    assert 'M3_W030' not in dev_cands
    assert 'M3_W240' not in dev_cands


# ---------------------------------------------------------------------------
# Test 16: each survivor has exactly N_dev DEV daily rows
# ---------------------------------------------------------------------------
def test_16_each_survivor_has_exactly_n_dev_dev_daily_rows(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_w30_fails(counts: Any, W: int | None = None) -> FittedModelResult:
        if W == 30:
            raise OptimizerNonConvergence('W30 fail')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_exact_n_dev_rows',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_w30_fails,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    with gzip.open(result.output_dir / 'daily_forecast_scores.csv.gz', 'rt') as f:
        daily_df = pd.read_csv(f)

    b0_dev_count = len(daily_df[(daily_df['stage'] == 'DEV') & (daily_df['candidate_id'] == 'B0_UNIFORM')])
    for cand in ['M3_W060', 'M3_W120', 'M3_W240', 'M3_W365']:
        cand_dev_count = len(daily_df[(daily_df['stage'] == 'DEV') & (daily_df['candidate_id'] == cand)])
        assert cand_dev_count == b0_dev_count


# ---------------------------------------------------------------------------
# Test 17: disqualified candidates have exactly zero scientific rows
# ---------------------------------------------------------------------------
def test_17_disqualified_candidates_have_exactly_zero_scientific_rows(repo_root: Path, real_spec_blob: bytes) -> None:
    def fitter_w30_w60_fail(counts: Any, W: int | None = None) -> FittedModelResult:
        if W in (30, 60):
            raise OptimizerNonConvergence(f'W{W} fail')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_disqualified_zero_rows',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_w30_w60_fail,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    with gzip.open(result.output_dir / 'daily_forecast_scores.csv.gz', 'rt') as f:
        daily_df = pd.read_csv(f)
    metrics_df = pd.read_csv(result.output_dir / 'forecast_metrics.csv')

    for disq in ['M3_W030', 'M3_W060']:
        assert len(daily_df[daily_df['candidate_id'] == disq]) == 0
        assert len(metrics_df[metrics_df['candidate_id'] == disq]) == 0


# ---------------------------------------------------------------------------
# Test 18: selected candidate belongs to complete_valid_candidates
# ---------------------------------------------------------------------------
def test_18_selected_candidate_belongs_to_complete_valid_candidates(repo_root: Path, real_spec_blob: bytes) -> None:
    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_selected_in_complete_valid',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=_mock_fitter,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    adj = json.loads((result.output_dir / 'development_adjudication.json').read_text(encoding='utf-8'))
    assert adj['selected_candidate_id'] in adj['complete_valid_candidates']


# ---------------------------------------------------------------------------
# Test 19: VAL failure produces no fallback
# ---------------------------------------------------------------------------
def test_19_val_failure_produces_no_fallback(repo_root: Path, real_spec_blob: bytes) -> None:
    call_count = 0

    def fitter_val_fails(counts: Any, W: int | None = None) -> FittedModelResult:
        nonlocal call_count
        call_count += 1
        # W=60 wins DEV; during VAL partition, fail on W=60
        if W == 60 and call_count > 300:
            raise OptimizerNonConvergence('VAL failure for selected candidate W60')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_val_fail_no_fallback',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_val_fails,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'FAILED'
    assert result.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION
    fail_data = json.loads((result.output_dir / FAILURE_ARTIFACT_NAME).read_text(encoding='utf-8'))
    assert fail_data['failed_stage'] == FailureStage.MODEL_FIT.value
    assert fail_data['error_type'] == 'OptimizerNonConvergence'


# ---------------------------------------------------------------------------
# Test 20: STABILITY failure produces no fallback
# ---------------------------------------------------------------------------
def test_20_stability_failure_produces_no_fallback(repo_root: Path, real_spec_blob: bytes) -> None:
    call_count = 0

    def fitter_stability_fails(counts: Any, W: int | None = None) -> FittedModelResult:
        nonlocal call_count
        call_count += 1
        # W=60 wins DEV; during STABILITY partition, fail on W=60
        if W == 60 and call_count > 350:
            raise OptimizerNonConvergence('STABILITY failure for selected candidate W60')
        return _mock_fitter(counts, W)

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_stability_fail_no_fallback',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=fitter_stability_fails,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'FAILED'
    assert result.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION
    fail_data = json.loads((result.output_dir / FAILURE_ARTIFACT_NAME).read_text(encoding='utf-8'))
    assert fail_data['failed_stage'] == FailureStage.MODEL_FIT.value
    assert fail_data['error_type'] == 'OptimizerNonConvergence'


# ---------------------------------------------------------------------------
# Test 21: authority has exactly 74 keys
# ---------------------------------------------------------------------------
def test_21_authority_has_exactly_74_keys() -> None:
    assert len(AUTHORITY_KEYS) == 74
    auth = _make_valid_authority()
    assert len(auth) == 74
    validate_authority(auth)


# ---------------------------------------------------------------------------
# Test 22: authority provenance remains unchanged
# ---------------------------------------------------------------------------
def test_22_authority_provenance_remains_unchanged() -> None:
    expected_provenance = (
        'RECONSTRUCTED_FROM_SURVIVING_APPROVED_ARTIFACTS_REPOSITORY_EVIDENCE_'
        'AND_CONTROL_PLANE_ADJUDICATION'
    )
    assert FROZEN_LITERALS['authority_provenance'] == expected_provenance


# ---------------------------------------------------------------------------
# Test 23: protocol fingerprint reflects new authority
# ---------------------------------------------------------------------------
def test_23_protocol_fingerprint_reflects_new_authority() -> None:
    expected_fingerprint = '197631ae43dd6d0deb8f5b92a257d4ef42ce05de883acb027c360e30343e65e9'
    auth = _make_valid_authority()
    actual_fingerprint = protocol_fingerprint(auth)
    assert actual_fingerprint == expected_fingerprint


# ---------------------------------------------------------------------------
# Test 24: success inventory remains exact 9
# ---------------------------------------------------------------------------
def test_24_success_inventory_remains_exact_9(repo_root: Path, real_spec_blob: bytes) -> None:
    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_success_inventory_exact_9',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=_mock_fitter,
        forecast_bootstrapper=_pass_bootstrapper,
        economic_evaluator=_mock_economic_eval,
    )
    assert result.status == 'COMPLETED'
    assert len(result.published_artifacts) == SUCCESS_ARTIFACT_COUNT == 9
    actual_filenames = {Path(p).name for p in result.published_artifacts}
    assert actual_filenames == set(SUCCESS_ARTIFACT_NAMES)
    validate_success_artifacts(result.output_dir)


# ---------------------------------------------------------------------------
# Test 25: failure inventory remains exact 1
# ---------------------------------------------------------------------------
def test_25_failure_inventory_remains_exact_1(repo_root: Path, real_spec_blob: bytes) -> None:
    def all_failing_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
        raise OptimizerNonConvergence('All fail')

    git_runner = _make_git_runner()
    result = _run_development_with_dependencies(
        repository_root=repo_root,
        output_root=repo_root / 'out',
        run_id='test_failure_inventory_exact_1',
        git_runner=git_runner,
        blob_reader=lambda spec: real_spec_blob,
        dataset_loader=lambda p: _make_synthetic_dataset(),
        model_fitter=all_failing_fitter,
    )
    assert result.status == 'FAILED'
    assert len(result.published_artifacts) == 1
    assert Path(result.published_artifacts[0]).name == FAILURE_ARTIFACT_NAME
    assert (result.output_dir / FAILURE_ARTIFACT_NAME).exists()
