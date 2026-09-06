"""Tests for XPIS v3 M3 development orchestration runner (Slice M3-13)."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any
import unittest.mock as mock

import numpy as np
import pytest

from src.m3_digit_factor.artifacts import (
    SUCCESS_ARTIFACT_COUNT,
)
from src.m3_digit_factor.authority import (
    CANONICAL_SPEC_RELATIVE_PATH,
)
from src.m3_digit_factor.bootstrap import (
    ForecastBootstrapResult,
)
from src.m3_digit_factor.contracts import (
    DevelopmentExitStatus,
    FailureExitStatus,
    FailureStage,
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
    OptimizerExecutionError,
)
from src.m3_digit_factor.initialization import (
    SVDLeadingSubspaceAmbiguity,
)
import src.m3_digit_factor.economics
import src.m3_digit_factor.runner
from src.m3_digit_factor.runner import (
    HistoricalExecutionNotAuthorizedError,
    RunIdValidationError,
    publish_artifacts,
    run_development,
)
from src.m3_digit_factor.validation import (
    ArtifactProtocolInconsistencyError,
    validate_success_artifacts,
)


def _make_synthetic_dataset(row_count: int = 605) -> CanonicalRawDataset:
    """Create a minimal valid synthetic dataset of 605 draws (240 eligible targets)."""
    dates = tuple(date(2010, 1, 1) + timedelta(days=i) for i in range(row_count))
    counts = np.zeros((row_count, 100), dtype=np.int16)
    counts[:, :27] = 1  # 27 hits per day summing to 27
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
    """Fast mock fitter producing valid mu of length 100 summing to 27.0."""
    mu = np.full(100, 0.27, dtype=np.float64)
    # Give window 60 a slightly lower deviance so it deterministically wins DEV
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


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Setup a mock repository root with canonical data file and spec blob."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    data_dir = repo / 'data'
    data_dir.mkdir()
    (data_dir / 'xsmb-2-digits.csv').write_bytes(b'dummy-canonical-data')
    return repo


@pytest.fixture
def real_spec_blob() -> bytes:
    """Read the canonical spec blob from real repository worktree."""
    real_spec_path = Path(CANONICAL_SPEC_RELATIVE_PATH)
    if not real_spec_path.exists():
        # Find it relative to worktree root
        real_spec_path = Path(__file__).resolve().parent.parent.parent / CANONICAL_SPEC_RELATIVE_PATH
    import subprocess
    return subprocess.check_output(['git', 'cat-file', 'blob', f'HEAD:{CANONICAL_SPEC_RELATIVE_PATH}'], cwd=real_spec_path.parent.parent.parent)


class TestAuthorityAndSourceIdentity:
    """Verify repository authority, git source identity, and clean checkout enforcement."""

    def test_source_commit_and_tree_captured_in_authority(self, repo_root: Path, real_spec_blob: bytes) -> None:
        git_runner = _make_git_runner()
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_auth_capture',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
            model_fitter=_mock_fitter,
        )
        assert result.status == 'COMPLETED'
        snap_path = result.output_dir / 'protocol_snapshot.json'
        snap_data = json.loads(snap_path.read_text(encoding='utf-8'))
        auth = snap_data['authority']
        assert auth['source_commit'] == 'a' * 40
        assert auth['source_tree'] == 'b' * 40
        assert auth['data_source_repository'] == 'Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis'

    def test_dirty_source_checkout_rejected(self, repo_root: Path, real_spec_blob: bytes) -> None:
        git_runner = _make_git_runner(status=' M modified_file.py')
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_dirty',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
        )
        assert result.status == 'FAILED'
        assert result.exit_status == FailureExitStatus.TECHNICAL_FAILURE
        assert result.published_artifacts == (FAILURE_ARTIFACT_NAME,)
        fail_path = result.output_dir / FAILURE_ARTIFACT_NAME
        fail_data = validate_failure_artifact(fail_path.read_bytes())
        assert fail_data['failed_stage'] == FailureStage.DATA_VALIDATION.value
        assert fail_data['protocol_snapshot'] is None  # Pre-authority null state

    def test_repository_identity_mismatch_rejected(self, repo_root: Path, real_spec_blob: bytes) -> None:
        git_runner = _make_git_runner(repo_name='other-org/other-repo')
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_repo_mismatch',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
        )
        assert result.status == 'FAILED'
        assert result.exit_status == FailureExitStatus.TECHNICAL_FAILURE
        fail_data = validate_failure_artifact((result.output_dir / FAILURE_ARTIFACT_NAME).read_bytes())
        assert fail_data['failed_stage'] == FailureStage.DATA_VALIDATION.value
        assert fail_data['protocol_snapshot'] is None

    def test_canonical_data_path_mismatch_rejected(self, repo_root: Path, real_spec_blob: bytes) -> None:
        git_runner = _make_git_runner(data_path='data/wrong_path.csv')
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_data_mismatch',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
        )
        assert result.status == 'FAILED'
        fail_data = validate_failure_artifact((result.output_dir / FAILURE_ARTIFACT_NAME).read_bytes())
        assert fail_data['failed_stage'] == FailureStage.DATA_VALIDATION.value
        assert fail_data['protocol_snapshot'] is None


class TestHistoricalExecutionGuard:
    """Verify safety boundary preventing real historical runs during M3-13."""

    def test_real_historical_run_refused_without_authorization(self, repo_root: Path, real_spec_blob: bytes) -> None:
        git_runner = _make_git_runner()
        # dataset_loader is None -> attempts to load canonical dataset
        with pytest.raises(HistoricalExecutionNotAuthorizedError, match='Real historical M3 development execution is NOT authorized'):
            run_development(
                repository_root=repo_root,
                output_root=repo_root / 'out',
                run_id='test_guard',
                git_runner=git_runner,
                blob_reader=lambda spec: real_spec_blob,
                authorize_historical_run=False,
            )


class TestCandidateEvaluationAndFreezing:
    """Verify DEV evaluates all 5 candidate windows, freezes winner, and isolates VAL."""

    def test_dev_evaluates_all_five_m3_candidates_and_freezes_winner(self, repo_root: Path, real_spec_blob: bytes) -> None:
        fitter_calls: list[int | None] = []

        def spy_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
            fitter_calls.append(W)
            return _mock_fitter(counts, W)

        git_runner = _make_git_runner()
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_freeze',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
            model_fitter=spy_fitter,
        )
        assert result.status == 'COMPLETED'
        assert result.selected_window == 60
        assert result.selected_candidate_id == 'M3_W060'

        # Verify all 5 windows were evaluated on DEV
        dev_windows_called = set(fitter_calls[: 120 * 5])
        assert dev_windows_called == {30, 60, 120, 240, 365}

        # Verify VAL evaluated ONLY winner window (60)
        val_calls = fitter_calls[120 * 5 : 120 * 5 + 60]
        assert len(val_calls) == 60
        assert set(val_calls) == {60}


class TestForecastGateBranching:
    """Verify forecast-failed short circuit and forecast-passed path."""

    def test_forecast_gate_failed_short_circuits_stability_and_economics(self, repo_root: Path, real_spec_blob: bytes) -> None:
        economic_evaluator_mock = mock.Mock()

        # Mock bootstrapper to return negative lower bound (forecast gate fail)
        def fail_bootstrapper(d: Any) -> ForecastBootstrapResult:
            return ForecastBootstrapResult(
                bootstrap_lower_bound=-0.01,
                bootstrap_replications=2000,
                replicate_means=np.array([-0.01]),
            )

        git_runner = _make_git_runner()
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_fc_fail',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
            model_fitter=_mock_fitter,
            forecast_bootstrapper=fail_bootstrapper,
            economic_evaluator=economic_evaluator_mock,
        )

        assert result.status == 'COMPLETED'
        assert result.exit_status == DevelopmentExitStatus.FORECAST_GATE_FAILED
        assert result.forecast_signal is False
        assert result.economic_signal is False

        # Verify STABILITY and economics were NEVER called
        economic_evaluator_mock.assert_not_called()

        # Verify exact 9 success artifacts produced and valid
        assert len(result.published_artifacts) == SUCCESS_ARTIFACT_COUNT
        validate_success_artifacts(result.output_dir)
        assert not (result.output_dir / FAILURE_ARTIFACT_NAME).exists()

        # Verify adjudication artifact indicates forecast gate failed
        adj_data = json.loads((result.output_dir / 'development_adjudication.json').read_text(encoding='utf-8'))
        assert adj_data['forecast_signal'] is False
        assert adj_data['development_exit_status'] == 'FORECAST_GATE_FAILED'
        assert adj_data['qualified_top_k'] == []
        assert adj_data['recommended_top_k'] == []

    def test_forecast_gate_passed_executes_stability_and_economics(self, repo_root: Path, real_spec_blob: bytes) -> None:
        def pass_bootstrapper(d: Any) -> ForecastBootstrapResult:
            return ForecastBootstrapResult(
                bootstrap_lower_bound=0.01,
                bootstrap_replications=2000,
                replicate_means=np.array([0.01]),
            )

        def mock_economic_eval(y: Any, mu: Any, blocks: Any, forecast_signal: bool = True) -> EconomicEvaluationResult:
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

        git_runner = _make_git_runner()
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_fc_pass',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
            model_fitter=_mock_fitter,
            forecast_bootstrapper=pass_bootstrapper,
            economic_evaluator=mock_economic_eval,
        )

        assert result.status == 'COMPLETED'
        assert result.exit_status == DevelopmentExitStatus.ECONOMIC_SIGNAL_FOUND
        assert result.forecast_signal is True
        assert result.economic_signal is True

        # Verify exact 9 success artifacts produced and valid
        assert len(result.published_artifacts) == SUCCESS_ARTIFACT_COUNT
        validate_success_artifacts(result.output_dir)

        adj_data = json.loads((result.output_dir / 'development_adjudication.json').read_text(encoding='utf-8'))
        assert adj_data['forecast_signal'] is True
        assert adj_data['economic_signal'] is True
        assert adj_data['qualified_top_k'] == [1, 3, 5]
        assert adj_data['recommended_top_k'] == [3]


class TestTerminalFailurePreservation:
    """Verify structured failure triplet and protocol snapshot preservation."""

    def test_scientific_failure_svd_ambiguity_preserves_triplet(self, repo_root: Path, real_spec_blob: bytes) -> None:
        def failing_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
            raise SVDLeadingSubspaceAmbiguity('Leading singular value gap below threshold')

        git_runner = _make_git_runner()
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_svd_fail',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
            model_fitter=failing_fitter,
        )
        assert result.status == 'FAILED'
        assert result.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION
        fail_path = result.output_dir / FAILURE_ARTIFACT_NAME
        fail_data = validate_failure_artifact(fail_path.read_bytes())
        assert fail_data['failed_stage'] == FailureStage.MODEL_INITIALIZATION.value
        assert fail_data['error_type'] == 'SVDLeadingSubspaceAmbiguity'
        assert fail_data['development_exit_status'] == FailureExitStatus.NEEDS_MODEL_REVISION.value
        # Post-authority complete snapshot
        assert fail_data['protocol_snapshot'] is not None
        assert len(fail_data['protocol_snapshot']['authority']) == 72

    def test_technical_failure_optimizer_execution_error_preserves_triplet(self, repo_root: Path, real_spec_blob: bytes) -> None:
        def failing_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
            raise OptimizerExecutionError('SLSQP internal solver exception')

        git_runner = _make_git_runner()
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_opt_fail',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
            model_fitter=failing_fitter,
        )
        assert result.status == 'FAILED'
        assert result.exit_status == FailureExitStatus.TECHNICAL_FAILURE
        fail_data = validate_failure_artifact((result.output_dir / FAILURE_ARTIFACT_NAME).read_bytes())
        assert fail_data['failed_stage'] == FailureStage.MODEL_FIT.value
        assert fail_data['error_type'] == 'OptimizerExecutionError'
        assert fail_data['development_exit_status'] == FailureExitStatus.TECHNICAL_FAILURE.value
        assert fail_data['protocol_snapshot'] is not None

    def test_artifact_protocol_inconsistency_error_preserves_triplet(self, repo_root: Path, real_spec_blob: bytes) -> None:
        git_runner = _make_git_runner()

        # Patch build_success_artifacts to inject an invalid schema
        with mock.patch('src.m3_digit_factor.runner.build_success_artifacts') as mock_artifacts_builder:
            mock_artifacts_builder.side_effect = ArtifactProtocolInconsistencyError('Artifact checksum mismatch')
            result = run_development(
                repository_root=repo_root,
                output_root=repo_root / 'out',
                run_id='test_art_fail',
                git_runner=git_runner,
                blob_reader=lambda spec: real_spec_blob,
                dataset_loader=lambda p: _make_synthetic_dataset(),
                model_fitter=_mock_fitter,
            )

        assert result.status == 'FAILED'
        assert result.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
        fail_data = validate_failure_artifact((result.output_dir / FAILURE_ARTIFACT_NAME).read_bytes())
        assert fail_data['failed_stage'] == FailureStage.ARTIFACT_VALIDATION.value
        assert fail_data['error_type'] == 'ArtifactProtocolInconsistencyError'
        assert fail_data['development_exit_status'] == FailureExitStatus.NEEDS_PROTOCOL_REVISION.value
        assert fail_data['protocol_snapshot'] is not None

    def test_artifact_protocol_validation_boundary_failure_end_to_end(
        self, repo_root: Path, real_spec_blob: bytes
    ) -> None:
        """Inject ArtifactProtocolInconsistencyError at validation boundary of otherwise valid success bundle.

        Verifies exact triplet:
        - failed_stage == FailureStage.ARTIFACT_VALIDATION
        - failure_policy_disposition == 'NEEDS_PROTOCOL_REVISION'
        - development_exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
        And verifies exactly 0 success files, exactly 1 failure file.
        """
        git_runner = _make_git_runner()

        orig_validate = src.m3_digit_factor.runner.validate_artifact_bundle_exclusivity

        def failing_validate_boundary(staging_dir: Any) -> Any:
            if not (Path(staging_dir) / FAILURE_ARTIFACT_NAME).exists():
                raise ArtifactProtocolInconsistencyError('Injected checksum violation at validation boundary')
            return orig_validate(staging_dir)

        with mock.patch(
            'src.m3_digit_factor.runner.validate_artifact_bundle_exclusivity',
            side_effect=failing_validate_boundary,
        ):
            result = run_development(
                repository_root=repo_root,
                output_root=repo_root / 'out',
                run_id='test_val_boundary_fail',
                git_runner=git_runner,
                blob_reader=lambda spec: real_spec_blob,
                dataset_loader=lambda p: _make_synthetic_dataset(),
                model_fitter=_mock_fitter,
            )

        assert result.status == 'FAILED'
        assert result.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
        assert result.published_artifacts == (FAILURE_ARTIFACT_NAME,)

        # Exact triplet verified
        fail_bytes = (result.output_dir / FAILURE_ARTIFACT_NAME).read_bytes()
        fail_data = validate_failure_artifact(fail_bytes)
        assert fail_data['failed_stage'] == FailureStage.ARTIFACT_VALIDATION.value
        assert fail_data['error_type'] == 'ArtifactProtocolInconsistencyError'
        assert fail_data['development_exit_status'] == FailureExitStatus.NEEDS_PROTOCOL_REVISION.value
        assert fail_data['protocol_snapshot'] is not None

        # Output directory must contain exactly 1 failure artifact and exactly 0 success artifacts
        published_files = [p.name for p in result.output_dir.iterdir()]
        assert published_files == [FAILURE_ARTIFACT_NAME]


class TestPublicationSafetyAndDirectoryMechanics:
    """Verify publication safety, existing directory fail-closed, and atomic isolation."""

    @pytest.mark.parametrize(
        'bad_run_id',
        [
            '',
            ' ',
            '   \t\n',
            '.',
            '..',
            '../traversal',
            '..\\traversal',
            'foo/bar',
            'foo\\bar',
            'nested/path/to/run',
            '/absolute/posix',
            '\\absolute\\windows',
            'C:\\absolute\\drive',
            'D:/absolute/drive',
            '\\\\server\\share\\run',
        ],
    )
    def test_run_id_rejects_empty_whitespace_slashes_traversal(
        self, bad_run_id: str, repo_root: Path, real_spec_blob: bytes
    ) -> None:
        """run_id must be a single safe child component without separators, drive specifiers, or traversal."""
        git_runner = _make_git_runner()
        out_root = repo_root / 'out'
        out_root.mkdir(parents=True, exist_ok=True)

        with pytest.raises(RunIdValidationError):
            run_development(
                repository_root=repo_root,
                output_root=out_root,
                run_id=bad_run_id,
                git_runner=git_runner,
                blob_reader=lambda spec: real_spec_blob,
                dataset_loader=lambda p: _make_synthetic_dataset(),
                model_fitter=_mock_fitter,
            )

        # Output root contains NO created directories or files from the bad run
        assert list(out_root.iterdir()) == []

    def test_existing_run_directory_fails_closed_without_modifying_contents(self, repo_root: Path, real_spec_blob: bytes) -> None:
        existing_dir = repo_root / 'out' / 'prior_run'
        existing_dir.mkdir(parents=True)
        prior_file = existing_dir / 'prior_evidence.txt'
        prior_file.write_text('important prior evidence')

        git_runner = _make_git_runner()
        with pytest.raises(FileExistsError, match='Target run directory already exists'):
            run_development(
                repository_root=repo_root,
                output_root=repo_root / 'out',
                run_id='prior_run',
                git_runner=git_runner,
                blob_reader=lambda spec: real_spec_blob,
                dataset_loader=lambda p: _make_synthetic_dataset(),
                model_fitter=_mock_fitter,
            )

        # Prior file preserved untouched
        assert prior_file.read_text() == 'important prior evidence'
        assert not (existing_dir / FAILURE_ARTIFACT_NAME).exists()
        assert [p.name for p in existing_dir.iterdir()] == ['prior_evidence.txt']

    def test_atomic_staging_leaves_no_partial_final_directory_on_failure(self, repo_root: Path) -> None:
        out_root = repo_root / 'out'
        target_dir = out_root / 'staged_run'
        # Pass invalid artifacts that will fail bundle exclusivity validation
        invalid_bundle = {'random_file.txt': b'bad content'}

        with pytest.raises(Exception):
            publish_artifacts(invalid_bundle, target_dir)

        assert not target_dir.exists()
        # Staging directories starting with .tmp_ must also be cleaned up
        staging_dirs = [p.name for p in out_root.iterdir() if p.name.startswith('.tmp_')]
        assert staging_dirs == []

    def test_validation_occurs_before_final_publication(self, repo_root: Path, real_spec_blob: bytes) -> None:
        """Verify that validate_success_artifacts runs on the bundle before target_dir is created."""
        validation_called = False

        def spy_validator(path_or_bundle: Any) -> Any:
            nonlocal validation_called
            validation_called = True
            # Assert target_dir does not exist yet at the time of validation
            target = repo_root / 'out' / 'test_val_order'
            assert not target.exists()
            return validate_success_artifacts(path_or_bundle)

        git_runner = _make_git_runner()
        with mock.patch('src.m3_digit_factor.runner.validate_artifact_bundle_exclusivity', side_effect=spy_validator):
            result = run_development(
                repository_root=repo_root,
                output_root=repo_root / 'out',
                run_id='test_val_order',
                git_runner=git_runner,
                blob_reader=lambda spec: real_spec_blob,
                dataset_loader=lambda p: _make_synthetic_dataset(),
                model_fitter=_mock_fitter,
            )

        assert validation_called is True
        assert result.output_dir.exists()


class TestOrchestrationOrderingAndStateIsolation:
    """Verify strict execution sequencing, call ordering, and state boundaries."""

    def test_orchestration_call_order_pinned(self, repo_root: Path, real_spec_blob: bytes) -> None:
        call_order: list[str] = []

        def spy_git(*args: str) -> str:
            if 'authority' not in call_order:
                call_order.append('authority')
            return _make_git_runner()(*args)

        def spy_dataset(p: Path) -> CanonicalRawDataset:
            call_order.append('dataset_loader')
            return _make_synthetic_dataset()

        def spy_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
            # Distinguish DEV vs VAL vs STABILITY by length of call_order
            if 'dev_fit' not in call_order:
                call_order.append('dev_fit')
            elif 'val_fit' not in call_order and 'forecast_bootstrap' not in call_order:
                call_order.append('val_fit')
            elif 'forecast_gate_passed' in call_order and 'stability_fit' not in call_order:
                call_order.append('stability_fit')
            return _mock_fitter(counts, W)

        def spy_bootstrapper(d: Any) -> ForecastBootstrapResult:
            call_order.append('forecast_bootstrap')
            return ForecastBootstrapResult(0.01, 2000, np.array([0.01]))

        def spy_economic_eval(y: Any, mu: Any, blocks: Any, forecast_signal: bool = True) -> EconomicEvaluationResult:
            call_order.append('economic_eval')
            per_k = {
                k: PerKEconomicEvaluation(k, 10.0, (10.0,) * 6, 6, 2.0, True)
                for k in [1, 3, 5, 10]
            }
            return EconomicEvaluationResult(
                per_k=per_k,
                economic_signal=True,
                qualified_top_k=[1, 3, 5, 10],
                recommended_top_k=[1],
                bootstrap_executed=True,
                economic_delta=np.ones((60, 4)),
            )

        # Hook to mark when forecast gate passes
        original_eval_gate = src.m3_digit_factor.runner.evaluate_forecast_gate

        def spy_gate(*args: Any, **kwargs: Any) -> Any:
            res = original_eval_gate(*args, **kwargs)
            call_order.append('forecast_gate_passed')
            return res

        with mock.patch('src.m3_digit_factor.runner.evaluate_forecast_gate', side_effect=spy_gate):
            result = run_development(
                repository_root=repo_root,
                output_root=repo_root / 'out',
                run_id='test_call_order',
                git_runner=spy_git,
                blob_reader=lambda spec: real_spec_blob,
                dataset_loader=spy_dataset,
                model_fitter=spy_fitter,
                forecast_bootstrapper=spy_bootstrapper,
                economic_evaluator=spy_economic_eval,
            )

        assert result.status == 'COMPLETED'
        expected_order = [
            'authority',
            'dataset_loader',
            'dev_fit',
            'val_fit',
            'forecast_bootstrap',
            'forecast_gate_passed',
            'stability_fit',
            'economic_eval',
        ]
        assert call_order == expected_order

    def test_economic_bootstrap_called_once_on_complete_matrix(self, repo_root: Path, real_spec_blob: bytes) -> None:
        """Verify economic bootstrap is executed exactly once during economic evaluation."""
        git_runner = _make_git_runner()

        def pass_bootstrapper(d: Any) -> ForecastBootstrapResult:
            return ForecastBootstrapResult(0.01, 2000, np.array([0.01]))

        bootstrap_call_count = 0
        original_econ_bootstrap = src.m3_digit_factor.economics.run_economic_bootstrap

        def spy_econ_bootstrap(delta_matrix: np.ndarray) -> Any:
            nonlocal bootstrap_call_count
            bootstrap_call_count += 1
            assert delta_matrix.shape == (60, 4)
            return original_econ_bootstrap(delta_matrix)

        with mock.patch('src.m3_digit_factor.economics.run_economic_bootstrap', side_effect=spy_econ_bootstrap):
            result = run_development(
                repository_root=repo_root,
                output_root=repo_root / 'out',
                run_id='test_econ_bs_once',
                git_runner=git_runner,
                blob_reader=lambda spec: real_spec_blob,
                dataset_loader=lambda p: _make_synthetic_dataset(),
                model_fitter=_mock_fitter,
                forecast_bootstrapper=pass_bootstrapper,
            )

        assert result.status == 'COMPLETED'
        assert bootstrap_call_count == 1

    def test_val_does_not_reopen_candidate_selection(self, repo_root: Path, real_spec_blob: bytes) -> None:
        """Verify VAL strictly evaluates only frozen candidate and does not evaluate other windows."""
        fitted_windows_in_val: list[int | None] = []

        # Target index 485 is the first VAL target index in a 605-row dataset (dev is 365..484)
        def tracking_fitter(counts: Any, W: int | None = None) -> FittedModelResult:
            # We track which windows are requested during VAL
            fitted_windows_in_val.append(W)
            return _mock_fitter(counts, W)

        git_runner = _make_git_runner()
        result = run_development(
            repository_root=repo_root,
            output_root=repo_root / 'out',
            run_id='test_val_freeze_only',
            git_runner=git_runner,
            blob_reader=lambda spec: real_spec_blob,
            dataset_loader=lambda p: _make_synthetic_dataset(),
            model_fitter=tracking_fitter,
        )

        assert result.status == 'COMPLETED'
        frozen_winner = result.selected_window
        assert frozen_winner == 60

        # After 120 * 5 DEV fits, VAL begins:
        val_fitter_calls = fitted_windows_in_val[120 * 5 : 120 * 5 + 60]
        assert len(val_fitter_calls) == 60
        assert all(w == frozen_winner for w in val_fitter_calls)

