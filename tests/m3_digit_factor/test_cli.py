"""Tests for thin CLI entrypoint backtests/m3_digit_factor_development.py (Slice M3-13)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
import unittest.mock as mock

import pytest

import backtests.m3_digit_factor_development as cli
from src.m3_digit_factor.contracts import DevelopmentExitStatus, FailureExitStatus
from src.m3_digit_factor.runner import (
    DevelopmentRunResult,
    HistoricalExecutionNotAuthorizedError,
)


class TestCLIStructureAndThinness:
    """Verify CLI thinness and lack of scientific / model / artifact code."""

    def test_cli_import_has_no_side_effects(self) -> None:
        """Importing CLI module does not trigger execution."""
        # Already imported without side effects
        assert hasattr(cli, 'main')
        assert hasattr(cli, 'build_parser')

    def test_cli_source_contains_no_scientific_formulas_or_model_fitting(self) -> None:
        """Inspect CLI AST to ensure no fitting, bootstrap, or economic calculations exist."""
        cli_file = Path(cli.__file__)
        tree = ast.parse(cli_file.read_text(encoding='utf-8'))

        prohibited_names = {
            'scipy',
            'fit_m3_model',
            'initialize_m3_from_counts',
            'run_forecast_bootstrap',
            'run_economic_bootstrap',
            'evaluate_economic_protocol',
            'build_success_artifacts',
            'build_failure_artifact',
        }

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.add(node.module)
                for alias in node.names:
                    imported_names.add(alias.name)

        overlap = imported_names.intersection(prohibited_names)
        assert not overlap, f'CLI contains prohibited scientific imports: {overlap}'


class TestCLIRunnerDispatch:
    """Verify CLI argument parsing and exit codes."""

    def test_historical_execution_guard_exits_with_code_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI without --authorize-historical-run exits with code 2 on historical attempt."""
        def mock_run_development(**kwargs: Any) -> DevelopmentRunResult:
            if not kwargs.get('authorize_historical_run', False):
                raise HistoricalExecutionNotAuthorizedError('Historical execution not authorized')
            return mock.Mock()

        monkeypatch.setattr('backtests.m3_digit_factor_development.run_development', mock_run_development)

        exit_code = cli.main([])
        assert exit_code == 2

    def test_cli_successful_run_returns_code_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI returns 0 on successful COMPLETED development run."""
        mock_res = DevelopmentRunResult(
            status='COMPLETED',
            exit_status=DevelopmentExitStatus.ECONOMIC_SIGNAL_FOUND,
            output_dir=tmp_path / 'out',
            run_id='test_cli_pass',
            selected_candidate_id='M3_W060',
            selected_window=60,
            forecast_signal=True,
            economic_signal=True,
            published_artifacts=('protocol_snapshot.json',),
        )

        monkeypatch.setattr('backtests.m3_digit_factor_development.run_development', lambda **kwargs: mock_res)

        exit_code = cli.main(['--output-root', str(tmp_path / 'out'), '--run-id', 'test_cli_pass'])
        assert exit_code == 0

    def test_cli_forecast_failed_run_returns_code_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ordinary forecast-gate failure is a completed development outcome and returns 0."""
        mock_res = DevelopmentRunResult(
            status='COMPLETED',
            exit_status=DevelopmentExitStatus.FORECAST_GATE_FAILED,
            output_dir=tmp_path / 'out',
            run_id='test_cli_fc_fail',
            selected_candidate_id='M3_W060',
            selected_window=60,
            forecast_signal=False,
            economic_signal=False,
            published_artifacts=('protocol_snapshot.json',),
        )

        monkeypatch.setattr('backtests.m3_digit_factor_development.run_development', lambda **kwargs: mock_res)

        exit_code = cli.main(['--output-root', str(tmp_path / 'out')])
        assert exit_code == 0

    def test_cli_terminal_failure_returns_code_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Terminal failure returning FAILED status exits with code 1."""
        mock_res = DevelopmentRunResult(
            status='FAILED',
            exit_status=FailureExitStatus.TECHNICAL_FAILURE,
            output_dir=tmp_path / 'out',
            run_id='test_cli_fail',
            failure_artifact={'error_type': 'OptimizerExecutionError'},
            published_artifacts=('development_run_FAILED.json',),
        )

        monkeypatch.setattr('backtests.m3_digit_factor_development.run_development', lambda **kwargs: mock_res)

        exit_code = cli.main(['--output-root', str(tmp_path / 'out')])
        assert exit_code == 1

    def test_cli_unhandled_exception_returns_code_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unhandled unexpected exception exits with code 1."""
        def error_runner(**kwargs: Any) -> DevelopmentRunResult:
            raise RuntimeError('Unexpected crash')

        monkeypatch.setattr('backtests.m3_digit_factor_development.run_development', error_runner)

        exit_code = cli.main([])
        assert exit_code == 1

    def test_cli_allowed_options_surface_strictly_bounded(self) -> None:
        """CLI options surface must contain zero scientific overrides and only allowed runner flags."""
        parser = cli.build_parser()
        allowed_flags = {'-h', '--help', '--repository-root', '--output-root', '--run-id', '--authorize-historical-run'}
        actual_flags = set()
        for action in parser._actions:
            actual_flags.update(action.option_strings)

        assert actual_flags == allowed_flags, f'CLI option surface mismatch: {actual_flags} vs {allowed_flags}'

        # Verify no scientific overrides can be passed
        prohibited_scientific_flags = {'--window', '--seed', '--k', '--csv', '--alpha', '--model'}
        assert not actual_flags.intersection(prohibited_scientific_flags)

    @pytest.mark.parametrize(
        'unknown_args',
        [
            ['--window', '60'],
            ['--seed', '20260831'],
            ['--k', '5'],
            ['--csv', 'data/xsmb-2-digits.csv'],
            ['--alpha', '0.05'],
            ['--unrecognized'],
            ['-x'],
        ],
    )
    def test_cli_unknown_option_rejection_returns_code_2(self, unknown_args: list[str]) -> None:
        """CLI must reject unknown options with exit code 2."""
        exit_code = cli.main(unknown_args)
        assert exit_code == 2

    @pytest.mark.parametrize(
        'bad_run_id',
        [
            '../traversal',
            '..\\traversal',
            'foo/bar',
            'foo\\bar',
            '.',
            '..',
        ],
    )
    def test_cli_invalid_run_id_returns_code_2(self, bad_run_id: str, tmp_path: Path) -> None:
        """CLI must reject invalid or traversing run_id with exit code 2."""
        exit_code = cli.main(['--output-root', str(tmp_path), '--run-id', bad_run_id])
        assert exit_code == 2

