"""Thin CLI entrypoint for XPIS v3 M3 digit-factor development (Slice M3-13).

This script parses execution options and delegates entirely to the
reviewed orchestration runner (src.m3_digit_factor.runner.run_development).
It contains NO scientific formulas, NO model fitting code, NO bootstrap code,
NO economics code, and NO artifact schema definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.m3_digit_factor.runner import (
    HistoricalExecutionNotAuthorizedError,
    run_development,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for development pipeline execution."""
    parser = argparse.ArgumentParser(
        description='XPIS v3 M3 Digit-Factor Development Pipeline Entrypoint',
    )
    parser.add_argument(
        '--repository-root',
        type=Path,
        default=None,
        help='Path to repository root (defaults to parent directory of backtests/)',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Root directory for development publication',
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Deterministic or custom run identifier',
    )
    parser.add_argument(
        '--authorize-historical-run',
        action='store_true',
        default=False,
        help='Explicit authorization token required for real historical development execution',
    )
    return parser


def main(args: list[str] | None = None) -> int:
    """Execute M3 development run from parsed CLI arguments."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    repo_root = parsed.repository_root
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    try:
        result = run_development(
            repository_root=repo_root,
            output_root=parsed.output_root,
            run_id=parsed.run_id,
            authorize_historical_run=parsed.authorize_historical_run,
        )
    except HistoricalExecutionNotAuthorizedError as exc:
        print(f'EXECUTION REFUSED: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:
        print(f'RUNNER EXECUTION ERROR: {exc}', file=sys.stderr)
        return 1

    print(f'M3 Development Run {result.status}: {result.exit_status.value}')
    print(f'Output Directory: {result.output_dir}')
    if result.status == 'COMPLETED':
        print(f'Selected Candidate: {result.selected_candidate_id} (W={result.selected_window})')
        print(f'Forecast Signal: {result.forecast_signal}')
        print(f'Economic Signal: {result.economic_signal}')
        return 0

    err_type = result.failure_artifact.get('error_type', 'Unknown') if result.failure_artifact else 'Unknown'
    print(f'Terminal Failure Error Type: {err_type}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
