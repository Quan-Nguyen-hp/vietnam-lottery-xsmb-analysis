"""Deterministic M3 development orchestration runner (Slice M3-13).

This module orchestrates the complete end-to-end XPIS v3 M3 development pipeline:
1. Repository and source authority verification (fail-closed if dirty or authority mismatch).
2. Protocol authority snapshot finalization (72-key closed authority).
3. Canonical dataset loading and chronological splits derivation (DEV / VAL / STABILITY).
4. DEV candidate evaluation across all 5 M3 windows [30, 60, 120, 240, 365] + B0 baseline.
5. Strict lexicographic DEV winner selection (PD, MAE, RMSE, W) and candidate freeze.
6. VAL evaluation on frozen M3 candidate + B0.
7. Stationary circular forecast bootstrap computation on VAL deviance differences.
8. Complete forecast gate evaluation.
9. Forecast-failed short circuit: skips STABILITY and economics, emits exact 9 success artifacts.
10. Forecast-passed path: evaluates STABILITY, runs economic protocol and stationary economic bootstrap once.
11. Safe atomic publication of either exact 9 success artifacts or single failure artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid

import numpy as np

from .artifacts import (
    SUCCESS_ARTIFACT_NAMES,
    build_success_artifacts,
)
from .authority import (
    CANONICAL_SPEC_SHA256,
    AuthoritySnapshot,
    AuthorityValidationError,
    build_authority,
    canonical_spec_sha256_from_git,
    collect_repository_identity,
    construct_authority_snapshot,
    protocol_fingerprint,
)
from .bootstrap import (
    ForecastBootstrapResult,
    run_forecast_bootstrap,
)
from .contracts import (
    CandidateID,
    DevelopmentExitStatus,
    DevelopmentStage,
    FailureExitStatus,
    ModelID,
)
from .dataset import CanonicalRawDataset, load_canonical_dataset
from .economics import (
    ECONOMIC_K_VALUES,
    EconomicEvaluationResult,
    evaluate_economic_protocol,
)
from .failure import (
    FAILURE_ARTIFACT_NAME,
    build_failure_artifact_from_exception,
    validate_artifact_bundle_exclusivity,
)
from .fitting import FittedModelResult, fit_m3_model
from .metrics import (
    CANDIDATE_WINDOWS,
    DailyForecastMetrics,
    StageForecastMetrics,
    candidate_id_for_window,
    compute_daily_metrics,
    compute_stage_metrics,
    create_b0_forecast,
    evaluate_forecast_gate,
    select_dev_winner,
)
from .splits import ChronologicalSplits, build_chronological_splits


class HistoricalExecutionNotAuthorizedError(RuntimeError):
    """Raised when real historical execution is attempted without Control Plane authorization."""


class RunIdValidationError(ValueError):
    """Raised when run_id violates single safe path component and traversal constraints."""


class ArtifactPublicationError(ValueError):
    """Raised when artifact publication constraints or basename safety checks are violated."""


class PostPublicationVerificationError(RuntimeError):
    """Raised when post-publication directory inventory or integrity check fails."""


class PostPublicationQuarantineError(PostPublicationVerificationError):
    """Raised when quarantining an invalid post-publication directory fails."""


def _validate_artifact_key_basename(key: str, staging_dir: Path) -> None:
    """Validate that artifact key is a single safe basename without path traversal.

    Rejects:
    - Non-string types, empty strings, strings with leading/trailing whitespace
    - '.' and '..'
    - Path separators ('/' and '\\') and drive specifiers (':')
    - Any path resolving outside or not an immediate child of staging_dir
    """
    if not isinstance(key, str):
        raise ArtifactPublicationError(f"Artifact key must be a string, got {type(key).__name__}")

    if not key or key.strip() != key:
        raise ArtifactPublicationError(f"Invalid artifact key {key!r}: cannot be empty or have surrounding whitespace")

    if key in {".", ".."}:
        raise ArtifactPublicationError(f"Invalid artifact key {key!r}: cannot be '.' or '..'")

    if "/" in key or "\\" in key or ":" in key:
        raise ArtifactPublicationError(f"Invalid artifact key {key!r}: path separators and drive specifiers are forbidden")

    staging_resolved = staging_dir.resolve()
    target_path = (staging_resolved / key).resolve()

    if target_path.parent != staging_resolved or target_path.name != key:
        raise ArtifactPublicationError(
            f"Invalid artifact key {key!r}: must resolve to an immediate child beneath staging directory"
        )


def validate_run_id(run_id: str, output_root: Path) -> str:
    """Validate that run_id is a single safe path component without traversal.

    Rejects:
    - Non-string types, empty strings, strings with leading/trailing whitespace
    - '.' and '..'
    - Path separators ('/' and '\\') and drive specifiers (':')
    - Any path resolving outside or not an immediate child of output_root
    """
    if not isinstance(run_id, str):
        raise RunIdValidationError(f'run_id must be a string, got {type(run_id).__name__}')

    if not run_id or run_id.strip() != run_id:
        raise RunIdValidationError(f'Invalid run_id {run_id!r}: cannot be empty or have surrounding whitespace')

    if run_id in {'.', '..'}:
        raise RunIdValidationError(f'Invalid run_id {run_id!r}: cannot be "." or ".."')

    if '/' in run_id or '\\' in run_id or ':' in run_id:
        raise RunIdValidationError(f'Invalid run_id {run_id!r}: path separators and drive specifiers are forbidden')

    out_root_resolved = Path(output_root).resolve()
    target_path = (out_root_resolved / run_id).resolve()

    if target_path.parent != out_root_resolved or target_path.name != run_id:
        raise RunIdValidationError(
            f'Invalid run_id {run_id!r}: must be an immediate single child component beneath output_root'
        )

    return run_id


@dataclass(frozen=True)
class DevelopmentRunResult:
    """Outcome metadata from an M3 development pipeline execution."""

    status: str
    exit_status: DevelopmentExitStatus | FailureExitStatus
    output_dir: Path
    run_id: str
    selected_candidate_id: str | None = None
    selected_window: int | None = None
    forecast_signal: bool = False
    economic_signal: bool = False
    failure_artifact: dict[str, Any] | None = None
    published_artifacts: tuple[str, ...] = ()


def check_clean_working_tree(
    repository_root: Path,
    *,
    git_runner: Callable[..., str] | None = None,
) -> None:
    """Verify that git working tree is clean."""
    if git_runner is not None:
        status = git_runner('status', '--porcelain')
    else:
        res = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise AuthorityValidationError('Failed to check Git working tree status')
        status = res.stdout.strip()

    if status:
        raise AuthorityValidationError(
            f'Source checkout is dirty; clean repository required for authority capture:\n{status}'
        )


def _quarantine_final_directory(final_run_dir: Path) -> Path:
    """Move an invalid published directory away to a quarantine sibling.

    Ensures that final_run_dir does not remain in place upon post-publication failure.
    Uses atomic same-filesystem rename without copy fallback.
    """
    quarantine_dir = (
        final_run_dir.parent / f'.quarantine_{final_run_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}'
    )
    try:
        os.rename(str(final_run_dir), str(quarantine_dir))
    except Exception as exc:
        raise PostPublicationQuarantineError(
            f'Failed to quarantine invalid post-publication directory {final_run_dir} to {quarantine_dir}: {exc}'
        ) from exc
    return quarantine_dir


def publish_artifacts(
    artifacts: Mapping[str, bytes],
    output_dir: Path,
) -> Path:
    """Safely publish an artifact bundle atomically to output_dir.

    Fails closed if output_dir already exists.
    Uses temporary staging directory in output_dir's parent, validates keys as safe basenames,
    writes all files, validates the bundle, and atomically renames to output_dir with no copy fallback.
    Performs post-publication inventory verification and byte revalidation.
    On post-publication failure, quarantines output_dir so no invalid directory remains in place.
    """
    if output_dir.exists():
        raise FileExistsError(f'Target run directory already exists: {output_dir}')

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f'.tmp_{output_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}'
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        for name, content in artifacts.items():
            _validate_artifact_key_basename(name, staging_dir)
            (staging_dir / name).write_bytes(content)

        # Validate bundle in staging directory (enforces mutual exclusivity and schema)
        validate_artifact_bundle_exclusivity(staging_dir)

        # Pre-rename race check: destination must not already exist
        if output_dir.exists():
            raise FileExistsError(f'Target run directory already exists: {output_dir}')

        # Atomic same-filesystem rename without copy fallback
        os.rename(str(staging_dir), str(output_dir))
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    try:
        # Post-publication inventory verification
        if not output_dir.is_dir():
            raise PostPublicationVerificationError(f'Published output directory is not a directory: {output_dir}')

        published_items = list(output_dir.iterdir())
        for item in published_items:
            if not item.is_file():
                raise PostPublicationVerificationError(f'Published directory contains non-file item: {item}')

        published_names = {item.name for item in published_items}
        expected_names = set(artifacts.keys())
        if published_names != expected_names:
            raise PostPublicationVerificationError(
                f'Post-publication inventory mismatch: expected {expected_names}, found {published_names}'
            )

        if expected_names != set(SUCCESS_ARTIFACT_NAMES) and expected_names != {FAILURE_ARTIFACT_NAME}:
            raise PostPublicationVerificationError(
                f'Post-publication inventory does not match canonical success (9) or failure (1) set: {expected_names}'
            )

        # Complete byte and schema revalidation on published directory
        try:
            validate_artifact_bundle_exclusivity(output_dir)
        except Exception as val_exc:
            raise PostPublicationVerificationError(
                f'Post-publication byte validation failed on {output_dir}: {val_exc}'
            ) from val_exc

    except Exception:
        if output_dir.exists():
            _quarantine_final_directory(output_dir)
        raise

    return output_dir


def run_development(
    repository_root: Path,
    output_root: Path | None = None,
    run_id: str | None = None,
    *,
    authorize_historical_run: bool = False,
) -> DevelopmentRunResult:
    """Execute the production deterministic M3 development run.

    Requires explicit authorize_historical_run=True from Control Plane.
    Accepts NO dependency injection parameters, ensuring that production execution
    invariants and lifecycle authority gates cannot be bypassed.
    """
    if not authorize_historical_run:
        raise HistoricalExecutionNotAuthorizedError(
            'Real historical M3 development execution is NOT authorized by Control Plane. '
            'Explicit authorization token required.'
        )

    return _run_development_with_dependencies(
        repository_root=repository_root,
        output_root=output_root,
        run_id=run_id,
        authorize_historical_run=True,
    )


def _detect_canonical_loader_reference(fn: Any, target_fn: Any) -> bool:
    """Detect if fn is or directly/indirectly wraps or references target_fn."""
    if fn is target_fn:
        return True
    wrapped = getattr(fn, '__wrapped__', None)
    if wrapped is not None and _detect_canonical_loader_reference(wrapped, target_fn):
        return True
    closure = getattr(fn, '__closure__', None)
    if closure:
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if val is target_fn or _detect_canonical_loader_reference(val, target_fn):
                return True
    fn_globals = getattr(fn, '__globals__', None)
    if fn_globals:
        code = getattr(fn, '__code__', None)
        if code and 'load_canonical_dataset' in code.co_names:
            val = fn_globals.get('load_canonical_dataset')
            if val is target_fn or getattr(val, '__name__', '') == 'load_canonical_dataset':
                return True
    return False


def _run_development_with_dependencies(
    repository_root: Path,
    output_root: Path | None = None,
    run_id: str | None = None,
    *,
    git_runner: Callable[..., str] | None = None,
    blob_reader: Callable[[str], bytes] | None = None,
    dataset_loader: Callable[[Path], CanonicalRawDataset] | None = None,
    model_fitter: Callable[[Sequence[float] | np.ndarray, int | None], FittedModelResult] | None = None,
    forecast_bootstrapper: Callable[[np.ndarray | Sequence[float]], ForecastBootstrapResult] | None = None,
    economic_evaluator: Callable[..., EconomicEvaluationResult] | None = None,
    authorize_historical_run: bool = False,
) -> DevelopmentRunResult:
    """Execute M3 development run with dependency injection (internal test harness only)."""
    repo_root = Path(repository_root).resolve()
    out_root = (
        Path(output_root).resolve()
        if output_root is not None
        else repo_root / 'research_artifacts' / 'xpis_v3_m3_digit_factor' / 'development'
    )
    r_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') if run_id is None else run_id
    validate_run_id(r_id, out_root)
    target_dir = out_root / r_id

    # Section 22: Existing run directory fail closed
    if target_dir.exists():
        raise FileExistsError(f'Target run directory already exists: {target_dir}')

    authority_snapshot: AuthoritySnapshot | None = None

    try:
        # 1. Establish repository/source authority & check clean working tree
        check_clean_working_tree(repo_root, git_runner=git_runner)

        if blob_reader is not None:
            spec_sha = canonical_spec_sha256_from_git(repo_root, blob_reader=blob_reader)
            if spec_sha != CANONICAL_SPEC_SHA256:
                raise AuthorityValidationError('canonical specification identity does not match frozen authority')
            identity = collect_repository_identity(repo_root, git_runner=git_runner)
            authority = build_authority(identity)
            authority_snapshot = AuthoritySnapshot(authority, protocol_fingerprint(authority))
        else:
            authority_snapshot = construct_authority_snapshot(repo_root, git_runner=git_runner)

        # 2. Historical execution guard (Sections 21-25)
        if not authorize_historical_run:
            from . import dataset as _dataset_module
            orig_canonical_loader = _dataset_module.load_canonical_dataset

            if dataset_loader is None or _detect_canonical_loader_reference(dataset_loader, orig_canonical_loader):
                raise HistoricalExecutionNotAuthorizedError(
                    'Real historical M3 development execution is NOT authorized by Control Plane. '
                    'Synthetic orchestration and testing requires an explicit synthetic dataset_loader.'
                )

        # 3. Load dataset (with guard against indirect canonical loading if unauthorized)
        if not authorize_historical_run:
            from . import dataset as _dataset_module
            orig_loader = _dataset_module.load_canonical_dataset

            def _forbidden_canonical_load(*args: Any, **kwargs: Any) -> Any:
                raise HistoricalExecutionNotAuthorizedError(
                    'Real historical M3 development execution is NOT authorized by Control Plane. '
                    'Direct or indirect invocation of load_canonical_dataset is forbidden when authorize_historical_run=False.'
                )

            _dataset_module.load_canonical_dataset = _forbidden_canonical_load
            try:
                dataset: CanonicalRawDataset = dataset_loader(repo_root)  # type: ignore[misc]
            finally:
                _dataset_module.load_canonical_dataset = orig_loader
        else:
            dataset = (dataset_loader or load_canonical_dataset)(repo_root)

        # 4 & 5. Derive eligible targets and chronological splits
        splits: ChronologicalSplits = build_chronological_splits(dataset)
        dev_indices = splits.dev_indices
        val_indices = splits.val_indices
        stability_indices = splits.stability_indices
        stability_blocks = splits.stability_blocks

        daily_score_rows: list[list[Any]] = []
        forecast_metric_rows: list[dict[str, Any]] = []

        # 6. Evaluate B0 on DEV
        b0_forecast = create_b0_forecast()
        b0_dev_daily: list[DailyForecastMetrics] = []
        for t in dev_indices:
            y_t = dataset.count_matrix[t]
            m_b0 = compute_daily_metrics(y_t, b0_forecast)
            b0_dev_daily.append(m_b0)
            daily_score_rows.append([
                str(dataset.recorded_dates[t]),
                DevelopmentStage.DEV.value,
                ModelID.B0.value,
                CandidateID.B0_UNIFORM.value,
                m_b0.poisson_deviance,
                m_b0.mae,
                m_b0.rmse,
            ])
        b0_dev_stage = compute_stage_metrics(b0_dev_daily)
        forecast_metric_rows.append({
            'stage': DevelopmentStage.DEV.value,
            'model_id': ModelID.B0.value,
            'candidate_id': CandidateID.B0_UNIFORM.value,
            'date_count': splits.N_dev,
            'poisson_deviance': b0_dev_stage.poisson_deviance,
            'mae': b0_dev_stage.mae,
            'rmse': b0_dev_stage.rmse,
        })

        # 7. Evaluate all five M3 candidates on DEV
        dev_stage_metrics: dict[int, StageForecastMetrics] = {}
        for W in CANDIDATE_WINDOWS:
            c_id = candidate_id_for_window(W)
            c_id_str = c_id.value if hasattr(c_id, 'value') else str(c_id)
            m3_dev_daily: list[DailyForecastMetrics] = []
            for t in dev_indices:
                counts = np.sum(dataset.count_matrix[t - W : t], axis=0)
                fit_res = (model_fitter or fit_m3_model)(counts, W=W)
                y_t = dataset.count_matrix[t]
                m_m3 = compute_daily_metrics(y_t, fit_res.mu)
                m3_dev_daily.append(m_m3)
                daily_score_rows.append([
                    str(dataset.recorded_dates[t]),
                    DevelopmentStage.DEV.value,
                    ModelID.M3.value,
                    c_id_str,
                    m_m3.poisson_deviance,
                    m_m3.mae,
                    m_m3.rmse,
                ])
            stage_m = compute_stage_metrics(m3_dev_daily)
            dev_stage_metrics[W] = stage_m
            forecast_metric_rows.append({
                'stage': DevelopmentStage.DEV.value,
                'model_id': ModelID.M3.value,
                'candidate_id': c_id_str,
                'date_count': splits.N_dev,
                'poisson_deviance': stage_m.poisson_deviance,
                'mae': stage_m.mae,
                'rmse': stage_m.rmse,
            })

        # 8 & 9. Select DEV winner and freeze
        dev_winner = select_dev_winner(dev_stage_metrics)
        winner_window = dev_winner.window
        winner_candidate_id = (
            dev_winner.candidate_id.value
            if hasattr(dev_winner.candidate_id, 'value')
            else str(dev_winner.candidate_id)
        )

        # 10. Evaluate B0 and frozen M3 on VAL
        b0_val_daily: list[DailyForecastMetrics] = []
        m3_val_daily: list[DailyForecastMetrics] = []
        deviance_diffs: list[float] = []

        for t in val_indices:
            y_t = dataset.count_matrix[t]
            target_date_str = str(dataset.recorded_dates[t])

            m_b0 = compute_daily_metrics(y_t, b0_forecast)
            b0_val_daily.append(m_b0)
            daily_score_rows.append([
                target_date_str,
                DevelopmentStage.VAL.value,
                ModelID.B0.value,
                CandidateID.B0_UNIFORM.value,
                m_b0.poisson_deviance,
                m_b0.mae,
                m_b0.rmse,
            ])

            counts = np.sum(dataset.count_matrix[t - winner_window : t], axis=0)
            fit_res = (model_fitter or fit_m3_model)(counts, W=winner_window)
            m_m3 = compute_daily_metrics(y_t, fit_res.mu)
            m3_val_daily.append(m_m3)
            daily_score_rows.append([
                target_date_str,
                DevelopmentStage.VAL.value,
                ModelID.M3.value,
                winner_candidate_id,
                m_m3.poisson_deviance,
                m_m3.mae,
                m_m3.rmse,
            ])

            deviance_diffs.append(m_b0.poisson_deviance - m_m3.poisson_deviance)

        b0_val_stage = compute_stage_metrics(b0_val_daily)
        m3_val_stage = compute_stage_metrics(m3_val_daily)

        forecast_metric_rows.append({
            'stage': DevelopmentStage.VAL.value,
            'model_id': ModelID.B0.value,
            'candidate_id': CandidateID.B0_UNIFORM.value,
            'date_count': splits.N_val,
            'poisson_deviance': b0_val_stage.poisson_deviance,
            'mae': b0_val_stage.mae,
            'rmse': b0_val_stage.rmse,
        })
        forecast_metric_rows.append({
            'stage': DevelopmentStage.VAL.value,
            'model_id': ModelID.M3.value,
            'candidate_id': winner_candidate_id,
            'date_count': splits.N_val,
            'poisson_deviance': m3_val_stage.poisson_deviance,
            'mae': m3_val_stage.mae,
            'rmse': m3_val_stage.rmse,
        })

        # 11. Compute forecast bootstrap on VAL
        d_arr = np.array(deviance_diffs, dtype=np.float64)
        observed_mean_improvement = float(np.mean(d_arr))
        fb_res = (forecast_bootstrapper or run_forecast_bootstrap)(d_arr)
        forecast_bootstrap_lower_bound = fb_res.bootstrap_lower_bound

        # 12. Evaluate complete forecast gate
        gate_res = evaluate_forecast_gate(
            pd_val_m3=m3_val_stage.poisson_deviance,
            pd_val_b0=b0_val_stage.poisson_deviance,
            mae_val_m3=m3_val_stage.mae,
            mae_val_b0=b0_val_stage.mae,
            rmse_val_m3=m3_val_stage.rmse,
            rmse_val_b0=b0_val_stage.rmse,
            forecast_bootstrap_lower_bound=forecast_bootstrap_lower_bound,
        )
        forecast_signal = gate_res.forecast_gate_pass

        # 13. If forecast gate fails: short-circuit STABILITY and economics
        assert authority_snapshot is not None
        assert authority_snapshot.authority is not None
        if not forecast_signal:
            success_artifacts = build_success_artifacts(
                authority=dict(authority_snapshot.authority),
                selected_candidate_id=winner_candidate_id,
                selected_window_days=winner_window,
                forecast_signal=False,
                forecast_bootstrap_lower_bound=forecast_bootstrap_lower_bound,
                observed_mean_improvement=observed_mean_improvement,
                daily_score_rows=daily_score_rows,
                forecast_metric_rows=forecast_metric_rows,
            )
            publish_artifacts(success_artifacts, target_dir)
            return DevelopmentRunResult(
                status='COMPLETED',
                exit_status=DevelopmentExitStatus.FORECAST_GATE_FAILED,
                output_dir=target_dir,
                run_id=r_id,
                selected_candidate_id=winner_candidate_id,
                selected_window=winner_window,
                forecast_signal=False,
                economic_signal=False,
                published_artifacts=SUCCESS_ARTIFACT_NAMES,
            )

        # 14. If forecast gate passes: evaluate STABILITY + economics
        b0_stab_daily: list[DailyForecastMetrics] = []
        m3_stab_daily: list[DailyForecastMetrics] = []
        y_stability: list[np.ndarray] = []
        mu_stability: list[np.ndarray] = []

        for t in stability_indices:
            y_t = dataset.count_matrix[t]
            target_date_str = str(dataset.recorded_dates[t])

            m_b0 = compute_daily_metrics(y_t, b0_forecast)
            b0_stab_daily.append(m_b0)
            daily_score_rows.append([
                target_date_str,
                DevelopmentStage.STABILITY.value,
                ModelID.B0.value,
                CandidateID.B0_UNIFORM.value,
                m_b0.poisson_deviance,
                m_b0.mae,
                m_b0.rmse,
            ])

            counts = np.sum(dataset.count_matrix[t - winner_window : t], axis=0)
            fit_res = (model_fitter or fit_m3_model)(counts, W=winner_window)
            m_m3 = compute_daily_metrics(y_t, fit_res.mu)
            m3_stab_daily.append(m_m3)
            daily_score_rows.append([
                target_date_str,
                DevelopmentStage.STABILITY.value,
                ModelID.M3.value,
                winner_candidate_id,
                m_m3.poisson_deviance,
                m_m3.mae,
                m_m3.rmse,
            ])

            y_stability.append(y_t)
            mu_stability.append(fit_res.mu)

        b0_stab_stage = compute_stage_metrics(b0_stab_daily)
        m3_stab_stage = compute_stage_metrics(m3_stab_daily)

        forecast_metric_rows.append({
            'stage': DevelopmentStage.STABILITY.value,
            'model_id': ModelID.B0.value,
            'candidate_id': CandidateID.B0_UNIFORM.value,
            'date_count': splits.N_stability,
            'poisson_deviance': b0_stab_stage.poisson_deviance,
            'mae': b0_stab_stage.mae,
            'rmse': b0_stab_stage.rmse,
        })
        forecast_metric_rows.append({
            'stage': DevelopmentStage.STABILITY.value,
            'model_id': ModelID.M3.value,
            'candidate_id': winner_candidate_id,
            'date_count': splits.N_stability,
            'poisson_deviance': m3_stab_stage.poisson_deviance,
            'mae': m3_stab_stage.mae,
            'rmse': m3_stab_stage.rmse,
        })

        # Evaluate economics using relative 0-indexed blocks into y_stability
        stab_start_idx = splits.stability_indices[0]
        relative_stability_blocks = tuple(
            tuple(idx - stab_start_idx for idx in blk) for blk in stability_blocks
        )
        y_stab_arr = np.array(y_stability, dtype=np.float64)
        mu_stab_arr = np.array(mu_stability, dtype=np.float64)
        econ_res = (economic_evaluator or evaluate_economic_protocol)(
            y_stab_arr, mu_stab_arr, relative_stability_blocks, forecast_signal=True
        )

        blocks_info = []
        for b_idx, blk in enumerate(stability_blocks):
            start_date = str(dataset.recorded_dates[blk[0]])
            end_date = str(dataset.recorded_dates[blk[-1]])
            blk_per_k = []
            for k in ECONOMIC_K_VALUES:
                m_delta = float(econ_res.per_k[k].block_mean_economic_delta[b_idx])
                blk_per_k.append({
                    'K': k,
                    'mean_economic_delta': m_delta,
                    'positive': bool(m_delta > 0.0),
                })
            blocks_info.append({
                'block_id': b_idx,
                'start_date': start_date,
                'end_date': end_date,
                'date_count': len(blk),
                'per_k': blk_per_k,
            })

        # 15, 16, 17. Construct, validate, and publish success artifacts
        success_artifacts = build_success_artifacts(
            authority=dict(authority_snapshot.authority),
            selected_candidate_id=winner_candidate_id,
            selected_window_days=winner_window,
            forecast_signal=True,
            forecast_bootstrap_lower_bound=forecast_bootstrap_lower_bound,
            observed_mean_improvement=observed_mean_improvement,
            daily_score_rows=daily_score_rows,
            forecast_metric_rows=forecast_metric_rows,
            stability_date_count=splits.N_stability,
            blocks_info=blocks_info,
            per_k_evals=econ_res.per_k,
            economic_signal=econ_res.economic_signal,
            qualified_top_k=econ_res.qualified_top_k,
            recommended_top_k=econ_res.recommended_top_k,
        )
        publish_artifacts(success_artifacts, target_dir)

        exit_status = (
            DevelopmentExitStatus.ECONOMIC_SIGNAL_FOUND
            if econ_res.economic_signal
            else DevelopmentExitStatus.ECONOMIC_GATE_FAILED
        )
        return DevelopmentRunResult(
            status='COMPLETED',
            exit_status=exit_status,
            output_dir=target_dir,
            run_id=r_id,
            selected_candidate_id=winner_candidate_id,
            selected_window=winner_window,
            forecast_signal=True,
            economic_signal=econ_res.economic_signal,
            published_artifacts=SUCCESS_ARTIFACT_NAMES,
        )

    except (FileExistsError, HistoricalExecutionNotAuthorizedError, RunIdValidationError, PostPublicationVerificationError):
        # Do not publish failure artifact for target directory collision, safety guard, invalid run_id, or post-publication verification failure
        raise

    except Exception as exc:
        snap = authority_snapshot.protocol_snapshot if authority_snapshot is not None else None
        failure_bytes = build_failure_artifact_from_exception(exc, protocol_snapshot=snap)
        failure_bundle = {FAILURE_ARTIFACT_NAME: failure_bytes}
        publish_artifacts(failure_bundle, target_dir)

        parsed = json.loads(failure_bytes.decode('utf-8'))
        return DevelopmentRunResult(
            status='FAILED',
            exit_status=FailureExitStatus(parsed['development_exit_status']),
            output_dir=target_dir,
            run_id=r_id,
            failure_artifact=parsed,
            published_artifacts=(FAILURE_ARTIFACT_NAME,),
        )
