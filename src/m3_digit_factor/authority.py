"""Closed authority snapshot construction for XPIS v3 M3."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from types import MappingProxyType


EXPECTED_REPOSITORY = 'Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis'
CANONICAL_DATA_PATH = 'data/xsmb-2-digits.csv'
CANONICAL_SPEC_SHA256 = '49396218fcd5387bb4682f9b1d83169adc9a8194c410694806c7d2a63cf3c977'
CANONICAL_SPEC_RELATIVE_PATH = 'docs/superpowers/specs/2026-09-03-xpis-v3-m3-digit-factor-canonical-reconstruction.md'


class AuthorityValidationError(ValueError):
    """Raised when an M3 authority snapshot cannot be validated exactly."""


FROZEN_LITERALS: Mapping[str, object] = MappingProxyType({
    'spec_version': 'XPIS_V3_M3_DIGIT_FACTOR_DESIGN_V3',
    'authority_provenance': 'RECONSTRUCTED_FROM_SURVIVING_APPROVED_ARTIFACTS_REPOSITORY_EVIDENCE_AND_CONTROL_PLANE_ADJUDICATION',
    'data_source_repository': EXPECTED_REPOSITORY,
    'data_source_path': CANONICAL_DATA_PATH,
    'artifact_contract_version': 'XPIS_V3_M3_ARTIFACT_CONTRACT_V1',
    'model_contract_version': 'XPIS_V3_M3_MODEL_CP1',
    'data_split_contract_version': 'XPIS_V3_M3_DATA_SPLIT_CP1',
    'candidate_contract_version': 'XPIS_V3_M3_CANDIDATE_CP1',
    'forecast_gate_contract_version': 'XPIS_V3_M3_FORECAST_GATE_CP1',
    'forecast_bootstrap_contract_version': 'XPIS_V3_M3_FORECAST_BOOTSTRAP_CP1',
    'economic_contract_version': 'XPIS_V3_M3_ECONOMIC_CP1',
    'success_artifact_schema_version': 'XPIS_V3_M3_SUCCESS_ARTIFACTS_CP1',
    'failure_artifact_schema_version': 'XPIS_V3_M3_FAILURE_ARTIFACT_CP1',
    'metric_definition_version': 'XPIS_V3_METRICS_V1',
    'poisson_log_base': 'NATURAL',
    'poisson_outcome_aggregation': 'MEAN_OVER_100_OUTCOMES',
    'poisson_zero_count_convention': 'Y_EQ_0_TERM_EQUALS_2_MU',
    'mae_outcome_aggregation': 'MEAN_OVER_100_OUTCOMES',
    'rmse_daily_aggregation': 'SQRT_MEAN_OVER_100_SQUARED_ERRORS',
    'stage_poisson_aggregation': 'ARITHMETIC_MEAN_DAILY',
    'stage_mae_aggregation': 'ARITHMETIC_MEAN_DAILY',
    'stage_rmse_aggregation': 'SQRT_ARITHMETIC_MEAN_DAILY_RMSE_SQUARED',
    'block_metric_aggregation': 'SAME_RULES_AS_STAGE',
    'forecast_sum_tolerance': 1e-10,
    'artifact_numeric_abs_tolerance': 1e-12,
    'artifact_numeric_rel_tolerance': 1e-12,
    'csv_float_format': '.17g',
    'baseline_id': 'B0_UNIFORM',
    'candidate_windows': (30, 60, 120, 240, 365),
    'minimum_eligible_targets': 240,
    'split_fractions': (0.5, 0.25, 0.25),
    'stability_block_count': 6,
    'svd_implementation': 'numpy.linalg.svd',
    'svd_full_matrices': False,
    'interaction_zero_tolerance': 1e-10,
    'svd_gap_tolerance': 1e-8,
    'svd_gap_definition': '(s0-s1)/s0',
    'svd_ambiguous_leading_subspace_policy': 'FAIL_NEEDS_MODEL_REVISION',
    'optimizer_implementation': 'scipy.optimize.minimize',
    'optimizer_method': 'SLSQP',
    'optimizer_max_iterations': 2000,
    'optimizer_ftol': 1e-12,
    'optimizer_constraint_tolerance': 1e-10,
    'forecast_bootstrap_rng_implementation': 'numpy.random.Generator',
    'forecast_bootstrap_bit_generator': 'PCG64',
    'forecast_bootstrap_seed': 20260831,
    'forecast_bootstrap_replications': 2000,
    'forecast_bootstrap_mean_block_length': 30,
    'forecast_bootstrap_restart_probability': 0.03333333333333333,
    'forecast_bootstrap_alpha': 0.05,
    'forecast_bootstrap_quantile_method': 'linear',
    'economic_k_values': (1, 3, 5, 10),
    'economic_cost_per_number_thousand_vnd': 27.0,
    'economic_payout_per_hit_thousand_vnd': 99.0,
    'economic_bootstrap_rng_implementation': 'numpy.random.Generator',
    'economic_bootstrap_bit_generator': 'PCG64',
    'economic_bootstrap_seed': 20260832,
    'economic_bootstrap_replications': 2000,
    'economic_bootstrap_mean_block_length': 30,
    'economic_bootstrap_restart_probability': 0.03333333333333333,
    'economic_bootstrap_quantile_method': 'linear',
    'economic_bootstrap_shared_resample_indices': True,
    'economic_familywise_alpha': 0.05,
    'economic_multiplicity_method': 'BONFERRONI',
    'economic_per_k_alpha': 0.0125,
    'economic_uncertainty_allowed_statuses': ('NOT_EVALUATED_FORECAST_GATE_FAILED', 'EVALUATED'),
})

AUTHORITY_KEYS = frozenset({
    *FROZEN_LITERALS,
    'canonical_spec_sha256',
    'source_commit',
    'source_tree',
    'data_source_commit',
    'data_source_blob',
    'data_source_sha256',
})

_HEX_40 = re.compile(r'[0-9a-f]{40}\Z')
_HEX_64 = re.compile(r'[0-9a-f]{64}\Z')


@dataclass(frozen=True)
class RepositoryIdentity:
    """Run-bound repository and canonical-data object identities."""

    repository: str
    data_path: str
    source_commit: str
    source_tree: str
    data_commit: str
    data_blob: str
    data_sha256: str


def _require_hex(value: object, *, field: str, size: int) -> None:
    matcher = _HEX_40 if size == 40 else _HEX_64
    if not isinstance(value, str) or matcher.fullmatch(value) is None:
        raise AuthorityValidationError(f'{field} must be a {size}-character lowercase hex value')


def _require_finite_float(value: object, *, field: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise AuthorityValidationError(f'{field} must be a finite float')


def _validate_literal(field: str, value: object, expected: object) -> None:
    if isinstance(expected, bool):
        if type(value) is not bool or value is not expected:
            raise AuthorityValidationError(f'{field} must equal its frozen boolean literal')
    elif isinstance(expected, int):
        if type(value) is not int or value != expected:
            raise AuthorityValidationError(f'{field} must equal its frozen integer literal')
    elif isinstance(expected, float):
        _require_finite_float(value, field=field)
        if value != expected:
            raise AuthorityValidationError(f'{field} must equal its frozen float literal')
    elif isinstance(expected, str):
        if type(value) is not str or value != expected:
            raise AuthorityValidationError(f'{field} must equal its frozen string literal')
    elif isinstance(expected, tuple):
        if type(value) is not list or value != list(expected):
            raise AuthorityValidationError(f'{field} must equal its frozen ordered array literal')
        for item, sample in zip(value, expected, strict=True):
            if type(item) is not type(sample):
                raise AuthorityValidationError(f'{field} contains an invalid element type')
    else:
        raise AssertionError(f'Unsupported frozen literal type for {field}')


def validate_authority(authority: Mapping[str, object]) -> None:
    """Validate the complete closed authority object without fallback values."""
    keys = set(authority)
    missing = AUTHORITY_KEYS - keys
    extra = keys - AUTHORITY_KEYS
    if missing:
        raise AuthorityValidationError(f'authority has missing keys: {sorted(missing)!r}')
    if extra:
        raise AuthorityValidationError(f'authority has extra keys: {sorted(extra)!r}')
    if len(authority) != 72:
        raise AuthorityValidationError('authority must contain exactly 72 unique keys')

    _require_hex(authority['canonical_spec_sha256'], field='canonical_spec_sha256', size=64)
    if authority['canonical_spec_sha256'] != CANONICAL_SPEC_SHA256:
        raise AuthorityValidationError('canonical_spec_sha256 does not match the locked canonical specification')
    for field in ('source_commit', 'source_tree', 'data_source_commit', 'data_source_blob'):
        _require_hex(authority[field], field=field, size=40)
    _require_hex(authority['data_source_sha256'], field='data_source_sha256', size=64)
    for field, expected in FROZEN_LITERALS.items():
        _validate_literal(field, authority[field], expected)


def canonical_authority_json_bytes(authority: Mapping[str, object]) -> bytes:
    """Return the frozen compact UTF-8 JSON input to the protocol fingerprint."""
    try:
        payload = json.dumps(authority, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AuthorityValidationError('authority cannot be encoded as canonical JSON') from error
    return payload.encode('utf-8')


def protocol_fingerprint(authority: Mapping[str, object]) -> str:
    """SHA-256 of only the canonical compact authority JSON bytes."""
    return hashlib.sha256(canonical_authority_json_bytes(authority)).hexdigest()


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class AuthoritySnapshot:
    """Immutable pre/post-``AUTHORITY_COMPLETE`` protocol-snapshot representation."""

    authority: Mapping[str, object] | None = None
    protocol_fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.authority is None and self.protocol_fingerprint_sha256 is None:
            return
        if self.authority is None or self.protocol_fingerprint_sha256 is None:
            raise AuthorityValidationError('authority and fingerprint must be present together')
        if not isinstance(self.authority, Mapping):
            raise AuthorityValidationError('authority must be a mapping')
        materialized = dict(self.authority)
        validate_authority(materialized)
        expected_fingerprint = protocol_fingerprint(materialized)
        if self.protocol_fingerprint_sha256 != expected_fingerprint:
            raise AuthorityValidationError('protocol fingerprint does not match canonical authority JSON')
        object.__setattr__(self, 'authority', MappingProxyType({key: _freeze_value(value) for key, value in materialized.items()}))

    @property
    def authority_complete(self) -> bool:
        return self.authority is not None and self.protocol_fingerprint_sha256 is not None

    @property
    def protocol_snapshot(self) -> dict[str, object] | None:
        if not self.authority_complete:
            return None
        assert self.authority is not None
        assert self.protocol_fingerprint_sha256 is not None
        return {
            'authority': _json_value(self.authority),
            'protocol_fingerprint_sha256': self.protocol_fingerprint_sha256,
        }


def build_authority(identity: RepositoryIdentity) -> dict[str, object]:
    """Construct the exact authority object from frozen literals and verified identities."""
    if identity.repository != EXPECTED_REPOSITORY:
        raise AuthorityValidationError('repository identity does not match frozen authority')
    if identity.data_path != CANONICAL_DATA_PATH:
        raise AuthorityValidationError('canonical data path does not match frozen authority')
    authority = {
        **{field: _json_value(value) for field, value in FROZEN_LITERALS.items()},
        'canonical_spec_sha256': CANONICAL_SPEC_SHA256,
        'source_commit': identity.source_commit,
        'source_tree': identity.source_tree,
        'data_source_commit': identity.data_commit,
        'data_source_blob': identity.data_blob,
        'data_source_sha256': identity.data_sha256,
    }
    validate_authority(authority)
    return authority


def _run_git(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ['git', *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AuthorityValidationError(f'Git identity verification failed: {" ".join(args)}')
    return result.stdout.strip()


def _read_git_blob(repository_root: Path, object_spec: str) -> bytes:
    result = subprocess.run(
        ['git', 'cat-file', 'blob', object_spec],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AuthorityValidationError(f'Git blob identity verification failed: {object_spec}')
    return result.stdout


def canonical_spec_sha256_from_git(
    repository_root: Path,
    *,
    blob_reader: Callable[[str], bytes] | None = None,
) -> str:
    """Hash the canonical specification's raw Git blob bytes, not checkout text."""
    root = Path(repository_root)
    reader = blob_reader if blob_reader is not None else lambda spec: _read_git_blob(root, spec)
    raw_bytes = reader(f'HEAD:{CANONICAL_SPEC_RELATIVE_PATH}')
    if not isinstance(raw_bytes, bytes):
        raise AuthorityValidationError('canonical specification Git blob reader must return bytes')
    return hashlib.sha256(raw_bytes).hexdigest()


def _repository_from_remote(remote_url: str) -> str:
    match = re.search(r'github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?\Z', remote_url)
    if match is None:
        raise AuthorityValidationError('repository remote is not a canonical GitHub repository identity')
    return match.group(1)


def collect_repository_identity(
    repository_root: Path,
    *,
    git_runner: Callable[..., str] | None = None,
) -> RepositoryIdentity:
    """Verify the canonical repository and data-path Git identities without parsing draws."""
    root = Path(repository_root)
    runner = git_runner if git_runner is not None else lambda *args: _run_git(root, *args)
    remote = runner('config', '--get', 'remote.origin.url')
    repository = _repository_from_remote(remote)
    if repository != EXPECTED_REPOSITORY:
        raise AuthorityValidationError('repository identity does not match frozen authority')

    data_file = root / CANONICAL_DATA_PATH
    if not data_file.is_file():
        raise AuthorityValidationError('canonical data path is absent')
    tracked_path = runner('ls-files', '--error-unmatch', CANONICAL_DATA_PATH)
    if tracked_path != CANONICAL_DATA_PATH:
        raise AuthorityValidationError('canonical data path is not tracked exactly')

    source_commit = runner('rev-parse', 'HEAD')
    source_tree = runner('rev-parse', 'HEAD^{tree}')
    data_blob = runner('rev-parse', f'HEAD:{CANONICAL_DATA_PATH}')
    working_blob = runner('hash-object', CANONICAL_DATA_PATH)
    if data_blob != working_blob:
        raise AuthorityValidationError('canonical data path does not match its tracked Git object')

    identity = RepositoryIdentity(
        repository=repository,
        data_path=CANONICAL_DATA_PATH,
        source_commit=source_commit,
        source_tree=source_tree,
        data_commit=source_commit,
        data_blob=data_blob,
        data_sha256=hashlib.sha256(data_file.read_bytes()).hexdigest(),
    )
    for field in ('source_commit', 'source_tree', 'data_commit', 'data_blob'):
        _require_hex(getattr(identity, field), field=field, size=40)
    _require_hex(identity.data_sha256, field='data_sha256', size=64)
    return identity


def construct_authority_snapshot(
    repository_root: Path,
    *,
    git_runner: Callable[..., str] | None = None,
) -> AuthoritySnapshot:
    """Build a complete snapshot after validating the canonical spec and Git identities."""
    root = Path(repository_root)
    if canonical_spec_sha256_from_git(root) != CANONICAL_SPEC_SHA256:
        raise AuthorityValidationError('canonical specification identity does not match frozen authority')
    authority = build_authority(collect_repository_identity(root, git_runner=git_runner))
    return AuthoritySnapshot(authority, protocol_fingerprint(authority))
