import hashlib
import json
import math
from pathlib import Path

import pytest

from src.m3_digit_factor.authority import (
    AuthoritySnapshot,
    AuthorityValidationError,
    FROZEN_LITERALS,
    RepositoryIdentity,
    build_authority,
    canonical_spec_sha256_from_git,
    canonical_authority_json_bytes,
    collect_repository_identity,
    protocol_fingerprint,
)


EXPECTED_AUTHORITY = {
    'spec_version': 'XPIS_V3_M3_DIGIT_FACTOR_DESIGN_V3',
    'authority_provenance': 'RECONSTRUCTED_FROM_SURVIVING_APPROVED_ARTIFACTS_REPOSITORY_EVIDENCE_AND_CONTROL_PLANE_ADJUDICATION',
    'canonical_spec_sha256': '49396218fcd5387bb4682f9b1d83169adc9a8194c410694806c7d2a63cf3c977',
    'source_commit': 'b' * 40,
    'source_tree': 'c' * 40,
    'data_source_repository': 'Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis',
    'data_source_path': 'data/xsmb-2-digits.csv',
    'data_source_commit': 'd' * 40,
    'data_source_blob': 'e' * 40,
    'data_source_sha256': 'f' * 64,
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
    'candidate_windows': [30, 60, 120, 240, 365],
    'minimum_eligible_targets': 240,
    'split_fractions': [0.5, 0.25, 0.25],
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
    'economic_k_values': [1, 3, 5, 10],
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
    'economic_uncertainty_allowed_statuses': ['NOT_EVALUATED_FORECAST_GATE_FAILED', 'EVALUATED'],
}

GOLDEN_FINGERPRINT = '7a8cf89bb8fce581d5eac8562ac7795a671c9a9458f221b09fd41bffa4719b98'


def make_identity(**overrides: object) -> RepositoryIdentity:
    values = {
        'repository': 'Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis',
        'data_path': 'data/xsmb-2-digits.csv',
        'source_commit': 'b' * 40,
        'source_tree': 'c' * 40,
        'data_commit': 'd' * 40,
        'data_blob': 'e' * 40,
        'data_sha256': 'f' * 64,
    }
    values.update(overrides)
    return RepositoryIdentity(**values)


def valid_authority() -> dict[str, object]:
    return dict(EXPECTED_AUTHORITY)


def test_authority_key_fixture_is_independent_complete_and_unique():
    assert len(EXPECTED_AUTHORITY) == 72
    assert len(set(EXPECTED_AUTHORITY)) == 72


def test_builder_preserves_the_exact_locked_authority_literals():
    assert build_authority(make_identity()) == EXPECTED_AUTHORITY


def test_frozen_literals_cannot_be_mutated_to_change_built_authority():
    with pytest.raises(TypeError):
        FROZEN_LITERALS['candidate_windows'] = (999,)  # type: ignore[index]
    assert FROZEN_LITERALS['candidate_windows'] == (30, 60, 120, 240, 365)
    assert build_authority(make_identity())['candidate_windows'] == [30, 60, 120, 240, 365]


def test_missing_and_extra_authority_keys_are_rejected():
    missing = valid_authority()
    del missing['optimizer_method']
    with pytest.raises(AuthorityValidationError, match='missing'):
        AuthoritySnapshot(missing, protocol_fingerprint(missing))

    extra = valid_authority()
    extra['unapproved_scientific_override'] = 'no'
    with pytest.raises(AuthorityValidationError, match='extra'):
        AuthoritySnapshot(extra, protocol_fingerprint(extra))


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('minimum_eligible_targets', '240'),
        ('minimum_eligible_targets', True),
        ('forecast_sum_tolerance', math.inf),
        ('optimizer_method', None),
        ('candidate_windows', [30, 60, 120, 365, 240]),
        ('economic_uncertainty_allowed_statuses', ['EVALUATED', 'NOT_EVALUATED_FORECAST_GATE_FAILED']),
    ],
)
def test_closed_type_and_domain_validation_rejects_invalid_authority_values(key: str, value: object):
    authority = valid_authority()
    authority[key] = value
    with pytest.raises(AuthorityValidationError):
        AuthoritySnapshot(authority, protocol_fingerprint(authority))


def test_repository_and_canonical_data_substitution_fail_closed():
    with pytest.raises(AuthorityValidationError, match='repository'):
        build_authority(make_identity(repository='other/project'))
    with pytest.raises(AuthorityValidationError, match='data path'):
        build_authority(make_identity(data_path='data/substitute.csv'))


def test_collect_repository_identity_collects_source_and_tree_and_data_identity(tmp_path: Path):
    data_path = tmp_path / 'data' / 'xsmb-2-digits.csv'
    data_path.parent.mkdir()
    data_path.write_bytes(b'raw-csv-identity')

    git_values = {
        ('config', '--get', 'remote.origin.url'): 'git@github.com:Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis.git',
        ('rev-parse', 'HEAD'): '1' * 40,
        ('rev-parse', 'HEAD^{tree}'): '2' * 40,
        ('ls-files', '--error-unmatch', 'data/xsmb-2-digits.csv'): 'data/xsmb-2-digits.csv',
        ('rev-parse', 'HEAD:data/xsmb-2-digits.csv'): '3' * 40,
        ('hash-object', 'data/xsmb-2-digits.csv'): '3' * 40,
    }

    identity = collect_repository_identity(tmp_path, git_runner=lambda *args: git_values[args])

    assert identity.source_commit == '1' * 40
    assert identity.source_tree == '2' * 40
    assert identity.data_commit == '1' * 40
    assert identity.data_blob == '3' * 40
    assert identity.data_sha256 == hashlib.sha256(b'raw-csv-identity').hexdigest()


def test_canonical_spec_hash_uses_raw_git_blob_bytes_not_checkout_text(tmp_path: Path):
    raw_blob = b'line-one\nline-two\n'

    assert canonical_spec_sha256_from_git(tmp_path, blob_reader=lambda spec: raw_blob) == hashlib.sha256(raw_blob).hexdigest()


def test_snapshot_schema_lifecycle_and_immutability_are_closed():
    incomplete = AuthoritySnapshot()
    assert incomplete.authority_complete is False
    assert incomplete.protocol_snapshot is None

    authority = valid_authority()
    snapshot = AuthoritySnapshot(authority, protocol_fingerprint(authority))
    assert snapshot.authority_complete is True
    assert snapshot.protocol_snapshot == {
        'authority': EXPECTED_AUTHORITY,
        'protocol_fingerprint_sha256': protocol_fingerprint(EXPECTED_AUTHORITY),
    }
    with pytest.raises(TypeError):
        snapshot.authority['optimizer_method'] = 'OTHER'  # type: ignore[index]


def test_fingerprint_contract_is_compact_sorted_utf8_and_excludes_wrapper():
    authority = valid_authority()
    compact = canonical_authority_json_bytes(authority)
    expected = json.dumps(authority, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

    assert compact == expected
    assert compact.endswith(b'\n') is False
    assert compact.index(b'"artifact_contract_version"') < compact.index(b'"spec_version"')
    assert protocol_fingerprint(authority) == hashlib.sha256(compact).hexdigest()
    assert protocol_fingerprint(authority) != hashlib.sha256(
        json.dumps({'authority': authority}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    ).hexdigest()


def test_fingerprint_is_deterministic_sensitive_and_matches_fixed_golden_value():
    authority = valid_authority()
    assert protocol_fingerprint(authority) == GOLDEN_FINGERPRINT
    changed = valid_authority()
    changed['source_commit'] = 'a' * 40
    assert protocol_fingerprint(changed) != protocol_fingerprint(authority)
