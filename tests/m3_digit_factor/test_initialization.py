"""Tests for XPIS v3 M3 deterministic model initialization (Slice M3-05)."""

import math
from typing import Any

import numpy as np
import pytest

from src.m3_digit_factor.contracts import FailureExitStatus, FailureStage
from src.m3_digit_factor.initialization import (
    CENTERED_VECTOR_NORM_TOLERANCE,
    SVD_LEADING_GAP_THRESHOLD,
    SVD_ZERO_TOLERANCE,
    CenteredSingularVectorDegeneracy,
    ModelInitializationError,
    ModelInitializationResult,
    SVDLeadingSubspaceAmbiguity,
    ZeroDigitMarginal,
    compute_digit_marginals,
    compute_empirical_distribution,
    compute_residual_matrix,
    initialize_m3_from_counts,
)
from src.m3_digit_factor.model import THETA_LENGTH, unpack_parameters


def test_initialization_constants() -> None:
    assert SVD_ZERO_TOLERANCE == 1e-10
    assert SVD_LEADING_GAP_THRESHOLD == 1e-8
    assert CENTERED_VECTOR_NORM_TOLERANCE == 1e-12


def test_hand_computable_fixture_empirical_q_marginals_residual_and_main_effects() -> None:
    # W = 10, total observations = 27 * 10 = 270
    row_weights = [1, 2, 3, 4, 5, 1, 2, 3, 4, 2]
    assert sum(row_weights) == 27
    W = 10
    total_T = 27 * W  # 270

    X = np.empty((10, 10), dtype=np.int64)
    for i in range(10):
        for j in range(10):
            X[i, j] = row_weights[i]
    assert int(np.sum(X)) == total_T

    # 1. Compute empirical q
    q = compute_empirical_distribution(X, W=W)
    assert q.shape == (10, 10)
    assert math.isclose(float(np.sum(q)), 1.0, abs_tol=1e-15)

    for i in range(10):
        for j in range(10):
            assert math.isclose(q[i, j], row_weights[i] / 270.0, abs_tol=1e-15)

    # 2. Compute marginals
    r, c = compute_digit_marginals(q)
    assert r.shape == (10,)
    assert c.shape == (10,)
    for i in range(10):
        expected_r = (row_weights[i] * 10) / 270.0
        assert math.isclose(r[i], expected_r, abs_tol=1e-15)
    for j in range(10):
        assert math.isclose(c[j], 0.1, abs_tol=1e-15)

    # 3. Compute residual matrix R
    R = compute_residual_matrix(q, r, c)
    assert R.shape == (10, 10)
    assert np.allclose(R, 0.0, atol=1e-15)

    # 4. In this zero-interaction case, s0 == 0 <= 1e-10, enters zero branch
    res = initialize_m3_from_counts(X, W=W)
    assert isinstance(res, ModelInitializationResult)
    assert res.gamma == 0.0

    # Check hand-computed a0 and b0
    ln_r = np.log(r)
    expected_a0 = ln_r - np.mean(ln_r)
    assert np.allclose(res.a, expected_a0, atol=1e-14)
    assert math.isclose(float(np.sum(res.a)), 0.0, abs_tol=1e-14)
    assert np.allclose(res.b, 0.0, atol=1e-14)


def test_zero_interaction_canonical_vector_independently_pinned() -> None:
    # Pin exact independent vector without importing production constant
    z = np.array([9.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64) / math.sqrt(90.0)

    # Verify mathematical properties
    assert math.isclose(float(np.sum(z)), 0.0, abs_tol=1e-15)
    assert math.isclose(float(np.sum(z**2)), 1.0, abs_tol=1e-15)

    # W = 100 -> total = 2700 counts (27 per cell)
    X_uniform = np.full((10, 10), 27, dtype=np.int64)
    res = initialize_m3_from_counts(X_uniform, W=100)

    assert res.gamma == 0.0
    assert np.allclose(res.u, z, atol=1e-15)
    assert np.allclose(res.v, z, atol=1e-15)
    assert np.allclose(res.a, 0.0, atol=1e-15)
    assert np.allclose(res.b, 0.0, atol=1e-15)


def test_input_count_accepts_both_1d_and_2d_and_does_not_mutate() -> None:
    # Construct 100 positive counts summing to 27 * 30 = 810
    base_counts = np.full(100, 8, dtype=np.int64)
    base_counts[:10] += 1  # 800 + 10 = 810
    assert int(np.sum(base_counts)) == 27 * 30

    orig_counts = base_counts.copy()

    # 1D input
    res_1d = initialize_m3_from_counts(base_counts, W=30)
    assert np.array_equal(base_counts, orig_counts)

    # 2D input
    counts_2d = base_counts.reshape((10, 10)).copy()
    orig_2d = counts_2d.copy()
    res_2d = initialize_m3_from_counts(counts_2d, W=30)
    assert np.array_equal(counts_2d, orig_2d)

    # Both must match exactly
    assert np.array_equal(res_1d.theta, res_2d.theta)


def test_input_count_sum_mismatch_rejected() -> None:
    counts = np.full(100, 8, dtype=np.int64)
    # sum is 800, but 27 * 30 = 810
    with pytest.raises((ValueError, ModelInitializationError)):
        initialize_m3_from_counts(counts, W=30)


@pytest.mark.parametrize("bad_shape", [np.zeros(99), np.zeros(101), np.zeros((5, 20)), np.zeros((10, 10, 1))])
def test_input_count_invalid_shape_rejected(bad_shape: np.ndarray) -> None:
    with pytest.raises((ValueError, ModelInitializationError)):
        initialize_m3_from_counts(bad_shape, W=10)


def test_zero_row_marginal_rejected_with_exact_failure_triplet() -> None:
    # W = 10 -> total 270.
    # 9 rows have 10 cells of 3 each = 270. Row 3 has all 0.
    counts = np.full((10, 10), 3, dtype=np.int64)
    counts[3, :] = 0
    assert int(np.sum(counts)) == 27 * 10

    with pytest.raises(ZeroDigitMarginal) as exc_info:
        initialize_m3_from_counts(counts, W=10)

    err = exc_info.value
    assert err.stage == FailureStage.MODEL_INITIALIZATION
    assert err.error_type == "ZeroDigitMarginal"
    assert err.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION


def test_zero_column_marginal_rejected_with_exact_failure_triplet() -> None:
    # W = 10 -> total 270.
    # 9 columns have 10 cells of 3 each = 270. Column 7 has all 0.
    counts = np.full((10, 10), 3, dtype=np.int64)
    counts[:, 7] = 0
    assert int(np.sum(counts)) == 27 * 10

    with pytest.raises(ZeroDigitMarginal) as exc_info:
        initialize_m3_from_counts(counts, W=10)

    err = exc_info.value
    assert err.stage == FailureStage.MODEL_INITIALIZATION
    assert err.error_type == "ZeroDigitMarginal"
    assert err.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION


def test_no_epsilon_or_pseudocount_behavior() -> None:
    # Verify that a zero marginal strictly raises ZeroDigitMarginal
    counts = np.full((10, 10), 3, dtype=np.int64)
    counts[5, :] = 0
    assert int(np.sum(counts)) == 270

    with pytest.raises(ZeroDigitMarginal):
        initialize_m3_from_counts(counts, W=10)


def test_svd_parameters_pinned_to_numpy_linalg_svd_and_full_matrices_false(monkeypatch: pytest.MonkeyPatch) -> None:
    called_args: dict[str, Any] = {}

    real_svd = np.linalg.svd

    def spy_svd(a: np.ndarray, full_matrices: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        called_args["full_matrices"] = full_matrices
        called_args["input_shape"] = a.shape
        return real_svd(a, full_matrices=full_matrices)

    monkeypatch.setattr(np.linalg, "svd", spy_svd)

    # Valid counts: W = 100, 27 counts per cell = 2700
    counts = np.full((10, 10), 27, dtype=np.int64)
    _ = initialize_m3_from_counts(counts, W=100)

    assert called_args.get("full_matrices") is False
    assert called_args.get("input_shape") == (10, 10)


def test_s0_boundary_zero_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = np.full((10, 10), 27, dtype=np.int64)

    # Case 1: s0 exactly 1e-10 -> enters zero branch
    def mock_svd_at_tol(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U = np.eye(10)
        s = np.array([1e-10, 0.5e-10] + [0.0] * 8)
        Vt = np.eye(10)
        return U, s, Vt

    monkeypatch.setattr(np.linalg, "svd", mock_svd_at_tol)
    res_at_tol = initialize_m3_from_counts(counts, W=100)
    assert res_at_tol.gamma == 0.0

    # Case 2: s0 slightly below 1e-10 -> enters zero branch
    def mock_svd_below_tol(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U = np.eye(10)
        s = np.array([0.99e-10, 0.1e-10] + [0.0] * 8)
        Vt = np.eye(10)
        return U, s, Vt

    monkeypatch.setattr(np.linalg, "svd", mock_svd_below_tol)
    res_below_tol = initialize_m3_from_counts(counts, W=100)
    assert res_below_tol.gamma == 0.0


def test_leading_subspace_ambiguity_threshold_and_exact_failure_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = np.full((10, 10), 27, dtype=np.int64)

    # Case 1: relative gap == 1e-8 exactly -> FAILS (strict inequality required: gap > 1e-8)
    # Using s0 = 1e8, s1 = 1e8 - 1.0 achieves exactly 1e-8 in float64 without roundoff error
    s0 = 1e8
    s1_equal = 1e8 - 1.0
    assert (s0 - s1_equal) / s0 == 1e-8

    def mock_svd_equal_gap(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U = np.eye(10)
        s = np.array([s0, s1_equal] + [0.0] * 8)
        Vt = np.eye(10)
        return U, s, Vt

    monkeypatch.setattr(np.linalg, "svd", mock_svd_equal_gap)
    with pytest.raises(SVDLeadingSubspaceAmbiguity) as exc_info:
        initialize_m3_from_counts(counts, W=100)
    err = exc_info.value
    assert err.stage == FailureStage.MODEL_INITIALIZATION
    assert err.error_type == "SVDLeadingSubspaceAmbiguity"
    assert err.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION

    # Case 2: relative gap < 1e-8 -> FAILS
    s1_smaller_gap = 1e8 - 0.5
    assert (s0 - s1_smaller_gap) / s0 < 1e-8

    def mock_svd_smaller_gap(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U = np.eye(10)
        s = np.array([s0, s1_smaller_gap] + [0.0] * 8)
        Vt = np.eye(10)
        return U, s, Vt

    monkeypatch.setattr(np.linalg, "svd", mock_svd_smaller_gap)
    with pytest.raises(SVDLeadingSubspaceAmbiguity):
        initialize_m3_from_counts(counts, W=100)

    # Case 3: relative gap > 1e-8 -> PASSES ambiguity check
    s1_larger_gap = 1e8 - 2.0
    assert (s0 - s1_larger_gap) / s0 > 1e-8

    # Construct U and Vt whose centered leading vectors are non-degenerate
    U_valid = np.eye(10)
    U_valid[:, 0] = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0])
    Vt_valid = np.eye(10)
    Vt_valid[0, :] = np.array([0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0])

    def mock_svd_larger_gap(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.array([s0, s1_larger_gap] + [0.0] * 8)
        return U_valid, s, Vt_valid

    monkeypatch.setattr(np.linalg, "svd", mock_svd_larger_gap)
    res = initialize_m3_from_counts(counts, W=100)
    assert res.gamma > 0.0


def test_centered_singular_vector_degeneracy_and_exact_failure_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = np.full((10, 10), 27, dtype=np.int64)
    s = np.array([1.0, 0.1] + [0.0] * 8)

    # Case 1: u_raw is constant -> centered norm nu == 0 <= 1e-12 -> FAILS
    U_deg_u = np.eye(10)
    U_deg_u[:, 0] = np.full(10, 1.0 / math.sqrt(10))  # constant vector
    Vt_valid = np.eye(10)
    Vt_valid[0, :] = np.array([0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0])

    def mock_deg_u(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return U_deg_u, s, Vt_valid

    monkeypatch.setattr(np.linalg, "svd", mock_deg_u)
    with pytest.raises(CenteredSingularVectorDegeneracy) as exc_info:
        initialize_m3_from_counts(counts, W=100)
    err = exc_info.value
    assert err.stage == FailureStage.MODEL_INITIALIZATION
    assert err.error_type == "CenteredSingularVectorDegeneracy"
    assert err.exit_status == FailureExitStatus.NEEDS_MODEL_REVISION

    # Case 2: v_raw is constant -> centered norm nv == 0 <= 1e-12 -> FAILS
    U_valid = np.eye(10)
    U_valid[:, 0] = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0])
    Vt_deg_v = np.eye(10)
    Vt_deg_v[0, :] = np.full(10, 1.0 / math.sqrt(10))

    def mock_deg_v(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return U_valid, s, Vt_deg_v

    monkeypatch.setattr(np.linalg, "svd", mock_deg_v)
    with pytest.raises(CenteredSingularVectorDegeneracy):
        initialize_m3_from_counts(counts, W=100)


def test_gamma_rescaling_formula_and_non_squared_norms(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = np.full((10, 10), 27, dtype=np.int64)
    s0 = 0.5
    s1 = 0.1

    # U leading vector:
    u_raw = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    nu_expected = math.sqrt(0.9)

    # Vt leading vector:
    v_raw = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    nv_expected = math.sqrt(0.9)

    U = np.eye(10)
    U[:, 0] = u_raw
    Vt = np.eye(10)
    Vt[0, :] = v_raw

    def mock_svd(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return U, np.array([s0, s1] + [0.0] * 8), Vt

    monkeypatch.setattr(np.linalg, "svd", mock_svd)
    res = initialize_m3_from_counts(counts, W=100)

    # Normative: gamma = s0 * nu * nv
    expected_gamma = s0 * nu_expected * nv_expected
    assert math.isclose(res.gamma, expected_gamma, abs_tol=1e-14)
    assert not math.isclose(res.gamma, s0, abs_tol=1e-5)
    assert not math.isclose(res.gamma, s0 * (nu_expected**2) * (nv_expected**2), abs_tol=1e-5)


def test_canonical_sign_rule_positive_unchanged_and_negative_flips_both_u_and_v(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = np.full((10, 10), 27, dtype=np.int64)
    s = np.array([1.0, 0.2] + [0.0] * 8)

    # Case 1: max |u| attained at index 0 with positive value -> no flip
    u_raw_pos = np.array([0.9, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
    v_raw = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0])

    U = np.eye(10)
    U[:, 0] = u_raw_pos
    Vt = np.eye(10)
    Vt[0, :] = v_raw

    def mock_svd_pos(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return U, s, Vt

    monkeypatch.setattr(np.linalg, "svd", mock_svd_pos)
    res_pos = initialize_m3_from_counts(counts, W=100)
    assert res_pos.u[0] > 0
    u_saved = res_pos.u.copy()
    v_saved = res_pos.v.copy()
    gamma_saved = res_pos.gamma

    # Case 2: negate both U and Vt in SVD output -> max |u| has negative value -> must flip both u and v back!
    U_neg = np.eye(10)
    U_neg[:, 0] = -u_raw_pos
    Vt_neg = np.eye(10)
    Vt_neg[0, :] = -v_raw

    def mock_svd_neg(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return U_neg, s, Vt_neg

    monkeypatch.setattr(np.linalg, "svd", mock_svd_neg)
    res_neg = initialize_m3_from_counts(counts, W=100)

    # Canonical sign ensures result matches res_pos!
    assert np.allclose(res_neg.u, u_saved, atol=1e-14)
    assert np.allclose(res_neg.v, v_saved, atol=1e-14)
    assert math.isclose(res_neg.gamma, gamma_saved, abs_tol=1e-14)


def test_canonical_sign_rule_tied_max_uses_smallest_index(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = np.full((10, 10), 27, dtype=np.int64)
    s = np.array([1.0, 0.2] + [0.0] * 8)

    # Tied max |u| at index 2 and index 5:
    # index 2 has -0.8
    # index 5 has +0.8
    u_raw = np.zeros(10)
    u_raw[2] = -0.8
    u_raw[5] = 0.8
    v_raw = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0])

    U = np.eye(10)
    U[:, 0] = u_raw
    Vt = np.eye(10)
    Vt[0, :] = v_raw

    def mock_svd_tie(a: np.ndarray, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return U, s, Vt

    monkeypatch.setattr(np.linalg, "svd", mock_svd_tie)
    res = initialize_m3_from_counts(counts, W=100)

    assert res.u[2] > 0
    assert res.u[5] < 0


def test_non_mocked_real_svd_successful_initialization() -> None:
    # Deterministic synthetic count matrix with rank-1 interaction and non-ambiguous gap
    # W = 60, total = 27 * 60 = 1620
    W = 60
    counts = np.full((10, 10), 10, dtype=np.int64)
    counts[0, 0] += 100
    counts[0, 1] += 50
    counts[1, 0] += 50
    counts[1, 1] += 25
    diff = (27 * W) - int(np.sum(counts))
    counts[2:, 2:] += diff // 64
    rem = (27 * W) - int(np.sum(counts))
    counts[2, 2] += rem
    assert int(np.sum(counts)) == 27 * W

    res = initialize_m3_from_counts(counts, W=W)

    assert isinstance(res, ModelInitializationResult)
    assert res.gamma > 0.0
    assert math.isclose(float(np.sum(res.a)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(np.sum(res.b)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(np.sum(res.u)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(np.sum(res.v)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(np.sum(res.u**2)), 1.0, abs_tol=1e-12)
    assert math.isclose(float(np.sum(res.v**2)), 1.0, abs_tol=1e-12)

    # Check theta packing matches unpack_parameters
    assert len(res.theta) == THETA_LENGTH
    a_unpacked, b_unpacked, u_unpacked, v_unpacked, gamma_unpacked = unpack_parameters(res.theta)
    assert np.array_equal(res.a, a_unpacked)
    assert np.array_equal(res.b, b_unpacked)
    assert np.array_equal(res.u, u_unpacked)
    assert np.array_equal(res.v, v_unpacked)
    assert res.gamma == gamma_unpacked


def test_initialization_is_strictly_deterministic() -> None:
    W = 30
    counts = np.full((10, 10), 8, dtype=np.int64)
    counts[0, :] += 1  # 800 + 10 = 810 = 27 * 30
    assert int(np.sum(counts)) == 27 * W

    res1 = initialize_m3_from_counts(counts, W=W)
    res2 = initialize_m3_from_counts(counts, W=W)

    assert np.array_equal(res1.theta, res2.theta)
    assert res1.gamma == res2.gamma
