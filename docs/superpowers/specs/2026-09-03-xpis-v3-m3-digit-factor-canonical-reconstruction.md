# XPIS v3 — M3 Digit-Factor Design Specification V3 — Canonical Reconstruction

```text
SPEC_VERSION = XPIS_V3_M3_DIGIT_FACTOR_DESIGN_V3
SPEC_STATUS = AUTHORITY_RECONSTRUCTION_CANDIDATE
AUTHORITY_PROVENANCE = RECONSTRUCTED_FROM_SURVIVING_APPROVED_ARTIFACTS_REPOSITORY_EVIDENCE_AND_CONTROL_PLANE_ADJUDICATION

HISTORICAL_M3_V2_SOURCE_ARTIFACT = UNAVAILABLE
HISTORICAL_M3_V2_RECOVERY_STATUS = EXHAUSTED_UNAVAILABLE
SCIENTIFIC_V3_CLOSURE_DECISIONS = PRESERVED
SURVIVING_SCIENTIFIC_CONTENT_REVIEW = PASS_UNCHANGED
NEW_CONTROL_PLANE_AUTHORITY_SCIENTIFIC_REVIEW = REQUIRED
TARGETED_SCIENTIFIC_REVIEW = PASS
SCIENTIFIC_REOPENED = NO
SCIENTIFIC_SEMANTICS_CHANGED = NO

CONTROL_PLANE_RECONSTRUCTION_ADJUDICATION = COMPLETE
CONTROL_PLANE_FINAL_LOCK_CORRECTION = COMPLETE
INPUT_PROTOCOL_CRITICAL_UNRESOLVED_COUNT = 22
RESOLVED_COUNT = 22
REMAINING_PROTOCOL_CRITICAL_UNRESOLVED_COUNT = 0
HIDDEN_IMPLEMENTATION_DECISION_COUNT = 0
SURVIVING_AUTHORITY_SEMANTIC_DRIFT = NONE

M3_AUTHORITY_RECONSTRUCTION = COMPLETE_ADJUDICATED
FORMAL_ARTIFACT_REVIEW = READY_FOR_FINAL_INDEPENDENT_RECHECK
EVIDENCE_IDENTITY_CHANGED = NO
LOCATOR_REPRESENTATION_REPAIRED = YES

FORMAL_SPEC_LOCK = NOT_AUTHORIZED
REPO_GROUNDED_PLANNING = NOT_AUTHORIZED
SOURCE_IMPLEMENTATION = NOT_AUTHORIZED
V3_HISTORICAL_RUN = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
HISTORICAL_EXECUTION = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
```

### 16.1 Final-lock fidelity audit

```text
FINAL_LOCK_FIDELITY_REPAIR = COMPLETE
FR-01 = PASS
FR-02 = PASS
FR-03 = PASS
FR-04 = PASS
FR-05 = PASS
FR-06 = PASS
FR-07 = PASS
FR-08 = PASS

CONTROL_PLANE_CP1_VALUE_MISMATCH_COUNT = 0
SURVIVING_FINGERPRINT_VALUE_MISMATCH_COUNT = 0
CONTROL_PLANE_AUTHORITY_LITERAL_MISMATCH_COUNT = 0
SURVIVING_AUTHORITY_LITERAL_MISMATCH_COUNT = 0
HIDDEN_IMPLEMENTATION_DECISION_COUNT = 0
SUCCESS_ARTIFACT_SCHEMA_SELF_CONTAINMENT = PASS
FAILURE_CONTRACT_SELF_CONTAINMENT = PASS
BOOTSTRAP_RNG_CONSUMPTION_ORDER_FULLY_SPECIFIED = YES
ALL_SCIENTIFIC_QUALIFICATION = UNROUNDED_IN_MEMORY_BINARY64_VALUES
SURVIVING_EVIDENCE_IDENTITY_REPRODUCIBLE = PASS
```

---

## 1. Status, purpose, and authority boundary

This document is the canonical V3 specification for Method 3 (Digit-Factor Interaction Model). It reconstructs the complete formal contract of M3 from surviving approved artifacts, repository evidence, and explicit Control Plane adjudication.

The historical M3 V2 source artifact was lost and is permanently unavailable. All 22 protocol-critical gaps identified during reconstruction have been formally adjudicated by the Control Plane, and 10 review blockers (B-01 through B-10) have been resolved by explicit Control Plane review corrections.

No scientific reopening has occurred. Surviving scientific content passes unchanged. All hidden implementation decisions have been eliminated. Formal spec lock remains NOT_AUTHORIZED pending independent recheck.

---

## 2. Repository and evidence baseline

### 2.1 Evidence identities

| Evidence ID | Repository Path / Locator | Commit / State | Git Blob Hash | SHA-256 Digest | Role / Description |
|---|---|---|---|---|---|
| `E-001` | `docs/superpowers/specs/2026-08-31-xpis-v3-closure-evidence.md` | `HISTORICAL_UNRESOLVED_COMMIT_LOCATOR: 9d5e3ef` | `9353a8f4940de4257f6c96ee5e309ff827adfe11` | `ead5f0d83957a5a43aed96941b6c8d28fb9ed34275d274f3109c539249e54286` | Surviving V3 closure evidence; authoritative source for metric mathematics, SVD core, protocol snapshot structure, and economic uncertainty |
| `E-002` | `docs/superpowers/specs/2026-08-30-xpis-v3-methodology-and-core-contracts.md` | `HISTORICAL_UNRESOLVED_COMMIT_LOCATOR: 9d5e3ef` | `9353a8f4940de4257f6c96ee5e309ff827adfe11` | `ead5f0d83957a5a43aed96941b6c8d28fb9ed34275d274f3109c539249e54286` | Content-identical duplicate file; evidence-only; establishes non-existence of distinct V2 content in this file |
| `E-003` | Work-Order Prompt `XPIS v3 M3 Reconstruction Work-Order` | `NOT_REPOSITORY_ARTIFACT` | `NOT_COMMITTED` | `7ecf23599a5b4650b7bd8e933e5a4a802ffaf554039aa4e981b9df3fd1276126` | Procedural work-order establishing scope, allowed evidence, and reconstruction classification boundaries |
| `E-004` | Historical locator: `docs/superpowers/specs/2026-08-31-xpis-v3-count-first-design-v2.md` at `11a7cbf` | `HISTORICAL_UNAVAILABLE_REFERENCE` | `HISTORICAL_REFERENCE_ONLY: 1e021a8d0092c730fc9ecf8e0258169ff5c9aaee` | `HISTORICAL_REFERENCE_ONLY: 7e169da464b584a20b080ef201d4a0be7b3c299cff33534d07b46ca52a514dcf` | Background only. Claimed commit-path tuple is unavailable; no current verified repository identity or M3 normative authority. |
| `E-005` | Historical locator: `docs/superpowers/plans/2026-08-31-xpis-v3-count-first-v2.md` at `11a7cbf` | `HISTORICAL_UNAVAILABLE_REFERENCE` | `HISTORICAL_REFERENCE_ONLY: 66874ebdb4979e2a74c2e68ce0db50b71d60927b` | `HISTORICAL_REFERENCE_ONLY: 37803dcf34988775f0535c8e312953258c7075c32729a6575191b72186dd4892` | Background only. Claimed commit-path tuple is unavailable; no current verified repository identity or M3 normative authority. |
| `E-006` | Historical branch locator: `codex/feat/xpis-v3-count-first-v2` | `HISTORICAL_UNAVAILABLE_REFERENCE` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | Background only. The claimed branch/ref is unavailable; no current verified repository identity or M3 normative authority. |
| `E-007` | Commit `11a7cbf` | `11a7cbf` | `11a7cbf349a8b8f4435b3c67c3f27aea78b9d5b4` | `NOT_APPLICABLE` | Base HEAD of current branch; repository anchor |
| `E-008` | Remote tracking ref `origin/main` | `11a7cbf` | `11a7cbf349a8b8f4435b3c67c3f27aea78b9d5b4` | `NOT_APPLICABLE` | Remote synchronization state; confirms zero unmerged M3 commits |
| `E-009` | `docs/superpowers/specs/2026-09-04-xpis-v3-m3-control-plane-reconstruction-adjudication.md` | `UNCOMMITTED_FINAL_BYTES` | `7e9c55caea96be92eec64f0e30a1d39af53c5070` | `235d4c530634ff4d60810f77388738a74d92d5551adeed7d240de86b9f19afa9` | Control Plane 22-decision reconstruction adjudication memorandum, final-lock review addendum (B-01..B-10), FR-07 serialization addendum, U-020 literal authority addendum (PROV-01/PROV-02), and U-018 row-order addendum; identity independently recomputed from current memorandum bytes |

---

### 2.2 Current evidence-resolution status

```text
E-004
EVIDENCE_STATUS = HISTORICAL_UNAVAILABLE_REFERENCE
CURRENT_RESOLUTION = UNAVAILABLE
CURRENT_GIT_OBJECT_VERIFIED = NO
NORMATIVE_AUTHORITY_DEPENDENCY = NO

E-005
EVIDENCE_STATUS = HISTORICAL_UNAVAILABLE_REFERENCE
CURRENT_RESOLUTION = UNAVAILABLE
CURRENT_GIT_OBJECT_VERIFIED = NO
NORMATIVE_AUTHORITY_DEPENDENCY = NO

E-006
EVIDENCE_STATUS = HISTORICAL_UNAVAILABLE_REFERENCE
CURRENT_RESOLUTION = UNAVAILABLE
CURRENT_GIT_OBJECT_VERIFIED = NO
NORMATIVE_AUTHORITY_DEPENDENCY = NO

NORMATIVE_AUTHORITY_INPUTS =
surviving Closure blob/content;
persisted Control Plane memorandum;
current canonical artifact;
explicit Control Plane decisions

NORMATIVE_DEPENDENCY_ON_UNAVAILABLE_BACKGROUND_EVIDENCE = NONE
```

The loose historical blob identifiers for E-004 and E-005 may resolve as Git objects, but their claimed `11a7cbf:path` locators do not. They are therefore not current reproducible repository evidence.

---

## 3. Evidence hierarchy and contract classification

The formal reconstruction follows a strict four-tier evidence hierarchy:
1. **Tier 1 (Preserved Scientific Authority)**: Unambiguous rules preserved from `E-001` (`P-001` through `P-038`).
2. **Tier 2 (Control Plane Adjudication Authority)**: Decisions resolving protocol-critical gaps (`U-001` through `U-020`, `U-023`, `U-024`), review corrections (`B-01` through `B-10`), FR-07 serialization addendum, U-020 literal authority addendum (`PROV-01`, `PROV-02`), and U-018 row-order addendum from `E-009`.
3. **Tier 3 (Reconstructed Non-Normative Consistency)**: Reconciliation records (`R-001`, `R-002`).
4. **Tier 4 (Deferred Lifecycle / Implementation)**: Run-bound and release-environment concerns (`U-021`, `U-022`).

---

## 4. Canonical authority schema state

### 4.1 `protocol_snapshot.json` top-level contract

`protocol_snapshot.json` is a closed object containing exactly two top-level key names. Their listed order is descriptive only; serialized JSON object-member order is lexicographic under `sort_keys = true`.
```json
{
  "authority": { ... },
  "protocol_fingerprint_sha256": "<64-character lowercase hex string>"
}
```

### 4.2 Complete closed `authority` schema

`protocol_snapshot.authority` is a closed object containing exactly the 72 keys defined below. Every key participates in the protocol fingerprint bytes.

For `spec_version`, schema membership and typing are governed by Control Plane reconstruction decision `U-020`, while the exact literal value `"XPIS_V3_M3_DIGIT_FACTOR_DESIGN_V3"` is preserved from surviving Closure evidence (`E-001:5`).

For `spec_version`, schema membership and typing are governed by Control Plane reconstruction decision `U-020`, while the exact literal value `"XPIS_V3_M3_DIGIT_FACTOR_DESIGN_V3"` is preserved from surviving Closure evidence (`E-001:5`).

| Field | Type / Encoding | Value or Canonical Rule | Status |
|---|---|---|---|
| `spec_version` | string | `"XPIS_V3_M3_DIGIT_FACTOR_DESIGN_V3"` | `PRESERVED (literal value from surviving closure) / CONTROL_PLANE_RECONSTRUCTION_DECISION (schema membership via U-020)` |
| `authority_provenance` | string | `"RECONSTRUCTED_FROM_SURVIVING_APPROVED_ARTIFACTS_REPOSITORY_EVIDENCE_AND_CONTROL_PLANE_ADJUDICATION"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `canonical_spec_sha256` | 64-char lowercase hex | SHA-256 of final accepted canonical specification bytes; populated at snapshot construction | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `source_commit` | 40-char lowercase hex | Mandatory run-bound source identity; frozen at implementation/release gate | `DEFERRED_RUN_BOUND_VALUE` |
| `source_tree` | 40-char lowercase hex | Mandatory run-bound source-tree identity; frozen at implementation/release gate | `DEFERRED_RUN_BOUND_VALUE` |
| `data_source_repository` | string | `"Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `data_source_path` | string | `"data/xsmb-2-digits.csv"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `data_source_commit` | 40-char lowercase hex | Run-bound commit of data source at execution | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `data_source_blob` | 40-char lowercase hex | Run-bound blob of data source at execution | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `data_source_sha256` | 64-char lowercase hex | SHA-256 of raw data source file at execution | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `artifact_contract_version` | string | `"XPIS_V3_M3_ARTIFACT_CONTRACT_V1"` | `PRESERVED` |
| `model_contract_version` | string | `"XPIS_V3_M3_MODEL_CP1"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `data_split_contract_version` | string | `"XPIS_V3_M3_DATA_SPLIT_CP1"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `candidate_contract_version` | string | `"XPIS_V3_M3_CANDIDATE_CP1"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `forecast_gate_contract_version` | string | `"XPIS_V3_M3_FORECAST_GATE_CP1"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `forecast_bootstrap_contract_version` | string | `"XPIS_V3_M3_FORECAST_BOOTSTRAP_CP1"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `economic_contract_version` | string | `"XPIS_V3_M3_ECONOMIC_CP1"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `success_artifact_schema_version` | string | `"XPIS_V3_M3_SUCCESS_ARTIFACTS_CP1"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `failure_artifact_schema_version` | string | `"XPIS_V3_M3_FAILURE_ARTIFACT_CP1"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `metric_definition_version` | string | `"XPIS_V3_METRICS_V1"` | `PRESERVED` |
| `poisson_log_base` | string | `"NATURAL"` | `PRESERVED` |
| `poisson_outcome_aggregation` | string | `"MEAN_OVER_100_OUTCOMES"` | `PRESERVED` |
| `poisson_zero_count_convention` | string | `"Y_EQ_0_TERM_EQUALS_2_MU"` | `PRESERVED` |
| `mae_outcome_aggregation` | string | `"MEAN_OVER_100_OUTCOMES"` | `PRESERVED` |
| `rmse_daily_aggregation` | string | `"SQRT_MEAN_OVER_100_SQUARED_ERRORS"` | `PRESERVED` |
| `stage_poisson_aggregation` | string | `"ARITHMETIC_MEAN_DAILY"` | `PRESERVED` |
| `stage_mae_aggregation` | string | `"ARITHMETIC_MEAN_DAILY"` | `PRESERVED` |
| `stage_rmse_aggregation` | string | `"SQRT_ARITHMETIC_MEAN_DAILY_RMSE_SQUARED"` | `PRESERVED` |
| `block_metric_aggregation` | string | `"SAME_RULES_AS_STAGE"` | `PRESERVED` |
| `forecast_sum_tolerance` | number | `1e-10` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `artifact_numeric_abs_tolerance` | number | `1e-12` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `artifact_numeric_rel_tolerance` | number | `1e-12` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `csv_float_format` | string | `".17g"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `baseline_id` | string | `"B0_UNIFORM"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `candidate_windows` | array of integer | `[30,60,120,240,365]` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `minimum_eligible_targets` | integer | `240` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `split_fractions` | array of number | `[0.5,0.25,0.25]` (nominal design fractions; actual integer counts defined by floor formulas have normative precedence) | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `stability_block_count` | integer | `6` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `svd_implementation` | string | `"numpy.linalg.svd"` | `PRESERVED` |
| `svd_full_matrices` | boolean | `false` | `PRESERVED` |
| `interaction_zero_tolerance` | number | `1e-10` | `PRESERVED` |
| `svd_gap_tolerance` | number | `1e-8` | `PRESERVED` |
| `svd_gap_definition` | string | `"(s0-s1)/s0"` | `PRESERVED` |
| `svd_ambiguous_leading_subspace_policy` | string | `"FAIL_NEEDS_MODEL_REVISION"` | `PRESERVED` |
| `optimizer_implementation` | string | `"scipy.optimize.minimize"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `optimizer_method` | string | `"SLSQP"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `optimizer_max_iterations` | integer | `2000` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `optimizer_ftol` | number | `1e-12` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `optimizer_constraint_tolerance` | number | `1e-10` (normatively: POST_SOLVER_MAX_ABS_EQUALITY_RESIDUAL_TOLERANCE; independent post-solver acceptance gate, not an SLSQP solver option) | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `forecast_bootstrap_rng_implementation` | string | `"numpy.random.Generator"` | `PRESERVED` |
| `forecast_bootstrap_bit_generator` | string | `"PCG64"` | `PRESERVED` |
| `forecast_bootstrap_seed` | integer | `20260831` | `PRESERVED` |
| `forecast_bootstrap_replications` | integer | `2000` | `CONTROL_PLANE_RECONSTRUCTION_DECISION (U-014)` |
| `forecast_bootstrap_mean_block_length` | integer | `30` | `CONTROL_PLANE_RECONSTRUCTION_DECISION (U-014)` |
| `forecast_bootstrap_restart_probability` | number | `0.03333333333333333` | `CONTROL_PLANE_RECONSTRUCTION_DECISION (U-014)` |
| `forecast_bootstrap_alpha` | number | `0.05` | `CONTROL_PLANE_RECONSTRUCTION_DECISION (U-014)` |
| `forecast_bootstrap_quantile_method` | string | `"linear"` | `CONTROL_PLANE_RECONSTRUCTION_DECISION (U-014)` |
| `economic_k_values` | array of integer | `[1,3,5,10]` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `economic_cost_per_number_thousand_vnd` | number | `27.0` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `economic_payout_per_hit_thousand_vnd` | number | `99.0` | `CONTROL_PLANE_RECONSTRUCTION_DECISION` |
| `economic_bootstrap_rng_implementation` | string | `"numpy.random.Generator"` | `PRESERVED` |
| `economic_bootstrap_bit_generator` | string | `"PCG64"` | `PRESERVED` |
| `economic_bootstrap_seed` | integer | `20260832` | `PRESERVED` |
| `economic_bootstrap_replications` | integer | `2000` | `PRESERVED` |
| `economic_bootstrap_mean_block_length` | integer | `30` | `PRESERVED` |
| `economic_bootstrap_restart_probability` | number | `0.03333333333333333` | `PRESERVED` |
| `economic_bootstrap_quantile_method` | string | `"linear"` | `PRESERVED` |
| `economic_bootstrap_shared_resample_indices` | boolean | `true` | `PRESERVED` |
| `economic_familywise_alpha` | number | `0.05` | `PRESERVED` |
| `economic_multiplicity_method` | string | `"BONFERRONI"` | `PRESERVED` |
| `economic_per_k_alpha` | number | `0.0125` | `PRESERVED` |
| `economic_uncertainty_allowed_statuses` | array of string | `["NOT_EVALUATED_FORECAST_GATE_FAILED","EVALUATED"]` | `PRESERVED` |

### 4.3 Closed-world schema guarantees

1. The key count of `protocol_snapshot.authority` is exactly 72.
2. No key outside this list may appear in `protocol_snapshot.authority`.
3. No key in this list may be omitted, null, empty string, or NaN.
4. All 72 keys participate in the canonical protocol fingerprint.

---

## 5. Exact metric authority

### 5.1 Daily Poisson deviance

For outcome `n in 0..99`, observed count `y >= 0`, and forecast `mu > 0`:

```text
PD_cell(y, mu) =
    2 * [y * ln(y / mu) - (y - mu)]  when y > 0
    2 * mu                             when y = 0
```
The daily Poisson deviance is:
$$PD = \frac{1}{100} \sum_{n=0}^{99} PD_{cell}(y_n, \mu_n)$$

### 5.2 Daily MAE

$$MAE = \frac{1}{100} \sum_{n=0}^{99} |y_n - \mu_n|$$

### 5.3 Daily RMSE

$$RMSE = \sqrt{\frac{1}{100} \sum_{n=0}^{99} (y_n - \mu_n)^2}$$

### 5.4 Stage aggregation

For stage with $T$ target dates:
- Stage Poisson Deviance: $PD_{stage} = \frac{1}{T} \sum_{t=1}^T PD_t$
- Stage MAE: $MAE_{stage} = \frac{1}{T} \sum_{t=1}^T MAE_t$
- Stage RMSE: $RMSE_{stage} = \sqrt{\frac{1}{T} \sum_{t=1}^T RMSE_t^2}$

### 5.5 Block metric aggregation

`BLOCK_METRIC_AGGREGATION = SAME_RULES_AS_STAGE`. Each of the 6 stability blocks computes block metrics using the exact stage aggregation formulas applied to its assigned dates.

### 5.6 Development-validation comparisons

On validation stage (VAL):
- Primary comparison: $PD_{VAL, M3} < PD_{VAL, B0}$
- Secondary comparison: $MAE_{VAL, M3} \le MAE_{VAL, B0} \lor RMSE_{VAL, M3} \le RMSE_{VAL, B0}$

### 5.7 Metric artifact consistency

Reconstructed and serialized metrics must compare with semantics exactly equivalent to:

```text
math.isclose(
    reconstructed,
    serialized,
    rel_tol=1e-12,
    abs_tol=1e-12,
)
```

`ARTIFACT_NUMERIC_ABS_TOLERANCE = 1e-12` and `ARTIFACT_NUMERIC_REL_TOLERANCE = 1e-12`. No additive tolerance rule with different semantics is permitted.

### 5.8 Metric authority constants

```text
metric_definition_version = "XPIS_V3_METRICS_V1"
poisson_log_base = "NATURAL"
poisson_outcome_aggregation = "MEAN_OVER_100_OUTCOMES"
poisson_zero_count_convention = "Y_EQ_0_TERM_EQUALS_2_MU"
mae_outcome_aggregation = "MEAN_OVER_100_OUTCOMES"
rmse_daily_aggregation = "SQRT_MEAN_OVER_100_SQUARED_ERRORS"
stage_poisson_aggregation = "ARITHMETIC_MEAN_DAILY"
stage_mae_aggregation = "ARITHMETIC_MEAN_DAILY"
stage_rmse_aggregation = "SQRT_ARITHMETIC_MEAN_DAILY_RMSE_SQUARED"
block_metric_aggregation = "SAME_RULES_AS_STAGE"
forecast_sum_tolerance = 1e-10
artifact_numeric_abs_tolerance = 1e-12
artifact_numeric_rel_tolerance = 1e-12
csv_float_format = ".17g"
ALL_SCIENTIFIC_QUALIFICATION = UNROUNDED_IN_MEMORY_BINARY64_VALUES
```

Serialized CSV and JSON values are reconstruction/reporting evidence only. Metric stage and block aggregation, every gate, and every qualification use unrounded in-memory binary64 daily values; no rounded CSV or JSON string may be used for qualification.

---

## 6. Deterministic SVD authority

### 6.1 Double-centered interaction matrix R construction

For candidate history window $W \in \{30, 60, 120, 240, 365\}$:
1. $X_{t,W}[i,j]$ is the empirical count of 2-digit outcome $(10 \cdot i + j)$ over the $W$ recorded draw dates strictly preceding target date $t$.
2. Total observations: $T = \sum_i \sum_j X[i,j] = 27 \cdot W$.
3. Empirical joint probabilities: $q[i,j] = X[i,j] / (27 \cdot W)$.
4. Digit marginals: $r[i] = \sum_{j=0}^9 q[i,j]$ for head digit $i \in \{0..9\}$, and $c[j] = \sum_{i=0}^9 q[i,j]$ for tail digit $j \in \{0..9\}$.
5. Strict marginal check: $r[i] > 0$ for all $i$, $c[j] > 0$ for all $j$. If any $r[i] = 0$ or $c[j] = 0$, fail immediately with `FAILED_STAGE=MODEL_INITIALIZATION`, `ERROR_TYPE=ZeroDigitMarginal`, `DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION`.
6. Double-centered matrix:
   $$R[i,j] = q[i,j] - r[i] \cdot c[j]$$
   Row and column sums of $R$ are structurally zero: $\sum_j R[i,j] = 0$, $\sum_i R[i,j] = 0$.
7. SVD execution: `U_svd, S_svd, Vt_svd = numpy.linalg.svd(R, full_matrices=False)`. Let $s_0 = S_{svd}[0]$ and $s_1 = S_{svd}[1]$.

### 6.2 Zero-interaction case

If `s0 <= 1e-10`:
- $\gamma_0 = 0.0$
- Canonical zero-interaction vectors:
  $$u_0 = v_0 = \frac{1}{\sqrt{90}} (9, -1, -1, -1, -1, -1, -1, -1, -1, -1)^T$$
- Initial main effects: $a_0[i] = \ln(r[i]) - \frac{1}{10} \sum_k \ln(r[k])$, $b_0[j] = \ln(c[j]) - \frac{1}{10} \sum_k \ln(c[k])$.

### 6.3 Leading-gap test and fail-closed result

If `s0 > 1e-10`:
- Test relative leading gap:
  `(s0 - s1) / s0 > 1e-8`
- If relative gap `<= 1e-8`: fail immediately with `FAILED_STAGE=MODEL_INITIALIZATION`, `ERROR_TYPE=SVDLeadingSubspaceAmbiguity`, `DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION`. No arbitrary basis selection, perturbation, retry, or fallback solver permitted.

### 6.4 Centering, normalization, initial main effects, and sign canonicalization

When `s0 > 1e-10` and relative gap `> 1e-8`:
1. Raw singular vectors: $u_{raw} = U_{svd}[:, 0]$, $v_{raw} = V^T_{svd}[0, :]$, $\gamma_{raw} = s_0$.
2. Center: $u_c = u_{raw} - \text{mean}(u_{raw})$, $v_c = v_{raw} - \text{mean}(v_{raw})$.
3. Norms: `nu = ||u_c||_2`, `nv = ||v_c||_2`.
4. Strict non-degeneracy check: `nu > 1e-12` and `nv > 1e-12`. Otherwise fail with `FAILED_STAGE=MODEL_INITIALIZATION`, `ERROR_TYPE=CenteredSingularVectorDegeneracy`, `DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION`.
5. Normalize: `u = u_c / nu`, `v = v_c / nv`, `gamma = gamma_raw * nu * nv`.
6. Canonical sign rule: let $i^* = \min \{ i \in \{0..9\} : |u[i]| = \max_k |u[k]| \}$. If $u[i^*] < 0$, negate both: $u \leftarrow -u$, $v \leftarrow -v$.
7. Initial main effects: $a_0[i] = \ln(r[i]) - \frac{1}{10} \sum_k \ln(r[k])$, $b_0[j] = \ln(c[j]) - \frac{1}{10} \sum_k \ln(c[k])$.

### 6.5 SVD authority constants

```text
svd_implementation = "numpy.linalg.svd"
svd_full_matrices = false
interaction_zero_tolerance = 1e-10
svd_gap_tolerance = 1e-8
svd_gap_definition = "(s0-s1)/s0"
svd_ambiguous_leading_subspace_policy = "FAIL_NEEDS_MODEL_REVISION"
```

---

## 7. Rank-1 Digit Interaction Multinomial Loglinear Model & Fit Objective

### 7.1 Model equation and latent score

For each target date $t$, let outcome $n \in \{0..99\}$ have head digit $i = \lfloor n / 10\rfloor$ and tail digit $j = n \bmod 10$ ($i, j \in \{0..9\}$).
The model family is `RANK1_DIGIT_INTERACTION_MULTINOMIAL_LOGLINEAR`.
The latent loglinear score is:
$$\eta_t[i,j] = a_t[i] + b_t[j] + \gamma_t \cdot u_t[i] \cdot v_t[j]$$
Softmax probabilities with mandatory numerical stabilization:
$$p_t[i,j] = \frac{\exp(\eta_t[i,j] - \max_{r,s} \eta_t[r,s])}{\sum_{r=0}^9 \sum_{s=0}^9 \exp(\eta_t[r,s] - \max_{r,s} \eta_t[r,s])}$$
Expected outcome count forecast:
$$\mu_t[10 \cdot i + j] = 27 \cdot p_t[i,j]$$
Structurally $\mu_t[n] > 0$ and $\sum_{n=0}^{99} \mu_t[n] = 27.0$ within tolerance `1e-10`. No post-scoring clipping, epsilon injection, or negative repair permitted.

### 7.2 Parameter domains, constraints, and identifiability

Parameters:
- Head main effects: $a \in \mathbb{R}^{10}$
- Tail main effects: $b \in \mathbb{R}^{10}$
- Head interaction factor: $u \in \mathbb{R}^{10}$
- Tail interaction factor: $v \in \mathbb{R}^{10}$
- Interaction scale: $\gamma \ge 0$

Exact equality constraints:
$$\sum_{i=0}^9 a[i] = 0, \quad \sum_{j=0}^9 b[j] = 0, \quad \sum_{i=0}^9 u[i] = 0, \quad \sum_{j=0}^9 v[j] = 0$$
$$\sum_{i=0}^9 u[i]^2 - 1.0 = 0, \quad \sum_{j=0}^9 v[j]^2 - 1.0 = 0$$

### 7.3 Exact SLSQP numerical fit contract (B-01)

1. **Optimization Parameter Vector**:
   Exact variable order:
   $$\theta = [a[0], \dots, a[9], b[0], \dots, b[9], u[0], \dots, u[9], v[0], \dots, v[9], \gamma]^T \in \mathbb{R}^{41}$$
   Length: exactly 41.
   Bounds:
   - $a[0..9]$: unbounded $(-\infty, +\infty)$
   - $b[0..9]$: unbounded $(-\infty, +\infty)$
   - $u[0..9]$: unbounded $(-\infty, +\infty)$
   - $v[0..9]$: unbounded $(-\infty, +\infty)$
   - $\gamma$: $[0, +\infty)$ using native SLSQP parameter bounds.

2. **Equality Residual Vector**:
   Exact 6-dimensional residual vector:
   ```text
   c(theta) = [
       sum_i a[i],
       sum_j b[j],
       sum_i u[i],
       sum_j v[j],
       sum_i u[i]^2 - 1.0,
       sum_j v[j]^2 - 1.0,
   ] = [0, 0, 0, 0, 0, 0]
   ```
   Norm constraints are strictly $\sum u[i]^2 - 1 = 0$ and $\sum v[j]^2 - 1 = 0$ (NOT $\|u\|_2 - 1$).

3. **Negative Log-Likelihood Objective**:
   $$L(\theta) = - \sum_{i=0}^9 \sum_{j=0}^9 X[i,j] \cdot \ln(p[i,j])$$
   Let $T = \sum_i \sum_j X[i,j] = 27 \cdot W$ and $G[i,j] = T \cdot p[i,j] - X[i,j]$.
   The objective Jacobian is strictly analytic:
   $$\frac{\partial L}{\partial a[i]} = \sum_{j=0}^9 G[i,j]$$
   $$\frac{\partial L}{\partial b[j]} = \sum_{i=0}^9 G[i,j]$$
   $$\frac{\partial L}{\partial u[i]} = \gamma \sum_{j=0}^9 G[i,j] \cdot v[j]$$
   $$\frac{\partial L}{\partial v[j]} = \gamma \sum_{i=0}^9 G[i,j] \cdot u[i]$$
   $$\frac{\partial L}{\partial \gamma} = \sum_{i=0}^9 \sum_{j=0}^9 G[i,j] \cdot u[i] \cdot v[j]$$

4. **Exact Analytic Constraint Jacobian**:
   Exact $6 \times 41$ matrix:
   - Row 0 ($\sum a = 0$): $1.0$ on coordinates $0..9$; $0.0$ elsewhere.
   - Row 1 ($\sum b = 0$): $1.0$ on coordinates $10..19$; $0.0$ elsewhere.
   - Row 2 ($\sum u = 0$): $1.0$ on coordinates $20..29$; $0.0$ elsewhere.
   - Row 3 ($\sum v = 0$): $1.0$ on coordinates $30..39$; $0.0$ elsewhere.
   - Row 4 ($\sum u^2 - 1 = 0$): $2.0 \cdot u[i]$ on coordinates $20..29$; $0.0$ elsewhere.
   - Row 5 ($\sum v^2 - 1 = 0$): $2.0 \cdot v[j]$ on coordinates $30..39$; $0.0$ elsewhere.

   ```text
   OBJECTIVE_JACOBIAN = ANALYTIC
   CONSTRAINT_JACOBIAN = ANALYTIC
   FINITE_DIFFERENCE_SCHEME = NOT_APPLICABLE
   FINITE_DIFFERENCE_STEP = NOT_APPLICABLE
   ```

5. **SciPy Call Semantics**:
   Normatively equivalent to:
   ```python
   scipy.optimize.minimize(
       fun=objective,
       x0=initial_theta,
       jac=objective_jacobian,
       method="SLSQP",
       bounds=bounds,
       constraints=[
           {
               "type": "eq",
               "fun": constraint_residuals,
               "jac": constraint_jacobian,
           }
       ],
       options={
           "maxiter": 2000,
           "ftol": 1e-12,
           "disp": False,
       },
   )
   ```
   Do NOT pass generic `tol` to `scipy.optimize.minimize`.
   The authority field `optimizer_constraint_tolerance = 1e-10` normatively represents:
   ```text
   POST_SOLVER_MAX_ABS_EQUALITY_RESIDUAL_TOLERANCE = 1e-10
   ```
   which is an independent post-solver acceptance gate, NOT an SLSQP solver option.

### 7.4 Exact post-solver canonicalization and forecast validation order (B-02)

Every fitted solution must execute the following 4 steps strictly in order:

1. **Step 1 — Raw Solver Validity**:
   Before ANY canonicalization, require:
   - `optimizer.success == true`
   - All raw parameters in $\theta_{raw}$ are finite (no NaN, no Inf)
   - Raw objective $L(\theta_{raw})$ is finite
   - `gamma_raw >= 0.0`
   If `gamma_raw < 0.0`: fail-closed with `FAILED_STAGE=MODEL_FIT`, `ERROR_TYPE=ConstraintViolation`, `DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION`. Zero-interaction canonicalization must never hide a negative raw gamma.

2. **Step 2 — Equality Feasibility**:
   Compute exact equality residual vector $c(\theta_{raw})$.
   Require:
   `max(k=0..5, abs(c_k(theta_raw))) <= 1e-10`
   Otherwise fail-closed with `FAILED_STAGE=MODEL_FIT`, `ERROR_TYPE=ConstraintViolation`, `DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION`.

3. **Step 3 — Interaction Canonicalization**:
   - If `0.0 <= gamma_raw <= 1e-10`:
     Set $\gamma = 0.0$, $u = v = \frac{1}{\sqrt{90}} (9, -1, -1, -1, -1, -1, -1, -1, -1, -1)^T$.
     Fitted main effects $a = a_{raw}$ and $b = b_{raw}$ remain unchanged.
   - Else (`gamma_raw > 1e-10`):
     Set $\gamma = \gamma_{raw}$, $u = u_{raw}$, $v = v_{raw}$.
     Apply surviving canonical sign rule: let $i^* = \min \{ i : |u[i]| = \max_k |u[k]| \}$. If $u[i^*] < 0$, negate both $u \leftarrow -u$ and $v \leftarrow -v$.
     Do NOT re-center or re-normalize the accepted non-zero fitted solution after solver acceptance.

4. **Step 4 — Final Forecast Validation**:
   Generate forecasts $\eta, p, \mu$ strictly from the canonicalized representation:
   - All $\eta[i,j]$, $p[i,j]$, $\mu[n]$ are finite
   - $\mu[n] > 0.0$ strictly for all $n \in \{0..99\}$
   - `abs(sum(n=0..99, mu[n]) - 27.0) <= 1e-10`
    Any numerical underflow producing $\mu[n] = 0.0$ is fail-closed: `ERROR_TYPE=ForecastContractViolation`.
    This final forecast is the sole authoritative forecast.

### 7.5 U-005 fit initialization and fit-failure classification

Every target-date/window fit is independent. For every fit:

```text
FIT_INITIALIZATION =
EXACT_U004_INITIALIZATION
```

No prior target fit may initialize any other fit. The following are normative prohibitions: random restart, alternate solver, fallback optimizer, warm-start chain, and tolerance relaxation.

A fit is accepted only if `optimizer.success = true`, the objective is finite, all raw model parameters are finite, raw `gamma >= 0`, all equality constraints satisfy the post-solver feasibility tolerance, and the final canonicalized forecast satisfies the forecast contract.

For every failure governed by U-005:

```text
FAILED_STAGE = MODEL_FIT
DEVELOPMENT_EXIT_STATUS = NEEDS_MODEL_REVISION
```

The stable `error_type` is selected without collapsing these cases:

- `OptimizerNonConvergence`: `optimizer.success != true`, unless a more-specific technical failure already applies before a solver result exists.
- `NonFiniteModelFit`: the raw objective or any raw fitted parameter is non-finite.
- `ConstraintViolation`: raw `gamma < 0`, `max(abs(c(theta_raw))) > 1e-10`, or another explicitly frozen fit constraint is violated.
- `ForecastContractViolation`: the accepted/canonicalized representation fails final forecast validation, including non-finite `eta`, `p`, or `mu`; `mu[n] <= 0`; or `abs(sum(mu)-27) > 1e-10`.

U005_FAILURE_MAPPING takes precedence for MODEL_FIT failures. U-019 supplies only the general failure-artifact envelope and may not broaden the U-005 exit status.

---

## 8. Dataset, Chronology, and Stability Partition

### 8.1 Dataset source and integrity (B-03)

1. Primary dataset path and repository identity:
   ```text
   DATA_SOURCE_PATH = data/xsmb-2-digits.csv
   DATA_SOURCE_REPOSITORY = Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis
   ```
   This is explicit Control Plane M3 authority under U-006 and U-020 (PROV-02). Repository-grounded planning may verify the selected authority, but may not silently replace either identity. If repository-grounded planning establishes that the repository or canonical dataset authority no longer matches the frozen reconstruction authority, stop and report back to the Control Plane:
   ```text
   STOP = REPOSITORY_AUTHORITY_MISMATCH
   ```
   ```text
   SILENT_REPOSITORY_SUBSTITUTION = FORBIDDEN
   SILENT_DATA_SOURCE_SUBSTITUTION = FORBIDDEN
   ```
   No other CSV, loader path, newer-looking file, inferred replacement, alternate repository slug, or change to the canonical source may be selected silently.
2. Format: CSV with header `date` followed by 27 prize columns containing 2-digit integer outcomes $0..99$.
3. Chronology: strictly increasing unique draw dates.
4. **Observed Draw Event Indexing**:
   M3 is strictly `OBSERVED_DRAW_EVENT_INDEXED`, not calendar-completeness inferred.
   `history_window_W` counts $W$ recorded draw dates / observations, NOT calendar days.
5. **No Missing Draw Semantics**:
   `NO_MISSING_DRAW` within M3 means:
   - Every PRESENT dataset row/date must contain exactly 27 valid integer outcomes in $\{0..99\}$.
   - No value may be null, absent, malformed, imputed, clipped, or silently repaired.
   Absence of a calendar date does NOT by itself constitute an M3 missing-draw violation.
6. **Scope Demarcation**:
   ```text
   CALENDAR_CONTINUITY_VALIDATION = OUT_OF_SCOPE_FOR_M3_MODEL_PROTOCOL
   OFFICIAL_DRAW_CALENDAR_COMPLETENESS = UPSTREAM_DATA_GOVERNANCE_CONCERN
   ```
   M3 operates exclusively on the frozen canonical recorded-event sequence. It must never synthesize or impute rows for absent calendar dates.
7. **Legacy Field Mapping**:
   `selected_window_days` in artifacts is a legacy field name semantically identical to:
   ```text
   COUNT_OF_PRECEDING_RECORDED_DRAW_DATES
   ```
8. Target date eligibility: a recorded target date $t$ is eligible if and only if it is preceded by at least $W_{max} = 365$ recorded draw dates.

### 8.2 Chronology and development split (B-04)

Let $N$ be the total count of eligible target dates sorted chronologically ascending.
Required: $N \ge 240$; otherwise fail with `FAILED_STAGE=DATA_VALIDATION`, `ERROR_TYPE=InsufficientDevelopmentSample`, `DEVELOPMENT_EXIT_STATUS=NEEDS_DATA_REVISION`.

Contiguous chronological partition:
- $N_{dev} = \lfloor N / 2\rfloor$
- $N_{val} = \lfloor N / 4\rfloor$
- $N_{stability} = N - N_{dev} - N_{val}$

```text
split_fractions = [0.5, 0.25, 0.25] (NOMINAL_DESIGN_FRACTIONS)
```
The nominal design fractions do not assert exact realized finite-sample proportions. The integer count formulas above have absolute normative precedence.

Partition cohorts:
- DEV: first $N_{dev}$ eligible target dates.
- VAL: next $N_{val}$ eligible target dates.
- STABILITY: final $N_{stability}$ eligible target dates.

### 8.3 Stability cohort and N_stability

For $N \ge 240$, $N_{stability} \ge 60$ strictly.
When economics is evaluated, `economic_uncertainty.per_k[*].date_count = N_stability` exactly.

### 8.4 Six stability blocks

`STABILITY_BLOCK_COUNT = 6`.
Let $q = \lfloor N_{stability} / 6\rfloor$ and $r = N_{stability} \bmod 6$.
- Blocks $1..r$ each contain $q + 1$ contiguous dates.
- Blocks $(r+1)..6$ each contain $q$ contiguous dates.
Every block contains at least 10 target dates.

---

## 9. Baseline, Candidate Family, and Forecast Gate

### 9.1 Baseline B0

Baseline ID: `B0_UNIFORM`.
For all target dates $t$ and numbers $n \in \{0..99\}$:
$$\mu_{B0}[t, n] = \frac{27}{100} = 0.27$$
No parameters, history window, or optimization.

### 9.2 Candidate family

Candidate hyperparameter space: trailing history window $W \in \{30, 60, 120, 240, 365\}$.
Candidate IDs: `M3_W030`, `M3_W060`, `M3_W120`, `M3_W240`, `M3_W365`.

### 9.3 Candidate selection and tie-breaking

Evaluated on DEV stage only. Select the single candidate minimizing the lexicographic tuple:
$$(PD_{DEV}, MAE_{DEV}, RMSE_{DEV}, W)$$
Zero tolerance. Strict lexicographic order.

### 9.4 Complete FORECAST_SIGNAL gate

Evaluated on VAL stage for the selected M3 candidate:
1. `PRIMARY_FORECAST_PASS`: $PD_{VAL, M3} < PD_{VAL, B0}$
2. `DEV_VAL_SECONDARY_PASS`: $MAE_{VAL, M3} \le MAE_{VAL, B0} \lor RMSE_{VAL, M3} \le RMSE_{VAL, B0}$
3. `FORECAST_BOOTSTRAP_PASS`: `forecast_bootstrap_lower_bound > 0.0`

`FORECAST_SIGNAL = true` if and only if ALL THREE conditions pass.

### 9.5 Forecast bootstrap

The VAL daily difference is `d_t = PD_t_B0 - PD_t_M3`. Instantiate this RNG exactly once before replicate 1 and do not reset it between replicates:

```text
rng =
numpy.random.Generator(
    numpy.random.PCG64(20260831)
)
```

For each of the 2000 replicates, generate one index vector of length `N_val` in this exact RNG-consumption order:

```text
idx[0] =
rng.integers(0, N_val)

for m = 1,...,N_val-1:
    u = rng.random()

    if u < 1/30:
        idx[m] =
        rng.integers(0, N_val)
    else:
        idx[m] =
        (idx[m-1] + 1) mod N_val
```

No `rng.integers()` draw occurs on a continuation step. For replicate `b`, `D_b = mean(d_t over idx)`. The lower bound is exactly:

```text
forecast_bootstrap_lower_bound =
numpy.quantile(
    [D_1,...,D_2000],
    0.05,
    method="linear"
)
```

Frozen constants are `FORECAST_BOOTSTRAP_REPLICATIONS = 2000`, `FORECAST_BOOTSTRAP_MEAN_BLOCK_LENGTH = 30`, `FORECAST_BOOTSTRAP_RESTART_PROBABILITY = 1/30`, `FORECAST_BOOTSTRAP_ALPHA = 0.05`, and `FORECAST_BOOTSTRAP_QUANTILE_METHOD = "linear"`.

---

## 10. Economic Protocol, Top-K, and Stationary Bootstrap

### 10.1 Economic construction and top-K

Evaluated on STABILITY stage if and only if `FORECAST_SIGNAL = true`.
- Portfolio sizes: $K \in \{1, 3, 5, 10\}$.
- Daily ranking: rank numbers $0..99$ by descending $\mu_t[n]$; break ties by smaller number index $n$. Select top $K$.
- Economics:
  - Cost per number bet: $27.0$ thousand VND
  - Payout per hit: $99.0$ thousand VND
  - Net PnL per outcome: $PnL[t, n] = 99.0 \cdot y[t, n] - 27.0$
  - Daily economic delta:
    ```text
    economic_delta[t,K] =
    Σ over selected K numbers of
    (99*y[t,n] - 27)
    ```
    Units remain thousand VND. `economic_delta` represents net daily PnL, not a B0-relative delta. Multiplicity is preserved.

#### Aggregation formulas across STABILITY and blocks (U-015):
```text
mean_economic_delta[K] =
(1 / N_stability)
* Σ_{t in STABILITY} economic_delta[t,K]

block_mean_economic_delta[b,K] =
(1 / |B_b|)
* Σ_{t in B_b} economic_delta[t,K]

positive_block_count[K] =
Σ_{b=1..6}
I(block_mean_economic_delta[b,K] > 0)
```

#### Aggregation Invariant:
```text
mean_economic_delta[K]
is the arithmetic mean across ALL STABILITY DATES.

It is NOT the unweighted arithmetic mean
of the six block means.
```

- Strict positive block rule: `block_mean_economic_delta[b,K] > 0.0` strictly (indicator function evaluates to 1 if strictly greater than 0, else 0).
- Artifact fields named `mean_economic_delta` and `positive_block_count` across all artifacts (including `economic_uncertainty.json` and `economic_summary.csv`) are directly linked to these exact definitions.

#### Economic qualification:
```text
mean_economic_delta[K] > 0

AND

positive_block_count[K] >= 5

AND

bootstrap_lower_bound[K] > 0
```

### 10.2 Recommendation eligibility domain and selection rule (B-09)

1. **Ranking Domain**:
   Ranking applies strictly to `qualified_top_k`. Non-qualifying K values are never considered for recommendation.

2. **Selection Rule**:
   - If `qualified_top_k == []`:
     `recommended_top_k = []`.
   - Else:
     $$\text{chosen\_K} = \arg\max_{K \in \text{qualified\_top\_k}} (\text{bootstrap\_lower\_bound}_K, \text{mean\_economic\_delta}_K, -K)$$
     `recommended_top_k = [chosen_K]`.

3. **Invariants**:
   - `recommended_top_k subset_of qualified_top_k`
   - `len(recommended_top_k) <= 1`
   - In `economic_summary.csv`, `recommended = true` if and only if its $K$ is the sole element of `recommended_top_k`.

### 10.3 Economic stationary bootstrap

Instantiate this RNG exactly once and do not reset it between K values or replicates:

```text
rng =
numpy.random.Generator(
    numpy.random.PCG64(20260832)
)
```

For each of the 2000 replicates, generate exactly one index vector of length `N_stability` using this exact sequence:

```text
idx[0] =
rng.integers(0, N_stability)

for m = 1,...,N_stability-1:
    u = rng.random()

    if u < 1/30:
        idx[m] =
        rng.integers(0, N_stability)
    else:
        idx[m] =
        (idx[m-1] + 1) mod N_stability
```

No `rng.integers()` draw occurs on a continuation step. That same one index vector is applied to every `K` in ordered `K = 1, 3, 5, 10`; this is the exact meaning of `shared_resample_indices = true`. For each K, `M_b[K] = mean(economic_delta[t,K] over idx)`, followed by:

```text
bootstrap_lower_bound[K] =
numpy.quantile(
    [M_1[K],...,M_2000[K]],
    0.0125,
    method="linear"
)
```

Frozen constants are 2000 replications, mean block length 30, restart probability `1/30`, familywise alpha `0.05`, multiplicity method `BONFERRONI`, per-K alpha `0.0125`, and quantile method `linear`.

### 10.4 Aggregate evaluated-state economic adjudication

When evaluated:
- `qualifies` for portfolio $K$ is `true` if and only if:
  $$\text{mean\_economic\_delta} > 0 \quad \land \quad \text{positive\_block\_count} \ge 5 \quad \land \quad \text{bootstrap\_lower\_bound} > 0$$
- `qualified_top_k` is the canonically ordered list of $K \in [1, 3, 5, 10]$ for which `qualifies == true`.
- `economic_signal = (len(qualified_top_k) > 0)`.

---

## 11. Exact Artifact Contract

### 11.1 Successful run inventory (B-07)

A successful run produces exactly 9 artifacts:

1. `protocol_snapshot.json`
2. `daily_forecast_scores.csv.gz`
3. `forecast_metrics.csv`
4. `forecast_uncertainty.json`
5. `stability_diagnostics.json`
6. `economic_uncertainty.json`
7. `economic_summary.csv`
8. `development_adjudication.json`
9. `artifact_manifest.json`

#### Common CSV encoding
- Encoding: UTF-8 without BOM
- Delimiter: comma `,`
- Quoting: double-quote `"`, minimal quoting
- Line terminator: LF (`0x0A`)
- Header row: mandatory
- Values: `null` -> empty field; `boolean` -> lowercase `true` / `false`; `integer` -> base-10 decimal; `finite float` -> `.17g`; `date` -> `YYYY-MM-DD`. No locale-specific formatting.

#### Common JSON encoding
- Encoding: UTF-8 without BOM
- Floats: `allow_nan = false`
- Character set: `ensure_ascii = false`
- Key order: `sort_keys = true`
- `JSON_OBJECT_KEY_ORDER = LEXICOGRAPHIC_BY_SORT_KEYS_TRUE`
- Object-member key lists in this specification are descriptive schema listings only; they do not override lexicographic serializer order.
- Whitespace: separators `(",", ":")`
- Terminator: exactly one final LF (`0x0A`) byte

#### Gzip encoding (`daily_forecast_scores.csv.gz`)
- Exactly one gzip member using DEFLATE at compression level `9`.
- The first ten bytes are exactly `1f 8b 08 00 00 00 00 00 02 ff`: `ID1=0x1f`, `ID2=0x8b`, `CM=0x08`, `FLG=0`, `MTIME=0`, `XFL=2`, and `OS=255`.
- `FEXTRA`, `FNAME`, `FCOMMENT`, and `FHCRC` are absent. If a compressor produces another OS byte, normalize byte offset 9 to `0xff` before hashing. Any other fixed-header mismatch fails artifact generation closed.
- Trailer `CRC32` and `ISIZE` must be correct for the uncompressed CSV payload. No producer/platform-derived OS byte has authority.

```text
GZIP_MEMBER_COUNT = 1
GZIP_COMPRESSION_METHOD = DEFLATE
GZIP_COMPRESSION_LEVEL = 9
GZIP_FLG = 0
GZIP_MTIME = 0
GZIP_XFL = 2
GZIP_OS = 255
GZIP_FEXTRA = ABSENT
GZIP_FNAME = ABSENT
GZIP_FCOMMENT = ABSENT
GZIP_FHCRC = ABSENT
```

#### Identifier mapping
- `B0 model_id = "B0"`
- `B0 candidate_id = "B0_UNIFORM"`
- `M3 model_id = "M3"`
- M3 candidate IDs: `M3_W030`, `M3_W060`, `M3_W120`, `M3_W240`, `M3_W365`. The selected candidate retains its candidate ID on VAL and STABILITY.

#### 1. `protocol_snapshot.json`
Closed 2-key schema: `authority` (72 keys) and `protocol_fingerprint_sha256`.

#### 2. `daily_forecast_scores.csv.gz`
- Row universe:
  - DEV: exactly 6 rows per DEV target date (B0 + all 5 M3 candidates).
  - VAL: exactly 2 rows per VAL target date (B0 + selected M3 candidate).
  - STABILITY:
    - If `FORECAST_SIGNAL == true`: exactly 2 rows per STABILITY target date (B0 + selected M3 candidate).
    - If `FORECAST_SIGNAL == false`: ZERO STABILITY rows.
- Columns: `target_date, stage, model_id, candidate_id, poisson_deviance, mae, rmse`.
- No nulls permitted in existing rows.
- Ordering: stage order (DEV, VAL, STABILITY), then `target_date` ascending, then `model_id` ascending, then `candidate_id` ascending.

#### 3. `forecast_metrics.csv`
- Row universe: exactly one aggregate row per `(stage, model_id, candidate_id)` combination present in daily scores:
  - DEV: exactly 6 rows.
  - VAL: exactly 2 rows.
  - STABILITY: exactly 2 rows if `FORECAST_SIGNAL == true`; ZERO rows if `FORECAST_SIGNAL == false`.
- Columns: `stage, model_id, candidate_id, date_count, poisson_deviance, mae, rmse`.
- Deterministic Row Order (U-018 Correction Addendum):
  ```text
  ROW_ORDER =
  1. stage exact order:
     DEV
     VAL
     STABILITY

  2. within stage:
     model_id ascending lexicographic

  3. within same stage and model_id:
     candidate_id ascending lexicographic
  ```

#### 4. `forecast_uncertainty.json`
Exact top-level key set, with no others. The list below is descriptive and does not prescribe JSON object-member order:

```text
status
bootstrap_rng_implementation
bootstrap_bit_generator
bootstrap_seed
bootstrap_replications
mean_block_length
restart_probability
alpha
quantile_method
observed_mean_improvement
bootstrap_lower_bound
```

`status` is the non-null string `"EVALUATED"`. `bootstrap_rng_implementation` and `bootstrap_bit_generator` are non-null strings; `bootstrap_seed`, `bootstrap_replications`, and `mean_block_length` are non-null integers; `restart_probability` and `alpha` are non-null numbers; and `quantile_method` is a non-null string. `observed_mean_improvement` and `bootstrap_lower_bound` are finite non-null binary64 numbers. `observed_mean_improvement = mean_t(PD_B0_t - PD_M3_t)` on VAL. Its constants and lower-bound calculation are exactly those in Section 9.5.

#### 5. `stability_diagnostics.json`
Exact top-level key set, with no others. The list below is descriptive and does not prescribe JSON object-member order:

```text
status
date_count
blocks
per_k
```

When evaluated, `status = "EVALUATED"`, `date_count` is the non-null integer `N_stability`, and `blocks` is an array of exactly six records ordered by integer `block_id` ascending. Each block has exactly `block_id` (integer), `start_date` (non-null `YYYY-MM-DD` string), `end_date` (non-null `YYYY-MM-DD` string), `date_count` (non-null integer), and `per_k` (array). Each block `per_k` has exactly four ordered records for K `1, 3, 5, 10`, each with exactly `K` (integer), `mean_economic_delta` (finite non-null binary64 number), and `positive` (boolean).

Top-level `per_k` is exactly four ordered records for K `1, 3, 5, 10`. Each record has exactly `K` (integer), `positive_block_count` (integer from 0 through 6), `bootstrap_lower_bound` (finite non-null binary64 number), and `qualifies` (boolean).

When the forecast gate fails, `status = "NOT_EVALUATED_FORECAST_GATE_FAILED"`, `date_count = null`, and `blocks = []`. The four ordered top-level K records remain present with `positive_block_count = null`, `bootstrap_lower_bound = null`, and `qualifies = false`.

#### 6. `economic_uncertainty.json` (B-05)
Self-contained closed 13-key top-level schema. The following object-key listing is descriptive only:
- Top-level keys:
  `status`, `familywise_alpha`, `multiplicity_method`, `per_k_alpha`, `bootstrap_rng_implementation`, `bootstrap_bit_generator`, `bootstrap_seed`, `bootstrap_replications`, `mean_block_length`, `restart_probability`, `quantile_method`, `shared_resample_indices`, `per_k`.
- Frozen constants:
  `familywise_alpha = 0.05`, `multiplicity_method = "BONFERRONI"`, `per_k_alpha = 0.0125`, `bootstrap_rng_implementation = "numpy.random.Generator"`, `bootstrap_bit_generator = "PCG64"`, `bootstrap_seed = 20260832`, `bootstrap_replications = 2000`, `mean_block_length = 30`, `restart_probability = 1/30`, `quantile_method = "linear"`, `shared_resample_indices = true`.
- `per_k` array: exactly 4 records ordered `1, 3, 5, 10`. Each record has 8 keys:
  `K, status, date_count, mean_economic_delta, positive_block_count, bonferroni_alpha, bootstrap_lower_bound, qualifies`.
- Evaluated state:
  `status = "EVALUATED"`, `date_count = N_stability`, `mean_economic_delta = float`, `positive_block_count = 0..6`, `bonferroni_alpha = 0.0125`, `bootstrap_lower_bound = float`, `qualifies = boolean`.
- Forecast-failed state:
  Top-level `status = "NOT_EVALUATED_FORECAST_GATE_FAILED"`.
  For each K: `status = "NOT_EVALUATED_FORECAST_GATE_FAILED"`, `date_count = null`, `mean_economic_delta = null`, `positive_block_count = null`, `bonferroni_alpha = 0.0125`, `bootstrap_lower_bound = null`, `qualifies = false`.
  `ECONOMIC_BOOTSTRAP_EXECUTED = NO`. Top-level constants remain populated.

#### 7. `economic_summary.csv`
Exactly 4 rows for $K \in [1, 3, 5, 10]$ in order.
Columns: `K, status, date_count, mean_economic_delta, positive_block_count, bonferroni_alpha, bootstrap_lower_bound, qualifies, recommended`.
When evaluated: valid typed entries; `recommended = true` iff $K$ is chosen recommended portfolio.
When forecast gate fails: `date_count=null`, `mean_economic_delta=null`, `positive_block_count=null`, `bootstrap_lower_bound=null`, `qualifies=false`, `recommended=false`, `bonferroni_alpha=0.0125`.

#### 8. `development_adjudication.json`
Exact top-level key set, with no others. The list below is descriptive and does not prescribe JSON object-member order:

```text
status
selected_candidate_id
selected_window_days
forecast_signal
forecast_bootstrap_lower_bound
economic_signal
qualified_top_k
recommended_top_k
development_exit_status
```

`status` is exactly the non-null string `"COMPLETED"`; `selected_candidate_id` is a non-null candidate-ID string; `selected_window_days` is a non-null integer and means `COUNT_OF_PRECEDING_RECORDED_DRAW_DATES`; `forecast_signal` and `economic_signal` are booleans; `forecast_bootstrap_lower_bound` is a finite non-null binary64 number; `qualified_top_k` and `recommended_top_k` are integer arrays in canonical K order; and `development_exit_status` is one of exactly `FORECAST_GATE_FAILED`, `ECONOMIC_GATE_FAILED`, or `ECONOMIC_SIGNAL_FOUND`.

The mappings are exact: `forecast_signal = false` maps to `FORECAST_GATE_FAILED`; `forecast_signal = true` and `economic_signal = false` maps to `ECONOMIC_GATE_FAILED`; and `forecast_signal = true` and `economic_signal = true` maps to `ECONOMIC_SIGNAL_FOUND`. In the forecast-failed case, `economic_signal = false`, `qualified_top_k = []`, and `recommended_top_k = []`.

#### 9. `artifact_manifest.json`
Its exact top-level key is `artifacts`; no other top-level key is permitted. `artifacts` is an array ordered lexicographically by filename and has exactly the other eight success artifacts, never a self-record. Every record has exactly `filename` (non-null string), `sha256` (64-character lowercase hexadecimal string), and `byte_size` (non-negative integer). The manifest is excluded from its own inventory and is hashed only after every listed artifact is final.

```text
artifact_manifest.artifacts = filename lexicographic ascending
```

### 11.2 General failure artifact contract (B-08)

If any stage fails, the pipeline produces exactly `development_run_FAILED.json`.
Its exact top-level keys, with no others, are `status`, `failed_stage`, `error_type`, `development_exit_status`, and `protocol_snapshot`. `status` is exactly `"FAILED"`. `failed_stage` is exactly one of:

```text
DATA_VALIDATION
MODEL_INITIALIZATION
MODEL_FIT
FORECAST_GENERATION
METRIC_EVALUATION
FORECAST_BOOTSTRAP
ECONOMIC_EVALUATION
ECONOMIC_BOOTSTRAP
ARTIFACT_VALIDATION
```

`error_type` is a non-empty stable protocol error identifier; it is not a globally closed enum. `development_exit_status` is exactly one of `NEEDS_DATA_REVISION`, `NEEDS_MODEL_REVISION`, `NEEDS_PROTOCOL_REVISION`, or `TECHNICAL_FAILURE`. `protocol_snapshot` has type `object | null`.

Definition of `AUTHORITY_COMPLETE`:
All required closed authority keys have valid non-null values and `protocol_fingerprint_sha256` can be computed.
- If failure occurs BEFORE `AUTHORITY_COMPLETE`:
  `protocol_snapshot = null`.
  No partial authority object, no placeholder, no sentinel SHA-256 string.
- If failure occurs AFTER `AUTHORITY_COMPLETE`:
  `protocol_snapshot = { "authority": <complete authority object>, "protocol_fingerprint_sha256": <valid fingerprint> }`.

The SVD ambiguity triplet remains exact: `failed_stage = MODEL_INITIALIZATION`, `error_type = SVDLeadingSubspaceAmbiguity`, and `development_exit_status = NEEDS_MODEL_REVISION`.

---

## 12. Cross-artifact economic invariants (B-06)

When `economic_uncertainty.status = "EVALUATED"`, the four K records must agree across:
- `economic_uncertainty.json`
- `economic_summary.csv`
- `stability_diagnostics.json`
- `development_adjudication.json`

on `K`, `positive_block_count`, `bootstrap_lower_bound`, `qualifies` **WHERE THOSE FIELDS ARE REPRESENTED**.

Per-K scalar fields are NOT added to `development_adjudication.json`. For `development_adjudication.json`, consistency is strictly:
- `qualified_top_k = ordered list of K for which qualifies == true`
- `recommended_top_k subset_of qualified_top_k` (`len(recommended_top_k) <= 1`)

When `economic_uncertainty.status = "NOT_EVALUATED_FORECAST_GATE_FAILED"`:
- All four artifacts reflect forecast-failed state.
- `economic_signal = false`
- `qualified_top_k = []`
- `recommended_top_k = []`
- `economic_summary.csv` has 4 rows with null deltas, `qualifies=false`, and `recommended=false`.

---

## 13. Protocol fingerprint

1. Construct compact sorted UTF-8 JSON representation of `protocol_snapshot.authority` with `allow_nan=False` and separators `(",", ":")`.
2. Compute SHA-256 digest of the resulting UTF-8 bytes.
3. Record as lowercase 64-character hex string in `protocol_snapshot.protocol_fingerprint_sha256`.

---

## 14. Control Plane reconstruction decisions register

All 22 protocol-critical items are resolved by Control Plane reconstruction decisions and refined by the final-lock review correction addendum (B-01 through B-10):

| ID | Subject | Status | Final Lock Review Correction | Canonical Section |
|---|---|---|---|---|
| `U-001` | Complete model equation & digit mapping | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed in B-01 / B-02 | Section 7.1 |
| `U-002` | Parameter definitions & bounds | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (SLSQP parameter bounds & domains frozen in B-01) | Section 7.2, 7.3 |
| `U-003` | Matrix R construction & double-centering | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 6.1 |
| `U-004` | Non-degenerate centering/normalization | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 6.4 |
| `U-005` | Fitting objective, optimizer & convergence | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (Exact U-004 initialization, SLSQP vector/Jacobians/options, post-solver order, and MODEL_FIT error mapping restored) | Section 7.3, 7.4, 7.5 |
| `U-006` | Dataset identity, repository authority & missing-draw semantics | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (Observed draw event indexing, no-missing-draw semantics, and repository-authority stop rule frozen in B-03) | Section 8.1 |
| `U-007` | Chronology & split fractions | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (Nominal design fractions vs integer floor counts frozen in B-04) | Section 8.2 |
| `U-008` | Stability cohort & exact N_stability | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 8.3 |
| `U-009` | Stability block partition | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 8.4 |
| `U-010` | Baseline B0 definition | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 9.1 |
| `U-011` | Candidate family & bounds | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 9.2 |
| `U-012` | Candidate selection & tie-breaking | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 9.3 |
| `U-013` | Complete forecast qualification | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 9.4 |
| `U-014` | Forecast bootstrap contract | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (RNG authority preserved; protocol constants classified as Control Plane selected) | Section 9.5 |
| `U-015` | Economic delta construction | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 10.1 |
| `U-016` | recommended_top_k rule | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (Recommendation domain restricted to qualified_top_k in B-09) | Section 10.2 |
| `U-017` | Numeric tolerances & float format | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 5.8 |
| `U-018` | Success artifact schemas & inventory | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (Complete row universes, gzip mtime, typings in B-07; full economic uncertainty in B-05) | Section 11.1 |
| `U-019` | Failure artifact & exit semantics | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (Pre- vs post-authority completion protocol_snapshot in B-08) | Section 11.2 |
| `U-020` | Closed authority key set & types | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | APPLIED (Exact 72 keys verified; optimizer_constraint_tolerance clarified in B-01) | Section 4.2 |
| `U-023` | Economic stationary bootstrap contract | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 10.3 |
| `U-024` | Evaluated-state aggregate economic adjudication | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Addressed | Section 10.4 |

---

## 15. Reconstruction classification summary

- **Preserved Contracts (`P-001` through `P-038`)**: 38
- **Reconstructed Records (`R-001`, `R-002`)**: 2
- **Control Plane Adjudicated Decisions (`U-001`..`U-020`, `U-023`, `U-024`)**: 22
- **Deferred Dispositions (`U-021`, `U-022`)**: 2
- **Non-Applicable Records (`N-001` through `N-005`)**: 5
- **Total Registered Contracts**: 69
- **Review Blockers Resolved (`B-01` through `B-10`)**: 10
- **Hidden Implementation Decisions**: 0
- **Surviving Authority Semantic Drift**: NONE

---

## 16. Canonical reconstruction gate state

```text
CONTROL_PLANE_RECONSTRUCTION_ADJUDICATION = COMPLETE
CONTROL_PLANE_FINAL_LOCK_CORRECTION = COMPLETE
PROTOCOL_CRITICAL_UNRESOLVED_COUNT = 0
HIDDEN_IMPLEMENTATION_DECISION_COUNT = 0
SURVIVING_AUTHORITY_SEMANTIC_DRIFT = NONE

FORMAL_ARTIFACT_REVIEW = READY_FOR_FINAL_INDEPENDENT_RECHECK
TARGETED_SCIENTIFIC_REVIEW = PASS
FORMAL_SPEC_LOCK = NOT_AUTHORIZED
REPO_GROUNDED_PLANNING = NOT_AUTHORIZED
SOURCE_IMPLEMENTATION = NOT_AUTHORIZED
V3_HISTORICAL_RUN = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
HISTORICAL_EXECUTION = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
```

### 16.1 Final-lock bounded correction (LOCK-01..LOCK-03)

```text
FINAL_LOCK_BOUND_CORRECTION = COMPLETE
LOCK-01 = PASS
LOCK-02 = PASS
LOCK-03 = PASS

JSON_OBJECT_KEY_ORDER = LEXICOGRAPHIC_BY_SORT_KEYS_TRUE
JSON_OBJECT_ORDER_CONTRADICTION_COUNT = 0
U005_CANONICAL_FIDELITY = PASS
STALE_CONTROL_PLANE_MEMORANDUM_BLOB_REFERENCE_COUNT = 0
UNRESOLVABLE_CURRENT_EVIDENCE_CLAIM_COUNT = 0
NORMATIVE_DEPENDENCY_ON_UNAVAILABLE_BACKGROUND_EVIDENCE = NONE

TARGETED_SCIENTIFIC_REVIEW = PASS
FORMAL_ARTIFACT_REVIEW = READY_FOR_FINAL_INDEPENDENT_RECHECK
FORMAL_SPEC_LOCK = NOT_AUTHORIZED
REPO_GROUNDED_PLANNING = NOT_AUTHORIZED
SOURCE_IMPLEMENTATION = NOT_AUTHORIZED
V3_HISTORICAL_RUN = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
```

### 16.2 Final fidelity repair (FLR-01..FLR-02)

```text
FLR-01 = PASS
FLR-02 = PASS
U006_CANONICAL_FIDELITY = PASS
REPOSITORY_AUTHORITY_MISMATCH_RULE_PRESENT = PASS
SILENT_DATA_SOURCE_SUBSTITUTION_FORBIDDEN = PASS
U014_PROVENANCE_CLASSIFICATION = PASS
FORECAST_BOOTSTRAP_CP_PROVENANCE_MISMATCH_COUNT = 0
CONTROL_PLANE_DECISIONS_LOCAL_FIDELITY = 22/22
HIDDEN_IMPLEMENTATION_DECISION_COUNT = 0
CANONICAL_SELF_CONTAINMENT = PASS
AUTHORITY_KEY_COUNT = 72

TARGETED_SCIENTIFIC_REVIEW = PASS
FORMAL_ARTIFACT_REVIEW = READY_FOR_FINAL_INDEPENDENT_RECHECK
FORMAL_SPEC_LOCK = NOT_AUTHORIZED
REPO_GROUNDED_PLANNING = NOT_AUTHORIZED
SOURCE_IMPLEMENTATION = NOT_AUTHORIZED
V3_HISTORICAL_RUN = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
```
