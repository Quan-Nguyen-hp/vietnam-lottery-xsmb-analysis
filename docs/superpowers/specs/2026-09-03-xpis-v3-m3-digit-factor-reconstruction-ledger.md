# XPIS v3 — M3 Digit-Factor Reconstruction Ledger

```text
LEDGER_VERSION = XPIS_V3_M3_RECONSTRUCTION_LEDGER_V3
LEDGER_STATUS = AUTHORITY_RECONSTRUCTION_CANDIDATE
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
HISTORICAL_EXECUTION = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
```

---

## 8. Final-lock bounded correction (LOCK-01..LOCK-03)

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
```

---

## 7. Final-lock fidelity repair record (FR-01..FR-08)

The following are bounded restoration and verification outcomes, not new `protocol_snapshot.authority` keys or new scientific decisions.

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

CONTROL_PLANE_DECISIONS_VERIFIED_LOCALLY = 22/22
CONTROL_PLANE_CP1_VALUE_MISMATCH_COUNT = 0
SURVIVING_FINGERPRINT_VALUE_MISMATCH_COUNT = 0
CONTROL_PLANE_AUTHORITY_LITERAL_MISMATCH_COUNT = 0
SURVIVING_AUTHORITY_LITERAL_MISMATCH_COUNT = 0
SURVIVING_AUTHORITY_SEMANTIC_DRIFT = NONE

SUCCESS_ARTIFACT_SCHEMA_SELF_CONTAINMENT = PASS
FAILURE_CONTRACT_SELF_CONTAINMENT = PASS
BOOTSTRAP_RNG_CONSUMPTION_ORDER_FULLY_SPECIFIED = YES
BOOTSTRAP_RNG_CONSUMPTION_ORDER = FULLY_DETERMINISTIC
NUMERIC_COMPARISON_SEMANTICS = FULLY_DETERMINISTIC
GZIP_FIXED_HEADER = FULLY_DETERMINISTIC
SUCCESS_SCHEMA = SELF_CONTAINED
FAILURE_SCHEMA = SELF_CONTAINED
SURVIVING_EVIDENCE_IDENTITY_REPRODUCIBLE = PASS
```

---

## 1. Evidence identities

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

## 1.0 Current background-evidence resolution

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

UNRESOLVABLE_CURRENT_EVIDENCE_CLAIM_COUNT = 0
NORMATIVE_DEPENDENCY_ON_UNAVAILABLE_BACKGROUND_EVIDENCE = NONE
```

R-001 and R-002 rely only on E-003, E-001, and E-009. The canonical authority relies only on the surviving Closure blob/content, the persisted Control Plane memorandum, the current canonical artifact, and explicit Control Plane decisions.

---

## 1.1 Literal fingerprint parity register

```text
artifact_contract_version = "XPIS_V3_M3_ARTIFACT_CONTRACT_V1"
model_contract_version = "XPIS_V3_M3_MODEL_CP1"
data_split_contract_version = "XPIS_V3_M3_DATA_SPLIT_CP1"
candidate_contract_version = "XPIS_V3_M3_CANDIDATE_CP1"
forecast_gate_contract_version = "XPIS_V3_M3_FORECAST_GATE_CP1"
forecast_bootstrap_contract_version = "XPIS_V3_M3_FORECAST_BOOTSTRAP_CP1"
economic_contract_version = "XPIS_V3_M3_ECONOMIC_CP1"
success_artifact_schema_version = "XPIS_V3_M3_SUCCESS_ARTIFACTS_CP1"
failure_artifact_schema_version = "XPIS_V3_M3_FAILURE_ARTIFACT_CP1"
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
svd_ambiguous_leading_subspace_policy = "FAIL_NEEDS_MODEL_REVISION"
```

This register is an audit representation of existing closed authority, not an addition to its 72-key schema.

---

## 2. Contract classification ledger

`U-021` and `U-022` are retained in this table as deferred lifecycle/release dispositions. All 22 protocol-critical items (`U-001..U-020, U-023, U-024`) are resolved by Control Plane reconstruction decisions and refined by final-lock review corrections. Remaining protocol-critical unresolved count is exactly `0`.

| CONTRACT_ID | SUBJECT | STATUS | CANONICAL_VALUE_OR_RULE | PRIMARY_EVIDENCE | SECONDARY_EVIDENCE | CONFLICT | CONTROL_PLANE_DECISION_REQUIRED |
|---|---|---|---|---|---|---|---|
| `P-001` | Metric domain and scoring preconditions | `PRESERVED` | `n=0..99`; finite positive `mu`; nonnegative `y`; observed total 27; natural log; `SURVIVING_AUTHORITY`: forecast outputs must satisfy forecast contract (`sum_n mu[t,n] = 27`) subject to tolerance-based validation condition, and scientific qualification and metric evaluation use unrounded forecast values; `NUMERIC_TOLERANCE_VALUE = 1e-10`; `NUMERIC_TOLERANCE_AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION`; `NUMERIC_TOLERANCE_DECISION_ID = U-017`; `NUMERIC_REPRESENTATION = binary64`; `NUMERIC_REPRESENTATION_AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION`; `NUMERIC_REPRESENTATION_DECISION_ID = U-017` (`SCIENTIFIC_QUALIFICATION_NUMERIC_REPRESENTATION = UNROUNDED_IN_MEMORY_BINARY64_VALUES`) | `E-001:16–43` | `E-003:86–157` | Forecast contract existence, tolerance validation requirement, and unrounded forecast values preserved from surviving Closure (`E-001:16–43`); numeric tolerance value `1e-10` and `binary64` numerical representation resolved by Control Plane decision `U-017` (`CONTROL_PLANE_RECONSTRUCTION_DECISION`) | `NO` |
| `P-002` | Poisson cell and zero-count rule | `PRESERVED` | Exact positive-count formula; `PD_cell(0,mu)=2*mu`; no epsilon in `y` | `E-001:45–87` | `E-003:86–157` | None | `NO` |
| `P-003` | Daily Poisson deviance | `PRESERVED` | Mean of 100 cell deviances | `E-001:45–87` | `E-003:86–157` | None | `NO` |
| `P-004` | Daily MAE | `PRESERVED` | Mean absolute error over 100 outcomes | `E-001:89–102` | `E-003:86–157` | None | `NO` |
| `P-005` | Daily RMSE | `PRESERVED` | Square root of mean squared error over 100 outcomes | `E-001:104–119` | `E-003:86–157` | None | `NO` |
| `P-006` | Stage Poisson aggregation | `PRESERVED` | Arithmetic mean of daily Poisson deviance | `E-001:121–133` | `E-003:86–157` | None | `NO` |
| `P-007` | Stage MAE aggregation | `PRESERVED` | Arithmetic mean of daily MAE | `E-001:135–143` | `E-003:86–157` | None | `NO` |
| `P-008` | Stage RMSE aggregation | `PRESERVED` | `sqrt(mean(daily_rmse^2))`, equivalent to pooled target-outcome RMSE | `E-001:145–181` | `E-003:86–157` | None | `NO` |
| `P-009` | Block metric mathematics | `PRESERVED` | Same mathematical rules as stage: daily means for Poisson/MAE, root mean daily squared RMSE; use in-memory unrounded values | `E-001:184–208` | `E-003:86–157` | Naming inconsistency resolved by `R-002` | `NO` |
| `P-010` | Development-validation secondary rule | `PRESERVED` | `MAE_M3 <= MAE_B0 OR RMSE_M3 <= RMSE_B0`; no tolerance | `E-001:210–229` | `E-003:86–157` | Integrated into complete gate by `U-013` | `NO` |
| `P-011` | Primary Poisson comparison | `PRESERVED` | `PD_STAGE_M3 < PD_STAGE_B0` | `E-001:230–235` | `E-003:86–157` | Baseline defined by `U-010` | `NO` |
| `P-012` | Daily forecast-score columns | `PRESERVED` | `poisson_deviance`, `mae`, `rmse` | `E-001:237–248` | `E-003:86–157` | Full schema resolved by `U-018` | `NO` |
| `P-013` | Forecast metric reconstruction | `PRESERVED` | Stage metrics reconstruct from daily metrics by the exact stage rules | `E-001:249–264` | `E-003:86–157` | Reconstruction tolerance resolved by `U-017` | `NO` |
| `P-014` | Metric authority constants | `PRESERVED` | Version, log base, outcome aggregations, and stage aggregations exactly as recorded in canonical Section 5.8 | `E-001:266–304` | `E-003:86–157` | Block field name normalized by `R-002` | `NO` |
| `P-015` | SVD implementation | `PRESERVED` | `numpy.linalg.svd(R, full_matrices=False)` | `E-001:306–340` | `E-003:86–157` | `R` construction resolved by `U-003` | `NO` |
| `P-016` | Interaction-zero tolerance | `PRESERVED` | `1e-10` | `E-001:342–351` | `E-003:86–157` | Serialized authority mapping resolved by `U-020` | `NO` |
| `P-017` | Zero-interaction representation | `PRESERVED` | `gamma=0`; `u=v=(9,-1,...,-1)/sqrt(90)` | `E-001:352–371` | `E-003:86–157` | Serialized authority encoding resolved by `U-020` | `NO` |
| `P-018` | Leading SVD gap | `PRESERVED` | `(s0-s1)/s0 > 1e-8` when `s0 > 1e-10` | `E-001:373–401` | `E-003:86–157` | None | `NO` |
| `P-019` | Near-tie failure and prohibited fallbacks | `PRESERVED` | Fail `MODEL_INITIALIZATION` / `SVDLeadingSubspaceAmbiguity` / `NEEDS_MODEL_REVISION`; no arbitrary basis, alternate vector, perturbation, tolerance change, random init, or retry | `E-001:402–429` | `E-003:86–157` | General failure schema resolved by `U-019` | `NO` |
| `P-020` | Non-degenerate SVD start | `PRESERVED` | `u0=U[:,0]`, `v0=Vt[0,:]`, `gamma0=s0` only after both strict tests | `E-001:431–451` | `E-003:86–157` | Centering/normalization resolved by `U-004` | `NO` |
| `P-021` | SVD sign canonicalization | `PRESERVED` | Smallest max-absolute `u` index must be positive; otherwise negate both `u` and `v` | `E-001:452–475` | `E-003:86–157` | Serialized authority encoding resolved by `U-020` | `NO` |
| `P-022` | Serialized SVD V3 fields | `PRESERVED` | Implementation, `full_matrices`, gap tolerance/definition, and ambiguous-subspace policy values | `E-001:477–498` | `E-003:86–157` | None | `NO` |
| `P-023` | Artifact contract version | `PRESERVED` | `XPIS_V3_M3_ARTIFACT_CONTRACT_V1`; exact schemas mandatory | `E-001:500–513` | `E-003:86–157` | Full success/failure schemas resolved by `U-018/U-019` | `NO` |
| `P-024` | Protocol snapshot top level | `PRESERVED` | Exactly `authority` and `protocol_fingerprint_sha256` | `E-001:515–534` | `E-003:86–157` | None | `NO` |
| `P-025` | Source identity placement | `PRESERVED` | `source_commit` and `source_tree` are mandatory inside `authority` and fingerprinted | `E-001:535–548` | `E-003:86–157` | Exact run values deferred to implementation/release gate via `U-021` | `NO` |
| `P-026` | V3-added authority field list | `PRESERVED` | Schema and field existence list preserved; exact literal value authority for reconstructed fields resolved by U-020 addendum (PROV-01, PROV-02); complete closed schema resolved by `U-020` | `E-001:550–616` | `E-003:382–435` | Whole closed schema resolved by `U-020` | `NO` |
| `P-027` | Bootstrap RNG constructors and seeds | `PRESERVED` | `Generator(PCG64(20260831))` forecast; `Generator(PCG64(20260832))` economic; no global/default/platform seed | `E-001:618–639` | `E-003:86–157` | Forecast law resolved by `U-014`; economic resampling resolved by `U-023` | `NO` |
| `P-028` | Economic uncertainty top-level schema | `PRESERVED` | Exactly 13 named keys in canonical Section 11.1.6; self-contained | `E-001:642–663` | `E-003:86–157` | Restored self-contained by B-05 | `NO` |
| `P-029` | Economic uncertainty constants | `PRESERVED` | Alpha, Bonferroni, seed, 2000 replications, block length 30, restart `1/30`, linear quantile, shared indices | `E-001:664–698` | `E-003:86–157` | Delta construction resolved by `U-015`; bootstrap estimator resolved by `U-023` | `NO` |
| `P-030` | Economic statuses and technical failure distinction | `PRESERVED` | Exactly `NOT_EVALUATED_FORECAST_GATE_FAILED` and `EVALUATED`; technical failure uses failure contract | `E-001:700–712` | `E-003:86–157` | Failure contract resolved by `U-019` | `NO` |
| `P-031` | Per-K representation | `PRESERVED` | JSON array ordered `1,3,5,10`; exact eight-key record | `E-001:714–740` | `E-003:86–157` | None | `NO` |
| `P-032` | Evaluated per-K value contract | `PRESERVED` | Exact status, types, date count, block-count range, alpha, lower bound, and boolean constraints | `E-001:742–773` | `E-003:86–157` | `N_stability` resolved by `U-008` | `NO` |
| `P-033` | Economic qualification | `PRESERVED` | Mean delta > 0 AND at least five positive blocks AND bootstrap lower bound > 0 | `E-001:774–790` | `E-003:86–157` | Delta construction resolved by `U-015` | `NO` |
| `P-034` | Forecast-fail economic representation | `PRESERVED` | Exact top-level/per-K null and false representation; no resampling | `E-001:793–835` | `E-003:86–157` | Forecast predicate resolved by `U-013` | `NO` |
| `P-035` | Forecast-fail cross-artifact invariant | `PRESERVED` | False signals, empty qualified/recommended K, four matching summary rows | `E-001:837–862` | `E-003:86–157` | Positive recommendation rule resolved by `U-016`; evaluated aggregate state resolved by `U-024` | `NO` |
| `P-036` | Evaluated cross-artifact invariant | `PRESERVED` | Four K records agree across uncertainty, summary, adjudication, and stability artifacts on represented fields | `E-001:864–889` | `E-003:86–157` | Full artifact schemas resolved by `U-018`; qualifier restored by B-06 | `NO` |
| `P-037` | Fingerprint canonicalization | `PRESERVED` | UTF-8 of sorted compact JSON with `allow_nan=False`, then SHA-256 | `E-001:891–927` | `E-003:86–157` | Closed authority key set resolved by `U-020` | `NO` |
| `P-038` | Scientific freeze and authorization guards | `PRESERVED` | Passed science unchanged; explicit preserved authorization guards: FORMAL_SPEC_LOCK=NOT_AUTHORIZED, REPO_GROUNDED_PLANNING=NOT_AUTHORIZED, SOURCE_IMPLEMENTATION=NOT_AUTHORIZED, V3_HISTORICAL_RUN=NOT_AUTHORIZED, PREREGISTRATION=NOT_AUTHORIZED, CONFIRMATORY_EXECUTION=NOT_AUTHORIZED | `E-001:929–993` | `E-003:565–595,770–839` | Preserved guards verified against E-001:972–993 | `NO` |
| `R-001` | Reconstruction provenance and gate status | `RECONSTRUCTED` | Explicit reconstruction candidate; historical source unavailable; science preserved; status `COMPLETE_ADJUDICATED` post control-plane adjudication, review corrections, and U-020 literal provenance closure; distinguishes schema existence from exact literal authority (PROV-01) | `E-003:44–84,473–516,770–839` | `E-001:1–14,968–993` | None | `NO` |
| `R-002` | Block metric authority name | `RECONSTRUCTED` | `BLOCK_METRIC_AGGREGATION=SAME_RULES_AS_STAGE`; serialized field `authority.block_metric_aggregation="SAME_RULES_AS_STAGE"`; no independent shorter alias | `E-003:437–470` | `E-001:184–208,266–304,550–616` | None | `NO` |
| `U-001` | Complete model equation and digit/index mapping | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Multinomial loglinear rank-1 digit interaction: `i=floor(n/10)`, `j=n%10`; `eta[i,j]=a[i]+b[j]+gamma*u[i]*v[j]`; `p[i,j]=softmax(eta)` with `max_eta` stabilization; `mu=27*p`; structural `mu>0`, `sum mu=27` | `E-009:Cluster A` | `E-001:16–43,306–475` | None | `NO (RESOLVED)` |
| `U-002` | Parameter definitions, domains, bounds, and identifiability constraints | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `a[i], b[j] ∈ R`; `sum a=0`, `sum b=0`; `sum u=0`, `sum v=0`; `dot(u,u)=dot(v,v)=1`; `gamma>=0`; canonical sign rule on smallest max-absolute `u` index; `gamma<=1e-10` canonicalized to `gamma=0`, `u=v=(9,-1,...,-1)/sqrt(90)`; refined by B-01/B-02 | `E-009:Cluster A` | `E-001:342–475` | None | `NO (RESOLVED)` |
| `U-003` | Interaction matrix R construction/double-centering | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `q[i,j]=X[i,j]/(27*W)`; digit marginals `r[i]=sum_j q`, `c[j]=sum_i q`; `r[i]>0, c[j]>0` strictly required (fail `ZeroDigitMarginal`); `R[i,j]=q[i,j]-r[i]*c[j]`; exact double-centered input to SVD | `E-009:Cluster A` | `E-001:306–340` | None | `NO (RESOLVED)` |
| `U-004` | Non-degenerate centering/normalization | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `a0[i]=ln(r[i])-mean(ln(r))`, `b0[j]=ln(c[j])-mean(ln(c))`; `u_c=u-mean(u)`, `v_c=v-mean(v)`; `nu,nv>1e-12` (fail `CenteredSingularVectorDegeneracy`); `u=u_c/nu`, `v=v_c/nv`, `gamma=s0*nu*nv`; sign canonicalization | `E-009:Cluster A` | `E-001:431–475` | None | `NO (RESOLVED)` |
| `U-005` | Fit objective/optimizer/convergence/general fit failure | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Independent per-target/window fit initialized exactly from U-004; no random restart, alternate solver, fallback optimizer, warm-start chain, or tolerance relaxation. Accepted only after the frozen successful-fit checks. Any U-005 failure is `MODEL_FIT` / `NEEDS_MODEL_REVISION` with exactly `OptimizerNonConvergence`, `NonFiniteModelFit`, `ConstraintViolation`, or `ForecastContractViolation`, as applicable. | `E-009:Cluster A` | `E-001:306–475` | None | `NO (RESOLVED)` |
| `U-006` | M3 dataset/preprocessing/snapshot identity | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `DATA_SOURCE_PATH=data/xsmb-2-digits.csv`; `DATA_SOURCE_REPOSITORY=Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis` (PROV-02); repository mismatch fails closed; `SILENT_REPOSITORY_SUBSTITUTION=FORBIDDEN`; `SILENT_DATA_SOURCE_SUBSTITUTION=FORBIDDEN`; 27 integer outcomes `0..99`; observed draw event indexed; no-missing-draw means present dates have 27 valid outcomes; calendar continuity out of scope; refined by B-03 and U-020 addendum | `E-009:Cluster B & U-020 Addendum` | `E-001:16–43` | None | `NO (RESOLVED)` |
| `U-007` | Development chronology and split | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Eligible dates sorted ascending; `N>=240`; contiguous split `N_dev=floor(N/2)`, `N_val=floor(N/4)`, `N_stability=N-N_dev-N_val`; nominal design fractions `[0.5,0.25,0.25]`; floor counts have precedence; refined by B-04 | `E-009:Cluster B` | `E-001:121–181,210–235` | None | `NO (RESOLVED)` |
| `U-008` | Stability cohort and N_stability | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `N_stability = N - floor(N/2) - floor(N/4) >= 60`; `economic_uncertainty.per_k[*].date_count = N_stability` exactly when evaluated | `E-009:Cluster B` | `E-001:742–790` | None | `NO (RESOLVED)` |
| `U-009` | Stability block partition | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `STABILITY_BLOCK_COUNT=6`; `q=N_stability//6`, `r=N_stability%6`; first `r` blocks have `q+1` dates, remaining `6-r` have `q` dates; chronological contiguous assignment | `E-009:Cluster B` | `E-001:742–790` | None | `NO (RESOLVED)` |
| `U-010` | Baseline B0 | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `B0 = UNIFORM_COUNT_BASELINE`; `mu_B0[t,n] = 27/100 = 0.27` for all `t,n`; no parameters, window, or optimization | `E-009:Cluster C` | `E-001:210–235` | None | `NO (RESOLVED)` |
| `U-011` | Candidate/hyperparameter space and bounds | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Single hyperparameter `history_window_W ∈ {30,60,120,240,365}`; IDs `M3_W030`, `M3_W060`, `M3_W120`, `M3_W240`, `M3_W365`; no other sweeps permitted | `E-009:Cluster C` | Repository-wide negative search | None | `NO (RESOLVED)` |
| `U-012` | Candidate selection and ties | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Evaluated on DEV; select single candidate by ascending tuple `(PD_DEV, MAE_DEV, RMSE_DEV, W)`; strictly lexicographic; zero tolerance | `E-009:Cluster C` | `E-001:210–235` | None | `NO (RESOLVED)` |
| `U-013` | Complete forecast qualification | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | On VAL: `PRIMARY_FORECAST_PASS iff PD_VAL_M3 < PD_VAL_B0`; `DEV_VAL_SECONDARY_PASS iff MAE_VAL_M3 <= MAE_VAL_B0 OR RMSE_VAL_M3 <= RMSE_VAL_B0`; `FORECAST_BOOTSTRAP_PASS iff forecast_bootstrap_lower_bound > 0`; all three required | `E-009:Cluster C` | `E-001:210–235,793–889` | None | `NO (RESOLVED)` |
| `U-014` | Complete forecast bootstrap | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `d_t = PD_t_B0 - PD_t_M3` on VAL; surviving preserved RNG authority is `Generator(PCG64(20260831))`; Control Plane selected constants are 2000 reps, block length 30, restart `1/30`, alpha 0.05, linear quantile; single RNG instantiation; stationary circular bootstrap | `E-009:Cluster C` | `E-001:577–639` | None | `NO (RESOLVED)` |
| `U-015` | Economic delta and top-K economic construction | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `K ∈ {1,3,5,10}`; rank `mu_M3` descending, tie-break smaller `n`; units thousand VND; cost 27, payout 99; net `PnL[t,n] = 99*y[t,n]-27`; daily `economic_delta = sum PnL`; `mean_economic_delta = arithmetic mean across ALL STABILITY dates` (not unweighted mean of block means); `block_mean_economic_delta = arithmetic mean within block`; `positive_block_count = count of blocks with block_mean > 0 strictly` | `E-009:Cluster D` | `E-001:642–889` | None | `NO (RESOLVED)` |
| `U-016` | Positive-state recommended_top_k rule | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Ranking applies strictly to `qualified_top_k`; if empty -> `recommended_top_k = []`; else select single K by descending tuple `(bootstrap_lower_bound, mean_economic_delta, -K)`; refined by B-09 | `E-009:Cluster D` | `E-001:837–889` | None | `NO (RESOLVED)` |
| `U-017` | Forecast-sum and metric-serialization tolerances | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `FORECAST_SUM_TOLERANCE = 1e-10`; `ARTIFACT_NUMERIC_ABS_TOLERANCE = 1e-12`, `ARTIFACT_NUMERIC_REL_TOLERANCE = 1e-12`; `CSV_FLOAT_FORMAT = .17g`; binary64 in-memory evaluation; `allow_nan = false` | `E-009:Cluster E` | `E-001:16–43,237–264` | None | `NO (RESOLVED)` |
| `U-018` | Complete success-artifact schemas and required success-artifact inventory | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Exactly 9 artifacts; complete row universes for daily scores and forecast metrics; deterministic 3-tier row order for forecast_metrics.csv (stage order DEV, VAL, STABILITY; then model_id asc; then candidate_id asc); gzip level 9 mtime 0; self-contained 13-key economic uncertainty schema; refined by B-05/B-07 and U-018 row-order addendum | `E-009:Cluster E & U-018 Addendum` | `E-001:237–264,515–548,837–889` | None | `NO (RESOLVED)` |
| `U-019` | General failure artifacts and semantics | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Exactly `development_run_FAILED.json`; `protocol_snapshot` is object if `AUTHORITY_COMPLETE`, else null; no sentinel SHA; refined by B-08 | `E-009:Cluster E` | `E-001:396–429,500–513,700–712` | None | `NO (RESOLVED)` |
| `U-020` | Closed authority key set/types/encodings/fingerprint coverage | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | Closed set of 72 authority keys inside `authority`; explicit types and encodings; `optimizer_constraint_tolerance` documented as post-solver check; refined by B-01/B-07; distinguishes SCHEMA_OR_FIELD_EXISTENCE_AUTHORITY from EXACT_LITERAL_VALUE_AUTHORITY for PROV-01 and PROV-02 via U-020 addendum; spec_version literal value preserved from surviving closure (E-001:5) while schema membership/typing resolved by U-020 | `E-009:Cluster E & U-020 Addendum` | `E-001:515–616,891–927` | None | `NO (RESOLVED)` |
| `U-021` | Run-bound source commit/tree values | `DEFERRED_RUN_BOUND_VALUE` | Placement and fingerprint semantics preserved; exact run value is filled at the approved implementation/release gate | `E-001:515–548` | `E-003` scope guard | Lifecycle/value deferral only; not a current design-gate unresolved decision | `NO` |
| `U-022` | Implementation/release reproducibility environment | `DEFERRED_IMPLEMENTATION_RELEASE_CONCERN` | Scientific API/RNG determinism is preserved; runtime/dependency/backend details remain a later release concern | `E-001:306–340,618–639,891–927` | Repository `pyproject.toml` is non-authoritative background | Do not promote package or hardware values to implicit M3 design authority | `NO` |
| `U-023` | Complete economic-bootstrap resampling and lower-bound estimator contract | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `Generator(PCG64(20260832))`; 2000 reps, block 30, restart `1/30`; shared circular indices across K; quantile `0.0125` linear; one-sided Bonferroni-adjusted percentile lower bound | `E-009:Cluster D` | `E-001:618–639,664–698,742–790` | None | `NO (RESOLVED)` |
| `U-024` | Evaluated-state aggregate economic adjudication | `CONTROL_PLANE_RECONSTRUCTION_DECISION` | `qualified_top_k = [K for K in [1,3,5,10] if per_k[K].qualifies]`; `economic_signal = (len(qualified_top_k) > 0)`; if empty -> signal=false, qualified=[], recommended=[]; else signal=true; preserved per-K rule | `E-009:Cluster D` | `E-001:774–790,837–889` | None | `NO (RESOLVED)` |
| `N-001` | V2-named duplicate as historical authority | `NOT_APPLICABLE` | Evidence-only duplicate; cannot establish historical authority | `E-002` | `E-001` | Same bytes contradict historical-V2 interpretation | `NO` |
| `N-002` | Count-First V2 specification/plan as M3 authority | `NOT_APPLICABLE` | Background architecture only | `E-004/E-005` | `E-003:260–291` | No M3 import/inheritance | `NO` |
| `N-003` | Count-First code/tests/feature branch as M3 authority | `NOT_APPLICABLE` | Executable Count-First evidence only | `E-006/E-007` | Repository-wide M3 negative search | Different research protocol | `NO` |
| `N-004` | General XPIS architecture/roadmap as missing M3 values | `NOT_APPLICABLE` | Context only; supplies no M3 contract | `AGENTS.md` blob `fa2b043dc038480a2a9b44f017371d554278b17b`; `ROADMAP.md` blob `f8dc508a3202215362ad1b4e544fc941207b98f7` at base | `E-003:260–291` | No explicit M3 applicability | `NO` |
| `N-005` | Remote holdout-count advancement as reconstruction evidence | `NOT_APPLICABLE` | Authority-neutral operational update | `E-008` | Exact remote diff from local base | No relevant content | `NO` |

---

## 3. Reconstructed decision records

### R-001 — Reconstruction provenance and gate status

```text
CONTRACT_ID = R-001
CANONICAL_RULE = AUTHORITY_RECONSTRUCTION_CANDIDATE; historical M3 V2 source unavailable; passed closure science preserved; status COMPLETE_ADJUDICATED post control-plane adjudication, review corrections, and U-020 literal authority closure. For PROV-01 (authority_provenance), SCHEMA_OR_FIELD_EXISTENCE_AUTHORITY is derived from closed authority specification, while EXACT_LITERAL_VALUE_AUTHORITY = CONTROL_PLANE_RECONSTRUCTION_DECISION / U-020 correction addendum; HISTORICAL_RECOVERY_STATUS = NOT_RECOVERED; SURVIVING_AUTHORITY_SELECTION = NO.
EVIDENCE_PATHS = E-003; E-001; E-009
EVIDENCE_COMMITS = NOT_REPOSITORY_ARTIFACT; NOT_COMMITTED; NOT_COMMITTED
EVIDENCE_BLOBS = work-order SHA256 7ecf23599a5b4650b7bd8e933e5a4a802ffaf554039aa4e981b9df3fd1276126; closure blob 9353a8f4940de4257f6c96ee5e309ff827adfe11; Control Plane blob 7e9c55caea96be92eec64f0e30a1d39af53c5070; Control Plane SHA256 235d4c530634ff4d60810f77388738a74d92d5551adeed7d240de86b9f19afa9
RECONSTRUCTION_REASONING_SUMMARY = Control-plane adjudication and final-lock review corrections resolve all 22 previously unresolved protocol-critical items and eliminate all 10 review blockers.
ALTERNATIVES_CONSIDERED = Claim exact historical recovery; maintain unresolved state despite completed adjudication.
WHY_ALTERNATIVES_REJECTED = Historical recovery is unsupported, and adjudication is now complete.
CONFIDENCE = HIGH
```

### R-002 — Block metric authority name

```text
CONTRACT_ID = R-002
CANONICAL_RULE = BLOCK_METRIC_AGGREGATION=SAME_RULES_AS_STAGE and authority.block_metric_aggregation="SAME_RULES_AS_STAGE"; no independent shorter alias
EVIDENCE_PATHS = E-003 lines 437–470; E-001 lines 184–208, 266–304, 550–616; E-009 Cluster E
EVIDENCE_COMMITS = NOT_REPOSITORY_ARTIFACT; NOT_COMMITTED; NOT_COMMITTED
EVIDENCE_BLOBS = work-order SHA256 7ecf23599a5b4650b7bd8e933e5a4a802ffaf554039aa4e981b9df3fd1276126; closure blob 9353a8f4940de4257f6c96ee5e309ff827adfe11; Control Plane blob 7e9c55caea96be92eec64f0e30a1d39af53c5070; Control Plane SHA256 235d4c530634ff4d60810f77388738a74d92d5551adeed7d240de86b9f19afa9
RECONSTRUCTION_REASONING_SUMMARY = The control plane fixes the canonical name, while the closure fixes the unchanged block mathematics and already carries the matching serialized field.
ALTERNATIVES_CONSIDERED = Retain the closure's shorter constant label; maintain two authority aliases.
WHY_ALTERNATIVES_REJECTED = Either alternative preserves formal inconsistency or creates duplicate authority names.
CONFIDENCE = HIGH
```

---

## 4. Control Plane reconstruction decision records

### U-001 — Complete model equation and digit/index mapping

```text
DECISION_ID = U-001
SUBJECT = Complete Digit-Factor model equation and digit/index mapping
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster A
CANONICAL_SECTION = Section 7.1
CANONICAL_RULE = For n in {0..99}, i=floor(n/10), j=n%10 (i,j in {0..9}). X_t,W[i,j] is sum of observed counts for (10*i+j) over W recorded draws strictly preceding t (sum X = 27*W). M3_MODEL_FAMILY = RANK1_DIGIT_INTERACTION_MULTINOMIAL_LOGLINEAR. Latent score: eta_t[i,j] = a_t[i] + b_t[j] + gamma_t * u_t[i] * v_t[j]. Probabilities: p_t[i,j] = exp(eta_t[i,j] - max_eta) / sum_{r,s} exp(eta_t[r,s] - max_eta) (max_eta subtraction is mandatory numerical stabilization). Forecast: mu[t,10*i+j] = 27 * p_t[i,j]. Structurally mu[t,n] > 0 and sum_n mu[t,n] = 27 within tolerance. No clipping, epsilon injection, post-scoring renormalization, or negative-value repair permitted.
```

### U-002 — Parameter definitions, domains, bounds, and identifiability constraints

```text
DECISION_ID = U-002
SUBJECT = Parameter definitions, domains, bounds, and identifiability constraints
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
FINAL_LOCK_REVIEW_CORRECTION = APPLIED
PRIMARY_EVIDENCE = E-009 Cluster A & Review Addendum B-01, B-02
CANONICAL_SECTION = Section 7.2, 7.3
CANONICAL_RULE = a[i] in R, b[j] in R, gamma >= 0. Constraints: sum_i a[i] = 0, sum_j b[j] = 0, sum_i u[i] = 0, sum_j v[j] = 0. Equality residuals: dot(u,u)-1=0, dot(v,v)-1=0. Canonical sign rule: i_star = smallest index attaining max_i |u[i]|; required u[i_star] > 0; otherwise u <- -u, v <- -v. If fitted gamma <= 1e-10, canonicalize to gamma = 0 and u = v = (9,-1,...,-1)/sqrt(90). No finite artificial bounds on a, b, u, v; gamma native lower bound 0.
```

### U-003 — Interaction matrix R construction and double-centering

```text
DECISION_ID = U-003
SUBJECT = Interaction matrix R construction and double-centering
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster A
CANONICAL_SECTION = Section 6.1
CANONICAL_RULE = For candidate window W: q[i,j] = X[i,j] / (27 * W). Marginals: r[i] = sum_j q[i,j], c[j] = sum_i q[i,j]. Required: r[i] > 0 for all i, c[j] > 0 for all j; otherwise fail FAILED_STAGE=MODEL_INITIALIZATION, ERROR_TYPE=ZeroDigitMarginal, DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION. Double-centered matrix: R[i,j] = q[i,j] - r[i] * c[j]. sum_j R[i,j] = 0, sum_i R[i,j] = 0. R is the exact matrix supplied to numpy.linalg.svd(R, full_matrices=False). No smoothing or pseudo-count.
```

### U-004 — Non-degenerate centering and normalization transforms

```text
DECISION_ID = U-004
SUBJECT = Non-degenerate centering and normalization transforms
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster A
CANONICAL_SECTION = Section 6.4
CANONICAL_RULE = Initial main effects: a0[i] = ln(r[i]) - mean_k ln(r[k]), b0[j] = ln(c[j]) - mean_k ln(c[k]). For non-degenerate SVD case (s0 > 1e-10 and relative_svd_gap > 1e-8): start with u_raw = U[:,0], v_raw = Vt[0,:], gamma_raw = s0. Center: u_c = u_raw - mean(u_raw), v_c = v_raw - mean(v_raw). Norms: nu = ||u_c||_2, nv = ||v_c||_2. Required: nu > 1e-12, nv > 1e-12; otherwise fail FAILED_STAGE=MODEL_INITIALIZATION, ERROR_TYPE=CenteredSingularVectorDegeneracy, DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION. Normalize: u = u_c / nu, v = v_c / nv, gamma = gamma_raw * nu * nv. Apply canonical sign rule after centering and normalization. For zero-interaction case (s0 <= 1e-10): gamma = 0, use canonical zero representation directly.
```

### U-005 — Fitting objective, optimizer, convergence, iteration limits, and non-SVD fit failure semantics

```text
DECISION_ID = U-005
SUBJECT = Fitting objective, optimizer, convergence, iteration limits, and non-SVD fit failure semantics
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
FINAL_LOCK_REVIEW_CORRECTION = APPLIED
PRIMARY_EVIDENCE = E-009 Cluster A & Review Addendum B-01, B-02
CANONICAL_SECTION = Section 7.3, 7.4, 7.5
CANONICAL_RULE = Every target/window fit is independent. FIT_INITIALIZATION = EXACT_U004_INITIALIZATION. No prior target fit may initialize any other fit. Prohibit random restart, alternate solver, fallback optimizer, warm-start chain, and tolerance relaxation. Objective: NLL without regularization. Optimization vector theta length 41 and residual c(theta) length 6 use the exact analytic objective and 6x41 constraint Jacobians. Optimizer: scipy.optimize.minimize(method="SLSQP", options={"maxiter":2000,"ftol":1e-12,"disp":False}). Accepted only if optimizer.success=true, objective and all raw parameters finite, raw gamma>=0, equality residual satisfies 1e-10, and final canonicalized forecast satisfies its contract. U-005 failure mapping is FAILED_STAGE=MODEL_FIT and DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION. Error identifiers are exactly OptimizerNonConvergence (optimizer.success != true absent a prior more-specific technical failure), NonFiniteModelFit (non-finite raw objective/parameter), ConstraintViolation (raw gamma<0, max(abs(c(theta_raw)))>1e-10, or frozen-constraint violation), and ForecastContractViolation (non-finite eta/p/mu, mu[n]<=0, or abs(sum(mu)-27)>1e-10). U005_FAILURE_MAPPING takes precedence for MODEL_FIT failures. U-019 supplies only the artifact envelope.
```

### U-006 — M3 dataset identity, preprocessing, date eligibility, and snapshot/hash contract

```text
DECISION_ID = U-006
SUBJECT = M3 dataset identity, preprocessing, date eligibility, and snapshot/hash contract
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
FINAL_LOCK_REVIEW_CORRECTION = APPLIED
PRIMARY_EVIDENCE = E-009 Cluster B & Review Addendum B-03 & U-020 Literal Addendum
CANONICAL_SECTION = Section 8.1
CANONICAL_RULE = DATA_SOURCE_PATH = data/xsmb-2-digits.csv and DATA_SOURCE_REPOSITORY = Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis. For PROV-02 (data_source_repository), SCHEMA_OR_FIELD_EXISTENCE_AUTHORITY is derived from closed authority specification, while EXACT_LITERAL_VALUE_AUTHORITY = CONTROL_PLANE_RECONSTRUCTION_DECISION / U-020 correction addendum, with EVIDENCE_BASIS = CURRENT_CANONICAL_REPOSITORY_IDENTITY; HISTORICAL_RECOVERY_STATUS = NOT_RECOVERED; SURVIVING_AUTHORITY_SELECTION = NO. This is explicit Control Plane M3 authority. Repository-grounded planning may verify this selected authority but may not silently replace either identity. If repository evidence establishes that the repository or canonical dataset authority no longer matches the frozen reconstruction authority: STOP = REPOSITORY_AUTHORITY_MISMATCH and report back to the Control Plane. REPOSITORY_AUTHORITY_MISMATCH = FAIL_CLOSED. SILENT_REPOSITORY_SUBSTITUTION = FORBIDDEN. SILENT_DATA_SOURCE_SUBSTITUTION = FORBIDDEN. No other CSV, loader path, newer-looking file, inferred replacement, alternate repository slug, or change to canonical source may be selected silently. 27 integer outcomes (0..99) per draw. M3 is strictly OBSERVED_DRAW_EVENT_INDEXED, not calendar-completeness inferred. history_window_W counts W recorded draw observations, NOT calendar days. NO_MISSING_DRAW requires all present dates have 27 valid outcomes without null, malformation, imputation, clipping, or repair. Calendar continuity validation is out of scope for M3 model protocol. selected_window_days is a legacy artifact field name meaning COUNT_OF_PRECEDING_RECORDED_DRAW_DATES. Target date eligible iff preceded by at least W_MAX = 365 recorded draws.
```

### U-007 — Development chronology and development/validation split

```text
DECISION_ID = U-007
SUBJECT = Development chronology and development/validation split
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
FINAL_LOCK_REVIEW_CORRECTION = APPLIED
PRIMARY_EVIDENCE = E-009 Cluster B & Review Addendum B-04
CANONICAL_SECTION = Section 8.2
CANONICAL_RULE = Eligible target dates sorted ascending. Required: N >= 240; otherwise fail InsufficientDevelopmentSample. Split: N_dev = floor(N/2), N_val = floor(N/4), N_stability = N - N_dev - N_val. split_fractions = [0.5, 0.25, 0.25] represents NOMINAL_DESIGN_FRACTIONS; actual integer counts defined by floor formulas have normative precedence. DEV, VAL, STABILITY are contiguous and strictly held out.
```

### U-008 — Stability cohort and exact N_stability

```text
DECISION_ID = U-008
SUBJECT = Stability cohort and exact N_stability
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster B
CANONICAL_SECTION = Section 8.3
CANONICAL_RULE = N_stability = N - floor(N/2) - floor(N/4). For N >= 240, N_stability >= 60. When economics is evaluated, economic_uncertainty.per_k[*].date_count = N_stability exactly.
```

### U-009 — Six stability-block boundaries and assignment rule

```text
DECISION_ID = U-009
SUBJECT = Six stability-block boundaries and assignment rule
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster B
CANONICAL_SECTION = Section 8.4
CANONICAL_RULE = STABILITY_BLOCK_COUNT = 6. Let q = N_stability // 6, r = N_stability mod 6. Blocks 1..r contain q+1 dates; blocks r+1..6 contain q dates. Assignments are chronological, contiguous, without overlap or omission. Every block contains at least 10 dates (since N_stability >= 60).
```

### U-010 — Baseline B0 model, fitting, and forecast contract

```text
DECISION_ID = U-010
SUBJECT = Baseline B0 model, fitting, and forecast contract
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster C
CANONICAL_SECTION = Section 9.1
CANONICAL_RULE = BASELINE_ID = B0_UNIFORM. For all dates and outcomes n in {0..99}: mu_B0[t,n] = 27/100 = 0.27. Exactly satisfies forecast preconditions: sum_n mu_B0 = 27, mu_B0 > 0. No parameters, history window, or optimization.
```

### U-011 — M3 candidate family, hyperparameter space, and parameter bounds

```text
DECISION_ID = U-011
SUBJECT = M3 candidate family, hyperparameter space, and parameter bounds
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster C
CANONICAL_SECTION = Section 9.2
CANONICAL_RULE = Single hyperparameter: history_window_W in {30, 60, 120, 240, 365}. Exactly 5 candidates: M3_W030, M3_W060, M3_W120, M3_W240, M3_W365. No other hyperparameter, sweep, regularization parameter, or multi-window ensemble permitted.
```

### U-012 — Candidate selection, ordering, and tie-breaking

```text
DECISION_ID = U-012
SUBJECT = Candidate selection, ordering, and tie-breaking
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster C
CANONICAL_SECTION = Section 9.3
CANONICAL_RULE = Candidate selection performed on DEV stage only. Candidate minimizing lexicographic tuple: (PD_DEV, MAE_DEV, RMSE_DEV, W). Exactly one winning candidate selected. Selected candidate is frozen for VAL and STABILITY without retuning.
```

### U-013 — Complete FORECAST_SIGNAL qualification predicate

```text
DECISION_ID = U-013
SUBJECT = Complete FORECAST_SIGNAL qualification predicate
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster C
CANONICAL_SECTION = Section 9.4
CANONICAL_RULE = Evaluated on VAL stage for selected M3 candidate: PRIMARY_FORECAST_PASS iff PD_VAL_M3 < PD_VAL_B0; DEV_VAL_SECONDARY_PASS iff MAE_VAL_M3 <= MAE_VAL_B0 OR RMSE_VAL_M3 <= RMSE_VAL_B0; FORECAST_BOOTSTRAP_PASS iff forecast_bootstrap_lower_bound > 0. FORECAST_SIGNAL = true iff all three conditions pass.
```

### U-014 — Forecast-bootstrap statistic, resampling, replications, block parameters, interval/quantile rule, multiplicity, and qualification

```text
DECISION_ID = U-014
SUBJECT = Forecast-bootstrap statistic, resampling, replications, block parameters, interval/quantile rule, multiplicity, and qualification
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster C
CANONICAL_SECTION = Section 9.5
CANONICAL_RULE = Daily differences d_t = PD_t_B0 - PD_t_M3 on VAL. SURVIVING_PRESERVED_RNG_AUTHORITY = forecast_bootstrap_rng_implementation="numpy.random.Generator"; forecast_bootstrap_bit_generator="PCG64"; forecast_bootstrap_seed=20260831. CONTROL_PLANE_SELECTED_BOOTSTRAP_PROTOCOL_CONSTANTS = forecast_bootstrap_replications=2000; forecast_bootstrap_mean_block_length=30; forecast_bootstrap_restart_probability=1/30; forecast_bootstrap_alpha=0.05; forecast_bootstrap_quantile_method="linear". Instantiate Generator(PCG64(20260831)) exactly once before replicate 1. For each replicate: draw idx[0] with rng.integers(0,N_val); for m=1..N_val-1 draw u=rng.random(), draw rng.integers(0,N_val) only if u<1/30, otherwise continue (idx[m-1]+1) mod N_val. Do not reset between replicates. D_b is the mean over idx. The lower bound is numpy.quantile([D_1,...,D_2000],0.05,method="linear").
```

### U-015 — Economic-delta construction, sign, aggregation unit, top-K selection/ties, and payout/cost mapping

```text
DECISION_ID = U-015
SUBJECT = Economic-delta construction, sign, aggregation unit, top-K selection/ties, and payout/cost mapping
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster D
CANONICAL_SECTION = Section 10.1
CANONICAL_RULE = Portfolios K in {1, 3, 5, 10}. Outcomes sorted descending by mu_t[n], ties broken by smaller n. Select top K. Unit: thousand VND. Cost = 27.0, payout = 99.0 per hit. Net outcome PnL = 99*y - 27. Daily delta = sum_{k in top_K} PnL[t, k]. Complete aggregation formulas: mean_economic_delta[K] = (1 / N_stability) * sum_{t in STABILITY} economic_delta[t, K]; block_mean_economic_delta[b, K] = (1 / |B_b|) * sum_{t in B_b} economic_delta[t, K]; positive_block_count[K] = sum_{b=1..6} I(block_mean_economic_delta[b, K] > 0). Explicit invariant: mean_economic_delta[K] is the arithmetic mean across ALL STABILITY DATES, NOT the unweighted arithmetic mean of the six block means. Strict positive block rule: block mean delta > 0 strictly.
```

### U-016 — Positive-state recommended_top_k construction and ordering

```text
DECISION_ID = U-016
SUBJECT = Positive-state recommended_top_k construction and ordering
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
FINAL_LOCK_REVIEW_CORRECTION = APPLIED
PRIMARY_EVIDENCE = E-009 Cluster D & Review Addendum B-09
CANONICAL_SECTION = Section 10.2
CANONICAL_RULE = Ranking applies strictly to qualified_top_k. If qualified_top_k == []: recommended_top_k = []. Else select single K maximizing lexicographic tuple: (bootstrap_lower_bound, mean_economic_delta, -K); recommended_top_k = [chosen_K]. Invariant: recommended_top_k subset_of qualified_top_k, length <= 1. economic_summary.csv.recommended = true iff K is chosen recommended portfolio.
```

### U-017 — Forecast-sum and metric-serialization tolerances

```text
DECISION_ID = U-017
SUBJECT = Forecast-sum and metric-serialization tolerances
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = NO
PRIMARY_EVIDENCE = E-009 Cluster E
CANONICAL_SECTION = Section 5.8
CANONICAL_RULE = FORECAST_SUM_TOLERANCE = 1e-10; ARTIFACT_NUMERIC_ABS_TOLERANCE = 1e-12; ARTIFACT_NUMERIC_REL_TOLERANCE = 1e-12; comparison is semantically equivalent to math.isclose(reconstructed, serialized, rel_tol=1e-12, abs_tol=1e-12); CSV_FLOAT_FORMAT = .17g; ALL_SCIENTIFIC_QUALIFICATION = UNROUNDED_IN_MEMORY_BINARY64_VALUES. JSON serialization uses allow_nan=false and is reporting/reconstruction evidence only.
```

### U-018 — Complete success-artifact schemas and required success-artifact inventory

```text
DECISION_ID = U-018
SUBJECT = Complete success-artifact schemas and required success-artifact inventory
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = NO
FINAL_LOCK_REVIEW_CORRECTION = APPLIED
PRIMARY_EVIDENCE = E-009 Cluster E, Review Addendum B-05, B-07, and U-018 Row Order Addendum
CANONICAL_SECTION = Section 11.1
CANONICAL_RULE = Exactly 9 artifacts: protocol_snapshot.json, daily_forecast_scores.csv.gz, forecast_metrics.csv, forecast_uncertainty.json, stability_diagnostics.json, economic_uncertainty.json, economic_summary.csv, development_adjudication.json, artifact_manifest.json. The four formerly abbreviated JSON schemas are closed with explicit types, nullability, and ordering in canonical Section 11.1. Gzip is one DEFLATE member at level 9 with fixed first bytes 1f 8b 08 00 00 00 00 00 02 ff. artifact_manifest contains exactly the other eight success artifacts and excludes itself. forecast_metrics.csv has deterministic three-tier row order (stage order DEV, VAL, STABILITY; within stage model_id ascending lexicographic; within same stage and model_id candidate_id ascending lexicographic) established by U-018 row-order addendum.
```

### U-019 — General failure artifact schema, filename mapping, and technical failure/exit semantics outside the SVD near-tie case

```text
DECISION_ID = U-019
SUBJECT = General failure artifact schema, filename mapping, and technical failure/exit semantics outside the SVD near-tie case
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = NO
FINAL_LOCK_REVIEW_CORRECTION = APPLIED
PRIMARY_EVIDENCE = E-009 Cluster E & Review Addendum B-08
CANONICAL_SECTION = Section 11.2
CANONICAL_RULE = Exactly development_run_FAILED.json. Top-level keys: status="FAILED", failed_stage, error_type, development_exit_status, protocol_snapshot. failed_stage and development_exit_status are each closed exact enums in canonical Section 11.2; error_type is a non-empty stable protocol identifier, not a globally closed enum. protocol_snapshot type is object | null. Before AUTHORITY_COMPLETE it is null; after AUTHORITY_COMPLETE it contains complete authority and valid fingerprint. The SVD ambiguity triplet is MODEL_INITIALIZATION / SVDLeadingSubspaceAmbiguity / NEEDS_MODEL_REVISION.
```

### U-020 — Closed-world authority key set, JSON types, exact encodings, and fingerprint coverage for all base M3 constants

```text
DECISION_ID = U-020
SUBJECT = Closed-world authority key set, JSON types, exact encodings, and fingerprint coverage for all base M3 constants
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = NO
FINAL_LOCK_REVIEW_CORRECTION = APPLIED
PRIMARY_EVIDENCE = E-009 Cluster E, Review Addendum B-01, B-07, and U-020 Literal Authority Addendum (PROV-01, PROV-02)
CANONICAL_SECTION = Section 4.2
CANONICAL_RULE = Closed set of exactly 72 authority keys inside protocol_snapshot.authority. All 72 keys participate in protocol fingerprint. optimizer_constraint_tolerance = 1e-10 represents POST_SOLVER_MAX_ABS_EQUALITY_RESIDUAL_TOLERANCE (independent post-solver acceptance gate, not SLSQP tol option). All keys verified non-null and typed. The U-020 authority framework explicitly distinguishes SCHEMA_OR_FIELD_EXISTENCE_AUTHORITY from EXACT_LITERAL_VALUE_AUTHORITY: for PROV-01 (authority_provenance), EXACT_LITERAL_VALUE_AUTHORITY = CONTROL_PLANE_RECONSTRUCTION_DECISION / U-020 correction addendum, HISTORICAL_RECOVERY_STATUS = NOT_RECOVERED, SURVIVING_AUTHORITY_SELECTION = NO; for PROV-02 (data_source_repository), EXACT_LITERAL_VALUE_AUTHORITY = CONTROL_PLANE_RECONSTRUCTION_DECISION / U-020 correction addendum, EVIDENCE_BASIS = CURRENT_CANONICAL_REPOSITORY_IDENTITY, HISTORICAL_RECOVERY_STATUS = NOT_RECOVERED, SURVIVING_AUTHORITY_SELECTION = NO; for spec_version, SCHEMA_MEMBERSHIP_AND_TYPE_AUTHORITY = CONTROL_PLANE_RECONSTRUCTION_DECISION / U-020, while EXACT_LITERAL_VALUE = XPIS_V3_M3_DIGIT_FACTOR_DESIGN_V3 and EXACT_LITERAL_VALUE_AUTHORITY = PRESERVED_FROM_SURVIVING_CLOSURE (E-001:5).
```

### U-023 — Complete economic-bootstrap resampling and lower-bound estimator contract

```text
DECISION_ID = U-023
SUBJECT = Complete economic-bootstrap resampling and lower-bound estimator contract
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster D
CANONICAL_SECTION = Section 10.3
CANONICAL_RULE = Daily deltas Delta_t[K] on STABILITY. Instantiate Generator(PCG64(20260832)) once. For each replicate draw exactly one N_stability index vector with idx[0]=rng.integers(0,N_stability), then for m=1..N_stability-1 consume rng.random() before conditionally consuming rng.integers(0,N_stability) only on restart; otherwise continue circularly. Do not reset between K or replicates. Apply that same vector to K=1,3,5,10. Per-K lower bound is numpy.quantile([M_1[K],...,M_2000[K]],0.0125,method="linear").
```

### U-024 — Evaluated-state aggregate economic adjudication

```text
DECISION_ID = U-024
SUBJECT = Evaluated-state aggregate economic adjudication
STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
FINAL_STATUS = CONTROL_PLANE_RECONSTRUCTION_DECISION
AUTHORITY_ORIGIN = CONTROL_PLANE_RECONSTRUCTION_DECISION
SCIENTIFIC_REREVIEW_REQUIRED = YES
PRIMARY_EVIDENCE = E-009 Cluster D
CANONICAL_SECTION = Section 10.4
CANONICAL_RULE = When economic_uncertainty.status = "EVALUATED": qualified_top_k = [K for K in [1,3,5,10] if per_k[K].qualifies == true] ordered canonically [1,3,5,10]. economic_signal = (len(qualified_top_k) > 0). If none qualify: economic_signal = false, qualified_top_k = [], recommended_top_k = []. If any qualify: economic_signal = true, and U-016 selects recommended_top_k. Preserved per-K rule: qualifies = true iff mean_economic_delta > 0 AND positive_block_count >= 5 AND bootstrap_lower_bound > 0.
```

---

## 5. Deferred lifecycle and release dispositions

```text
DECISION_ID = U-021
SUBJECT = Run-bound source_commit and source_tree values
STATUS = DEFERRED_RUN_BOUND_VALUE
FINAL_STATUS = DEFERRED_RUN_BOUND_VALUE
AUTHORITY_ORIGIN = PRESERVED_DESIGN_REPRESENTATION
PROTOCOL_CRITICAL_AT_CURRENT_DESIGN_GATE = NO
CONTROL_PLANE_DECISION_REQUIRED_NOW = NO
CANONICAL_SECTION = Section 4.2
CANONICAL_RULE = authority.source_commit and authority.source_tree remain mandatory fields included in the protocol fingerprint; exact values are frozen to the approved implementation commit/tree before historical V3 execution.
AVAILABLE_EVIDENCE = E-001 lines 515–548; E-003 scope guard; E-009 Cluster E
POSSIBLE_VALUES = Exact future approved implementation commit and tree, selected at the later approved implementation/release gate
```

```text
DECISION_ID = U-022
SUBJECT = Implementation/release reproducibility environment
STATUS = DEFERRED_IMPLEMENTATION_RELEASE_CONCERN
FINAL_STATUS = DEFERRED_IMPLEMENTATION_RELEASE_CONCERN
AUTHORITY_ORIGIN = PRESERVED_DESIGN_REPRESENTATION
PROTOCOL_CRITICAL_AT_CURRENT_DESIGN_GATE = NO
CONTROL_PLANE_DECISION_REQUIRED_NOW = NO
CANONICAL_SECTION = Section 4.2
CANONICAL_RULE = Scientific API/RNG determinism is preserved; runtime, dependency, numerical-backend, hardware, thread, and environment-record details may be recorded or frozen at the implementation/release gate without becoming implicit M3 design authority.
AVAILABLE_EVIDENCE = E-001 lines 306–340, 618–639, and 891–927; repository package metadata is background only; E-009 Cluster E
POSSIBLE_VALUES = Later implementation/release record; no value selected here
```

---

## 6. Ledger totals and gate effect

```text
PRESERVED_CONTRACT_COUNT = 38
RECONSTRUCTED_CONTRACT_COUNT = 2
CONTROL_PLANE_DECISION_COUNT = 22
DEFERRED_DISPOSITION_COUNT = 2
NOT_APPLICABLE_COUNT = 5
TOTAL_LEDGER_ROWS = 69

INPUT_PROTOCOL_CRITICAL_UNRESOLVED_COUNT = 22
RESOLVED_COUNT = 22
REMAINING_PROTOCOL_CRITICAL_UNRESOLVED_COUNT = 0
REVIEW_BLOCKERS_RESOLVED = 10
HIDDEN_IMPLEMENTATION_DECISION_COUNT = 0
SURVIVING_AUTHORITY_SEMANTIC_DRIFT = NONE

SURVIVING_SCIENTIFIC_CONTENT_REVIEW = PASS_UNCHANGED
NEW_CONTROL_PLANE_AUTHORITY_SCIENTIFIC_REVIEW = REQUIRED
TARGETED_SCIENTIFIC_REVIEW = PASS

SURVIVING_V3_AUTHORITY_PRESERVED = PASS
BLOCK_METRIC_AUTHORITY_NORMALIZED = PASS
AUTHORITY_KEY_NORMALIZED = PASS
STALE_EXTERNAL_V2_NORMATIVE_DEPENDENCY = NONE
AMBIGUOUS_CANONICAL_DUPLICATE = NONE
DOCS_ONLY_SCOPE = PASS
EVIDENCE_IDENTITY_CHANGED = NO
LOCATOR_REPRESENTATION_REPAIRED = YES

AUTHORITY_SCHEMA_COMPLETE = PASS
AUTHORITY_KEY_COUNT = 72
ARTIFACT_SCHEMA_COMPLETE = PASS

CONTROL_PLANE_RECONSTRUCTION_ADJUDICATION = COMPLETE
CONTROL_PLANE_FINAL_LOCK_CORRECTION = COMPLETE
FORMAL_ARTIFACT_REVIEW = READY_FOR_FINAL_INDEPENDENT_RECHECK
FORMAL_SPEC_LOCK = NOT_AUTHORIZED
REPO_GROUNDED_PLANNING = NOT_AUTHORIZED
SOURCE_IMPLEMENTATION = NOT_AUTHORIZED
V3_HISTORICAL_RUN = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
HISTORICAL_EXECUTION = NOT_AUTHORIZED
PREREGISTRATION = NOT_AUTHORIZED
CONFIRMATORY_EXECUTION = NOT_AUTHORIZED
```

### Final fidelity repair (FLR-01..FLR-04)

```text
FLR-01 = PASS
FLR-02 = PASS
FLR-03 = PASS
FLR-04 = PASS
P001_TOLERANCE_PROVENANCE = PASS
P001_BINARY64_PROVENANCE = PASS
U017_PROVENANCE_FIDELITY = PASS
SURVIVING_CLOSURE_SELECTS_NUMERIC_FORECAST_SUM_TOLERANCE = NO
CONTROL_PLANE_U017_SELECTS_1E_MINUS_10 = YES
SURVIVING_CLOSURE_REQUIRES_UNROUNDED_VALUES = YES
SURVIVING_CLOSURE_SELECTS_BINARY64 = NO
CONTROL_PLANE_U017_SELECTS_BINARY64 = YES
U006_CANONICAL_FIDELITY = PASS
REPOSITORY_AUTHORITY_MISMATCH_RULE_PRESENT = PASS
SILENT_DATA_SOURCE_SUBSTITUTION_FORBIDDEN = PASS
U014_PROVENANCE_CLASSIFICATION = PASS
FORECAST_BOOTSTRAP_CP_PROVENANCE_MISMATCH_COUNT = 0
CONTROL_PLANE_DECISIONS_LOCAL_FIDELITY = 22/22
CONTROL_PLANE_AUTHORITY_LITERAL_MISMATCH_COUNT = 0
SURVIVING_AUTHORITY_LITERAL_MISMATCH_COUNT = 0
SURVIVING_AUTHORITY_SEMANTIC_DRIFT = NONE
HIDDEN_IMPLEMENTATION_DECISION_COUNT = 0
REMAINING_PROTOCOL_CRITICAL_UNRESOLVED_COUNT = 0
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
```
