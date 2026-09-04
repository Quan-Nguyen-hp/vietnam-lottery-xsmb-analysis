# CHATGPT CONTROL PLANE DECISION MEMORANDUM

## XPIS v3 M3 Digit-Factor — Reconstruction Authority Adjudication

```text
CONTROL_PLANE_RECONSTRUCTION_ADJUDICATION =
COMPLETE

INPUT_PROTOCOL_CRITICAL_UNRESOLVED_COUNT =
22

RESOLVED_COUNT =
22

REMAINING_PROTOCOL_CRITICAL_UNRESOLVED_COUNT =
0

SURVIVING_SCIENTIFIC_CONTENT_REVIEW =
PASS_UNCHANGED

NEW_CONTROL_PLANE_AUTHORITY_SCIENTIFIC_REVIEW =
REQUIRED

FORMAL_SPEC_LOCK =
NOT_AUTHORIZED

REPO_GROUNDED_PLANNING =
NOT_AUTHORIZED

SOURCE_IMPLEMENTATION =
NOT_AUTHORIZED

V3_HISTORICAL_RUN =
NOT_AUTHORIZED
```

Every newly selected contract below has:

```text
AUTHORITY_ORIGIN =
CONTROL_PLANE_RECONSTRUCTION_DECISION
```

unless explicitly identified as surviving preserved authority.

Do NOT represent any of these decisions as recovered historical V2 authority.

---

# CLUSTER A — M3 MATHEMATICAL MODEL

## U-001 — Complete model equation and digit/index mapping

For:

```text
n ∈ {0,...,99}
```

define:

```text
i = floor(n / 10)
j = n mod 10
```

where:

```text
i = tens digit
j = ones digit
i,j ∈ {0,...,9}
```

For target date `t` and history window `W`:

```text
X_t,W[i,j]
=
sum of observed counts for number (10*i+j)
over the W recorded draw dates strictly preceding t
```

Therefore:

```text
sum_i sum_j X_t,W[i,j] = 27 * W
```

M3 is:

```text
M3_MODEL_FAMILY =
RANK1_DIGIT_INTERACTION_MULTINOMIAL_LOGLINEAR
```

with latent score:

```text
eta_t[i,j]
=
a_t[i]
+
b_t[j]
+
gamma_t * u_t[i] * v_t[j]
```

Probability:

```text
p_t[i,j]
=
exp(eta_t[i,j] - max_eta)
/
sum_r sum_s exp(eta_t[r,s] - max_eta)
```

`max_eta` subtraction is mandatory numerical stabilization only.

Forecast:

```text
mu[t,10*i+j]
=
27 * p_t[i,j]
```

Thus structurally:

```text
mu[t,n] > 0

sum_n mu[t,n] = 27
within floating-point tolerance
```

No clipping, epsilon injection, post-scoring renormalization, or negative-value repair is permitted.

```text
U-001 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-002 — Parameter domains, constraints and identifiability

Freeze:

```text
a[i] ∈ R
b[j] ∈ R

gamma >= 0
```

Constraints:

```text
sum_i a[i] = 0
sum_j b[j] = 0

sum_i u[i] = 0
sum_j v[j] = 0
```

For non-zero interaction:

```text
||u||_2 = 1
||v||_2 = 1
gamma > 0
```

Preserve canonical sign authority:

```text
i_star =
smallest index attaining max_i |u[i]|

u[i_star] > 0
```

otherwise:

```text
u <- -u
v <- -v
```

If fitted:

```text
gamma <= INTERACTION_ZERO_TOL
```

canonicalize to:

```text
gamma = 0

u = v =
(9,-1,-1,-1,-1,-1,-1,-1,-1,-1)
/
sqrt(90)
```

Preserve:

```text
INTERACTION_ZERO_TOL = 1e-10
```

No finite artificial bounds on `a` or `b`.

```text
U-002 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-003 — Interaction matrix R construction

For candidate window `W`:

```text
q[i,j]
=
X[i,j] / (27 * W)
```

Digit marginals:

```text
r[i] = sum_j q[i,j]

c[j] = sum_i q[i,j]
```

Required:

```text
r[i] > 0 for all i
c[j] > 0 for all j
```

Otherwise:

```text
FAILED_STAGE = MODEL_INITIALIZATION
ERROR_TYPE = ZeroDigitMarginal
DEVELOPMENT_EXIT_STATUS = NEEDS_MODEL_REVISION
```

Define:

```text
R[i,j]
=
q[i,j] - r[i] * c[j]
```

Therefore:

```text
sum_j R[i,j] = 0
sum_i R[i,j] = 0
```

`R` is exactly the matrix supplied to:

```text
numpy.linalg.svd(
    R,
    full_matrices=False
)
```

No smoothing or pseudo-count is used in `R`.

```text
U-003 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-004 — Centering and normalization transforms

Initial main effects:

```text
a0[i]
=
ln(r[i])
-
mean_k ln(r[k])

b0[j]
=
ln(c[j])
-
mean_k ln(c[k])
```

For the surviving non-degenerate SVD case:

```text
s0 > 1e-10

AND

relative_svd_gap > 1e-8
```

start with:

```text
u_raw = U[:,0]
v_raw = Vt[0,:]
gamma_raw = s0
```

Then:

```text
u_c = u_raw - mean(u_raw)
v_c = v_raw - mean(v_raw)

nu = ||u_c||_2
nv = ||v_c||_2
```

Required:

```text
nu > 1e-12
nv > 1e-12
```

Otherwise:

```text
FAILED_STAGE = MODEL_INITIALIZATION
ERROR_TYPE = CenteredSingularVectorDegeneracy
DEVELOPMENT_EXIT_STATUS = NEEDS_MODEL_REVISION
```

Normalize:

```text
u = u_c / nu
v = v_c / nv

gamma =
gamma_raw * nu * nv
```

Apply the surviving canonical sign rule after centering and normalization.

For the surviving zero-interaction SVD case:

```text
gamma = 0
```

and use the canonical zero representation directly.

```text
U-004 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-005 — Fit objective, optimizer, convergence and failures

Fit each target forecast independently from its trailing window.

No warm-start chain has protocol authority.

Objective:

```text
NLL
=
- sum_i sum_j
X[i,j] * ln(p[i,j])
```

No regularization.

Freeze:

```text
OPTIMIZER_IMPLEMENTATION =
scipy.optimize.minimize

OPTIMIZER_METHOD =
SLSQP

OPTIMIZER_MAX_ITERATIONS =
2000

OPTIMIZER_FTOL =
1e-12

OPTIMIZER_CONSTRAINT_TOL =
1e-10
```

Initialization is exactly U-004.

Successful fit requires:

```text
optimizer.success = true

objective finite

all parameters finite

all equality constraints satisfied
within 1e-10

gamma >= 0

forecast contract satisfied
```

Forbidden:

```text
random restart
alternate solver
fallback optimizer
warm-start chain
tolerance relaxation
```

Failure:

```text
FAILED_STAGE = MODEL_FIT
DEVELOPMENT_EXIT_STATUS = NEEDS_MODEL_REVISION
```

with stable `ERROR_TYPE` chosen as applicable from:

```text
OptimizerNonConvergence
NonFiniteModelFit
ConstraintViolation
ForecastContractViolation
```

```text
U-005 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

# CLUSTER B — DATA / CHRONOLOGY / STABILITY

## U-006 — Dataset / preprocessing / eligibility / snapshot

Adopt:

```text
DATA_SOURCE_PATH =
data/xsmb-2-digits.csv
```

If repository-grounded planning later verifies this path is no longer canonical:

```text
STOP = REPOSITORY_AUTHORITY_MISMATCH
```

No silent substitution.

Each observation date contains exactly 27 integer outcomes:

```text
0 <= draw <= 99
```

Convert to:

```text
C[t,n]
```

with:

```text
sum_n C[t,n] = 27
```

Required:

```text
dates unique
dates strictly increasing
no missing draw
no imputation
no clipping
no silent row deletion
no silent repair
```

Forecast history:

```text
history_date < target_date
```

A target date is eligible only if it has at least:

```text
W_MAX = 365
```

preceding recorded draw dates.

Before historical execution record:

```text
data_source_repository
data_source_path
data_source_commit
data_source_blob
data_source_sha256
```

Exact source object values are run-bound.

```text
U-006 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-007 — Development chronology and split

Let eligible target dates after the 365-history requirement be sorted ascending.

Let:

```text
N =
number of eligible target dates
```

Required:

```text
N >= 240
```

Otherwise:

```text
FAILED_STAGE = DATA_VALIDATION
ERROR_TYPE = InsufficientDevelopmentSample
DEVELOPMENT_EXIT_STATUS = NEEDS_DATA_REVISION
```

Define:

```text
N_dev = floor(N / 2)

N_val = floor(N / 4)

N_stability =
N - N_dev - N_val
```

Stages:

```text
DEV =
first N_dev eligible dates

VAL =
next N_val eligible dates

STABILITY =
remaining final N_stability dates
```

No shuffle.

No stratification.

No random split.

Candidate selection uses DEV only.

VAL remains untouched until candidate freeze.

STABILITY remains untouched until `FORECAST_SIGNAL` is adjudicated.

Walk-forward historical observations strictly preceding each target may be used.

```text
U-007 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-008 — N_stability

Freeze:

```text
N_stability =
N - floor(N/2) - floor(N/4)
```

With `N >= 240`:

```text
N_stability >= 60
```

When economics is evaluated:

```text
economic_uncertainty.per_k[*].date_count =
N_stability
```

exactly.

```text
U-008 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-009 — Six stability blocks

Freeze:

```text
STABILITY_BLOCK_COUNT = 6
```

Let:

```text
q = N_stability // 6
r = N_stability mod 6
```

Blocks `1..r` contain:

```text
q + 1
```

dates.

Blocks `r+1..6` contain:

```text
q
```

dates.

Assignments are chronological and contiguous.

No overlap.

No omission.

Given `N_stability >= 60`, every block has at least 10 dates.

```text
U-009 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

# CLUSTER C — BASELINE / CANDIDATES / FORECAST GATE

## U-010 — B0 baseline

Freeze:

```text
B0 =
UNIFORM_COUNT_BASELINE
```

with:

```text
mu_B0[t,n] =
27 / 100 =
0.27
```

for every target and number.

B0 has:

```text
no fitted parameter
no historical window
no optimization
```

This is now explicit M3 Control Plane authority.

Do not represent it as recovered M3 V2 authority.

```text
U-010 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-011 — Candidate family

M3 V3 has exactly one development hyperparameter:

```text
history_window_W
```

Candidate set:

```text
W ∈ {
    30,
    60,
    120,
    240,
    365
}
```

Candidate IDs:

```text
M3_W030
M3_W060
M3_W120
M3_W240
M3_W365
```

Forbidden additional sweeps:

```text
interaction rank
regularization
optimizer
link function
SVD implementation
hidden hyperparameters
```

```text
U-011 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-012 — Candidate selection and tie-breaking

Compute DEV metrics on the identical DEV target-date set.

Select one candidate by ascending tuple:

```text
(
    PD_DEV,
    MAE_DEV,
    RMSE_DEV,
    W
)
```

Priority:

1. lowest DEV Poisson deviance;
2. lowest DEV MAE;
3. lowest DEV RMSE;
4. smaller W.

No tolerance in candidate comparison.

No human discretion.

VAL must not influence candidate selection.

```text
U-012 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-013 — Complete FORECAST_SIGNAL

On VAL define:

```text
PRIMARY_FORECAST_PASS
iff

PD_VAL_M3 < PD_VAL_B0
```

Preserve secondary comparison semantics:

```text
DEV_VAL_SECONDARY_PASS
iff

MAE_VAL_M3 <= MAE_VAL_B0

OR

RMSE_VAL_M3 <= RMSE_VAL_B0
```

The historical name is retained, but its stage authority here is the VAL evaluation of the DEV-selected candidate.

Define:

```text
FORECAST_BOOTSTRAP_PASS
iff

forecast_bootstrap_lower_bound > 0
```

Complete gate:

```text
FORECAST_SIGNAL =
PRIMARY_FORECAST_PASS

AND

DEV_VAL_SECONDARY_PASS

AND

FORECAST_BOOTSTRAP_PASS
```

No additional forecast condition.

```text
U-013 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-014 — Forecast bootstrap

For each VAL date:

```text
d_t =
PD_t_B0 - PD_t_M3
```

Positive means M3 improves Poisson deviance.

Freeze:

```text
FORECAST_BOOTSTRAP_REPLICATIONS =
2000

FORECAST_BOOTSTRAP_MEAN_BLOCK_LENGTH =
30

FORECAST_BOOTSTRAP_RESTART_PROBABILITY =
1/30

FORECAST_BOOTSTRAP_ALPHA =
0.05

FORECAST_BOOTSTRAP_QUANTILE_METHOD =
linear
```

Preserve exact RNG:

```text
numpy.random.Generator(
    numpy.random.PCG64(20260831)
)
```

For one replicate of length `N_val`:

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

Instantiate RNG exactly once before replicate 1.

Do not reset between replicates.

For each replicate:

```text
D_b =
mean of d_t over bootstrap indices
```

Lower bound:

```text
forecast_bootstrap_lower_bound =
numpy.quantile(
    [D_1,...,D_2000],
    0.05,
    method="linear"
)
```

Forbidden:

```text
studentization
BCa
bias correction
alternative interval
```

```text
U-014 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

# CLUSTER D — ECONOMIC PROTOCOL

## U-015 — Economic construction and top-K

Freeze:

```text
K_VALUES =
[1,3,5,10]
```

For each STABILITY date and K:

rank all 100 numbers by:

```text
mu_M3[t,n]
```

descending.

Exact ties:

```text
smaller numeric n first
```

Select first K.

Economic units:

```text
thousand VND
```

Freeze:

```text
COST_PER_NUMBER =
27

PAYOUT_PER_HIT =
99
```

Per selected number:

```text
PnL[t,n]
=
99 * y[t,n] - 27
```

Multiplicity is preserved.

Daily economic delta:

```text
economic_delta[t,K]
=
sum over selected n
(
    99 * y[t,n] - 27
)
```

`economic_delta` is net daily PnL, not a B0-relative delta.

Mean:

```text
mean_economic_delta[K]
=
mean_t economic_delta[t,K]
```

For each stability block:

```text
block_mean_economic_delta[b,K]
=
mean economic_delta[t,K]
within block b
```

Then:

```text
positive_block_count[K]
=
count_b(
    block_mean_economic_delta[b,K] > 0
)
```

Strict `> 0`.

```text
U-015 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-016 — recommended_top_k

Determine U-024 `qualified_top_k` first.

If:

```text
qualified_top_k = []
```

then:

```text
recommended_top_k = []
```

Otherwise choose one K by descending tuple:

```text
(
    bootstrap_lower_bound,
    mean_economic_delta,
    -K
)
```

Priority:

1. highest bootstrap lower bound;
2. highest mean economic delta;
3. smaller K.

Representation:

```text
recommended_top_k =
[chosen_K]
```

exactly one element.

```text
U-016 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-023 — Economic stationary bootstrap

Preserve:

```text
bootstrap_rng_implementation =
numpy.random.Generator

bootstrap_bit_generator =
PCG64

bootstrap_seed =
20260832

bootstrap_replications =
2000

mean_block_length =
30

restart_probability =
1/30

quantile_method =
linear

shared_resample_indices =
true

familywise_alpha =
0.05

multiplicity_method =
BONFERRONI

per_k_alpha =
0.0125
```

Instantiate once:

```text
rng =
numpy.random.Generator(
    numpy.random.PCG64(20260832)
)
```

For each replicate of length `N_stability`, generate indices with the same stationary circular rule as U-014, with restart probability `1/30`.

The same index vector for one replicate is used for all four K values.

This is the exact meaning of:

```text
shared_resample_indices = true
```

Do not reset RNG between K values or replicates.

For each replicate and K:

```text
M_b[K]
=
mean economic_delta[t,K]
over shared bootstrap indices
```

Then:

```text
bootstrap_lower_bound[K]
=
numpy.quantile(
    [M_1[K],...,M_2000[K]],
    0.0125,
    method="linear"
)
```

This is a one-sided Bonferroni-adjusted percentile lower bound.

Forbidden:

```text
studentization
BCa
basic bootstrap interval
bias correction
independent per-K resampling
```

```text
U-023 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

## U-024 — Aggregate evaluated-state economic adjudication

When:

```text
economic_uncertainty.status =
EVALUATED
```

define:

```text
qualified_top_k =
[
    K
    for K in [1,3,5,10]
    if per_k[K].qualifies = true
]
```

Ordering is always canonical ascending:

```text
[1,3,5,10]
```

Define:

```text
economic_signal =
(len(qualified_top_k) > 0)
```

If none qualify:

```text
economic_signal = false
qualified_top_k = []
recommended_top_k = []
```

If any qualify:

```text
economic_signal = true
```

and U-016 selects the recommendation.

Preserve exact per-K rule:

```text
qualifies = true

iff

mean_economic_delta > 0

AND

positive_block_count >= 5

AND

bootstrap_lower_bound > 0
```

No additional per-K qualification condition.

```text
U-024 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = YES
```

---

# CLUSTER E — NUMERICAL / ARTIFACT / AUTHORITY CONTRACT

## U-017 — Numerical and serialization tolerances

Freeze:

```text
FORECAST_SUM_TOLERANCE =
1e-10
```

Required:

```text
abs(sum_n mu[t,n] - 27)
<= 1e-10
```

Artifact metric reconstruction:

```text
ARTIFACT_NUMERIC_ABS_TOLERANCE =
1e-12

ARTIFACT_NUMERIC_REL_TOLERANCE =
1e-12
```

with semantics equivalent to:

```text
math.isclose(
    reconstructed,
    serialized,
    rel_tol=1e-12,
    abs_tol=1e-12
)
```

CSV serialization:

```text
CSV_FLOAT_FORMAT =
.17g
```

All qualification uses unrounded in-memory binary64 values.

JSON requires finite values only:

```text
allow_nan = false
```

Existing protocol fingerprint JSON semantics remain unchanged.

```text
U-017 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = NO
```

---

## U-018 — Exact successful-run artifact inventory

A technically successful development run produces exactly:

```text
1. protocol_snapshot.json
2. daily_forecast_scores.csv.gz
3. forecast_metrics.csv
4. forecast_uncertainty.json
5. stability_diagnostics.json
6. economic_uncertainty.json
7. economic_summary.csv
8. development_adjudication.json
9. artifact_manifest.json
```

### protocol_snapshot.json

Exact top-level keys:

```text
authority
protocol_fingerprint_sha256
```

No others.

### daily_forecast_scores.csv.gz

Exact columns:

```text
target_date
stage
model_id
candidate_id
poisson_deviance
mae
rmse
```

Stage order:

```text
DEV
VAL
STABILITY
```

Rows:

* all M3 candidates + B0 on DEV;
* selected M3 + B0 on VAL;
* selected M3 + B0 on STABILITY only when forecast gate passes.

Sort:

```text
stage order
target_date ascending
model_id ascending
candidate_id ascending
```

### forecast_metrics.csv

Exact columns:

```text
stage
model_id
candidate_id
date_count
poisson_deviance
mae
rmse
```

Values reconstruct from daily scores using frozen stage aggregation.

### forecast_uncertainty.json

Exact top-level keys:

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

Successful status:

```text
EVALUATED
```

where:

```text
observed_mean_improvement =
mean_t(PD_B0_t - PD_M3_t)
```

on VAL.

### stability_diagnostics.json

Exact top-level keys:

```text
status
date_count
blocks
per_k
```

When evaluated:

```text
status = EVALUATED
date_count = N_stability
```

`blocks` has exactly six records:

```text
block_id
start_date
end_date
date_count
per_k
```

Each block `per_k` has ordered records:

```text
K
mean_economic_delta
positive
```

Top-level `per_k` has:

```text
K
positive_block_count
bootstrap_lower_bound
qualifies
```

When forecast gate fails:

```text
status =
NOT_EVALUATED_FORECAST_GATE_FAILED

date_count = null

blocks = []
```

and all four K records remain represented with null unavailable values and `qualifies=false`.

### economic_uncertainty.json

Retains exactly the surviving V3 schema.

No added keys.

### economic_summary.csv

Exact columns:

```text
K
status
date_count
mean_economic_delta
positive_block_count
bonferroni_alpha
bootstrap_lower_bound
qualifies
recommended
```

Exactly four rows ordered:

```text
1
3
5
10
```

`recommended=true` for at most one row.

### development_adjudication.json

Exact top-level keys:

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

Successful status:

```text
COMPLETED
```

Allowed successful `development_exit_status`:

```text
FORECAST_GATE_FAILED
ECONOMIC_GATE_FAILED
ECONOMIC_SIGNAL_FOUND
```

Mappings:

```text
forecast_signal=false
->
FORECAST_GATE_FAILED
```

```text
forecast_signal=true
economic_signal=false
->
ECONOMIC_GATE_FAILED
```

```text
forecast_signal=true
economic_signal=true
->
ECONOMIC_SIGNAL_FOUND
```

### artifact_manifest.json

Exact top-level key:

```text
artifacts
```

Array ordered lexicographically by filename.

Each record:

```text
filename
sha256
byte_size
```

Manifest excludes itself and lists the other eight artifacts.

```text
U-018 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = NO
```

---

## U-019 — General failure artifact contract

Technical/protocol execution failure produces exactly:

```text
development_run_FAILED.json
```

No finalized success artifact may coexist in a finalized failure directory.

Exact top-level keys:

```text
status
failed_stage
error_type
development_exit_status
protocol_snapshot
```

Required:

```text
status = FAILED
```

Allowed `failed_stage`:

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

`error_type` is a non-empty stable protocol identifier.

Allowed `development_exit_status`:

```text
NEEDS_DATA_REVISION
NEEDS_MODEL_REVISION
NEEDS_PROTOCOL_REVISION
TECHNICAL_FAILURE
```

`protocol_snapshot` contains exactly:

```text
authority
protocol_fingerprint_sha256
```

Preserve exact SVD ambiguity failure:

```text
failed_stage =
MODEL_INITIALIZATION

error_type =
SVDLeadingSubspaceAmbiguity

development_exit_status =
NEEDS_MODEL_REVISION
```

```text
U-019 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = NO
```

---

## U-020 — Closed authority schema

`protocol_snapshot.authority` is a closed object.

No undeclared key permitted.

Every declared key participates in fingerprint bytes.

Exact key set:

```text
spec_version
authority_provenance
canonical_spec_sha256

source_commit
source_tree

data_source_repository
data_source_path
data_source_commit
data_source_blob
data_source_sha256

artifact_contract_version
model_contract_version
data_split_contract_version
candidate_contract_version
forecast_gate_contract_version
forecast_bootstrap_contract_version
economic_contract_version
success_artifact_schema_version
failure_artifact_schema_version

metric_definition_version
poisson_log_base
poisson_outcome_aggregation
poisson_zero_count_convention
mae_outcome_aggregation
rmse_daily_aggregation
stage_poisson_aggregation
stage_mae_aggregation
stage_rmse_aggregation
block_metric_aggregation

forecast_sum_tolerance
artifact_numeric_abs_tolerance
artifact_numeric_rel_tolerance
csv_float_format

baseline_id
candidate_windows
minimum_eligible_targets
split_fractions
stability_block_count

svd_implementation
svd_full_matrices
interaction_zero_tolerance
svd_gap_tolerance
svd_gap_definition
svd_ambiguous_leading_subspace_policy

optimizer_implementation
optimizer_method
optimizer_max_iterations
optimizer_ftol
optimizer_constraint_tolerance

forecast_bootstrap_rng_implementation
forecast_bootstrap_bit_generator
forecast_bootstrap_seed
forecast_bootstrap_replications
forecast_bootstrap_mean_block_length
forecast_bootstrap_restart_probability
forecast_bootstrap_alpha
forecast_bootstrap_quantile_method

economic_k_values
economic_cost_per_number_thousand_vnd
economic_payout_per_hit_thousand_vnd

economic_bootstrap_rng_implementation
economic_bootstrap_bit_generator
economic_bootstrap_seed
economic_bootstrap_replications
economic_bootstrap_mean_block_length
economic_bootstrap_restart_probability
economic_bootstrap_quantile_method
economic_bootstrap_shared_resample_indices

economic_familywise_alpha
economic_multiplicity_method
economic_per_k_alpha
economic_uncertainty_allowed_statuses
```

Freeze contract versions:

```text
model_contract_version =
XPIS_V3_M3_MODEL_CP1

data_split_contract_version =
XPIS_V3_M3_DATA_SPLIT_CP1

candidate_contract_version =
XPIS_V3_M3_CANDIDATE_CP1

forecast_gate_contract_version =
XPIS_V3_M3_FORECAST_GATE_CP1

forecast_bootstrap_contract_version =
XPIS_V3_M3_FORECAST_BOOTSTRAP_CP1

economic_contract_version =
XPIS_V3_M3_ECONOMIC_CP1

success_artifact_schema_version =
XPIS_V3_M3_SUCCESS_ARTIFACTS_CP1

failure_artifact_schema_version =
XPIS_V3_M3_FAILURE_ARTIFACT_CP1
```

Types:

```text
*_version
*_implementation
*_method
*_definition
*_policy
*_path
*_repository
baseline_id
authority_provenance
csv_float_format

=
JSON string
```

```text
source_commit
data_source_commit

=
40-character lowercase Git commit hex
```

```text
source_tree
data_source_blob

=
40-character lowercase Git object hex
```

```text
canonical_spec_sha256
data_source_sha256

=
64-character lowercase SHA-256 hex
```

```text
candidate_windows
economic_k_values

=
ordered JSON arrays of integers
```

Freeze:

```text
split_fractions =
[0.5,0.25,0.25]
```

All tolerances/probabilities:

```text
finite JSON number
```

All counts/seeds:

```text
JSON integer
```

Booleans:

```text
JSON boolean
```

Exact status array:

```text
economic_uncertainty_allowed_statuses =
[
  "NOT_EVALUATED_FORECAST_GATE_FAILED",
  "EVALUATED"
]
```

Run-bound:

```text
source_commit
source_tree

data_source_commit
data_source_blob
data_source_sha256
```

must be frozen before historical execution.

`canonical_spec_sha256` is the SHA-256 of the final canonical specification bytes accepted by final spec-lock review.

It is populated at run-authority snapshot construction rather than self-embedded as a literal inside the canonical document.

No authority-bearing configuration may exist outside:

1. an explicit `authority` field above; or
2. normative specification content covered by `canonical_spec_sha256`.

```text
U-020 = RESOLVED
SCIENTIFIC_REREVIEW_REQUIRED = NO
```

---

# CONTROL PLANE FINAL ADJUDICATION STATUS

```text
U-001 = RESOLVED
U-002 = RESOLVED
U-003 = RESOLVED
U-004 = RESOLVED
U-005 = RESOLVED

U-006 = RESOLVED
U-007 = RESOLVED
U-008 = RESOLVED
U-009 = RESOLVED

U-010 = RESOLVED
U-011 = RESOLVED
U-012 = RESOLVED
U-013 = RESOLVED
U-014 = RESOLVED

U-015 = RESOLVED
U-016 = RESOLVED
U-023 = RESOLVED
U-024 = RESOLVED

U-017 = RESOLVED
U-018 = RESOLVED
U-019 = RESOLVED
U-020 = RESOLVED
```

Therefore:

```text
INPUT_UNRESOLVED_COUNT =
22

CONTROL_PLANE_DECISIONS_MADE =
22

REMAINING_PROTOCOL_CRITICAL_UNRESOLVED_COUNT =
0

CONTROL_PLANE_RECONSTRUCTION_ADJUDICATION =
COMPLETE
```

Because U-001 through U-016 plus U-023/U-024 introduce new scientific authority:

```text
TARGETED_SCIENTIFIC_REREVIEW_REQUIRED =
YES
```

Therefore:

```text
FORMAL_SPEC_LOCK =
NOT_AUTHORIZED

REPO_GROUNDED_PLANNING =
NOT_AUTHORIZED

SOURCE_IMPLEMENTATION =
NOT_AUTHORIZED

V3_HISTORICAL_RUN =
NOT_AUTHORIZED
```

---

# EXECUTION AUTHORITY FOR THIS SESSION

You may now resume the previously assigned docs-only patch.

Modify only:

```text
docs/superpowers/specs/2026-09-03-xpis-v3-m3-digit-factor-canonical-reconstruction.md

docs/superpowers/specs/2026-09-03-xpis-v3-m3-digit-factor-reconstruction-ledger.md
```

Apply all 22 Control Plane decisions exactly.

Do NOT redesign or reinterpret them.

Do NOT modify source, tests, data, backtests, or implementation planning.

Do NOT commit.

Do NOT push.

Do NOT open PR.

Do NOT merge.

Post-patch expected lifecycle:

```text
CONTROL_PLANE_RECONSTRUCTION_ADJUDICATION =
COMPLETE

PROTOCOL_CRITICAL_UNRESOLVED_COUNT =
0

TARGETED_SCIENTIFIC_REREVIEW_REQUIRED =
YES

FORMAL_ARTIFACT_REVIEW =
READY_FOR_TARGETED_SCIENTIFIC_AND_FINAL_LOCK_REVIEW

FORMAL_SPEC_LOCK =
NOT_AUTHORIZED

REPO_GROUNDED_PLANNING =
NOT_AUTHORIZED

SOURCE_IMPLEMENTATION =
NOT_AUTHORIZED

V3_HISTORICAL_RUN =
NOT_AUTHORIZED
```

Return the previously requested patch-integrity report with recomputed SHA-256 values.

# CORE RULE

```text
APPLY CONTROL PLANE AUTHORITY EXACTLY.

DO NOT RECONSTRUCT IT FROM MEMORY.

DO NOT REDESIGN IT.

DO NOT ADVANCE THE LIFECYCLE.
```

---

# CONTROL PLANE FINAL-LOCK REVIEW CORRECTION ADDENDUM

## XPIS v3 M3 Digit-Factor — Review Blocker Adjudication (B-01 .. B-10)

```text
CONTROL_PLANE_MEMORANDUM_ORIGIN = CHATGPT_GPT_5_6_SOL_CONTROL_PLANE
HISTORICAL_M3_V2_SOURCE = UNAVAILABLE
AUTHORITY_TYPE = EXPLICIT_CONTROL_PLANE_RECONSTRUCTION_AUTHORITY
ADDENDUM_DATE = 2026-09-04
ADDENDUM_STATUS = NORMATIVE_ADJUDICATED
REPRESENTED_BLOCKERS = B-01, B-02, B-03, B-04, B-05, B-06, B-07, B-08, B-09, B-10
HIDDEN_IMPLEMENTATION_DECISIONS = ELIMINATED (COUNT = 0)
```

### B-01 — Freeze Exact SLSQP Numerical Fit Contract

1. **Optimization Parameter Vector**:
   Exact variable order:
   ```text
   theta = [
     a[0], ..., a[9],
     b[0], ..., b[9],
     u[0], ..., u[9],
     v[0], ..., v[9],
     gamma
   ]
   ```
   Length: `41`.
   Bounds:
   - `a[0..9]`: unbounded `(-inf, +inf)`
   - `b[0..9]`: unbounded `(-inf, +inf)`
   - `u[0..9]`: unbounded `(-inf, +inf)`
   - `v[0..9]`: unbounded `(-inf, +inf)`
   - `gamma`: `[0, +inf)` using native SLSQP parameter bounds.

2. **Equality Residual Vector**:
   Define exactly:
   ```text
   c(theta) = [
     sum_i a[i],
     sum_j b[j],
     sum_i u[i],
     sum_j v[j],
     dot(u, u) - 1.0,
     dot(v, v) - 1.0
   ]
   ```
   Norm equality residuals are strictly `dot(u, u) - 1 = 0` and `dot(v, v) - 1 = 0` (NOT `norm(u) - 1`).

3. **Objective & Analytic Jacobian**:
   Objective:
   ```text
   L(theta) = - sum_i sum_j X[i,j] * ln(p[i,j])
   ```
   where `T = sum_i sum_j X[i,j] = 27 * W`, and:
   ```text
   G[i,j] = T * p[i,j] - X[i,j]
   ```
   The objective Jacobian is strictly analytic:
   ```text
   dL/da[i]   = sum_j G[i,j]
   dL/db[j]   = sum_i G[i,j]
   dL/du[i]   = gamma * sum_j (G[i,j] * v[j])
   dL/dv[j]   = gamma * sum_i (G[i,j] * u[i])
   dL/dgamma  = sum_i sum_j (G[i,j] * u[i] * v[j])
   ```

4. **Constraint Jacobian**:
   Exact analytic `6 x 41` matrix:
   - Row 0 (sum a = 0): `1.0` for all 10 `a` coordinates; `0.0` elsewhere.
   - Row 1 (sum b = 0): `1.0` for all 10 `b` coordinates; `0.0` elsewhere.
   - Row 2 (sum u = 0): `1.0` for all 10 `u` coordinates; `0.0` elsewhere.
   - Row 3 (sum v = 0): `1.0` for all 10 `v` coordinates; `0.0` elsewhere.
   - Row 4 (dot u = 1): `2.0 * u[i]` for each `u[i]` coordinate; `0.0` elsewhere.
   - Row 5 (dot v = 1): `2.0 * v[j]` for each `v[j]` coordinate; `0.0` elsewhere.

   ```text
   OBJECTIVE_JACOBIAN = ANALYTIC
   CONSTRAINT_JACOBIAN = ANALYTIC
   FINITE_DIFFERENCE_SCHEME = NOT_APPLICABLE
   FINITE_DIFFERENCE_STEP = NOT_APPLICABLE
   ```

5. **SciPy Solver Call Semantics**:
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
   The authority field `optimizer_constraint_tolerance = 1e-10` represents:
   ```text
   POST_SOLVER_FEASIBILITY_TOLERANCE = 1e-10
   ```
   which is an independent post-solver acceptance gate, NOT an SLSQP option.

---

### B-02 — Freeze Exact Final-Fit Canonicalization Order

1. **Step 1 — Raw Solver Validity**:
   Before ANY canonicalization, require:
   - `optimizer.success == true`
   - All raw parameters finite
   - Raw objective finite
   - `raw_gamma >= 0.0`
   If `raw_gamma < 0.0`: fail-closed with `FAILED_STAGE=MODEL_FIT`, `ERROR_TYPE=ConstraintViolation`, `DEVELOPMENT_EXIT_STATUS=NEEDS_MODEL_REVISION`. Zero-interaction canonicalization must never hide a negative raw gamma.

2. **Step 2 — Equality Feasibility**:
   Compute exact equality residual vector `c(theta_raw)`.
   Require:
   ```text
   max(abs(c(theta_raw))) <= 1e-10
   ```
   Otherwise fail with `ConstraintViolation`.

3. **Step 3 — Interaction Canonicalization**:
   If `0.0 <= gamma_raw <= 1e-10`:
   - Canonicalize to zero interaction:
     `gamma = 0.0`
     `u = v = (9, -1, -1, -1, -1, -1, -1, -1, -1, -1) / sqrt(90)`
   - `a` and `b` remain the accepted fitted main effects.
   Else (`gamma_raw > 1e-10`):
   - `gamma = gamma_raw`, `u = u_raw`, `v = v_raw`.
   - Apply surviving canonical sign rule to the final fitted `(u, v)` pair: let `i_star` be the smallest index attaining `max_i |u[i]|`; if `u[i_star] < 0`, negate both `u <- -u` and `v <- -v`.
   - Do NOT re-center or re-normalize the accepted non-zero fitted solution after solver acceptance.

4. **Step 4 — Final Forecast Validation**:
   Generate forecasts exclusively from the final canonicalized representation:
   - All `eta`, `p`, `mu` finite
   - `mu[n] > 0.0` strictly for all `n in {0..99}`
   - `abs(sum_n mu[n] - 27.0) <= 1e-10`
   Any numerical underflow producing `mu[n] == 0.0` is fail-closed:
   `ERROR_TYPE = ForecastContractViolation`.
   This final forecast is the sole authoritative forecast.

---

### B-03 — Missing-Draw Semantics & Event-Indexed Chronology

1. **Event-Indexed Model**:
   M3 is strictly `OBSERVED_DRAW_EVENT_INDEXED`, not calendar-completeness inferred.
   `history_window_W` counts `W` recorded draw dates / observations, NOT calendar days.

2. **No Missing Draw Definition**:
   `NO_MISSING_DRAW` within M3 means:
   - Every PRESENT dataset row/date must contain exactly 27 valid integer outcomes.
   - No value may be null, absent, malformed, imputed, clipped, or silently repaired.
   Absence of a calendar date does NOT by itself constitute an M3 missing-draw violation.

3. **Scope Demarcation**:
   ```text
   CALENDAR_CONTINUITY_VALIDATION = OUT_OF_SCOPE_FOR_M3_MODEL_PROTOCOL
   OFFICIAL_DRAW_CALENDAR_COMPLETENESS = UPSTREAM_DATA_GOVERNANCE_CONCERN
   ```
   M3 operates exclusively on the frozen canonical recorded-event sequence. It must never synthesize or impute rows for absent calendar dates.

4. **Legacy Field Mapping**:
   `selected_window_days` in artifacts is a legacy field name semantically identical to:
   ```text
   COUNT_OF_PRECEDING_RECORDED_DRAW_DATES
   ```

---

### B-04 — split_fractions Semantics

`split_fractions = [0.5, 0.25, 0.25]` represents:
```text
split_fractions = NOMINAL_DESIGN_FRACTIONS
```
It does NOT assert exact realized finite-sample proportions. Actual cohort sizes are defined strictly by integer floor formulas:
```text
N_dev = floor(N / 2)
N_val = floor(N / 4)
N_stability = N - N_dev - N_val
```
These count formulas have absolute normative precedence over nominal design fractions.

---

### B-05 — Self-Contained economic_uncertainty.json Contract

1. **Top-Level Keys (Exactly 13)**:
   `status`, `familywise_alpha`, `multiplicity_method`, `per_k_alpha`, `bootstrap_rng_implementation`, `bootstrap_bit_generator`, `bootstrap_seed`, `bootstrap_replications`, `mean_block_length`, `restart_probability`, `quantile_method`, `shared_resample_indices`, `per_k`.

2. **Frozen Protocol Constants**:
   ```text
   familywise_alpha = 0.05
   multiplicity_method = "BONFERRONI"
   per_k_alpha = 0.0125
   bootstrap_rng_implementation = "numpy.random.Generator"
   bootstrap_bit_generator = "PCG64"
   bootstrap_seed = 20260832
   bootstrap_replications = 2000
   mean_block_length = 30
   restart_probability = 1/30
   quantile_method = "linear"
   shared_resample_indices = true
   ```

3. **Top-Level Status Enum**:
   `NOT_EVALUATED_FORECAST_GATE_FAILED` or `EVALUATED`.

4. **per_k Array Schema**:
   Exactly 4 records ordered `1, 3, 5, 10`. Each record has exactly 8 keys:
   `K`, `status`, `date_count`, `mean_economic_delta`, `positive_block_count`, `bonferroni_alpha`, `bootstrap_lower_bound`, `qualifies`.

   - When evaluated:
     `status = "EVALUATED"`
     `date_count = N_stability`
     `mean_economic_delta = finite JSON float`
     `positive_block_count = integer in {0,1,2,3,4,5,6}`
     `bonferroni_alpha = 0.0125`
     `bootstrap_lower_bound = finite JSON float`
     `qualifies = boolean` (true iff mean_delta > 0 AND positive_blocks >= 5 AND lower_bound > 0).

   - When forecast gate fails:
     Top-level `status = "NOT_EVALUATED_FORECAST_GATE_FAILED"`.
     For each K:
     `status = "NOT_EVALUATED_FORECAST_GATE_FAILED"`
     `date_count = null`
     `mean_economic_delta = null`
     `positive_block_count = null`
     `bonferroni_alpha = 0.0125`
     `bootstrap_lower_bound = null`
     `qualifies = false`
     In this state: `ECONOMIC_BOOTSTRAP_EXECUTED = NO`. Top-level constants remain populated.

---

### B-06 — Exact Cross-Artifact Consistency Qualifier

When `economic_uncertainty.status = "EVALUATED"`, the four K records must agree across:
- `economic_uncertainty.json`
- `economic_summary.csv`
- `stability_diagnostics.json`
- `development_adjudication.json`

on `K`, `positive_block_count`, `bootstrap_lower_bound`, `qualifies` **WHERE THOSE FIELDS ARE REPRESENTED**.

Per-K scalar fields are NOT added to `development_adjudication.json`. For `development_adjudication.json`, consistency is strictly:
```text
qualified_top_k = ordered list of K for which qualifies == true
recommended_top_k subset_of qualified_top_k
```

---

### B-07 — Complete Success-Artifact Row Universe, Typings, and Encodings

1. **Common CSV Encoding**:
   UTF-8 without BOM, comma delimiter, double-quote `"` quoting, minimal quoting, LF terminator, header row mandatory.
   Types: `null` -> empty field; `boolean` -> lowercase `true` / `false`; `integer` -> base-10 decimal; `float` -> `.17g`; `date` -> `YYYY-MM-DD`.

2. **Common JSON Encoding**:
   UTF-8 without BOM, `allow_nan=false`, `ensure_ascii=false`, `sort_keys=true`, separators `(",",":")`, exactly one trailing LF byte.

3. **Gzip Encoding (`daily_forecast_scores.csv.gz`)**:
   Compression gzip, `compression_level = 9`, `mtime = 0`. No original filename, header comment, or extra fields.

4. **Identifier Namespace**:
   `B0 model_id = "B0"`, `B0 candidate_id = "B0_UNIFORM"`.
   `M3 model_id = "M3"`, candidate IDs: `M3_W030`, `M3_W060`, `M3_W120`, `M3_W240`, `M3_W365`.

5. **`daily_forecast_scores.csv.gz` Row Universe**:
   - DEV: exactly 6 rows per target date (B0 + all 5 M3 candidates).
   - VAL: exactly 2 rows per target date (B0 + selected M3 candidate).
   - STABILITY:
     - When `FORECAST_SIGNAL == true`: exactly 2 rows per target date (B0 + selected M3 candidate).
     - When `FORECAST_SIGNAL == false`: ZERO STABILITY rows.
   Columns: `target_date, stage, model_id, candidate_id, poisson_deviance, mae, rmse`. No null permitted.
   Ordering: stage order (DEV, VAL, STABILITY), then target_date ascending, then model_id ascending, then candidate_id ascending.

6. **`forecast_metrics.csv` Row Universe**:
   Exactly one aggregate row for every `(stage, model_id, candidate_id)` combination present in daily scores:
   - DEV: exactly 6 rows.
   - VAL: exactly 2 rows.
   - STABILITY: exactly 2 rows if `FORECAST_SIGNAL == true`; ZERO rows if `FORECAST_SIGNAL == false`.
   Columns: `stage, model_id, candidate_id, date_count, poisson_deviance, mae, rmse`.

7. **`economic_summary.csv` Row Universe**:
   Exactly 4 rows for `K in [1, 3, 5, 10]` in order.
   Columns: `K, status, date_count, mean_economic_delta, positive_block_count, bonferroni_alpha, bootstrap_lower_bound, qualifies, recommended`.
   When forecast gate fails: `date_count=null`, `mean_economic_delta=null`, `positive_block_count=null`, `bootstrap_lower_bound=null`, `qualifies=false`, `recommended=false`, `bonferroni_alpha=0.0125`.

8. **`development_adjudication.json` Schema**:
   Keys: `status`, `selected_candidate_id`, `selected_window_days`, `forecast_signal`, `forecast_bootstrap_lower_bound`, `economic_signal`, `qualified_top_k`, `recommended_top_k`, `development_exit_status`.
   When forecast gate fails:
   `economic_signal = false`, `qualified_top_k = []`, `recommended_top_k = []`, `development_exit_status = "FORECAST_GATE_FAILED"`.

9. **Deterministic Array Ordering**:
   - `economic per_k`: `[1, 3, 5, 10]`
   - `qualified_top_k`: canonical K ascending
   - `recommended_top_k`: `[]` or `[chosen_K]`
   - `stability blocks`: `block_id` ascending (`1..6`)
   - `artifact_manifest.artifacts`: `filename` lexicographic ascending.

---

### B-08 — Failure Snapshot Before vs After Authority Completion

In `development_run_FAILED.json`:
- Top-level keys: `status`, `failed_stage`, `error_type`, `development_exit_status`, `protocol_snapshot`.
- `protocol_snapshot` type: `object | null`.
- Definition:
  ```text
  AUTHORITY_COMPLETE =
  all required closed authority keys have valid non-null values
  and protocol_fingerprint_sha256 can be computed
  ```
- If failure occurs BEFORE `AUTHORITY_COMPLETE`:
  `protocol_snapshot = null`.
  No partial authority object, no placeholder, no sentinel SHA-256 string.
- If failure occurs AFTER `AUTHORITY_COMPLETE`:
  `protocol_snapshot = { "authority": <complete authority object>, "protocol_fingerprint_sha256": <valid fingerprint> }`.

---

### B-09 — Recommendation Eligibility Domain & Invariants

1. **Ranking Domain**:
   Ranking applies strictly to `qualified_top_k`. Non-qualifying K values are never considered for recommendation.

2. **Selection Rule**:
   If `qualified_top_k == []`:
   `recommended_top_k = []`.
   Else:
   ```text
   chosen_K = argmax_{K in qualified_top_k} (bootstrap_lower_bound, mean_economic_delta, -K)
   recommended_top_k = [chosen_K]
   ```

3. **Invariants**:
   - `recommended_top_k subset_of qualified_top_k`
   - `len(recommended_top_k) <= 1`
   - In `economic_summary.csv`, `recommended = true` iff its `K` is the sole element of `recommended_top_k`.

---

### B-10 — Authority Artifact Identity & Provenance

This memorandum and addendum document is permanently persisted in the repository as:
```text
PATH = docs/superpowers/specs/2026-09-04-xpis-v3-m3-control-plane-reconstruction-adjudication.md
```
All references to `E-009` across the canonical specification and reconstruction ledger bind to this file's path, SHA-256 digest, and Git blob hash.

---

## CONTROL PLANE FR-07 GZIP HEADER ADJUDICATION

```text
AUTHORITY_ORIGIN =
CONTROL_PLANE_FINAL_LOCK_SERIALIZATION_DECISION
```

For `daily_forecast_scores.csv.gz`, exactly one gzip member is permitted. Its first 10 bytes are exactly:

```text
1f 8b 08 00 00 00 00 00 02 ff
```

```text
ID1   = 0x1f
ID2   = 0x8b
CM    = 0x08
FLG   = 0x00
MTIME = 0x00000000
XFL   = 0x02
OS    = 0xff
```

The complete fixed serialization contract is:

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

No producer/platform-derived OS byte has authority. If the runtime compressor produces a different OS byte, the producer must normalize byte offset 9 to `0xff` before final artifact hashing. If it produces any other fixed-header mismatch, artifact generation fails closed. Trailer CRC32 and ISIZE must be correct for the uncompressed CSV payload under the gzip format. This adjudication does not reopen U-022 environment policy.

### FR-01..FR-08 Correction Provenance

The final-lock fidelity repairs restore literal Control Plane and surviving Closure authority, complete the U-018/U-019 self-contained artifact contracts, specify the previously abbreviated stationary-bootstrap RNG consumption order, normalize authority-file bytes to UTF-8 LF-only content with no forbidden control characters, and correct reproducible surviving-evidence identity. They do not revise the original 22-decision memorandum or reopen scientific choices.
---

## Control Plane Correction Addendum — U-020 Literal Authority

```text
AUTHORITY_ORIGIN =
CONTROL_PLANE_RECONSTRUCTION_DECISION

SUBORDINATE_TO =
U-020

DECISION_COUNT_IMPACT =
NONE_REMAINS_22_OF_22
```

This addendum is subordinate to Decision `U-020` (Closed authority schema). It establishes the exact normative literal selections for `authority_provenance` (`PROV-01`) and `data_source_repository` (`PROV-02`).
This addendum does NOT create a 23rd protocol-critical Control Plane decision, does NOT increase the Control Plane decision count beyond 22, does NOT reopen historical V2 recovery, and does NOT redesign scientific choices.

### PROV-01 — authority_provenance Canonical Literal Authority

```text
FIELD =
authority_provenance

TYPE =
string

VALUE =
RECONSTRUCTED_FROM_SURVIVING_APPROVED_ARTIFACTS_REPOSITORY_EVIDENCE_AND_CONTROL_PLANE_ADJUDICATION

AUTHORITY_ORIGIN =
CONTROL_PLANE_RECONSTRUCTION_DECISION

DECISION =
U-020 correction addendum

HISTORICAL_RECOVERY_STATUS =
NOT_RECOVERED

SURVIVING_AUTHORITY_SELECTION =
NO
```

The string literal itself is explicitly selected NOW by ChatGPT Control Plane.
The literal MUST NOT be described as preserved from Closure evidence, nor as recovered historical V2 authority.
Its purpose is the canonical self-description of the reconstructed authority provenance.

---

### PROV-02 — data_source_repository Canonical Literal Authority

```text
FIELD =
data_source_repository

TYPE =
string

VALUE =
Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis

AUTHORITY_ORIGIN =
CONTROL_PLANE_RECONSTRUCTION_DECISION

DECISION =
U-020 correction addendum

EVIDENCE_BASIS =
CURRENT_CANONICAL_REPOSITORY_IDENTITY

HISTORICAL_RECOVERY_STATUS =
NOT_RECOVERED

SURVIVING_AUTHORITY_SELECTION =
NO
```

This exact repository literal is explicitly selected NOW by ChatGPT Control Plane.
It is grounded in current repository evidence, but the literal selection itself is a Control Plane reconstruction decision.
The literal MUST NOT be relabeled as surviving authority merely because the repository exists.

---

### U-006 Interaction and Fail-Closed Repository Authority Invariant

The new `data_source_repository` literal does NOT replace or weaken Decision `U-006`.
The existing data path remains:

```text
data/xsmb-2-digits.csv
```

The existing fail-closed rule remains normative:

```text
IF repository-grounded planning establishes that
the repository or canonical dataset authority no longer matches
the frozen reconstruction authority

THEN

STOP =
REPOSITORY_AUTHORITY_MISMATCH
```

Require:

```text
SILENT_REPOSITORY_SUBSTITUTION =
FORBIDDEN

SILENT_DATA_SOURCE_SUBSTITUTION =
FORBIDDEN
```

Downstream agents and workflows MUST NOT silently rewrite either the repository slug or dataset path based on future repository state.
Any mismatch returns to Control Plane.
---

## Control Plane Correction Addendum — U-018 forecast_metrics.csv Row Order Authority

```text
AUTHORITY_ORIGIN =
CONTROL_PLANE_RECONSTRUCTION_DECISION

SUBORDINATE_TO =
U-018

DECISION_COUNT_IMPACT =
NONE_REMAINS_22_OF_22
```

This addendum is subordinate to Decision `U-018` (Exact successful-run artifact inventory). It establishes the explicit, deterministic row-ordering contract for `forecast_metrics.csv`.
This addendum applies specifically to `forecast_metrics.csv` and does NOT silently alter or weaken ordering contracts for any other artifact.
This addendum does NOT create a 23rd protocol-critical Control Plane decision, does NOT increase the Control Plane decision count beyond 22, and does NOT redesign scientific choices.

### forecast_metrics.csv Deterministic Row Order Contract

```text
ARTIFACT =
forecast_metrics.csv

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

All rows in `forecast_metrics.csv` MUST strictly follow this three-tier deterministic sort order:
1. Primary sort: stage order fixed as `DEV`, then `VAL`, then `STABILITY`.
2. Secondary sort: within each stage, `model_id` ascending lexicographically (`"B0"` before `"M3"`).
3. Tertiary sort: within the same stage and `model_id`, `candidate_id` ascending lexicographically (e.g., for DEV M3: `"M3_W030"`, `"M3_W060"`, `"M3_W120"`, `"M3_W240"`, `"M3_W365"`).

No producer, platform, operating system, or filesystem sorting variation is permitted.
