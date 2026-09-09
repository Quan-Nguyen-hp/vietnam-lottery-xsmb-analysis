# XPIS v3 M3 — Final Scientific Qualification Failure Closeout

**Document ID**: `XPIS_V3_M3_QUALIFICATION_FAILURE_CLOSEOUT`
**Date**: 2026-09-09
**Adjudication Authority**: ChatGPT GPT-5.6 Sol Control Plane
**Executor**: Gemini 3.8 Flash (`M3_SCIENTIFIC_FAILURE_CLOSEOUT_EXECUTOR`)
**Status**: `FINAL_CLOSED` / `NEEDS_MODEL_REVISION`

---

## 1. Executive Summary & Lifecycle Adjudication

Under the governance of the ChatGPT GPT-5.6 Sol Control Plane, the XPIS v3 M3 Digit Factor development pipeline has completed its definitive scientific evaluation.

```text
CONTROL_PLANE_M3_V3_FINAL_CLOSURE =
CLOSED_NEEDS_MODEL_REVISION

P3_PROTOCOL_STATUS =
ACCEPTED

DEV_WINNER =
M3_W365

VAL_QUALIFICATION =
FAIL

M3_V3_SCIENTIFIC_QUALIFICATION =
FAIL

DEVELOPMENT_EXIT_STATUS =
NEEDS_MODEL_REVISION

VAL_STATUS =
CONSUMED_BY_M3_V3

STABILITY =
CLOSED_NOT_EXECUTED

ECONOMICS =
CLOSED_NOT_EXECUTED

M3_V3_LIFECYCLE =
FINAL_CLOSED
```

---

## 2. Frozen Repository & Data Identities

```text
SOURCE_COMMIT =
27a45d3553fe448b016788305a1774054d6c1123

SOURCE_TREE =
e1f5b325688de13635fd5cb8f7d2ba6c12c581f1

CANONICAL_DATA_PATH =
data/xsmb-2-digits.csv

CANONICAL_DATA_BLOB =
8474b4e7c10802c683080779bead75d7e56cf358

CANONICAL_DATA_SHA256 =
3ae8970bfb2d54d9ad72d18aad1a1e3022c6231bc349dcfd01a78bc20e6a75c7

PROTOCOL_FINGERPRINT_SHA256 =
585b057f2e35153ceca58cc6f4dde338aea071d1827b46a3bffb705c1c38a57e

AUTHORITY_KEY_COUNT =
74
```

### Chronological Split Partitions

| Partition | Index Range | Target Count | Dates Covered | Status |
| :--- | :--- | :--- | :--- | :--- |
| **ELIGIBLE** | 365 – 7,541 | 7,177 | 2006-10-03 – 2026-07-15 | Canonical universe |
| **DEV** | 365 – 3,952 | 3,588 | 2006-10-03 – 2016-09-04 | Completed & Frozen |
| **VAL** | 3,953 – 5,746 | 1,794 | 2016-09-05 – 2021-08-04 | Evaluated & Consumed |
| **STABILITY** | 5,747 – 7,541 | 1,795 | 2021-08-05 – 2026-07-15 | **Closed / Not Executed** |

---

## 3. P3 Protocol Execution & Frozen DEV Decision

The candidate-level qualification protocol (P3) operated strictly as designed. Five historical lookback windows $W \in \{30, 60, 120, 240, 365\}$ and the uninformative uniform baseline $B_0$ were evaluated on the 3,588 DEV targets.

### Candidate Qualification Summary

```text
M3_W030 =
DISQUALIFIED_MODEL_FIT
(First failure: target_index 680 [2007-08-16], OptimizerNonConvergence)

COMPLETE_VALID =
M3_W060
M3_W120
M3_W240
M3_W365
```

### DEV Metric Ranking (Lexicographic: Ascending PD, MAE, RMSE, W)

| Rank | Candidate | Lookback ($W$) | Poisson Deviance | MAE | RMSE | Status |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- |
| **1** | **M3_W365** | **365** | **0.79808829434519313** | **0.41113920925380609** | **0.51527236122323972** | **FROZEN DEV WINNER** |
| 2 | M3_W240 | 240 | 0.79880285004903584 | 0.41114644718282228 | 0.51545815610333301 | Complete Valid |
| 3 | M3_W120 | 120 | 0.80018725622945408 | 0.41100671215990631 | 0.51581716626982954 | Complete Valid |
| 4 | M3_W060 | 60 | 0.80423334050360540 | 0.41100937674624588 | 0.51681306427398088 | Complete Valid |
| — | B0_UNIFORM | — | 0.79693894917559904 | 0.41119464882943152 | 0.51497496989752689 | Reference Baseline |

```text
FROZEN_DEV_WINNER =
M3_W365

DEV_SELECTED_W =
365
```

Candidate selection was permanently locked prior to accessing the validation partition.

---

## 4. Frozen VAL Evaluation & Forecast Qualification Gates

In accordance with strict out-of-sample discipline:
- Exactly **one** candidate (`M3_W365`) entered VAL against `B0_UNIFORM`.
- Lookback candidates `M3_W030`, `M3_W060`, `M3_W120`, and `M3_W240` were strictly barred from VAL access.
- All 1,794 daily targets were fitted with production SLSQP (`maxiter=2000`, `ftol=1e-12`, analytic Jacobians, U-004 initialization) with zero technical or optimizer failures.

### VAL Stage Forecast Metrics

| Metric | B0_UNIFORM | M3_W365 | Delta ($B_0 - M_3$) | Direction |
| :--- | :--- | :--- | :--- | :--- |
| **Poisson Deviance (PD)** | `0.79673067905918049` | `0.79802747443250688` | `-0.0012967953733262988` | $M_3$ Worse |
| **Mean Absolute Error (MAE)** | `0.41114046822742478` | `0.41116446264953266` | `-0.00002399442210788` | $M_3$ Worse |
| **Root Mean Squared Error (RMSE)** | `0.51489378281430520` | `0.51523137446863909` | `-0.00033759165433389` | $M_3$ Worse |

### Three-Part Qualification Gate Evaluation

```text
1. PRIMARY FORECAST GATE (Strict PD superiority):
   Predicate: PD_VAL_M3 < PD_VAL_B0
   Actual:    0.79802747443250688 < 0.79673067905918049 -> FALSE
   Result:    FAIL

2. SECONDARY FORECAST GATE (Weak secondary error non-inferiority):
   Predicate: (MAE_VAL_M3 <= MAE_VAL_B0) OR (RMSE_VAL_M3 <= RMSE_VAL_B0)
   Actual:    (0.41116446264953266 <= 0.41114046822742478) OR (0.51523137446863909 <= 0.51489378281430520) -> FALSE
   Result:    FAIL

3. FORECAST BOOTSTRAP GATE (Stationary circular block bootstrap lower bound):
   RNG / BitGen:        numpy.random.Generator / PCG64
   Seed:                20260831
   Replications:        2,000
   Mean Block Length:   30 (Restart probability = 1/30)
   Quantile / Method:   q = 0.05 / linear
   Lower Bound:         -0.0015550337802230211
   Predicate:           FORECAST_BOOTSTRAP_LOWER > 0.0
   Actual:              -0.0015550337802230211 > 0.0 -> FALSE
   Result:              FAIL
```

### Gate Verdict

```text
VAL_QUALIFICATION =
FAIL

M3_V3_SCIENTIFIC_QUALIFICATION =
FAIL

DEVELOPMENT_EXIT_STATUS =
NEEDS_MODEL_REVISION
```

`M3_W365` failed all three qualification criteria simultaneously.

---

## 5. Authoritative Scientific Interpretation

The scientific finding is recorded with the following exact, adjudicated language:

> **M3 v3, under its frozen model specification, DEV candidate-selection procedure, and VAL qualification criteria, failed to demonstrate out-of-sample predictive improvement over B0.**
>
> **This result does not establish that digit-factor persistence is impossible under other model specifications or future independently validated protocols.**

### Core Operational Conclusion

```text
P3 behaved as designed.

M3_W365 was the best COMPLETE_VALID DEV candidate
under the frozen DEV ranking rule.

The frozen VAL stage rejected M3_W365.

Therefore M3 v3 failed scientific qualification.
```

- The P3 candidate-selection protocol correctly handled the observed candidate-local optimizer failure for W=30 and isolated the remaining COMPLETE_VALID candidates without introducing partial-evidence contamination.
- The failure at VAL was neither technical nor numerical; all 1,794 models converged cleanly.
- Rejection is purely empirical: `M3_W365` exhibited inferior predictive deviance and error relative to the uninformative baseline $B_0$ on the independent validation sample.

---

## 6. Holdout Boundary & Methodology Constraints

```text
VAL_STATUS =
CONSUMED_BY_M3_V3
```

To preserve strict scientific integrity:

```text
SAME_VAL_AS_FRESH_VALIDATION =
FORBIDDEN

W240_VAL_FALLBACK =
FORBIDDEN

W120_VAL_FALLBACK =
FORBIDDEN

W60_VAL_FALLBACK =
FORBIDDEN

RETUNE_FROM_OBSERVED_VAL_AND_RETEST_SAME_VAL =
NOT_INDEPENDENT_VALIDATION
```

Any subsequent model, candidate, or protocol revision informed or influenced by these observed VAL results must be formally classified as **post-VAL model development**. It cannot claim a rerun on this validation partition (2016-09-05 – 2021-08-04) as an independent or unbiased test. Any future qualification requires a separately authorized, fresh validation architecture.

---

## 7. Downstream Stages Closure

In accordance with fail-closed governance:

```text
STABILITY_EXECUTED =
NO

ECONOMICS_EXECUTED =
NO

ECONOMIC_BOOTSTRAP_EXECUTED =
NO

HISTORICAL_M3_EXECUTION =
NO

STABILITY =
CLOSED_NOT_EXECUTED

ECONOMICS =
CLOSED_NOT_EXECUTED

SECOND_VAL_EXECUTION =
NOT_AUTHORIZED

VAL_RESELECTION =
FORBIDDEN
```

STABILITY and economic evaluations were completely short-circuited. No STABILITY outcomes were accessed or inspected.

---

## 8. Persisted Evidence Inventory & Cryptographic Hashes

All underlying evidence has been persisted and verified with exact SHA-256 digests:

| Artifact Name | Relative Path | Size (bytes) | Count | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| **DEV Daily Scores** | `scratch/dev_evidence_rematerialization_daily_scores.csv` | 1,616,703 | 17,940 rows | `701db285821365a083725b0fef60ff2dc66df6d69ba580d67a6c6fe3d7df03b7` |
| **DEV Summary** | `scratch/dev_evidence_rematerialization_summary.json` | 2,502 | — | `d711ef5ed4b1ad75a4a6ac4e9ed6719c863cd70874bad04cc678194f8b9cab3b` |
| **VAL Daily Scores** | `scratch/val_daily_scores.csv` | 337,962 | 3,588 rows | `08bbbebd24f5cece9ef5717ba419fe1fa32ce0af0cbea19228e70a42f221383b` |
| **VAL Bootstrap Evidence** | `scratch/val_forecast_bootstrap_evidence.json` | 45,765 | 2,000 reps | `4c1184e5bdc4d97be9d14d8839540cb334747be1c8bb343bd1c31eb808a71fd0` |
| **VAL Summary** | `scratch/val_summary.json` | 1,955 | — | `db67501ba27ab3feb82df39846d8ef79b53d13ca5ba7572991e5fa0f54a7f9da` |

---

## 9. Final Repository Integrity Audit

```text
DOCUMENT_CREATION =
COMPLETED (docs/superpowers/specs/2026-09-09-xpis-v3-m3-qualification-failure-closeout.md)

SCIENTIFIC_SOURCE_CHANGE =
NONE (src/ untouched)

AUTHORITY_SPEC_CHANGE =
NONE (canonical reconstruction spec untouched)

DATA_CHANGE =
NONE (data/ untouched)

MODEL_EXECUTION =
NONE

COMMIT =
NOT_EXECUTED (awaiting explicit Control Plane authorization)

PUSH =
FORBIDDEN
```
