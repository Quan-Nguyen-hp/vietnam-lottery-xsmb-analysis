# XPIS v3 M3 — Final Negative-Result Archival and Research-Line Retirement

**Document ID**: `XPIS_V3_M3_FINAL_NEGATIVE_RESULT_RETIREMENT`
**Date**: 2026-09-09
**Adjudication Authority**: ChatGPT GPT-5.6 Sol Control Plane
**Executor**: Gemini 3.8 Flash (`M3_FINAL_RETIREMENT_ARCHIVAL_EXECUTOR`)
**Status**: `FINAL_CLOSED_NEGATIVE_RESULT`

---

## 1. Executive Summary & Frozen Retirement Decision

Under the governance of the ChatGPT GPT-5.6 Sol Control Plane, the XPIS v3 M3 Digit Factor research line has concluded all evaluation and characterization activities. Following the independent validation failure of M3 v3 and the definitive outcome of the bounded additive characterization M3-R1, the Control Plane records the permanent retirement of the current M3 research line.

```text
M3_V3 =
FAILED_INDEPENDENT_VAL

M3_R1_ADDITIVE_CHARACTERIZATION =
NO_SUPPORTED_ADDITIVE_CANDIDATE

M3_RETIREMENT =
RETIRE_CURRENT_M3_RESEARCH_LINE

ADDITIONAL_M3_SEARCH =
FORBIDDEN

M3_FINAL_STATUS =
FINAL_CLOSED_NEGATIVE_RESULT
```

---

## 2. Frozen Repository Identity

```text
SOURCE_COMMIT =
bbbbae07aa2bfb87c546b21d36eec2e5272a6399

SOURCE_TREE =
3fe155aa214c97c165622f10313aee4196d7a368

BRANCH =
codex/impl/xpis-v3-m3-development

CANONICAL_DATA_PATH =
data/xsmb-2-digits.csv

CANONICAL_DATA_SHA256 =
3ae8970bfb2d54d9ad72d18aad1a1e3022c6231bc349dcfd01a78bc20e6a75c7

CANONICAL_DATA_BLOB =
8474b4e7c10802c683080779bead75d7e56cf358
```

---

## 3. M3 v3 Lifecycle Summary

The original M3 v3 digit-factor model underwent strict chronological evaluation across pre-registered development and validation partitions:

1. **DEV Evaluation ($N=3,588$, target indices 365..3952)**:
   - Evaluated 5 historical lookback windows $W \in \{30, 60, 120, 240, 365\}$ against the uninformative baseline $B_0$.
   - Candidate `M3_W030` was disqualified due to optimizer non-convergence (`target_index` 680).
   - Under the frozen lexicographic ranking rule (ascending Poisson deviance, MAE, RMSE, $W$), `M3_W365` was the best `COMPLETE_VALID` candidate and was selected as `FROZEN_DEV_WINNER`.
2. **VAL Evaluation ($N=1,794$, target indices 3953..5746)**:
   - Exactly one candidate (`M3_W365`) entered VAL against $B_0$.
   - All 1,794 optimization fits converged cleanly.
   - `M3_W365` failed all three qualification gates simultaneously:
     - **Primary Forecast Gate (Poisson Deviance)**: $\text{PD}_{M3} = 0.798027 > \text{PD}_{B0} = 0.796731 \implies \mathbf{FAIL}$ ($M_3$ worse than $B_0$).
     - **Secondary Forecast Gate (MAE/RMSE)**: Both MAE ($0.411164$ vs $0.411140$) and RMSE ($0.515231$ vs $0.514894$) were strictly worse than $B_0 \implies \mathbf{FAIL}$.
     - **Forecast Bootstrap Gate**: Stationary circular block bootstrap lower bound ($q=0.05$) was $-0.001555 \le 0.0 \implies \mathbf{FAIL}$.
3. **Partition Status**:
   - The VAL partition (indices 3953..5746) was consumed by M3 v3 (`CONSUMED_BY_M3_V3`).
   - Downstream STABILITY and economics stages were closed without execution.

---

## 4. M3-R1 Bounded Additive Characterization Summary

To address whether the M3 v3 failure was an artifact of non-convex SLSQP optimization or whether the underlying marginal digit-factor hypothesis lacked statistical support, a bounded additive characterization (M3-R1) was pre-registered and executed:

1. **Model Specification**:
   - Closed-form Laplace-smoothed marginal probabilities: $r_i = (H_i + 1)/(27W + 10)$, $c_j = (C_j + 1)/(27W + 10)$.
   - Outer product: $p_{\text{add},ij} = r_i \cdot c_j$.
   - Linear shrinkage toward uniform: $p_{ij} = (1 - \alpha) \cdot 0.01 + \alpha \cdot p_{\text{add},ij}$.
   - Poisson rate: $\mu_{ij} = 27 \cdot p_{ij}$.
   - Completely free of SVD, SLSQP, numerical optimizers, clipping, epsilon repairs, and post-hoc normalization.
2. **Candidate Universe**:
   - Exactly 9 pre-frozen additive candidates across $W \in \{180, 365, 730\}$, $\beta = 1$, $\alpha \in \{0.05, 0.10, 0.20\}$:
     - `M3R1_ADD_W180_A005`, `M3R1_ADD_W180_A010`, `M3R1_ADD_W180_A020`
     - `M3R1_ADD_W365_A005`, `M3R1_ADD_W365_A010`, `M3R1_ADD_W365_A020`
     - `M3R1_ADD_W730_A005`, `M3R1_ADD_W730_A010`, `M3R1_ADD_W730_A020`
3. **Evaluation Sample**:
   - Common development target pool: $N = 5,017$ (indices 730..5746).
   - Total daily evaluation rows: $50,170$ ($5,017 \times 10$ candidates including $B_0$).
   - Zero duplicate pairs, zero missing pairs.
4. **Empirical Findings**:
   - All 9 candidates exhibited positive but miniscule point estimates for paired Poisson deviance delta ($\bar{d} = \text{PD}_{B0} - \text{PD}_{\text{cand}} \in [+3.94 \times 10^{-6}, +1.73 \times 10^{-5}]$).
   - Multiplicity-adjusted stationary circular block bootstrap ($B=20,000$ replicates, mean block length $30$, seed $20260910$, shared indices across all candidates):
     - Under Bonferroni correction for 9 hypotheses ($q = 0.05 / 9 \approx 0.005555555555555556$), all 9 bootstrap lower bounds were strictly non-positive (ranging from $-7.76 \times 10^{-7}$ to $-3.84 \times 10^{-5}$).
   - Under the pre-registered support rule ($\bar{d} > 0 \land q_{\text{lower}} > 0$):
     - `SUPPORTED_CANDIDATE_COUNT = 0`.
     - `CHARACTERIZATION_OUTCOME = NO_SUPPORTED_ADDITIVE_CANDIDATE`.

---

## 5. Independent Final Audit Findings

An independent scientific audit conducted by Claude Opus 4.6 (Thinking) under zero execution authority verified all characterization artifacts prior to archival:

| Audit Item | Verification Method | Result | Discrepancy |
| :--- | :--- | :---: | :---: |
| **Tracked Worktree State** | Git status & tree hash comparison | **PASS** | 0 mutations |
| **Script Identity** | SHA-256 against frozen hash | **PASS** | 0 bits |
| **Fresh Holdout Boundary** | Range check (max target index = 5746) | **PASS** | 0 holdout rows |
| **Candidate Universe** | Pre-frozen 9-candidate check | **PASS** | Exact 9 candidates |
| **Formula Implementation** | Code audit (lines 214–258) | **PASS** | Exact analytical match |
| **Common Dataset Rows** | Count verification ($5,017 \times 10 = 50,170$) | **PASS** | 0 missing, 0 dupes |
| **Evidence File Hashes** | Cryptographic SHA-256 computation | **PASS** | Exact match |
| **Stage Metrics Recomputation** | Recomputed from CSV | **PASS** | `0.00e+00` |
| **Paired PD Deltas** | Recomputed from CSV | **PASS** | `0.00e+00` |
| **Bootstrap Protocol** | Seed, RNG, block length, shared indices | **PASS** | Fully verified |
| **Bootstrap Lower Bounds** | Linear quantile recomputation ($q=0.05/9$) | **PASS** | `0.00e+00` |
| **Support Rule Evaluation** | $\bar{d} > 0 \land q_{\text{lower}} > 0$ | **PASS** | 0 candidates supported |

- **Audit Severity Inventory**:
  - `BLOCKER_COUNT = 0`
  - `HIGH_COUNT = 0`
  - `MEDIUM_COUNT = 0`
  - `LOW_COUNT = 1` (Execution provenance supported via filesystem timestamps and output gate, rather than git-signed commit)
  - `OBSERVATION_COUNT = 2` (Point estimates positive but statistically indistinguishable from zero under multiplicity control; evidence archival required)

---

## 6. Fresh Holdout Boundary Discipline

Strict segregation of the fresh holdout partition has been maintained throughout the entire M3 lifecycle:

```text
FRESH_HOLDOUT_RANGE =
5747..7541

FRESH_HOLDOUT_ACCESSED_BY_M3_R1 =
NO

FRESH_HOLDOUT_STATUS =
UNUSED_BY_M3_R1
```

- Target indices 5747 through 7541 were neither read, fitted, evaluated, nor inspected by M3 v3 or M3-R1.
- The fresh holdout remains completely uncompromised, unpolluted, and sealed.
- In accordance with task governance, the fresh holdout is **not repurposed** in this task.

---

## 7. Permitted Scientific Conclusion & Retirement Boundary

The scientific adjudication of the M3 research line is bounded by the following explicit formulation:

> **The current M3 research line is retired after failure of M3 v3 independent validation and failure of the preregistered bounded additive characterization to establish multiplicity-adjusted paired-PD superiority over B0.**
>
> **This is a research-line retirement decision, not proof that all conceivable digit-factor hypotheses are impossible.**

### Negative Boundary Constraints

In keeping with rigorous scientific epistemology, the following claims are explicitly **disallowed**:
- It is **NOT** claimed that all digit-factor signals are impossible.
- It is **NOT** claimed that XSMB outcomes are proven to be IID uniform.
- It is **NOT** claimed that all future digit hypotheses are invalid.

### Operational Enforcement

- The current M3 digit-factor research line is permanently terminated.
- Additional search over lookback windows ($W$), shrinkage rates ($\alpha$), smoothing parameters ($\beta$), numerical optimization schemes, or alternative additive formulations within this line is strictly `FORBIDDEN`.

---

## 8. Persisted Archival Evidence Inventory

All evidence artifacts associated with the final M3-R1 characterization are permanently archived under:

`research_artifacts/xpis_v3_m3_digit_factor/m3_r1_final_additive_characterization/`

| Artifact | File Size | SHA-256 Hash |
| :--- | :---: | :--- |
| `run_m3_r1_additive_characterization.py` | 23,472 bytes | `3f071fc593933da1b2d5154b676bb221bddb009a469ac2fbfc55fc280e4773a7` |
| `daily_scores.csv` | 6,192,345 bytes | `b411529c14a92934b8ad53da4c4592698dc4e1de76da2a07b7c2fc420664b912` |
| `bootstrap_replicate_means.json` | 4,053,484 bytes | `d31dc24eb49a18b14819d799f49b2bc44757c665d14fe7474a13e4b5c4d24d21` |
| `characterization_summary.json` | 2,974 bytes | `f080adcd98540bbedea37286568bf550f1f1bd2df1a22a2b9233cd8501719703` |
| `evidence_manifest.json` | 1,448 bytes | `24e34d2d3bb7b5e40276b20d328e7e5abc04f5bdd658d4970c026e79dc95a5df` |

Archival verification confirms byte-exact identity between original execution outputs and persisted repository artifacts.
