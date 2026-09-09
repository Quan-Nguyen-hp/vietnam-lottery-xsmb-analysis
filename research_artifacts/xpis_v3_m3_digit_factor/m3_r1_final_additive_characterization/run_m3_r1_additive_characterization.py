"""M3-R1 Bounded Additive Characterization Execution.

Role: M3_R1_BOUNDED_ADDITIVE_CHARACTERIZATION_EXECUTOR
Scientific Authority: EXACTLY_ONE_DEVELOPMENT_ONLY_CHARACTERIZATION
Reference: B0_UNIFORM
Candidates: 9 pre-frozen additive candidates (W in {180, 365, 730}, beta=1, alpha in {0.05, 0.10, 0.20})
Target Range: 730..5746 (5017 common targets)
Fresh Holdout: 5747..7541 (SEALED, NOT ACCESSED)
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import scipy

from src.m3_digit_factor.authority import (
    CANONICAL_DATA_PATH,
    collect_repository_identity,
)
from src.m3_digit_factor.dataset import load_canonical_dataset
from src.m3_digit_factor.metrics import (
    DailyForecastMetrics,
    StageForecastMetrics,
    compute_daily_metrics,
    compute_stage_metrics,
    create_b0_forecast,
)
from src.m3_digit_factor.serialization import serialize_csv, serialize_json

EXPECTED_HEAD = "bbbbae07aa2bfb87c546b21d36eec2e5272a6399"
EXPECTED_TREE = "3fe155aa214c97c165622f10313aee4196d7a368"
EXPECTED_DATA_SHA256 = "3ae8970bfb2d54d9ad72d18aad1a1e3022c6231bc349dcfd01a78bc20e6a75c7"
EXPECTED_DATA_BLOB = "8474b4e7c10802c683080779bead75d7e56cf358"

COMMON_TARGET_START = 730
COMMON_TARGET_END = 5746
COMMON_TARGET_COUNT = 5017

FRESH_HOLDOUT_MIN_INDEX = 5747
FRESH_HOLDOUT_MAX_INDEX = 7541

OUTPUT_DIR_NAME = "m3_r1_additive_characterization_01"

FWER_ALPHA = 0.05
CANDIDATE_COUNT = 9
BONFERRONI_Q = FWER_ALPHA / CANDIDATE_COUNT  # 0.005555555555555556

BOOTSTRAP_SEED = 20260910
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_MEAN_BLOCK_LENGTH = 30
BOOTSTRAP_RESTART_PROBABILITY = 1.0 / 30.0
BOOTSTRAP_QUANTILE_METHOD = "linear"

ABS_TOL = 1e-12

CANDIDATE_CONFIGS = [
    # (candidate_id, W, beta, alpha)
    ("M3R1_ADD_W180_A005", 180, 1, 0.05),
    ("M3R1_ADD_W180_A010", 180, 1, 0.10),
    ("M3R1_ADD_W180_A020", 180, 1, 0.20),
    ("M3R1_ADD_W365_A005", 365, 1, 0.05),
    ("M3R1_ADD_W365_A010", 365, 1, 0.10),
    ("M3R1_ADD_W365_A020", 365, 1, 0.20),
    ("M3R1_ADD_W730_A005", 730, 1, 0.05),
    ("M3R1_ADD_W730_A010", 730, 1, 0.10),
    ("M3R1_ADD_W730_A020", 730, 1, 0.20),
]


def run_characterization() -> int:
    t_start = time.time()
    print("=" * 75)
    print("XPIS v3 M3-R1: Bounded Additive Characterization Execution")
    print("=" * 75)

    # 1. Repository Identity & Working Tree Verification
    identity = collect_repository_identity(repo_root)
    print(f"START_HEAD: {identity.source_commit}")
    print(f"SOURCE_TREE: {identity.source_tree}")

    res = subprocess.run(
        ["git", "status", "--porcelain", "-uno"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    worktree_status_before = res.stdout.strip()
    if worktree_status_before:
        print(f"STOP = SOURCE_IDENTITY_MISMATCH: Tracked worktree dirty:\n{worktree_status_before}")
        return 1

    if identity.source_commit != EXPECTED_HEAD or identity.source_tree != EXPECTED_TREE:
        print("STOP = SOURCE_IDENTITY_MISMATCH: Commit/Tree mismatch")
        print(f"  Got commit: {identity.source_commit} (expected {EXPECTED_HEAD})")
        print(f"  Got tree:   {identity.source_tree} (expected {EXPECTED_TREE})")
        return 1

    # 2. Canonical Data Verification
    csv_file = repo_root / CANONICAL_DATA_PATH
    raw_data_bytes = csv_file.read_bytes()
    data_sha256 = hashlib.sha256(raw_data_bytes).hexdigest()
    print(f"CANONICAL_DATA_PATH: {CANONICAL_DATA_PATH}")
    print(f"CANONICAL_DATA_SHA256: {data_sha256}")
    print(f"CANONICAL_DATA_BLOB: {identity.data_blob}")

    if data_sha256 != EXPECTED_DATA_SHA256 or identity.data_blob != EXPECTED_DATA_BLOB:
        print("STOP = CANONICAL_DATA_GATE: Data identity mismatch")
        return 1
    print("CANONICAL_DATA_GATE: PASS")

    # 3. Environment & Script Identity
    print(f"PYTHON_VERSION: {sys.version}")
    print(f"NUMPY_VERSION: {np.__version__}")
    print(f"SCIPY_VERSION: {scipy.__version__}")

    script_path = Path(__file__).resolve()
    script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
    print(f"SCRIPT_PATH: {script_path}")
    print(f"SCRIPT_SHA256: {script_sha256}")

    # 4. Output Directory Gate
    output_dir = repo_root / "scratch" / OUTPUT_DIR_NAME
    if output_dir.exists():
        print(f"STOP = CHARACTERIZATION_OUTPUT_ALREADY_EXISTS: {output_dir}")
        return 1
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIRECTORY_CREATED: {output_dir}")

    # 5. Ingest Canonical Dataset
    dataset = load_canonical_dataset(repo_root)
    print(f"CANONICAL_DATASET_ROW_COUNT: {dataset.row_count}")

    # Target indices check
    target_indices = tuple(range(COMMON_TARGET_START, COMMON_TARGET_END + 1))
    if len(target_indices) != COMMON_TARGET_COUNT:
        print(f"STOP = COMMON_TARGET_COUNT_MISMATCH: {len(target_indices)} != {COMMON_TARGET_COUNT}")
        return 1
    if target_indices[0] != COMMON_TARGET_START or target_indices[-1] != COMMON_TARGET_END:
        print(f"STOP = COMMON_TARGET_RANGE_MISMATCH: [{target_indices[0]}, {target_indices[-1]}]")
        return 1
    if max(target_indices) >= FRESH_HOLDOUT_MIN_INDEX:
        print(f"STOP = FRESH_HOLDOUT_VIOLATION: max target index {max(target_indices)} >= {FRESH_HOLDOUT_MIN_INDEX}")
        return 1
    print(f"COMMON_TARGET_RANGE: {COMMON_TARGET_START}..{COMMON_TARGET_END} ({COMMON_TARGET_COUNT} targets)")
    print(f"FRESH_HOLDOUT_SEAL: PASS (targets <= {COMMON_TARGET_END} < {FRESH_HOLDOUT_MIN_INDEX})")

    # 6. Evaluate B0 Uniform Baseline
    print("\n--- Evaluating Reference: B0_UNIFORM ---")
    b0_forecast = create_b0_forecast()
    if b0_forecast.shape != (100,):
        print("STOP = B0_FORECAST_CONTRACT_FAILURE: shape mismatch")
        return 1
    if not np.all(b0_forecast > 0):
        print("STOP = B0_FORECAST_CONTRACT_FAILURE: non-positive elements")
        return 1
    if abs(float(np.sum(b0_forecast)) - 27.0) > 1e-12:
        print("STOP = B0_FORECAST_CONTRACT_FAILURE: sum != 27")
        return 1

    b0_daily_metrics: list[DailyForecastMetrics] = []
    b0_csv_rows: list[list[object]] = []

    for t in target_indices:
        target_date = str(dataset.recorded_dates[t])
        y_t = dataset.count_matrix[t]
        m = compute_daily_metrics(y_t, b0_forecast)
        b0_daily_metrics.append(m)
        b0_csv_rows.append([
            t,
            target_date,
            "B0",
            "B0_UNIFORM",
            None,
            None,
            None,
            m.poisson_deviance,
            m.mae,
            m.rmse,
        ])

    b0_stage = compute_stage_metrics(b0_daily_metrics)
    print(f"B0 Stage Metrics: PD={b0_stage.poisson_deviance:.17g}, MAE={b0_stage.mae:.17g}, RMSE={b0_stage.rmse:.17g}")

    # 7. Evaluate 9 Additive Candidates
    candidate_ids = [cfg[0] for cfg in CANDIDATE_CONFIGS]
    cand_daily_metrics: dict[str, list[DailyForecastMetrics]] = {cid: [] for cid in candidate_ids}
    cand_csv_rows: dict[str, list[list[object]]] = {cid: [] for cid in candidate_ids}

    # Group configs by W to avoid redundant marginal computation
    windows = (180, 365, 730)
    w_to_configs: dict[int, list[tuple[str, int, int, float]]] = {w: [] for w in windows}
    for cfg in CANDIDATE_CONFIGS:
        w_to_configs[cfg[1]].append(cfg)

    print("\n--- Evaluating Additive Candidates ---")
    for t_idx, t in enumerate(target_indices):
        target_date = str(dataset.recorded_dates[t])
        y_t = dataset.count_matrix[t]

        for W in windows:
            # W recorded dates strictly before target t
            history_counts = dataset.count_matrix[t - W : t]
            if history_counts.shape != (W, 100):
                print(f"STOP = ADDITIVE_CHARACTERIZATION_CONTRACT_FAILURE: history shape {history_counts.shape} != ({W}, 100)")
                return 1

            X_flat = np.sum(history_counts, axis=0)  # (100,)
            if int(np.sum(X_flat)) != 27 * W:
                print(f"STOP = ADDITIVE_CHARACTERIZATION_CONTRACT_FAILURE: X sum {np.sum(X_flat)} != {27 * W}")
                return 1

            X = X_flat.reshape((10, 10))
            H = X.sum(axis=1)  # Head marginal (10,)
            C = X.sum(axis=0)  # Tail marginal (10,)

            if int(np.sum(H)) != 27 * W or int(np.sum(C)) != 27 * W:
                print("STOP = ADDITIVE_CHARACTERIZATION_CONTRACT_FAILURE: Marginal sums != 27*W")
                return 1

            # Frozen smoothing: BETA = 1
            denom = 27.0 * W + 10.0
            r = (H.astype(np.float64) + 1.0) / denom
            c = (C.astype(np.float64) + 1.0) / denom

            if abs(float(np.sum(r)) - 1.0) > 1e-15:
                print(f"STOP = ADDITIVE_CHARACTERIZATION_CONTRACT_FAILURE: sum(r) deviation {abs(float(np.sum(r)) - 1.0)} > 1e-15")
                return 1
            if abs(float(np.sum(c)) - 1.0) > 1e-15:
                print(f"STOP = ADDITIVE_CHARACTERIZATION_CONTRACT_FAILURE: sum(c) deviation {abs(float(np.sum(c)) - 1.0)} > 1e-15")
                return 1

            p_add = np.outer(r, c)  # (10, 10)

            for cid, _, beta, alpha in w_to_configs[W]:
                p = (1.0 - alpha) * 0.01 + alpha * p_add
                mu = 27.0 * p
                mu_1d = mu.ravel()

                if not np.all(mu_1d > 0.0):
                    print(f"STOP = ADDITIVE_CHARACTERIZATION_CONTRACT_FAILURE: mu not strictly positive for {cid}")
                    return 1
                if abs(float(np.sum(mu_1d)) - 27.0) > 1e-12:
                    print(f"STOP = ADDITIVE_CHARACTERIZATION_CONTRACT_FAILURE: abs(sum(mu)-27) = {abs(float(np.sum(mu_1d)) - 27.0)} > 1e-12")
                    return 1

                m = compute_daily_metrics(y_t, mu_1d)
                cand_daily_metrics[cid].append(m)
                cand_csv_rows[cid].append([
                    t,
                    target_date,
                    "M3_R1",
                    cid,
                    W,
                    beta,
                    alpha,
                    m.poisson_deviance,
                    m.mae,
                    m.rmse,
                ])

    cand_stages: dict[str, StageForecastMetrics] = {}
    for cid in candidate_ids:
        cand_stages[cid] = compute_stage_metrics(cand_daily_metrics[cid])
        print(f"Candidate {cid}: PD={cand_stages[cid].poisson_deviance:.17g}, MAE={cand_stages[cid].mae:.17g}, RMSE={cand_stages[cid].rmse:.17g}")

    # 8. Persist daily_scores.csv
    csv_header = [
        "target_index",
        "target_date",
        "model_id",
        "candidate_id",
        "window_days",
        "beta",
        "alpha",
        "poisson_deviance",
        "mae",
        "rmse",
    ]
    all_csv_rows: list[list[object]] = []
    all_csv_rows.extend(b0_csv_rows)
    for cid in candidate_ids:
        all_csv_rows.extend(cand_csv_rows[cid])

    if len(all_csv_rows) != 50170:
        print(f"STOP = DAILY_SCORE_ROW_COUNT_MISMATCH: {len(all_csv_rows)} != 50170")
        return 1

    daily_scores_path = output_dir / "daily_scores.csv"
    csv_bytes = serialize_csv(csv_header, all_csv_rows)
    daily_scores_path.write_bytes(csv_bytes)
    daily_scores_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    daily_scores_file_size = len(csv_bytes)
    print(f"\nPersisted daily_scores.csv: {daily_scores_path}")
    print(f"DAILY_SCORES_SHA256: {daily_scores_sha256}")
    print(f"DAILY_SCORES_FILE_SIZE: {daily_scores_file_size} bytes")
    print(f"DAILY_SCORE_ROW_COUNT: {len(all_csv_rows)}")

    # 9. Compute Paired PD Deltas
    b0_pds = np.array([m.poisson_deviance for m in b0_daily_metrics], dtype=np.float64)
    paired_d: dict[str, np.ndarray] = {}
    observed_mean_deltas: dict[str, float] = {}

    print("\n--- Paired PD Mean Deltas (PD_B0 - PD_candidate) ---")
    for cid in candidate_ids:
        cand_pds = np.array([m.poisson_deviance for m in cand_daily_metrics[cid]], dtype=np.float64)
        d_vec = b0_pds - cand_pds
        paired_d[cid] = d_vec
        mean_delta = float(np.mean(d_vec))
        observed_mean_deltas[cid] = mean_delta
        print(f"  {cid}: observed_mean_delta = {mean_delta:.17g}")

    # 10. Multiplicity-Adjusted Stationary Circular Bootstrap
    print("\n--- Running Multiplicity-Adjusted Stationary Circular Bootstrap ---")
    print(f"  Seed: {BOOTSTRAP_SEED}")
    print(f"  Replicates: {BOOTSTRAP_REPLICATES}")
    print(f"  Mean Block Length: {BOOTSTRAP_MEAN_BLOCK_LENGTH}")
    print(f"  Restart Probability: {BOOTSTRAP_RESTART_PROBABILITY}")
    print(f"  FWER Alpha: {FWER_ALPHA}")
    print(f"  Bonferroni q: {BONFERRONI_Q} (0.05 / 9)")
    print(f"  Shared indices across all 9 candidates: TRUE")

    t_boot_start = time.time()
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    n = COMMON_TARGET_COUNT
    replications = BOOTSTRAP_REPLICATES
    p_restart = BOOTSTRAP_RESTART_PROBABILITY

    indices = np.empty((replications, n), dtype=np.int32)
    rand = rng.random
    randint = rng.integers

    for b in range(replications):
        prev = int(randint(0, n))
        indices[b, 0] = prev
        for m in range(1, n):
            if rand() < p_restart:
                prev = int(randint(0, n))
            else:
                prev = (prev + 1) % n
            indices[b, m] = prev

    print(f"  Bootstrap indices generated in {time.time() - t_boot_start:.2f}s")

    replicate_means: dict[str, np.ndarray] = {
        cid: np.empty(replications, dtype=np.float64) for cid in candidate_ids
    }
    chunk_size = 2000
    for start in range(0, replications, chunk_size):
        end = min(start + chunk_size, replications)
        sub_indices = indices[start:end]
        for cid in candidate_ids:
            replicate_means[cid][start:end] = paired_d[cid][sub_indices].mean(axis=1)

    print(f"  Replicate means computed in {time.time() - t_boot_start:.2f}s total")

    # Compute Bonferroni Adjusted Lower Bounds
    bootstrap_lowers: dict[str, float] = {}
    for cid in candidate_ids:
        rep_arr = replicate_means[cid]
        lower_val = float(np.quantile(rep_arr, q=BONFERRONI_Q, method=BOOTSTRAP_QUANTILE_METHOD))
        bootstrap_lowers[cid] = lower_val
        print(f"  {cid}: bootstrap_lower (q={BONFERRONI_Q:.17g}) = {lower_val:.17g}")

    # 11. Persist bootstrap_replicate_means.json
    bootstrap_path = output_dir / "bootstrap_replicate_means.json"
    bootstrap_evidence = {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "mean_block_length": BOOTSTRAP_MEAN_BLOCK_LENGTH,
        "restart_probability": BOOTSTRAP_RESTART_PROBABILITY,
        "FWER_alpha": FWER_ALPHA,
        "candidate_count": CANDIDATE_COUNT,
        "candidate_q": BONFERRONI_Q,
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "shared_indices": True,
        "candidates": {
            cid: [float(x) for x in replicate_means[cid]]
            for cid in candidate_ids
        },
    }
    bootstrap_bytes = serialize_json(bootstrap_evidence)
    bootstrap_path.write_bytes(bootstrap_bytes)
    bootstrap_evidence_sha256 = hashlib.sha256(bootstrap_bytes).hexdigest()
    bootstrap_evidence_file_size = len(bootstrap_bytes)
    print(f"\nPersisted bootstrap_replicate_means.json: {bootstrap_path}")
    print(f"BOOTSTRAP_EVIDENCE_SHA256: {bootstrap_evidence_sha256}")
    print(f"BOOTSTRAP_EVIDENCE_FILE_SIZE: {bootstrap_evidence_file_size} bytes")

    # 12. Support Rule & Characterization Outcome
    cand_lookup = {cfg[0]: {"W": cfg[1], "beta": cfg[2], "alpha": cfg[3]} for cfg in CANDIDATE_CONFIGS}
    pd_supported: dict[str, bool] = {}
    for cid in candidate_ids:
        is_sup = bool(observed_mean_deltas[cid] > 0.0 and bootstrap_lowers[cid] > 0.0)
        pd_supported[cid] = is_sup

    supported_candidates = [cid for cid in candidate_ids if pd_supported[cid]]
    supported_candidate_count = len(supported_candidates)

    if supported_candidate_count == 0:
        characterization_outcome = "NO_SUPPORTED_ADDITIVE_CANDIDATE"
        diagnostic_ranking: list[dict[str, object]] = []
    else:
        characterization_outcome = "SUPPORTED_ADDITIVE_CANDIDATE_EXISTS"
        # Rank by: 1. bootstrap_lower desc, 2. observed_mean_delta desc, 3. W asc, 4. alpha asc
        ranked_cids = sorted(
            supported_candidates,
            key=lambda c: (
                -bootstrap_lowers[c],
                -observed_mean_deltas[c],
                cand_lookup[c]["W"],
                cand_lookup[c]["alpha"],
            ),
        )
        diagnostic_ranking = [
            {
                "rank": idx + 1,
                "candidate_id": c,
                "window_days": cand_lookup[c]["W"],
                "beta": cand_lookup[c]["beta"],
                "alpha": cand_lookup[c]["alpha"],
                "bootstrap_lower": bootstrap_lowers[c],
                "observed_mean_delta": observed_mean_deltas[c],
            }
            for idx, c in enumerate(ranked_cids)
        ]

    print(f"\nPD_SUPPORTED_CANDIDATES: {supported_candidates}")
    print(f"SUPPORTED_CANDIDATE_COUNT: {supported_candidate_count}")
    print(f"CHARACTERIZATION_OUTCOME: {characterization_outcome}")
    print(f"DIAGNOSTIC_RANKING: {diagnostic_ranking}")

    # 13. Persist characterization_summary.json
    summary_path = output_dir / "characterization_summary.json"
    summary_payload = {
        "source_commit": identity.source_commit,
        "source_tree": identity.source_tree,
        "canonical_data_sha256": data_sha256,
        "script_sha256": script_sha256,
        "common_target_bounds": [COMMON_TARGET_START, COMMON_TARGET_END],
        "common_target_count": COMMON_TARGET_COUNT,
        "candidate_universe": candidate_ids,
        "b0_metrics": {
            "poisson_deviance": b0_stage.poisson_deviance,
            "mae": b0_stage.mae,
            "rmse": b0_stage.rmse,
        },
        "candidate_metrics": {
            cid: {
                "poisson_deviance": cand_stages[cid].poisson_deviance,
                "mae": cand_stages[cid].mae,
                "rmse": cand_stages[cid].rmse,
            }
            for cid in candidate_ids
        },
        "all_observed_paired_pd_means": observed_mean_deltas,
        "all_adjusted_bootstrap_lowers": bootstrap_lowers,
        "pd_supported_per_candidate": pd_supported,
        "supported_candidate_count": supported_candidate_count,
        "diagnostic_ranking": diagnostic_ranking,
        "characterization_outcome": characterization_outcome,
        "fresh_holdout_accessed": False,
    }
    summary_bytes = serialize_json(summary_payload)
    summary_path.write_bytes(summary_bytes)
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    summary_file_size = len(summary_bytes)
    print(f"\nPersisted characterization_summary.json: {summary_path}")
    print(f"CHARACTERIZATION_SUMMARY_SHA256: {summary_sha256}")
    print(f"SUMMARY_FILE_SIZE: {summary_file_size} bytes")

    # 14. Independent Read-Back Verification (Section 20)
    print("\n--- Independent Read-Back Verification ---")
    readback_scores: dict[str, dict[str, list]] = {
        "B0_UNIFORM": {"pd": [], "mae": [], "rmse": []}
    }
    for cid in candidate_ids:
        readback_scores[cid] = {"pd": [], "mae": [], "rmse": []}

    row_count = 0
    with daily_scores_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            cid = row["candidate_id"]
            if cid not in readback_scores:
                print(f"FAIL = READBACK: Unknown candidate in CSV: {cid}")
                return 1
            readback_scores[cid]["pd"].append(float(row["poisson_deviance"]))
            readback_scores[cid]["mae"].append(float(row["mae"]))
            readback_scores[cid]["rmse"].append(float(row["rmse"]))

    print(f"  Verified row count: {row_count} (expected: 50170)")
    if row_count != 50170:
        print(f"FAIL = READBACK_ROW_COUNT_MISMATCH: {row_count} != 50170")
        return 1

    max_abs_metric_discrepancy = 0.0

    # Verify B0
    b0_rb_pd = float(np.mean(readback_scores["B0_UNIFORM"]["pd"]))
    b0_rb_mae = float(np.mean(readback_scores["B0_UNIFORM"]["mae"]))
    b0_rb_rmse = math.sqrt(float(np.mean(np.array(readback_scores["B0_UNIFORM"]["rmse"]) ** 2)))
    max_abs_metric_discrepancy = max(
        max_abs_metric_discrepancy,
        abs(b0_rb_pd - b0_stage.poisson_deviance),
        abs(b0_rb_mae - b0_stage.mae),
        abs(b0_rb_rmse - b0_stage.rmse),
    )

    # Verify Candidates Stage Metrics & Deltas
    b0_rb_pd_arr = np.array(readback_scores["B0_UNIFORM"]["pd"], dtype=np.float64)
    readback_deltas: dict[str, float] = {}

    for cid in candidate_ids:
        c_rb_pd = float(np.mean(readback_scores[cid]["pd"]))
        c_rb_mae = float(np.mean(readback_scores[cid]["mae"]))
        c_rb_rmse = math.sqrt(float(np.mean(np.array(readback_scores[cid]["rmse"]) ** 2)))
        max_abs_metric_discrepancy = max(
            max_abs_metric_discrepancy,
            abs(c_rb_pd - cand_stages[cid].poisson_deviance),
            abs(c_rb_mae - cand_stages[cid].mae),
            abs(c_rb_rmse - cand_stages[cid].rmse),
        )

        c_rb_pd_arr = np.array(readback_scores[cid]["pd"], dtype=np.float64)
        delta_arr = b0_rb_pd_arr - c_rb_pd_arr
        mean_d = float(np.mean(delta_arr))
        readback_deltas[cid] = mean_d
        max_abs_metric_discrepancy = max(
            max_abs_metric_discrepancy,
            abs(mean_d - observed_mean_deltas[cid]),
        )

    # Re-open bootstrap evidence
    with bootstrap_path.open("r", encoding="utf-8") as f:
        loaded_bootstrap = json.load(f)

    readback_bootstrap_lowers: dict[str, float] = {}
    readback_pd_supported: dict[str, bool] = {}

    for cid in candidate_ids:
        loaded_reps = np.array(loaded_bootstrap["candidates"][cid], dtype=np.float64)
        if len(loaded_reps) != BOOTSTRAP_REPLICATES:
            print(f"FAIL = READBACK: Bootstrap replicate count mismatch for {cid}")
            return 1
        q_rb = float(np.quantile(loaded_reps, q=BONFERRONI_Q, method=BOOTSTRAP_QUANTILE_METHOD))
        readback_bootstrap_lowers[cid] = q_rb
        max_abs_metric_discrepancy = max(
            max_abs_metric_discrepancy,
            abs(q_rb - bootstrap_lowers[cid]),
        )
        readback_pd_supported[cid] = bool(readback_deltas[cid] > 0.0 and q_rb > 0.0)

    rb_supported_count = sum(1 for v in readback_pd_supported.values() if v)
    if rb_supported_count == 0:
        rb_outcome = "NO_SUPPORTED_ADDITIVE_CANDIDATE"
    else:
        rb_outcome = "SUPPORTED_ADDITIVE_CANDIDATE_EXISTS"

    if rb_outcome != characterization_outcome:
        print(f"FAIL = READBACK: Outcome mismatch {rb_outcome} != {characterization_outcome}")
        return 1

    print(f"  MAX_ABS_METRIC_DISCREPANCY: {max_abs_metric_discrepancy:.17g}")
    if max_abs_metric_discrepancy > ABS_TOL:
        print(f"FAIL = READBACK: max discrepancy {max_abs_metric_discrepancy} > {ABS_TOL}")
        return 1

    print("PERSISTED_READBACK_VERIFICATION: PASS")
    print(f"\nTotal Execution Time: {time.time() - t_start:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(run_characterization())
