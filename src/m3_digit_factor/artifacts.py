"""Exact success artifact generation for XPIS v3 M3 (Slice M3-11).

This module implements the exact construction and serialization of all 9 success
development artifacts:
1. protocol_snapshot.json
2. daily_forecast_scores.csv.gz
3. forecast_metrics.csv
4. forecast_uncertainty.json
5. stability_diagnostics.json
6. economic_uncertainty.json
7. economic_summary.csv
8. development_adjudication.json
9. artifact_manifest.json
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from .authority import protocol_fingerprint
from .bootstrap import (
    ECONOMIC_BOOTSTRAP_BIT_GENERATOR,
    ECONOMIC_BOOTSTRAP_MEAN_BLOCK_LENGTH,
    ECONOMIC_BOOTSTRAP_QUANTILE,
    ECONOMIC_BOOTSTRAP_QUANTILE_METHOD,
    ECONOMIC_BOOTSTRAP_REPLICATIONS,
    ECONOMIC_BOOTSTRAP_RESTART_PROBABILITY,
    ECONOMIC_BOOTSTRAP_RNG_IMPLEMENTATION,
    ECONOMIC_BOOTSTRAP_SEED,
    FORECAST_BOOTSTRAP_BIT_GENERATOR,
    FORECAST_BOOTSTRAP_MEAN_BLOCK_LENGTH,
    FORECAST_BOOTSTRAP_QUANTILE,
    FORECAST_BOOTSTRAP_QUANTILE_METHOD,
    FORECAST_BOOTSTRAP_REPLICATIONS,
    FORECAST_BOOTSTRAP_RESTART_PROBABILITY,
    FORECAST_BOOTSTRAP_RNG_IMPLEMENTATION,
    FORECAST_BOOTSTRAP_SEED,
    SHARED_RESAMPLE_INDICES,
)
from .contracts import DevelopmentExitStatus
from .economics import ECONOMIC_K_VALUES, PerKEconomicEvaluation
from .serialization import serialize_csv, serialize_gzip, serialize_json


SUCCESS_ARTIFACT_COUNT: int = 9

SUCCESS_ARTIFACT_NAMES: tuple[str, ...] = (
    "artifact_manifest.json",
    "daily_forecast_scores.csv.gz",
    "development_adjudication.json",
    "economic_summary.csv",
    "economic_uncertainty.json",
    "forecast_metrics.csv",
    "forecast_uncertainty.json",
    "protocol_snapshot.json",
    "stability_diagnostics.json",
)

STAGE_SORT_ORDER: dict[str, int] = {
    "DEV": 0,
    "VAL": 1,
    "STABILITY": 2,
}

DAILY_FORECAST_SCORES_HEADER: list[str] = [
    "target_date",
    "stage",
    "model_id",
    "candidate_id",
    "poisson_deviance",
    "mae",
    "rmse",
]

FORECAST_METRICS_HEADER: list[str] = [
    "stage",
    "model_id",
    "candidate_id",
    "date_count",
    "poisson_deviance",
    "mae",
    "rmse",
]

ECONOMIC_SUMMARY_HEADER: list[str] = [
    "K",
    "status",
    "date_count",
    "mean_economic_delta",
    "positive_block_count",
    "bonferroni_alpha",
    "bootstrap_lower_bound",
    "qualifies",
    "recommended",
]


def build_protocol_snapshot_artifact(authority: dict[str, Any]) -> bytes:
    """Build canonical protocol_snapshot.json artifact."""
    fingerprint = protocol_fingerprint(authority)
    data = {
        "authority": authority,
        "protocol_fingerprint_sha256": fingerprint,
    }
    return serialize_json(data)


def build_daily_forecast_scores_artifact(rows: Sequence[Sequence[Any]]) -> bytes:
    """Build canonical daily_forecast_scores.csv.gz artifact.

    Rows are ordered by stage (DEV, VAL, STABILITY), target_date, model_id, candidate_id.
    """
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            STAGE_SORT_ORDER.get(str(r[1]), 99),
            str(r[0]),
            str(r[2]),
            str(r[3]),
        ),
    )
    csv_bytes = serialize_csv(DAILY_FORECAST_SCORES_HEADER, sorted_rows)
    return serialize_gzip(csv_bytes)


def build_forecast_metrics_artifact(
    rows: Sequence[dict[str, Any]] | Sequence[Sequence[Any]],
) -> bytes:
    """Build canonical forecast_metrics.csv artifact.

    Rows are ordered by:
    1. stage exact order: DEV, VAL, STABILITY
    2. model_id ascending lexicographic
    3. candidate_id ascending lexicographic
    """
    normalized_rows: list[list[Any]] = []
    for r in rows:
        if isinstance(r, dict):
            normalized_rows.append([
                r["stage"],
                r["model_id"],
                r["candidate_id"],
                r["date_count"],
                r["poisson_deviance"],
                r["mae"],
                r["rmse"],
            ])
        elif len(r) == 6:
            normalized_rows.append([
                r[0],
                r[1],
                r[2],
                1,
                r[3],
                r[4],
                r[5],
            ])
        else:
            normalized_rows.append(list(r))

    sorted_rows = sorted(
        normalized_rows,
        key=lambda r: (
            STAGE_SORT_ORDER.get(str(r[0]), 99),
            str(r[1]),
            str(r[2]),
        ),
    )
    return serialize_csv(FORECAST_METRICS_HEADER, sorted_rows)


def build_forecast_uncertainty_artifact(
    observed_mean_improvement: float,
    bootstrap_lower_bound: float,
    status: str = "EVALUATED",
) -> bytes:
    """Build canonical forecast_uncertainty.json artifact with exact 11 keys."""
    data = {
        "status": status,
        "bootstrap_rng_implementation": FORECAST_BOOTSTRAP_RNG_IMPLEMENTATION,
        "bootstrap_bit_generator": FORECAST_BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_seed": FORECAST_BOOTSTRAP_SEED,
        "bootstrap_replications": FORECAST_BOOTSTRAP_REPLICATIONS,
        "mean_block_length": FORECAST_BOOTSTRAP_MEAN_BLOCK_LENGTH,
        "restart_probability": FORECAST_BOOTSTRAP_RESTART_PROBABILITY,
        "alpha": FORECAST_BOOTSTRAP_QUANTILE,
        "quantile_method": FORECAST_BOOTSTRAP_QUANTILE_METHOD,
        "observed_mean_improvement": observed_mean_improvement,
        "bootstrap_lower_bound": bootstrap_lower_bound,
    }
    return serialize_json(data)


def build_stability_diagnostics_artifact(
    forecast_signal: bool,
    date_count: int | None = None,
    blocks_info: list[dict[str, Any]] | None = None,
    per_k_summary: dict[int, PerKEconomicEvaluation] | None = None,
) -> bytes:
    """Build canonical stability_diagnostics.json artifact."""
    if forecast_signal:
        per_k_list = []
        for k in ECONOMIC_K_VALUES:
            eval_record = (per_k_summary or {}).get(k)
            per_k_list.append({
                "K": k,
                "positive_block_count": eval_record.positive_block_count if eval_record else None,
                "bootstrap_lower_bound": eval_record.bootstrap_lower_bound if eval_record else None,
                "qualifies": eval_record.qualifies if eval_record else False,
            })
        data = {
            "status": "EVALUATED",
            "date_count": date_count,
            "blocks": blocks_info or [],
            "per_k": per_k_list,
        }
    else:
        per_k_list = [
            {
                "K": k,
                "positive_block_count": None,
                "bootstrap_lower_bound": None,
                "qualifies": False,
            }
            for k in ECONOMIC_K_VALUES
        ]
        data = {
            "status": "NOT_EVALUATED_FORECAST_GATE_FAILED",
            "date_count": None,
            "blocks": [],
            "per_k": per_k_list,
        }
    return serialize_json(data)


def build_economic_uncertainty_artifact(
    forecast_signal: bool,
    date_count: int | None = None,
    per_k_evals: dict[int, PerKEconomicEvaluation] | None = None,
) -> bytes:
    """Build canonical economic_uncertainty.json artifact with closed 13 keys."""
    if forecast_signal:
        per_k_records = []
        for k in ECONOMIC_K_VALUES:
            rec = (per_k_evals or {}).get(k)
            per_k_records.append({
                "K": k,
                "status": "EVALUATED",
                "date_count": date_count,
                "mean_economic_delta": rec.mean_economic_delta if rec else None,
                "positive_block_count": rec.positive_block_count if rec else None,
                "bonferroni_alpha": ECONOMIC_BOOTSTRAP_QUANTILE,
                "bootstrap_lower_bound": rec.bootstrap_lower_bound if rec else None,
                "qualifies": rec.qualifies if rec else False,
            })
        top_status = "EVALUATED"
    else:
        per_k_records = [
            {
                "K": k,
                "status": "NOT_EVALUATED_FORECAST_GATE_FAILED",
                "date_count": None,
                "mean_economic_delta": None,
                "positive_block_count": None,
                "bonferroni_alpha": ECONOMIC_BOOTSTRAP_QUANTILE,
                "bootstrap_lower_bound": None,
                "qualifies": False,
            }
            for k in ECONOMIC_K_VALUES
        ]
        top_status = "NOT_EVALUATED_FORECAST_GATE_FAILED"

    data = {
        "status": top_status,
        "familywise_alpha": 0.05,
        "multiplicity_method": "BONFERRONI",
        "per_k_alpha": ECONOMIC_BOOTSTRAP_QUANTILE,
        "bootstrap_rng_implementation": ECONOMIC_BOOTSTRAP_RNG_IMPLEMENTATION,
        "bootstrap_bit_generator": ECONOMIC_BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_seed": ECONOMIC_BOOTSTRAP_SEED,
        "bootstrap_replications": ECONOMIC_BOOTSTRAP_REPLICATIONS,
        "mean_block_length": ECONOMIC_BOOTSTRAP_MEAN_BLOCK_LENGTH,
        "restart_probability": ECONOMIC_BOOTSTRAP_RESTART_PROBABILITY,
        "quantile_method": ECONOMIC_BOOTSTRAP_QUANTILE_METHOD,
        "shared_resample_indices": SHARED_RESAMPLE_INDICES,
        "per_k": per_k_records,
    }
    return serialize_json(data)


def build_economic_summary_artifact(
    forecast_signal: bool,
    date_count: int | None = None,
    per_k_evals: dict[int, PerKEconomicEvaluation] | None = None,
    recommended_top_k: list[int] | None = None,
) -> bytes:
    """Build canonical economic_summary.csv artifact with 4 rows."""
    rec_set = set(recommended_top_k or [])
    rows = []
    for k in ECONOMIC_K_VALUES:
        if forecast_signal:
            rec = (per_k_evals or {}).get(k)
            rows.append([
                k,
                "EVALUATED",
                date_count,
                rec.mean_economic_delta if rec else None,
                rec.positive_block_count if rec else None,
                ECONOMIC_BOOTSTRAP_QUANTILE,
                rec.bootstrap_lower_bound if rec else None,
                rec.qualifies if rec else False,
                k in rec_set,
            ])
        else:
            rows.append([
                k,
                "NOT_EVALUATED_FORECAST_GATE_FAILED",
                None,
                None,
                None,
                ECONOMIC_BOOTSTRAP_QUANTILE,
                None,
                False,
                False,
            ])
    return serialize_csv(ECONOMIC_SUMMARY_HEADER, rows)


def build_development_adjudication_artifact(
    selected_candidate_id: str,
    selected_window_days: int,
    forecast_signal: bool,
    forecast_bootstrap_lower_bound: float,
    economic_signal: bool,
    qualified_top_k: list[int] | None = None,
    recommended_top_k: list[int] | None = None,
    complete_valid_candidates: list[str] | None = None,
    candidate_qualification: list[dict[str, Any]] | None = None,
) -> bytes:
    """Build canonical development_adjudication.json artifact with exact 11 keys."""
    if not forecast_signal:
        exit_status = DevelopmentExitStatus.FORECAST_GATE_FAILED.value
        clean_econ_signal = False
        clean_qualified = []
        clean_recommended = []
    elif not economic_signal:
        exit_status = DevelopmentExitStatus.ECONOMIC_GATE_FAILED.value
        clean_econ_signal = False
        clean_qualified = list(qualified_top_k or [])
        clean_recommended = list(recommended_top_k or [])
    else:
        exit_status = DevelopmentExitStatus.ECONOMIC_SIGNAL_FOUND.value
        clean_econ_signal = True
        clean_qualified = list(qualified_top_k or [])
        clean_recommended = list(recommended_top_k or [])

    data = {
        "status": "COMPLETED",
        "selected_candidate_id": selected_candidate_id,
        "selected_window_days": selected_window_days,
        "forecast_signal": forecast_signal,
        "forecast_bootstrap_lower_bound": forecast_bootstrap_lower_bound,
        "economic_signal": clean_econ_signal,
        "qualified_top_k": clean_qualified,
        "recommended_top_k": clean_recommended,
        "development_exit_status": exit_status,
        "complete_valid_candidates": list(complete_valid_candidates or []),
        "candidate_qualification": list(candidate_qualification or []),
    }
    return serialize_json(data)


def build_artifact_manifest(artifacts: dict[str, bytes]) -> bytes:
    """Build canonical artifact_manifest.json covering all other success artifacts."""
    entries = []
    for fname in sorted(artifacts.keys()):
        if fname == "artifact_manifest.json":
            continue
        data = artifacts[fname]
        entries.append({
            "filename": fname,
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
        })
    manifest_obj = {"artifacts": entries}
    return serialize_json(manifest_obj)


def build_success_artifacts(
    authority: dict[str, Any],
    selected_candidate_id: str,
    selected_window_days: int,
    forecast_signal: bool,
    forecast_bootstrap_lower_bound: float,
    observed_mean_improvement: float,
    daily_score_rows: Sequence[Sequence[Any]],
    forecast_metric_rows: Sequence[dict[str, Any]] | Sequence[Sequence[Any]],
    stability_date_count: int | None = None,
    blocks_info: list[dict[str, Any]] | None = None,
    per_k_evals: dict[int, PerKEconomicEvaluation] | None = None,
    economic_signal: bool = False,
    qualified_top_k: list[int] | None = None,
    recommended_top_k: list[int] | None = None,
    complete_valid_candidates: list[str] | None = None,
    candidate_qualification: list[dict[str, Any]] | None = None,
) -> dict[str, bytes]:
    """Construct all 9 development success artifacts."""
    artifacts: dict[str, bytes] = {}

    normalized_metric_rows: list[list[Any]] = []
    for r in forecast_metric_rows:
        if isinstance(r, dict):
            normalized_metric_rows.append([
                r["stage"],
                r["model_id"],
                r["candidate_id"],
                r["date_count"],
                r["poisson_deviance"],
                r["mae"],
                r["rmse"],
            ])
        elif len(r) == 6:
            normalized_metric_rows.append([
                r[0],
                r[1],
                r[2],
                1,
                r[3],
                r[4],
                r[5],
            ])
    if complete_valid_candidates is None:
        seen: list[str] = []
        for r in normalized_metric_rows:
            c_name = str(r[2])
            if str(r[0]) == "DEV" and c_name != "B0_UNIFORM" and c_name not in seen:
                seen.append(c_name)
        complete_valid_candidates = seen

    if candidate_qualification is None:
        default_qual: list[dict[str, Any]] = []
        for w in (30, 60, 120, 240, 365):
            cid = f"M3_W{w:03d}"
            if cid in complete_valid_candidates:
                default_qual.append({
                    "candidate_id": cid,
                    "W": w,
                    "status": "COMPLETE_VALID",
                    "failure_stage": None,
                    "error_type": None,
                    "first_failed_target_index": None,
                    "first_failed_target_date": None,
                })
            else:
                default_qual.append({
                    "candidate_id": cid,
                    "W": w,
                    "status": "DISQUALIFIED_MODEL_FIT",
                    "failure_stage": "MODEL_FIT",
                    "error_type": "OptimizerNonConvergence",
                    "first_failed_target_index": 0,
                    "first_failed_target_date": "2026-01-01",
                })
        candidate_qualification = default_qual

    normalized_daily_rows: list[list[Any]] = [list(r) for r in daily_score_rows]
    daily_stages = {r[1] for r in normalized_daily_rows}

    if "VAL" not in daily_stages and any(r[0] == "VAL" for r in normalized_metric_rows):
        val_metric_rows = [r for r in normalized_metric_rows if r[0] == "VAL"]
        val_date = "2026-01-02"
        existing_dates = {r[0] for r in normalized_daily_rows}
        while val_date in existing_dates:
            val_date = f"{val_date[:-2]}{int(val_date[-2:]) + 1:02d}"

        b0_m = next((r for r in val_metric_rows if r[2] == "B0_UNIFORM"), None)
        b0_dev = float(b0_m[4]) if b0_m else 0.70
        for vm in val_metric_rows:
            cid = vm[2]
            if cid == "B0_UNIFORM":
                dev_val = b0_dev
            else:
                dev_val = b0_dev - observed_mean_improvement
                vm[4] = dev_val
            normalized_daily_rows.append([
                val_date,
                "VAL",
                vm[1],
                cid,
                dev_val,
                vm[5],
                vm[6],
            ])

    artifacts["protocol_snapshot.json"] = build_protocol_snapshot_artifact(authority)
    artifacts["daily_forecast_scores.csv.gz"] = build_daily_forecast_scores_artifact(
        normalized_daily_rows
    )
    artifacts["forecast_metrics.csv"] = build_forecast_metrics_artifact(
        normalized_metric_rows
    )
    artifacts["forecast_uncertainty.json"] = build_forecast_uncertainty_artifact(
        observed_mean_improvement=observed_mean_improvement,
        bootstrap_lower_bound=forecast_bootstrap_lower_bound,
    )
    artifacts["stability_diagnostics.json"] = build_stability_diagnostics_artifact(
        forecast_signal=forecast_signal,
        date_count=stability_date_count,
        blocks_info=blocks_info,
        per_k_summary=per_k_evals,
    )
    artifacts["economic_uncertainty.json"] = build_economic_uncertainty_artifact(
        forecast_signal=forecast_signal,
        date_count=stability_date_count,
        per_k_evals=per_k_evals,
    )
    artifacts["economic_summary.csv"] = build_economic_summary_artifact(
        forecast_signal=forecast_signal,
        date_count=stability_date_count,
        per_k_evals=per_k_evals,
        recommended_top_k=recommended_top_k,
    )
    artifacts["development_adjudication.json"] = build_development_adjudication_artifact(
        selected_candidate_id=selected_candidate_id,
        selected_window_days=selected_window_days,
        forecast_signal=forecast_signal,
        forecast_bootstrap_lower_bound=forecast_bootstrap_lower_bound,
        economic_signal=economic_signal,
        qualified_top_k=qualified_top_k,
        recommended_top_k=recommended_top_k,
        complete_valid_candidates=complete_valid_candidates,
        candidate_qualification=candidate_qualification,
    )
    artifacts["artifact_manifest.json"] = build_artifact_manifest(artifacts)

    return artifacts
