"""Tests for XPIS v3 M3 successful development artifact composition (Slice M3-11)."""

from __future__ import annotations

import gzip
import json
from typing import Any

import pytest

from src.m3_digit_factor.artifacts import (
    SUCCESS_ARTIFACT_COUNT,
    SUCCESS_ARTIFACT_NAMES,
    build_artifact_manifest,
    build_daily_forecast_scores_artifact,
    build_development_adjudication_artifact,
    build_economic_summary_artifact,
    build_economic_uncertainty_artifact,
    build_forecast_metrics_artifact,
    build_forecast_uncertainty_artifact,
    build_protocol_snapshot_artifact,
    build_stability_diagnostics_artifact,
    build_success_artifacts,
)
from src.m3_digit_factor.authority import (
    RepositoryIdentity,
    build_authority,
    protocol_fingerprint,
)
from src.m3_digit_factor.economics import PerKEconomicEvaluation


def _sample_authority() -> dict[str, Any]:
    identity = RepositoryIdentity(
        repository="Quan-Nguyen-hp/vietnam-lottery-xsmb-analysis",
        data_path="data/xsmb-2-digits.csv",
        source_commit="b" * 40,
        source_tree="c" * 40,
        data_commit="d" * 40,
        data_blob="e" * 40,
        data_sha256="f" * 64,
    )
    return build_authority(identity)


def test_success_artifact_inventory_and_count() -> None:
    """Exactly 9 success artifacts are defined."""
    assert SUCCESS_ARTIFACT_COUNT == 9
    assert len(SUCCESS_ARTIFACT_NAMES) == 9
    assert SUCCESS_ARTIFACT_NAMES == (
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


# --- 1. protocol_snapshot.json Tests ---


def test_protocol_snapshot_artifact_schema_and_fingerprint() -> None:
    """protocol_snapshot.json has exactly 2 keys: authority and protocol_fingerprint_sha256."""
    auth = _sample_authority()
    raw_bytes = build_protocol_snapshot_artifact(auth)

    assert raw_bytes.endswith(b"\n")
    data = json.loads(raw_bytes.decode("utf-8"))

    assert set(data.keys()) == {"authority", "protocol_fingerprint_sha256"}
    assert len(data["authority"]) == 72
    assert data["protocol_fingerprint_sha256"] == protocol_fingerprint(auth)


# --- 2. daily_forecast_scores.csv.gz Tests ---


def test_daily_forecast_scores_artifact_gzip_and_csv_schema() -> None:
    """daily_forecast_scores.csv.gz is canonical level-9 gzip with exact CSV header."""
    rows = [
        ["2026-01-01", "DEV", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
        ["2026-01-01", "DEV", "M3", "M3_W030", 0.69, 0.34, 0.44],
    ]
    gz_bytes = build_daily_forecast_scores_artifact(rows)

    assert gz_bytes[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    csv_bytes = gzip.decompress(gz_bytes)
    assert csv_bytes.endswith(b"\n")
    header_line = csv_bytes.decode("utf-8").split("\n")[0]
    assert header_line == "target_date,stage,model_id,candidate_id,poisson_deviance,mae,rmse"


# --- 3. forecast_metrics.csv Tests ---


def test_forecast_metrics_artifact_row_ordering() -> None:
    """Row order: DEV, VAL, STABILITY -> model_id ascending -> candidate_id ascending."""
    # Provide intentionally scrambled rows
    rows = [
        {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 50, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
        {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 100, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 100, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
        {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 100, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
        {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 50, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
    ]
    csv_bytes = build_forecast_metrics_artifact(rows)
    lines = csv_bytes.decode("utf-8").strip().split("\n")
    assert lines[0] == "stage,model_id,candidate_id,date_count,poisson_deviance,mae,rmse"

    # Sorted order check:
    # 1. DEV, B0, B0_UNIFORM
    # 2. DEV, M3, M3_W030
    # 3. DEV, M3, M3_W060
    # 4. VAL, B0, B0_UNIFORM
    # 5. VAL, M3, M3_W060
    assert lines[1].startswith("DEV,B0,B0_UNIFORM")
    assert lines[2].startswith("DEV,M3,M3_W030")
    assert lines[3].startswith("DEV,M3,M3_W060")
    assert lines[4].startswith("VAL,B0,B0_UNIFORM")
    assert lines[5].startswith("VAL,M3,M3_W060")


# --- 4. forecast_uncertainty.json Tests ---


def test_forecast_uncertainty_artifact_schema() -> None:
    """Exact 11-key schema for forecast_uncertainty.json."""
    raw_bytes = build_forecast_uncertainty_artifact(
        observed_mean_improvement=0.015,
        bootstrap_lower_bound=0.005,
    )
    data = json.loads(raw_bytes.decode("utf-8"))
    expected_keys = {
        "status",
        "bootstrap_rng_implementation",
        "bootstrap_bit_generator",
        "bootstrap_seed",
        "bootstrap_replications",
        "mean_block_length",
        "restart_probability",
        "alpha",
        "quantile_method",
        "observed_mean_improvement",
        "bootstrap_lower_bound",
    }
    assert set(data.keys()) == expected_keys
    assert data["status"] == "EVALUATED"
    assert data["bootstrap_seed"] == 20260831
    assert data["bootstrap_replications"] == 2000
    assert data["mean_block_length"] == 30
    assert data["alpha"] == 0.05
    assert data["quantile_method"] == "linear"
    assert data["observed_mean_improvement"] == 0.015
    assert data["bootstrap_lower_bound"] == 0.005


# --- 5. stability_diagnostics.json Tests ---


def test_stability_diagnostics_evaluated_state() -> None:
    """stability_diagnostics.json schema when forecast gate passes."""
    blocks_info = [
        {
            "block_id": b,
            "start_date": f"2026-03-0{b+1}",
            "end_date": f"2026-03-1{b+1}",
            "date_count": 10,
            "per_k": [
                {"K": k, "mean_economic_delta": 5.0, "positive": True}
                for k in [1, 3, 5, 10]
            ],
        }
        for b in range(6)
    ]
    per_k_summary = {
        1: PerKEconomicEvaluation(k=1, mean_economic_delta=5.0, block_mean_economic_delta=(5.0,)*6, positive_block_count=6, bootstrap_lower_bound=1.0, qualifies=True),
        3: PerKEconomicEvaluation(k=3, mean_economic_delta=5.0, block_mean_economic_delta=(5.0,)*6, positive_block_count=6, bootstrap_lower_bound=1.0, qualifies=True),
        5: PerKEconomicEvaluation(k=5, mean_economic_delta=5.0, block_mean_economic_delta=(5.0,)*6, positive_block_count=6, bootstrap_lower_bound=1.0, qualifies=True),
        10: PerKEconomicEvaluation(k=10, mean_economic_delta=5.0, block_mean_economic_delta=(5.0,)*6, positive_block_count=6, bootstrap_lower_bound=1.0, qualifies=True),
    }

    raw_bytes = build_stability_diagnostics_artifact(
        forecast_signal=True,
        date_count=60,
        blocks_info=blocks_info,
        per_k_summary=per_k_summary,
    )
    data = json.loads(raw_bytes.decode("utf-8"))
    assert set(data.keys()) == {"status", "date_count", "blocks", "per_k"}
    assert data["status"] == "EVALUATED"
    assert data["date_count"] == 60
    assert len(data["blocks"]) == 6
    assert len(data["per_k"]) == 4


def test_stability_diagnostics_forecast_gate_failed_state() -> None:
    """stability_diagnostics.json schema when forecast gate fails."""
    raw_bytes = build_stability_diagnostics_artifact(
        forecast_signal=False,
        date_count=None,
        blocks_info=None,
        per_k_summary=None,
    )
    data = json.loads(raw_bytes.decode("utf-8"))
    assert data["status"] == "NOT_EVALUATED_FORECAST_GATE_FAILED"
    assert data["date_count"] is None
    assert data["blocks"] == []
    assert len(data["per_k"]) == 4
    for item in data["per_k"]:
        assert item["positive_block_count"] is None
        assert item["bootstrap_lower_bound"] is None
        assert item["qualifies"] is False


# --- 6. economic_uncertainty.json Tests ---


def test_economic_uncertainty_evaluated_state() -> None:
    """economic_uncertainty.json 13-key schema when evaluated."""
    per_k_evals = {
        1: PerKEconomicEvaluation(k=1, mean_economic_delta=10.0, block_mean_economic_delta=(10.0,)*6, positive_block_count=6, bootstrap_lower_bound=2.0, qualifies=True),
        3: PerKEconomicEvaluation(k=3, mean_economic_delta=10.0, block_mean_economic_delta=(10.0,)*6, positive_block_count=6, bootstrap_lower_bound=2.0, qualifies=True),
        5: PerKEconomicEvaluation(k=5, mean_economic_delta=10.0, block_mean_economic_delta=(10.0,)*6, positive_block_count=6, bootstrap_lower_bound=2.0, qualifies=True),
        10: PerKEconomicEvaluation(k=10, mean_economic_delta=10.0, block_mean_economic_delta=(10.0,)*6, positive_block_count=6, bootstrap_lower_bound=2.0, qualifies=True),
    }
    raw_bytes = build_economic_uncertainty_artifact(
        forecast_signal=True,
        date_count=60,
        per_k_evals=per_k_evals,
    )
    data = json.loads(raw_bytes.decode("utf-8"))
    expected_keys = {
        "status",
        "familywise_alpha",
        "multiplicity_method",
        "per_k_alpha",
        "bootstrap_rng_implementation",
        "bootstrap_bit_generator",
        "bootstrap_seed",
        "bootstrap_replications",
        "mean_block_length",
        "restart_probability",
        "quantile_method",
        "shared_resample_indices",
        "per_k",
    }
    assert set(data.keys()) == expected_keys
    assert data["status"] == "EVALUATED"
    assert data["familywise_alpha"] == 0.05
    assert data["multiplicity_method"] == "BONFERRONI"
    assert data["per_k_alpha"] == 0.0125
    assert data["bootstrap_seed"] == 20260832
    assert data["shared_resample_indices"] is True
    assert len(data["per_k"]) == 4


def test_economic_uncertainty_forecast_gate_failed_state() -> None:
    """economic_uncertainty.json when forecast gate fails."""
    raw_bytes = build_economic_uncertainty_artifact(
        forecast_signal=False,
        date_count=None,
        per_k_evals=None,
    )
    data = json.loads(raw_bytes.decode("utf-8"))
    assert data["status"] == "NOT_EVALUATED_FORECAST_GATE_FAILED"
    assert data["bootstrap_seed"] == 20260832
    assert len(data["per_k"]) == 4
    for rec in data["per_k"]:
        assert rec["status"] == "NOT_EVALUATED_FORECAST_GATE_FAILED"
        assert rec["date_count"] is None
        assert rec["mean_economic_delta"] is None
        assert rec["positive_block_count"] is None
        assert rec["bonferroni_alpha"] == 0.0125
        assert rec["bootstrap_lower_bound"] is None
        assert rec["qualifies"] is False


# --- 7. economic_summary.csv Tests ---


def test_economic_summary_evaluated_and_failed_states() -> None:
    """economic_summary.csv header, 4 rows, and recommendation marking."""
    per_k_evals = {
        1: PerKEconomicEvaluation(k=1, mean_economic_delta=10.0, block_mean_economic_delta=(10.0,)*6, positive_block_count=6, bootstrap_lower_bound=2.0, qualifies=True),
        3: PerKEconomicEvaluation(k=3, mean_economic_delta=15.0, block_mean_economic_delta=(15.0,)*6, positive_block_count=6, bootstrap_lower_bound=3.0, qualifies=True),
        5: PerKEconomicEvaluation(k=5, mean_economic_delta=12.0, block_mean_economic_delta=(12.0,)*6, positive_block_count=6, bootstrap_lower_bound=2.5, qualifies=True),
        10: PerKEconomicEvaluation(k=10, mean_economic_delta=-5.0, block_mean_economic_delta=(-5.0,)*6, positive_block_count=0, bootstrap_lower_bound=-10.0, qualifies=False),
    }
    csv_bytes = build_economic_summary_artifact(
        forecast_signal=True,
        date_count=60,
        per_k_evals=per_k_evals,
        recommended_top_k=[3],
    )
    lines = csv_bytes.decode("utf-8").strip().split("\n")
    assert lines[0] == "K,status,date_count,mean_economic_delta,positive_block_count,bonferroni_alpha,bootstrap_lower_bound,qualifies,recommended"
    assert len(lines) == 5  # header + 4 rows
    assert lines[2].startswith("3,EVALUATED,60,15,6,0.012500000000000001,3,true,true")
    assert lines[1].endswith(",false")  # K=1 not recommended


# --- 8. development_adjudication.json Tests ---


@pytest.mark.parametrize(
    "fc_signal,econ_signal,expected_status",
    [
        (False, False, "FORECAST_GATE_FAILED"),
        (True, False, "ECONOMIC_GATE_FAILED"),
        (True, True, "ECONOMIC_SIGNAL_FOUND"),
    ],
)
def test_development_adjudication_exit_status_mappings(
    fc_signal: bool, econ_signal: bool, expected_status: str
) -> None:
    """Verify exact 9-key schema and status mapping for development_adjudication.json."""
    raw_bytes = build_development_adjudication_artifact(
        selected_candidate_id="M3_W060",
        selected_window_days=60,
        forecast_signal=fc_signal,
        forecast_bootstrap_lower_bound=0.005 if fc_signal else -0.002,
        economic_signal=econ_signal,
        qualified_top_k=[1, 3] if econ_signal else [],
        recommended_top_k=[3] if econ_signal else [],
    )
    data = json.loads(raw_bytes.decode("utf-8"))
    assert set(data.keys()) == {
        "status",
        "selected_candidate_id",
        "selected_window_days",
        "forecast_signal",
        "forecast_bootstrap_lower_bound",
        "economic_signal",
        "qualified_top_k",
        "recommended_top_k",
        "development_exit_status",
    }
    assert data["status"] == "COMPLETED"
    assert data["development_exit_status"] == expected_status


# --- 9. artifact_manifest.json Tests ---


def test_artifact_manifest_schema_and_lexicographical_coverage() -> None:
    """Manifest contains exactly 8 other artifacts sorted lexicographically by filename."""
    artifacts = {
        "protocol_snapshot.json": b"proto",
        "daily_forecast_scores.csv.gz": b"scores",
        "forecast_metrics.csv": b"metrics",
        "forecast_uncertainty.json": b"fc_unc",
        "stability_diagnostics.json": b"stab_diag",
        "economic_uncertainty.json": b"econ_unc",
        "economic_summary.csv": b"econ_sum",
        "development_adjudication.json": b"adjudication",
    }
    manifest_bytes = build_artifact_manifest(artifacts)
    data = json.loads(manifest_bytes.decode("utf-8"))

    assert set(data.keys()) == {"artifacts"}
    entries = data["artifacts"]
    assert len(entries) == 8

    # Must be sorted by filename ascending
    filenames = [e["filename"] for e in entries]
    assert filenames == sorted(filenames)
    assert "artifact_manifest.json" not in filenames  # Excluded from self


def test_build_success_artifacts_returns_all_nine() -> None:
    """build_success_artifacts produces all 9 required artifacts."""
    auth = _sample_authority()
    artifacts = build_success_artifacts(
        authority=auth,
        selected_candidate_id="M3_W060",
        selected_window_days=60,
        forecast_signal=False,
        forecast_bootstrap_lower_bound=-0.005,
        observed_mean_improvement=-0.001,
        daily_score_rows=[],
        forecast_metric_rows=[],
    )
    assert len(artifacts) == 9
    assert tuple(sorted(artifacts.keys())) == SUCCESS_ARTIFACT_NAMES
