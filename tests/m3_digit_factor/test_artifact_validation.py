"""Tests for XPIS v3 M3 success artifact validation (Slice M3-11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.m3_digit_factor.artifacts import (
    build_artifact_manifest,
    build_forecast_metrics_artifact,
    build_success_artifacts,
)
from src.m3_digit_factor.authority import (
    RepositoryIdentity,
    build_authority,
)
from src.m3_digit_factor.contracts import FailureExitStatus, FailureStage
from src.m3_digit_factor.economics import PerKEconomicEvaluation
from src.m3_digit_factor.model import ModelValidationError
from src.m3_digit_factor.serialization import GZIP_HEADER_BYTES
from src.m3_digit_factor.validation import (
    ArtifactProtocolInconsistencyError,
    ArtifactTechnicalFailureError,
    ArtifactValidationError,
    validate_success_artifacts,
)


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


def _create_forecast_failed_bundle() -> dict[str, bytes]:
    """Builds a complete, valid bundle of 9 artifacts where forecast gate failed."""
    auth = _sample_authority()
    return build_success_artifacts(
        authority=auth,
        selected_candidate_id="M3_W060",
        selected_window_days=60,
        forecast_signal=False,
        forecast_bootstrap_lower_bound=-0.002,
        observed_mean_improvement=-0.001,
        daily_score_rows=[
            ["2026-01-01", "DEV", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
            ["2026-01-01", "DEV", "M3", "M3_W030", 0.69, 0.34, 0.44],
            ["2026-01-01", "DEV", "M3", "M3_W060", 0.68, 0.33, 0.43],
            ["2026-01-01", "DEV", "M3", "M3_W120", 0.69, 0.34, 0.44],
            ["2026-01-01", "DEV", "M3", "M3_W240", 0.70, 0.35, 0.45],
            ["2026-01-01", "DEV", "M3", "M3_W365", 0.71, 0.36, 0.46],
            ["2026-02-01", "VAL", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
            ["2026-02-01", "VAL", "M3", "M3_W060", 0.701, 0.351, 0.451],
        ],
        forecast_metric_rows=[
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.701, "mae": 0.351, "rmse": 0.451},
        ],
    )


def _create_economic_signal_bundle() -> dict[str, bytes]:
    """Builds a complete, valid bundle of 9 artifacts where economic signal was found."""
    auth = _sample_authority()
    per_k_evals = {
        1: PerKEconomicEvaluation(k=1, mean_economic_delta=10.0, block_mean_economic_delta=(10.0,)*6, positive_block_count=6, bootstrap_lower_bound=2.0, qualifies=True),
        3: PerKEconomicEvaluation(k=3, mean_economic_delta=15.0, block_mean_economic_delta=(15.0,)*6, positive_block_count=6, bootstrap_lower_bound=3.0, qualifies=True),
        5: PerKEconomicEvaluation(k=5, mean_economic_delta=12.0, block_mean_economic_delta=(12.0,)*6, positive_block_count=6, bootstrap_lower_bound=2.5, qualifies=True),
        10: PerKEconomicEvaluation(k=10, mean_economic_delta=-5.0, block_mean_economic_delta=(-5.0,)*6, positive_block_count=0, bootstrap_lower_bound=-10.0, qualifies=False),
    }
    blocks_info = [
        {
            "block_id": b,
            "start_date": f"2026-03-0{b+1}",
            "end_date": f"2026-03-0{b+1}",
            "date_count": 1,
            "per_k": [
                {"K": 1, "mean_economic_delta": 10.0, "positive": True},
                {"K": 3, "mean_economic_delta": 15.0, "positive": True},
                {"K": 5, "mean_economic_delta": 12.0, "positive": True},
                {"K": 10, "mean_economic_delta": -5.0, "positive": False},
            ],
        }
        for b in range(6)
    ]
    stability_daily_rows = []
    for b in range(6):
        d_str = f"2026-03-0{b+1}"
        stability_daily_rows.append([d_str, "STABILITY", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45])
        stability_daily_rows.append([d_str, "STABILITY", "M3", "M3_W060", 0.69, 0.34, 0.44])

    return build_success_artifacts(
        authority=auth,
        selected_candidate_id="M3_W060",
        selected_window_days=60,
        forecast_signal=True,
        forecast_bootstrap_lower_bound=0.005,
        observed_mean_improvement=0.010,
        daily_score_rows=[
            ["2026-01-01", "DEV", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
            ["2026-01-01", "DEV", "M3", "M3_W030", 0.69, 0.34, 0.44],
            ["2026-01-01", "DEV", "M3", "M3_W060", 0.68, 0.33, 0.43],
            ["2026-01-01", "DEV", "M3", "M3_W120", 0.69, 0.34, 0.44],
            ["2026-01-01", "DEV", "M3", "M3_W240", 0.70, 0.35, 0.45],
            ["2026-01-01", "DEV", "M3", "M3_W365", 0.71, 0.36, 0.46],
            ["2026-02-01", "VAL", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
            ["2026-02-01", "VAL", "M3", "M3_W060", 0.69, 0.34, 0.44],
            *stability_daily_rows,
        ],
        forecast_metric_rows=[
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 6, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 6, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        ],
        stability_date_count=6,
        blocks_info=blocks_info,
        per_k_evals=per_k_evals,
        economic_signal=True,
        qualified_top_k=[1, 3, 5],
        recommended_top_k=[3],
    )


def test_valid_bundles_pass_validation(tmp_path: Path) -> None:
    """Canonical valid bundles pass validation in-memory and from directory."""
    bundle_failed = _create_forecast_failed_bundle()
    validate_success_artifacts(bundle_failed)

    bundle_econ = _create_economic_signal_bundle()
    validate_success_artifacts(bundle_econ)

    # Directory input test
    for fname, b in bundle_failed.items():
        (tmp_path / fname).write_bytes(b)
    validate_success_artifacts(tmp_path)


def test_error_attributes_and_inheritance() -> None:
    """ArtifactValidationError conforms to protocol failure taxonomy and distinguishes protocol vs technical."""
    # Protocol inconsistency error defaults to NEEDS_PROTOCOL_REVISION
    proto_err = ArtifactProtocolInconsistencyError("Invalid bundle", error_type="ARTIFACT_COUNT_MISMATCH")
    assert isinstance(proto_err, ArtifactValidationError)
    assert isinstance(proto_err, ModelValidationError)
    assert proto_err.stage == FailureStage.ARTIFACT_VALIDATION
    assert proto_err.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    proto = proto_err.as_protocol_failure()
    assert proto.stage == FailureStage.ARTIFACT_VALIDATION
    assert proto.error_type == "ARTIFACT_COUNT_MISMATCH"
    assert proto.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION

    # Technical failure error defaults to TECHNICAL_FAILURE
    tech_err = ArtifactTechnicalFailureError("Disk failure", error_type="ARTIFACT_FILE_READ_ERROR")
    assert isinstance(tech_err, ArtifactValidationError)
    assert isinstance(tech_err, ModelValidationError)
    assert tech_err.stage == FailureStage.ARTIFACT_VALIDATION
    assert tech_err.exit_status == FailureExitStatus.TECHNICAL_FAILURE
    tech_proto = tech_err.as_protocol_failure()
    assert tech_proto.stage == FailureStage.ARTIFACT_VALIDATION
    assert tech_proto.error_type == "ARTIFACT_FILE_READ_ERROR"
    assert tech_proto.exit_status == FailureExitStatus.TECHNICAL_FAILURE


# --- Inventory and Filename Tests ---


def test_missing_artifact_raises() -> None:
    """Missing any one of the 9 artifacts raises ARTIFACT_VALIDATION with NEEDS_PROTOCOL_REVISION."""
    bundle = _create_forecast_failed_bundle()
    del bundle["economic_summary.csv"]
    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "Missing required artifact" in str(exc_info.value)


def test_extra_artifact_raises() -> None:
    """Extraneous artifact (e.g. failure artifact or tmp file) fails validation with NEEDS_PROTOCOL_REVISION."""
    bundle = _create_forecast_failed_bundle()
    bundle["development_run_FAILED.json"] = b'{"status":"FAILED"}'
    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "Unexpected extra artifact" in str(exc_info.value)


# --- Canonical Byte Exactness Tests ---


def test_json_byte_exactness_rejection() -> None:
    """JSON with extra whitespace, keys out of order, or missing LF yields NEEDS_PROTOCOL_REVISION."""
    bundle = _create_forecast_failed_bundle()

    # Tamper with formatting of protocol_snapshot.json (e.g. add trailing space before \n)
    tampered = bundle["protocol_snapshot.json"][:-1] + b" \n"
    bundle["protocol_snapshot.json"] = tampered
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "Canonical JSON byte-exactness mismatch" in str(exc_info.value)


def test_csv_crlf_rejection() -> None:
    """CSV containing CRLF violates canonical encoding and yields NEEDS_PROTOCOL_REVISION."""
    bundle = _create_forecast_failed_bundle()
    raw_csv = bundle["forecast_metrics.csv"]
    tampered_csv = raw_csv.replace(b"\n", b"\r\n")
    bundle["forecast_metrics.csv"] = tampered_csv
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "Canonical CSV byte-exactness mismatch" in str(exc_info.value)


def test_gzip_header_tampering_rejection() -> None:
    """Gzip header with non-0xff OS byte fails validation with NEEDS_PROTOCOL_REVISION."""
    bundle = _create_forecast_failed_bundle()
    raw_gz = bytearray(bundle["daily_forecast_scores.csv.gz"])
    raw_gz[9] = 0x03  # Unix OS instead of 0xff
    bundle["daily_forecast_scores.csv.gz"] = bytes(raw_gz)
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "Gzip fixed header mismatch" in str(exc_info.value)


def test_gzip_multi_member_rejection() -> None:
    """Gzip with appended extra member fails validation with NEEDS_PROTOCOL_REVISION."""
    bundle = _create_forecast_failed_bundle()
    raw_gz = bundle["daily_forecast_scores.csv.gz"]
    tampered_gz = raw_gz + raw_gz
    bundle["daily_forecast_scores.csv.gz"] = tampered_gz
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "multiple gzip members" in str(exc_info.value)


# --- Manifest Verification Tests ---


def test_manifest_self_inclusion_rejection() -> None:
    """Manifest cannot include itself (NEEDS_PROTOCOL_REVISION)."""
    bundle = _create_forecast_failed_bundle()
    data = json.loads(bundle["artifact_manifest.json"].decode("utf-8"))
    data["artifacts"].append({"filename": "artifact_manifest.json", "sha256": "0" * 64, "byte_size": 100})
    from src.m3_digit_factor.serialization import serialize_json
    bundle["artifact_manifest.json"] = serialize_json(data)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "Manifest must not contain itself" in str(exc_info.value)


def test_manifest_sha256_mismatch_rejection() -> None:
    """Manifest sha256 mismatch raises ARTIFACT_VALIDATION with NEEDS_PROTOCOL_REVISION."""
    bundle = _create_forecast_failed_bundle()
    data = json.loads(bundle["artifact_manifest.json"].decode("utf-8"))
    data["artifacts"][0]["sha256"] = "f" * 64
    from src.m3_digit_factor.serialization import serialize_json
    bundle["artifact_manifest.json"] = serialize_json(data)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "SHA256 mismatch" in str(exc_info.value)


def test_manifest_not_lexicographically_sorted_rejection() -> None:
    """Manifest entries must be sorted lexicographically by filename (NEEDS_PROTOCOL_REVISION)."""
    bundle = _create_forecast_failed_bundle()
    data = json.loads(bundle["artifact_manifest.json"].decode("utf-8"))
    data["artifacts"].reverse()
    from src.m3_digit_factor.serialization import serialize_json
    bundle["artifact_manifest.json"] = serialize_json(data)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "not sorted lexicographically" in str(exc_info.value)


# --- Cross-Artifact Consistency Tests ---


def test_fingerprint_mismatch_rejection() -> None:
    """protocol_fingerprint_sha256 must match recomputed fingerprint of authority (NEEDS_PROTOCOL_REVISION)."""
    bundle = _create_forecast_failed_bundle()
    data = json.loads(bundle["protocol_snapshot.json"].decode("utf-8"))
    data["protocol_fingerprint_sha256"] = "0" * 64
    from src.m3_digit_factor.serialization import serialize_json
    bundle["protocol_snapshot.json"] = serialize_json(data)
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "Protocol fingerprint mismatch" in str(exc_info.value)


def test_forecast_failed_stability_row_violation() -> None:
    """If forecast gate failed, STABILITY rows in daily scores or metrics yield NEEDS_PROTOCOL_REVISION."""
    bundle = _create_forecast_failed_bundle()

    # Add a stability row to forecast_metrics.csv while keeping DEV/VAL candidates valid
    tampered_metrics = build_forecast_metrics_artifact([
        {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
        {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
        {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
        {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
        {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
        {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.701, "mae": 0.351, "rmse": 0.451},
        {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
    ])
    bundle["forecast_metrics.csv"] = tampered_metrics
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "STABILITY rows found when forecast gate failed" in str(exc_info.value)


def test_economic_signal_cross_artifact_inconsistency() -> None:
    """Inconsistency in economic delta or qualified_top_k between artifacts yields NEEDS_PROTOCOL_REVISION."""
    bundle = _create_economic_signal_bundle()
    data = json.loads(bundle["development_adjudication.json"].decode("utf-8"))
    # Change recommended_top_k to [10] which does not qualify
    data["recommended_top_k"] = [10]
    from src.m3_digit_factor.serialization import serialize_json
    bundle["development_adjudication.json"] = serialize_json(data)
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "recommended_top_k must be a subset of qualified_top_k" in str(exc_info.value)


def test_economic_summary_point_estimate_mismatch() -> None:
    """Point estimate discrepancy between economic_summary.csv and economic_uncertainty.json yields NEEDS_PROTOCOL_REVISION."""
    bundle = _create_economic_signal_bundle()
    data = json.loads(bundle["economic_uncertainty.json"].decode("utf-8"))
    # Alter K=1 mean_economic_delta
    data["per_k"][0]["mean_economic_delta"] = 999.0
    from src.m3_digit_factor.serialization import serialize_json
    bundle["economic_uncertainty.json"] = serialize_json(data)
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_success_artifacts(bundle)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.NEEDS_PROTOCOL_REVISION
    assert "Discrepancy in mean_economic_delta" in str(exc_info.value)


# --- Technical Failure Tests ---


def test_unexpected_io_read_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected I/O failure while reading artifacts from filesystem yields TECHNICAL_FAILURE."""
    bundle = _create_forecast_failed_bundle()
    for fname, b in bundle.items():
        (tmp_path / fname).write_bytes(b)

    orig_read_bytes = Path.read_bytes

    def _failing_read_bytes(self: Path) -> bytes:
        if self.name == "economic_summary.csv":
            raise OSError("Simulated filesystem I/O hardware read error")
        return orig_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _failing_read_bytes)

    with pytest.raises(ArtifactTechnicalFailureError) as exc_info:
        validate_success_artifacts(tmp_path)

    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.TECHNICAL_FAILURE
    proto = exc_info.value.as_protocol_failure()
    assert proto.stage == FailureStage.ARTIFACT_VALIDATION
    assert proto.exit_status == FailureExitStatus.TECHNICAL_FAILURE
    assert "Unexpected I/O error reading artifact" in str(exc_info.value)


def test_unexpected_gzip_library_failure() -> None:
    """Unexpected decompression error from corrupt compressed body yields TECHNICAL_FAILURE."""
    bundle = _create_forecast_failed_bundle()
    # Keep valid 10-byte fixed header, but append corrupt stream payload that triggers zlib decompression error
    bundle["daily_forecast_scores.csv.gz"] = GZIP_HEADER_BYTES + b"\xff\xff\xff\xff_corrupt_stream"
    bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

    with pytest.raises(ArtifactTechnicalFailureError) as exc_info:
        validate_success_artifacts(bundle)

    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.TECHNICAL_FAILURE
    proto = exc_info.value.as_protocol_failure()
    assert proto.stage == FailureStage.ARTIFACT_VALIDATION
    assert proto.exit_status == FailureExitStatus.TECHNICAL_FAILURE
    assert "Unexpected compression library failure" in str(exc_info.value)


def test_artifact_directory_not_found(tmp_path: Path) -> None:
    """Non-existent directory path yields TECHNICAL_FAILURE."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(ArtifactTechnicalFailureError) as exc_info:
        validate_success_artifacts(non_existent)
    assert exc_info.value.stage == FailureStage.ARTIFACT_VALIDATION
    assert exc_info.value.exit_status == FailureExitStatus.TECHNICAL_FAILURE


# --- Section 15: Mandatory HIGH-01 Adversarial Remanifested Bundle Tests ---


class TestAdversarialFailClosedIntegrity:
    """Validate that tampered scientific/protocol bundles fail closed even with recomputed manifests."""

    def test_adversarial_wrong_forecast_seed(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["forecast_uncertainty.json"].decode("utf-8"))
        data["bootstrap_seed"] = 20260312  # Tamper seed
        from src.m3_digit_factor.serialization import serialize_json
        bundle["forecast_uncertainty.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_wrong_forecast_replication_count(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["forecast_uncertainty.json"].decode("utf-8"))
        data["bootstrap_replications"] = 9999  # Tamper replications
        from src.m3_digit_factor.serialization import serialize_json
        bundle["forecast_uncertainty.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_wrong_quantile_method(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["forecast_uncertainty.json"].decode("utf-8"))
        data["quantile_method"] = "nearest"  # Tamper method away from frozen 'linear'
        from src.m3_digit_factor.serialization import serialize_json
        bundle["forecast_uncertainty.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_candidate_window_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["development_adjudication.json"].decode("utf-8"))
        data["selected_window_days"] = 120  # Tamper window days (selected candidate is M3_W060 -> window must be 60)
        from src.m3_digit_factor.serialization import serialize_json
        bundle["development_adjudication.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_missing_dev_candidate_row(self) -> None:
        bundle = _create_economic_signal_bundle()
        # Drop M3_W365 from forecast_metrics.csv
        rows = [
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            # Missing M3_W365
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 6, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 6, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        ]
        bundle["forecast_metrics.csv"] = build_forecast_metrics_artifact(rows)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_duplicate_dev_candidate_row(self) -> None:
        bundle = _create_economic_signal_bundle()
        rows = [
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 6, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 6, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        ]
        bundle["forecast_metrics.csv"] = build_forecast_metrics_artifact(rows)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_unexpected_val_candidate(self) -> None:
        bundle = _create_economic_signal_bundle()
        rows = [
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 6, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 6, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        ]
        bundle["forecast_metrics.csv"] = build_forecast_metrics_artifact(rows)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_date_count_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        rows = [
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 999, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 999, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 6, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 6, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        ]
        bundle["forecast_metrics.csv"] = build_forecast_metrics_artifact(rows)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_daily_to_stage_pd_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        rows = [
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.99, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 6, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 6, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        ]
        bundle["forecast_metrics.csv"] = build_forecast_metrics_artifact(rows)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_daily_to_stage_mae_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        rows = [
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.88, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 6, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 6, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        ]
        bundle["forecast_metrics.csv"] = build_forecast_metrics_artifact(rows)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_daily_to_stage_pooled_rmse_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        rows = [
            {"stage": "DEV", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.99},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W030", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.68, "mae": 0.33, "rmse": 0.43},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W120", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W240", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "DEV", "model_id": "M3", "candidate_id": "M3_W365", "date_count": 1, "poisson_deviance": 0.71, "mae": 0.36, "rmse": 0.46},
            {"stage": "VAL", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 1, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "VAL", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 1, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
            {"stage": "STABILITY", "model_id": "B0", "candidate_id": "B0_UNIFORM", "date_count": 6, "poisson_deviance": 0.70, "mae": 0.35, "rmse": 0.45},
            {"stage": "STABILITY", "model_id": "M3", "candidate_id": "M3_W060", "date_count": 6, "poisson_deviance": 0.69, "mae": 0.34, "rmse": 0.44},
        ]
        bundle["forecast_metrics.csv"] = build_forecast_metrics_artifact(rows)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_forecast_gate_adjudication_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["development_adjudication.json"].decode("utf-8"))
        data["forecast_signal"] = False
        from src.m3_digit_factor.serialization import serialize_json
        bundle["development_adjudication.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_economic_qualifies_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["economic_uncertainty.json"].decode("utf-8"))
        for pk in data["per_k"]:
            if pk["K"] == 10:
                pk["qualifies"] = True
        from src.m3_digit_factor.serialization import serialize_json
        bundle["economic_uncertainty.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_economic_signal_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["development_adjudication.json"].decode("utf-8"))
        data["economic_signal"] = False
        from src.m3_digit_factor.serialization import serialize_json
        bundle["development_adjudication.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_qualified_top_k_mismatch(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["development_adjudication.json"].decode("utf-8"))
        data["qualified_top_k"] = [1, 3, 5, 10]
        from src.m3_digit_factor.serialization import serialize_json
        bundle["development_adjudication.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_recommended_top_k_selecting_unqualified_k(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["development_adjudication.json"].decode("utf-8"))
        data["recommended_top_k"] = [10]
        from src.m3_digit_factor.serialization import serialize_json
        bundle["development_adjudication.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

    def test_adversarial_forecast_restart_probability_tiny_drift(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["forecast_uncertainty.json"].decode("utf-8"))
        data["restart_probability"] = data["restart_probability"] + 5e-13
        from src.m3_digit_factor.serialization import serialize_json
        bundle["forecast_uncertainty.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError, match="forecast_uncertainty restart_probability mismatch"):
            validate_success_artifacts(bundle)

    def test_adversarial_economic_quantile_tiny_drift(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["economic_uncertainty.json"].decode("utf-8"))
        data["per_k_alpha"] = data["per_k_alpha"] + 1e-14
        from src.m3_digit_factor.serialization import serialize_json
        bundle["economic_uncertainty.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError, match="economic_uncertainty per_k_alpha mismatch"):
            validate_success_artifacts(bundle)

    def test_adversarial_forecast_uncertainty_invalid_status(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["forecast_uncertainty.json"].decode("utf-8"))
        data["status"] = "NOT_EVALUATED_FORECAST_GATE_FAILED"
        from src.m3_digit_factor.serialization import serialize_json
        bundle["forecast_uncertainty.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError, match="forecast_uncertainty status"):
            validate_success_artifacts(bundle)

    def test_adversarial_observed_mean_improvement_tampered(self) -> None:
        bundle = _create_economic_signal_bundle()
        data = json.loads(bundle["forecast_uncertainty.json"].decode("utf-8"))
        data["observed_mean_improvement"] = 999.0
        from src.m3_digit_factor.serialization import serialize_json
        bundle["forecast_uncertainty.json"] = serialize_json(data)
        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError, match="observed_mean_improvement"):
            validate_success_artifacts(bundle)

    def test_adversarial_bogus_stage_rejected(self) -> None:
        import csv
        import gzip
        from src.m3_digit_factor.artifacts import build_daily_forecast_scores_artifact
        bundle = _create_economic_signal_bundle()

        decomp = gzip.decompress(bundle["daily_forecast_scores.csv.gz"]).decode("utf-8")
        daily_reader = list(csv.reader(decomp.splitlines()))
        daily_rows = daily_reader[1:]
        daily_rows.append(["2026-01-01", "BOGUS", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45])
        bundle["daily_forecast_scores.csv.gz"] = build_daily_forecast_scores_artifact(daily_rows)

        metric_csv = bundle["forecast_metrics.csv"].decode("utf-8")
        metric_reader = list(csv.reader(metric_csv.splitlines()))
        metric_rows = metric_reader[1:]
        metric_rows.append(["BOGUS", "B0", "B0_UNIFORM", 1, 0.70, 0.35, 0.45])
        bundle["forecast_metrics.csv"] = build_forecast_metrics_artifact(metric_rows)

        bundle["artifact_manifest.json"] = build_artifact_manifest(bundle)

        with pytest.raises(ArtifactValidationError):
            validate_success_artifacts(bundle)

