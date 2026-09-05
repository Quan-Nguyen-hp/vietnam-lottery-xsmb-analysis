"""Exact development success artifact validation for XPIS v3 M3 (Slice M3-11).

This module implements complete, fail-closed validation of the 9 development success
artifacts:
1. Exact file inventory and count (no missing, no extraneous files, no failed artifacts).
2. Canonical byte-exactness (JSON sort_keys/compact/LF, CSV minimal quoting/.17g/LF, Gzip header/single member).
3. Self-contained schema conformity for all JSON and CSV files.
4. Cryptographic manifest integrity (SHA-256 hashes and byte lengths match emitted artifacts).
5. Cross-artifact consistency across protocol, forecast gate, stability, economics, and adjudication.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import zlib

from .artifacts import (
    DAILY_FORECAST_SCORES_HEADER,
    ECONOMIC_SUMMARY_HEADER,
    FORECAST_METRICS_HEADER,
    STAGE_SORT_ORDER,
    SUCCESS_ARTIFACT_COUNT,
    SUCCESS_ARTIFACT_NAMES,
)
from .authority import (
    protocol_fingerprint,
    validate_authority,
)
from .contracts import (
    DevelopmentExitStatus,
    FailureExitStatus,
    FailureStage,
    ProtocolFailure,
)
from .economics import ECONOMIC_K_VALUES
from .model import ModelValidationError
from .serialization import (
    GZIP_HEADER_BYTES,
    serialize_csv,
    serialize_json,
)


NUMERIC_ABS_TOLERANCE: float = 1e-12
NUMERIC_REL_TOLERANCE: float = 1e-12


class ArtifactValidationError(ModelValidationError):
    """Base error for all failures during artifact validation."""

    stage: FailureStage = FailureStage.ARTIFACT_VALIDATION
    exit_status: FailureExitStatus = FailureExitStatus.NEEDS_PROTOCOL_REVISION

    def __init__(
        self,
        message: str,
        error_type: str = "ArtifactValidationError",
        exit_status: FailureExitStatus = FailureExitStatus.NEEDS_PROTOCOL_REVISION,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.exit_status = exit_status

    def as_protocol_failure(self) -> ProtocolFailure:
        return ProtocolFailure(
            stage=self.stage,
            error_type=self.error_type,
            exit_status=self.exit_status,
        )


class ArtifactProtocolInconsistencyError(ArtifactValidationError):
    """Raised when an artifact bundle violates protocol or artifact contract requirements."""

    stage: FailureStage = FailureStage.ARTIFACT_VALIDATION
    exit_status: FailureExitStatus = FailureExitStatus.NEEDS_PROTOCOL_REVISION

    def __init__(
        self,
        message: str,
        error_type: str = "ArtifactProtocolInconsistencyError",
    ) -> None:
        super().__init__(
            message,
            error_type=error_type,
            exit_status=FailureExitStatus.NEEDS_PROTOCOL_REVISION,
        )


class ArtifactTechnicalFailureError(ArtifactValidationError):
    """Raised when an unexpected filesystem, I/O, or library exception occurs during validation."""

    stage: FailureStage = FailureStage.ARTIFACT_VALIDATION
    exit_status: FailureExitStatus = FailureExitStatus.TECHNICAL_FAILURE

    def __init__(
        self,
        message: str,
        error_type: str = "ArtifactTechnicalFailureError",
    ) -> None:
        super().__init__(
            message,
            error_type=error_type,
            exit_status=FailureExitStatus.TECHNICAL_FAILURE,
        )


def _load_raw_artifacts(artifacts: Mapping[str, bytes] | str | Path) -> dict[str, bytes]:
    """Load artifacts into an in-memory dictionary of raw bytes."""
    if isinstance(artifacts, (str, Path)):
        directory = Path(artifacts)
        if not directory.is_dir():
            raise ArtifactTechnicalFailureError(
                f"Artifact path is not a directory or does not exist: {directory}",
                error_type="ARTIFACT_DIRECTORY_NOT_FOUND",
            )
        raw_bundle: dict[str, bytes] = {}
        try:
            for p in directory.iterdir():
                if p.is_dir():
                    raise ArtifactProtocolInconsistencyError(
                        f"Unexpected directory in artifacts: {p.name}",
                        error_type="ARTIFACT_UNEXPECTED_DIRECTORY",
                    )
                if p.is_file():
                    try:
                        raw_bundle[p.name] = p.read_bytes()
                    except OSError as err:
                        raise ArtifactTechnicalFailureError(
                            f"Unexpected I/O error reading artifact {p.name}: {err}",
                            error_type="ARTIFACT_FILE_READ_ERROR",
                        ) from err
        except OSError as err:
            raise ArtifactTechnicalFailureError(
                f"Unexpected I/O error accessing artifact directory {directory}: {err}",
                error_type="ARTIFACT_DIRECTORY_ACCESS_ERROR",
            ) from err
        return raw_bundle
    return dict(artifacts)


def _validate_inventory(artifacts: dict[str, bytes]) -> None:
    """Verify exact count and filenames of the 9 success artifacts."""
    present_keys = set(artifacts.keys())
    required_keys = set(SUCCESS_ARTIFACT_NAMES)

    missing = required_keys - present_keys
    if missing:
        raise ArtifactValidationError(
            f"Missing required artifact(s): {sorted(missing)}",
            error_type="MISSING_SUCCESS_ARTIFACT",
        )

    extra = present_keys - required_keys
    if extra:
        raise ArtifactValidationError(
            f"Unexpected extra artifact(s): {sorted(extra)}",
            error_type="UNEXPECTED_ARTIFACT",
        )

    if len(artifacts) != SUCCESS_ARTIFACT_COUNT:
        raise ArtifactValidationError(
            f"Expected exactly {SUCCESS_ARTIFACT_COUNT} artifacts, got {len(artifacts)}",
            error_type="ARTIFACT_COUNT_MISMATCH",
        )


def _validate_canonical_json(raw_bytes: bytes, filename: str) -> Any:
    """Validate that raw bytes match canonical JSON serialization bitwise."""
    if not raw_bytes.endswith(b"\n"):
        raise ArtifactValidationError(
            f"JSON artifact {filename} does not terminate with LF",
            error_type="CANONICAL_ENCODING_MISMATCH",
        )
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except Exception as err:
        raise ArtifactValidationError(
            f"Failed to parse JSON artifact {filename}: {err}",
            error_type="JSON_DECODE_ERROR",
        ) from err

    re_serialized = serialize_json(parsed)
    if re_serialized != raw_bytes:
        raise ArtifactValidationError(
            f"Canonical JSON byte-exactness mismatch for {filename}",
            error_type="CANONICAL_ENCODING_MISMATCH",
        )
    return parsed


def _parse_and_validate_canonical_csv(
    raw_bytes: bytes,
    filename: str,
    expected_header: list[str],
) -> list[list[str]]:
    """Validate canonical CSV encoding and return string rows (excluding header)."""
    if b"\r" in raw_bytes:
        raise ArtifactValidationError(
            f"Canonical CSV byte-exactness mismatch (CRLF detected) in {filename}",
            error_type="CANONICAL_ENCODING_MISMATCH",
        )
    if not raw_bytes.endswith(b"\n"):
        raise ArtifactValidationError(
            f"CSV artifact {filename} does not terminate with LF",
            error_type="CANONICAL_ENCODING_MISMATCH",
        )
    try:
        text = raw_bytes.decode("utf-8")
        reader = list(csv.reader(text.splitlines()))
    except Exception as err:
        raise ArtifactValidationError(
            f"Failed to parse CSV artifact {filename}: {err}",
            error_type="CSV_DECODE_ERROR",
        ) from err

    if not reader:
        raise ArtifactValidationError(
            f"CSV artifact {filename} is empty",
            error_type="CSV_EMPTY_ERROR",
        )
    header = reader[0]
    if header != expected_header:
        raise ArtifactValidationError(
            f"CSV header mismatch in {filename}: expected {expected_header}, got {header}",
            error_type="CSV_HEADER_MISMATCH",
        )

    rows = reader[1:]

    # Re-serialization exactness check
    re_serialized = serialize_csv(header, rows)
    if re_serialized != raw_bytes:
        raise ArtifactValidationError(
            f"Canonical CSV byte-exactness mismatch for {filename}",
            error_type="CANONICAL_ENCODING_MISMATCH",
        )
    return rows


def _validate_manifest(
    artifacts: dict[str, bytes],
    manifest_data: dict[str, Any],
) -> None:
    """Validate artifact_manifest.json structure, coverage, ordering, and hashes."""
    if set(manifest_data.keys()) != {"artifacts"}:
        raise ArtifactValidationError(
            "Manifest must have exactly one top-level key: 'artifacts'",
            error_type="MANIFEST_SCHEMA_ERROR",
        )

    entries = manifest_data["artifacts"]
    if not isinstance(entries, list):
        raise ArtifactValidationError(
            "Manifest 'artifacts' must be an array",
            error_type="MANIFEST_SCHEMA_ERROR",
        )

    filenames = [e.get("filename") for e in entries if isinstance(e, dict)]
    if "artifact_manifest.json" in filenames:
        raise ArtifactValidationError(
            "Manifest must not contain itself",
            error_type="MANIFEST_SELF_INCLUSION",
        )

    if len(entries) != SUCCESS_ARTIFACT_COUNT - 1:
        raise ArtifactValidationError(
            f"Manifest must contain exactly {SUCCESS_ARTIFACT_COUNT - 1} entries, got {len(entries)}",
            error_type="MANIFEST_COUNT_MISMATCH",
        )

    expected_fnames = sorted(fn for fn in SUCCESS_ARTIFACT_NAMES if fn != "artifact_manifest.json")
    if filenames != expected_fnames:
        if filenames != sorted(filenames):
            raise ArtifactValidationError(
                "Manifest entries are not sorted lexicographically by filename",
                error_type="MANIFEST_ORDERING_ERROR",
            )
        raise ArtifactValidationError(
            f"Manifest entries do not match expected artifacts: {filenames} vs {expected_fnames}",
            error_type="MANIFEST_COVERAGE_MISMATCH",
        )

    for entry in entries:
        fname = entry.get("filename")
        sha = entry.get("sha256")
        size = entry.get("byte_size")

        if not isinstance(fname, str) or not isinstance(sha, str) or not isinstance(size, int):
            raise ArtifactValidationError(
                f"Invalid entry structure in manifest for {fname}",
                error_type="MANIFEST_ENTRY_STRUCTURE_ERROR",
            )

        actual_bytes = artifacts[fname]
        actual_sha = hashlib.sha256(actual_bytes).hexdigest()
        actual_size = len(actual_bytes)

        if sha != actual_sha:
            raise ArtifactValidationError(
                f"SHA256 mismatch for {fname} in manifest: expected {actual_sha}, got {sha}",
                error_type="MANIFEST_SHA256_MISMATCH",
            )
        if size != actual_size:
            raise ArtifactValidationError(
                f"Byte size mismatch for {fname} in manifest: expected {actual_size}, got {size}",
                error_type="MANIFEST_BYTE_SIZE_MISMATCH",
            )


def _validate_gzip_scores(raw_gz: bytes) -> list[list[str]]:
    """Validate daily_forecast_scores.csv.gz single member, header, and CSV contents."""
    if len(raw_gz) < 10:
        raise ArtifactValidationError(
            "Gzip artifact too short to contain valid header",
            error_type="GZIP_TOO_SHORT",
        )
    if raw_gz[:10] != GZIP_HEADER_BYTES:
        raise ArtifactValidationError(
            f"Gzip fixed header mismatch: expected {GZIP_HEADER_BYTES.hex()}, got {raw_gz[:10].hex()}",
            error_type="GZIP_HEADER_MISMATCH",
        )

    # Decompress and verify exactly one gzip member
    d = zlib.decompressobj(wbits=31)
    try:
        decompressed = d.decompress(raw_gz)
    except Exception as err:
        raise ArtifactTechnicalFailureError(
            f"Unexpected compression library failure during gzip decompression: {err}",
            error_type="GZIP_DECOMPRESS_ERROR",
        ) from err

    if d.unused_data:
        raise ArtifactProtocolInconsistencyError(
            "daily_forecast_scores.csv.gz has multiple gzip members or trailing data",
            error_type="GZIP_MULTI_MEMBER",
        )

    # Standard gzip decompress to verify trailer CRC32 and ISIZE
    try:
        std_decompressed = gzip.decompress(raw_gz)
    except Exception as err:
        raise ArtifactTechnicalFailureError(
            f"Gzip trailer or CRC validation failed: {err}",
            error_type="GZIP_CORRUPT_TRAILER",
        ) from err

    if std_decompressed != decompressed:
        raise ArtifactTechnicalFailureError(
            "Gzip decompress mismatch",
            error_type="GZIP_DECOMPRESS_MISMATCH",
        )

    # Validate the inner CSV
    rows = _parse_and_validate_canonical_csv(
        decompressed,
        "daily_forecast_scores.csv.gz (decompressed)",
        DAILY_FORECAST_SCORES_HEADER,
    )

    # Verify daily scores row ordering: stage order, target_date asc, model_id asc, candidate_id asc
    expected_sorted = sorted(
        rows,
        key=lambda r: (
            STAGE_SORT_ORDER.get(r[1], 99),
            r[0],
            r[2],
            r[3],
        ),
    )
    if rows != expected_sorted:
        raise ArtifactValidationError(
            "daily_forecast_scores.csv.gz rows are not sorted canonically",
            error_type="DAILY_SCORES_ROW_ORDER_MISMATCH",
        )

    return rows


def _compare_floats(val1: float, val2: float, name: str) -> None:
    if not math.isclose(val1, val2, abs_tol=NUMERIC_ABS_TOLERANCE, rel_tol=NUMERIC_REL_TOLERANCE):
        raise ArtifactValidationError(
            f"Discrepancy in {name}: {val1} vs {val2} exceeds numeric tolerance {NUMERIC_ABS_TOLERANCE}",
            error_type="CROSS_ARTIFACT_NUMERIC_MISMATCH",
        )


def validate_success_artifacts(artifacts: Mapping[str, bytes] | str | Path) -> None:
    """Validate all 9 development success artifacts fail-closed."""
    raw_artifacts = _load_raw_artifacts(artifacts)
    _validate_inventory(raw_artifacts)

    # 1. protocol_snapshot.json
    proto_data = _validate_canonical_json(raw_artifacts["protocol_snapshot.json"], "protocol_snapshot.json")
    if set(proto_data.keys()) != {"authority", "protocol_fingerprint_sha256"}:
        raise ArtifactValidationError(
            "protocol_snapshot.json must have exactly 2 keys: authority and protocol_fingerprint_sha256",
            error_type="PROTOCOL_SNAPSHOT_SCHEMA_ERROR",
        )
    auth = proto_data["authority"]
    validate_authority(auth)
    computed_fingerprint = protocol_fingerprint(auth)
    if proto_data["protocol_fingerprint_sha256"] != computed_fingerprint:
        raise ArtifactValidationError(
            "Protocol fingerprint mismatch in protocol_snapshot.json",
            error_type="PROTOCOL_FINGERPRINT_MISMATCH",
        )

    # 2. forecast_uncertainty.json
    fc_unc = _validate_canonical_json(raw_artifacts["forecast_uncertainty.json"], "forecast_uncertainty.json")
    expected_fc_keys = {
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
    if set(fc_unc.keys()) != expected_fc_keys:
        raise ArtifactValidationError(
            f"forecast_uncertainty.json keys mismatch: expected {expected_fc_keys}, got {set(fc_unc.keys())}",
            error_type="FORECAST_UNCERTAINTY_SCHEMA_ERROR",
        )

    # 3. stability_diagnostics.json
    stab_diag = _validate_canonical_json(raw_artifacts["stability_diagnostics.json"], "stability_diagnostics.json")
    if set(stab_diag.keys()) != {"status", "date_count", "blocks", "per_k"}:
        raise ArtifactValidationError(
            "stability_diagnostics.json keys mismatch",
            error_type="STABILITY_DIAGNOSTICS_SCHEMA_ERROR",
        )

    # 4. economic_uncertainty.json
    econ_unc = _validate_canonical_json(raw_artifacts["economic_uncertainty.json"], "economic_uncertainty.json")
    expected_econ_keys = {
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
    if set(econ_unc.keys()) != expected_econ_keys:
        raise ArtifactValidationError(
            f"economic_uncertainty.json keys mismatch: expected {expected_econ_keys}, got {set(econ_unc.keys())}",
            error_type="ECONOMIC_UNCERTAINTY_SCHEMA_ERROR",
        )

    # 5. development_adjudication.json
    adjudication = _validate_canonical_json(
        raw_artifacts["development_adjudication.json"], "development_adjudication.json"
    )
    expected_adj_keys = {
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
    if set(adjudication.keys()) != expected_adj_keys:
        raise ArtifactValidationError(
            f"development_adjudication.json keys mismatch: expected {expected_adj_keys}, got {set(adjudication.keys())}",
            error_type="DEVELOPMENT_ADJUDICATION_SCHEMA_ERROR",
        )
    if adjudication["status"] != "COMPLETED":
        raise ArtifactValidationError(
            f"development_adjudication.json status must be 'COMPLETED', got {adjudication['status']}",
            error_type="DEVELOPMENT_ADJUDICATION_STATUS_ERROR",
        )

    # 6. forecast_metrics.csv
    metric_rows = _parse_and_validate_canonical_csv(
        raw_artifacts["forecast_metrics.csv"],
        "forecast_metrics.csv",
        FORECAST_METRICS_HEADER,
    )
    expected_metric_sorted = sorted(
        metric_rows,
        key=lambda r: (
            STAGE_SORT_ORDER.get(r[0], 99),
            r[1],
            r[2],
        ),
    )
    if metric_rows != expected_metric_sorted:
        raise ArtifactValidationError(
            "forecast_metrics.csv rows are not sorted canonically",
            error_type="METRIC_ROW_ORDER_MISMATCH",
        )

    # 7. economic_summary.csv
    econ_sum_rows = _parse_and_validate_canonical_csv(
        raw_artifacts["economic_summary.csv"],
        "economic_summary.csv",
        ECONOMIC_SUMMARY_HEADER,
    )
    if len(econ_sum_rows) != len(ECONOMIC_K_VALUES):
        raise ArtifactValidationError(
            f"economic_summary.csv must have exactly {len(ECONOMIC_K_VALUES)} rows, got {len(econ_sum_rows)}",
            error_type="ECONOMIC_SUMMARY_ROW_COUNT_MISMATCH",
        )
    for idx, expected_k in enumerate(ECONOMIC_K_VALUES):
        if int(econ_sum_rows[idx][0]) != expected_k:
            raise ArtifactValidationError(
                f"economic_summary.csv row {idx} K mismatch: expected {expected_k}, got {econ_sum_rows[idx][0]}",
                error_type="ECONOMIC_SUMMARY_K_MISMATCH",
            )

    # 8. daily_forecast_scores.csv.gz
    daily_score_rows = _validate_gzip_scores(raw_artifacts["daily_forecast_scores.csv.gz"])

    # 9. artifact_manifest.json
    manifest_data = _validate_canonical_json(raw_artifacts["artifact_manifest.json"], "artifact_manifest.json")
    _validate_manifest(raw_artifacts, manifest_data)

    # --- Cross-Artifact Consistency Checks ---
    fc_signal: bool = adjudication["forecast_signal"]
    econ_signal: bool = adjudication["economic_signal"]
    exit_status: str = adjudication["development_exit_status"]
    qualified_top_k: list[int] = adjudication["qualified_top_k"]
    recommended_top_k: list[int] = adjudication["recommended_top_k"]
    selected_candidate: str = adjudication["selected_candidate_id"]

    # Recommended must be subset of qualified
    if not set(recommended_top_k).issubset(set(qualified_top_k)):
        raise ArtifactValidationError(
            "recommended_top_k must be a subset of qualified_top_k",
            error_type="RECOMMENDATION_NOT_QUALIFIED",
        )
    if len(recommended_top_k) > 1:
        raise ArtifactValidationError(
            f"At most one recommended portfolio allowed, got {len(recommended_top_k)}",
            error_type="MULTIPLE_RECOMMENDATIONS_ERROR",
        )

    # Lower bound in adjudication vs forecast_uncertainty
    _compare_floats(
        adjudication["forecast_bootstrap_lower_bound"],
        fc_unc["bootstrap_lower_bound"],
        "forecast_bootstrap_lower_bound",
    )

    val_m3_rows = [r for r in metric_rows if r[0] == "VAL" and r[1] == "M3"]
    if val_m3_rows and val_m3_rows[0][2] != selected_candidate:
        raise ArtifactValidationError(
            f"VAL M3 candidate_id mismatch: expected {selected_candidate}, got {val_m3_rows[0][2]}",
            error_type="VAL_CANDIDATE_MISMATCH",
        )

    # Stability rows check in daily scores and forecast metrics
    daily_stability_rows = [r for r in daily_score_rows if r[1] == "STABILITY"]
    metric_stability_rows = [r for r in metric_rows if r[0] == "STABILITY"]

    if not fc_signal:
        # Forecast gate failed case
        if exit_status != DevelopmentExitStatus.FORECAST_GATE_FAILED.value:
            raise ArtifactValidationError(
                f"Exit status must be FORECAST_GATE_FAILED when forecast_signal is False, got {exit_status}",
                error_type="INCONSISTENT_EXIT_STATUS",
            )
        if econ_signal:
            raise ArtifactValidationError(
                "economic_signal must be False when forecast_signal is False",
                error_type="INCONSISTENT_ECONOMIC_SIGNAL",
            )
        if qualified_top_k != [] or recommended_top_k != []:
            raise ArtifactValidationError(
                "qualified_top_k and recommended_top_k must be empty when forecast_signal is False",
                error_type="INCONSISTENT_PORTFOLIOS",
            )
        if daily_stability_rows:
            raise ArtifactValidationError(
                "STABILITY rows found when forecast gate failed in daily scores",
                error_type="FORECAST_GATE_FAILED_INVARIANT_VIOLATION",
            )
        if metric_stability_rows:
            raise ArtifactValidationError(
                "STABILITY rows found when forecast gate failed in forecast metrics",
                error_type="FORECAST_GATE_FAILED_INVARIANT_VIOLATION",
            )
        if stab_diag["status"] != "NOT_EVALUATED_FORECAST_GATE_FAILED":
            raise ArtifactValidationError(
                f"stability_diagnostics status must be NOT_EVALUATED_FORECAST_GATE_FAILED, got {stab_diag['status']}",
                error_type="INCONSISTENT_STABILITY_STATUS",
            )
        if stab_diag["date_count"] is not None or stab_diag["blocks"] != []:
            raise ArtifactValidationError(
                "stability_diagnostics date_count must be null and blocks empty when forecast gate failed",
                error_type="INCONSISTENT_STABILITY_DIAGNOSTICS",
            )
        if econ_unc["status"] != "NOT_EVALUATED_FORECAST_GATE_FAILED":
            raise ArtifactValidationError(
                f"economic_uncertainty status must be NOT_EVALUATED_FORECAST_GATE_FAILED, got {econ_unc['status']}",
                error_type="INCONSISTENT_ECONOMIC_STATUS",
            )
        for row in econ_sum_rows:
            if row[1] != "NOT_EVALUATED_FORECAST_GATE_FAILED" or row[7] != "false" or row[8] != "false":
                raise ArtifactValidationError(
                    "economic_summary rows must be NOT_EVALUATED_FORECAST_GATE_FAILED and false when forecast gate failed",
                    error_type="INCONSISTENT_ECONOMIC_SUMMARY",
                )
    else:
        # Forecast gate passed case
        if econ_signal:
            if exit_status != DevelopmentExitStatus.ECONOMIC_SIGNAL_FOUND.value:
                raise ArtifactValidationError(
                    f"Exit status must be ECONOMIC_SIGNAL_FOUND, got {exit_status}",
                    error_type="INCONSISTENT_EXIT_STATUS",
                )
        else:
            if exit_status != DevelopmentExitStatus.ECONOMIC_GATE_FAILED.value:
                raise ArtifactValidationError(
                    f"Exit status must be ECONOMIC_GATE_FAILED, got {exit_status}",
                    error_type="INCONSISTENT_EXIT_STATUS",
                )

        if not daily_stability_rows:
            raise ArtifactValidationError(
                "Missing STABILITY rows in daily forecast scores when forecast gate passed",
                error_type="MISSING_STABILITY_SCORES",
            )
        if not metric_stability_rows:
            raise ArtifactValidationError(
                "Missing STABILITY rows in forecast metrics when forecast gate passed",
                error_type="MISSING_STABILITY_METRICS",
            )

        if stab_diag["status"] != "EVALUATED":
            raise ArtifactValidationError(
                f"stability_diagnostics status must be EVALUATED, got {stab_diag['status']}",
                error_type="INCONSISTENT_STABILITY_STATUS",
            )
        if econ_unc["status"] != "EVALUATED":
            raise ArtifactValidationError(
                f"economic_uncertainty status must be EVALUATED, got {econ_unc['status']}",
                error_type="INCONSISTENT_ECONOMIC_STATUS",
            )

        # Cross check per_k stats between economic_uncertainty, stability_diagnostics, and economic_summary
        econ_unc_map = {item["K"]: item for item in econ_unc["per_k"]}
        stab_diag_map = {item["K"]: item for item in stab_diag["per_k"]}

        for row in econ_sum_rows:
            k = int(row[0])
            unc_rec = econ_unc_map.get(k)
            diag_rec = stab_diag_map.get(k)
            if not unc_rec or not diag_rec:
                raise ArtifactValidationError(
                    f"Missing per-K record for K={k}",
                    error_type="MISSING_PER_K_RECORD",
                )

            # Check mean_economic_delta
            row_mean = float(row[3])
            _compare_floats(row_mean, unc_rec["mean_economic_delta"], f"mean_economic_delta for K={k}")

            # Check positive_block_count
            row_blocks = int(row[4])
            if row_blocks != unc_rec["positive_block_count"] or row_blocks != diag_rec["positive_block_count"]:
                raise ArtifactValidationError(
                    f"positive_block_count mismatch for K={k}: {row_blocks} vs {unc_rec['positive_block_count']}",
                    error_type="CROSS_ARTIFACT_BLOCK_COUNT_MISMATCH",
                )

            # Check bootstrap_lower_bound
            row_lb = float(row[6])
            _compare_floats(row_lb, unc_rec["bootstrap_lower_bound"], f"bootstrap_lower_bound for K={k}")
            _compare_floats(row_lb, diag_rec["bootstrap_lower_bound"], f"stability diag bootstrap_lower_bound for K={k}")

            # Check qualifies
            row_qualifies = row[7].lower() == "true"
            if row_qualifies != unc_rec["qualifies"] or row_qualifies != diag_rec["qualifies"]:
                raise ArtifactValidationError(
                    f"qualifies mismatch for K={k}",
                    error_type="CROSS_ARTIFACT_QUALIFIES_MISMATCH",
                )
            if row_qualifies != (k in qualified_top_k):
                raise ArtifactValidationError(
                    f"qualifies mismatch with adjudication qualified_top_k for K={k}",
                    error_type="QUALIFIED_TOP_K_MISMATCH",
                )

            # Check recommended
            row_recommended = row[8].lower() == "true"
            if row_recommended != (k in recommended_top_k):
                raise ArtifactValidationError(
                    f"recommended mismatch with adjudication recommended_top_k for K={k}",
                    error_type="RECOMMENDED_TOP_K_MISMATCH",
                )
