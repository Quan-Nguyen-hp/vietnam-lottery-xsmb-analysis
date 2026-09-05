"""Tests for XPIS v3 M3 failure lifecycle and development_run_FAILED.json (Slice M3-12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.m3_digit_factor.artifacts import build_success_artifacts
from src.m3_digit_factor.authority import (
    AuthoritySnapshot,
    RepositoryIdentity,
    build_authority,
    protocol_fingerprint,
)
from src.m3_digit_factor.contracts import (
    FailureExitStatus,
    FailureStage,
    ProtocolFailure,
)
from src.m3_digit_factor.failure import (
    FAILURE_ALLOWED_EXIT_STATUSES,
    FAILURE_ALLOWED_STAGES,
    FAILURE_ARTIFACT_COUNT,
    FAILURE_ARTIFACT_NAME,
    FailureValidationError,
    build_failure_artifact,
    build_failure_artifact_from_exception,
    build_failure_artifact_from_failure,
    build_failure_artifact_set,
    build_failure_payload,
    validate_artifact_bundle_exclusivity,
    validate_failure_artifact,
    validate_failure_artifacts,
)
from src.m3_digit_factor.fitting import (
    ConstraintViolation,
    NonFiniteModelFit,
    OptimizerExecutionError,
    OptimizerNonConvergence,
)
from src.m3_digit_factor.initialization import (
    CenteredSingularVectorDegeneracy,
    SVDLeadingSubspaceAmbiguity,
    ZeroDigitMarginal,
)
from src.m3_digit_factor.model import ForecastContractViolation
from src.m3_digit_factor.validation import (
    ArtifactProtocolInconsistencyError,
    ArtifactTechnicalFailureError,
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


def _sample_authority_snapshot() -> AuthoritySnapshot:
    auth = _sample_authority()
    return AuthoritySnapshot(authority=auth, protocol_fingerprint_sha256=protocol_fingerprint(auth))


class TestFailureConstants:
    """Verify closed constants for M3 failure lifecycle."""

    def test_artifact_name_and_count(self) -> None:
        assert FAILURE_ARTIFACT_NAME == "development_run_FAILED.json"
        assert FAILURE_ARTIFACT_COUNT == 1

    def test_closed_stages(self) -> None:
        expected_stages = {
            "DATA_VALIDATION",
            "MODEL_INITIALIZATION",
            "MODEL_FIT",
            "FORECAST_GENERATION",
            "METRIC_EVALUATION",
            "FORECAST_BOOTSTRAP",
            "ECONOMIC_EVALUATION",
            "ECONOMIC_BOOTSTRAP",
            "ARTIFACT_VALIDATION",
        }
        assert set(FAILURE_ALLOWED_STAGES) == expected_stages
        assert len(FAILURE_ALLOWED_STAGES) == 9
        for s in expected_stages:
            assert FailureStage(s).value == s

    def test_closed_exit_statuses(self) -> None:
        expected_statuses = {
            "NEEDS_DATA_REVISION",
            "NEEDS_MODEL_REVISION",
            "NEEDS_PROTOCOL_REVISION",
            "TECHNICAL_FAILURE",
        }
        assert set(FAILURE_ALLOWED_EXIT_STATUSES) == expected_statuses
        assert len(FAILURE_ALLOWED_EXIT_STATUSES) == 4
        for s in expected_statuses:
            assert FailureExitStatus(s).value == s


class TestFailurePayloadAndArtifact:
    """Verify exact 5-key schema and byte-exact canonical serialization."""

    def test_pre_authority_payload_schema(self) -> None:
        payload = build_failure_payload(
            failed_stage=FailureStage.DATA_VALIDATION,
            error_type="InsufficientDevelopmentSample",
            development_exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
            protocol_snapshot=None,
        )
        assert set(payload.keys()) == {
            "status",
            "failed_stage",
            "error_type",
            "development_exit_status",
            "protocol_snapshot",
        }
        assert payload["status"] == "FAILED"
        assert payload["failed_stage"] == "DATA_VALIDATION"
        assert payload["error_type"] == "InsufficientDevelopmentSample"
        assert payload["development_exit_status"] == "NEEDS_DATA_REVISION"
        assert payload["protocol_snapshot"] is None

    def test_post_authority_payload_schema(self) -> None:
        snap = _sample_authority_snapshot()
        payload = build_failure_payload(
            failed_stage=FailureStage.MODEL_FIT,
            error_type="ConstraintViolation",
            development_exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
            protocol_snapshot=snap,
        )
        assert payload["protocol_snapshot"] is not None
        assert set(payload["protocol_snapshot"].keys()) == {"authority", "protocol_fingerprint_sha256"}
        assert len(payload["protocol_snapshot"]["authority"]) == 72
        assert len(payload["protocol_snapshot"]["protocol_fingerprint_sha256"]) == 64

    def test_canonical_byte_exactness(self) -> None:
        raw_bytes = build_failure_artifact(
            failed_stage=FailureStage.MODEL_INITIALIZATION,
            error_type="SVDLeadingSubspaceAmbiguity",
            development_exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
            protocol_snapshot=None,
        )
        assert raw_bytes.endswith(b"\n")
        assert not raw_bytes.startswith(b"\xef\xbb\xbf")  # No UTF-8 BOM
        text = raw_bytes.decode("utf-8")
        assert ": " not in text  # compact separators
        assert ", " not in text

        # Bitwise roundtrip validation
        parsed = validate_failure_artifact(raw_bytes)
        assert parsed["status"] == "FAILED"
        assert parsed["error_type"] == "SVDLeadingSubspaceAmbiguity"
        assert parsed["failed_stage"] == "MODEL_INITIALIZATION"
        assert parsed["development_exit_status"] == "NEEDS_MODEL_REVISION"
        assert parsed["protocol_snapshot"] is None

    def test_failure_artifact_set(self) -> None:
        artifact_set = build_failure_artifact_set(
            failed_stage=FailureStage.METRIC_EVALUATION,
            error_type="EvaluationDiscrepancy",
            development_exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
        assert set(artifact_set.keys()) == {FAILURE_ARTIFACT_NAME}
        assert len(artifact_set) == FAILURE_ARTIFACT_COUNT
        validate_failure_artifacts(artifact_set)


class TestFailurePreservationAndMappings:
    """Verify specific known failure mappings across pipeline stages."""

    @pytest.mark.parametrize(
        ("stage", "status"),
        [
            (FailureStage.DATA_VALIDATION, FailureExitStatus.NEEDS_DATA_REVISION),
            (FailureStage.MODEL_INITIALIZATION, FailureExitStatus.NEEDS_MODEL_REVISION),
            (FailureStage.MODEL_FIT, FailureExitStatus.NEEDS_MODEL_REVISION),
            (FailureStage.FORECAST_GENERATION, FailureExitStatus.NEEDS_MODEL_REVISION),
            (FailureStage.METRIC_EVALUATION, FailureExitStatus.NEEDS_MODEL_REVISION),
            (FailureStage.FORECAST_BOOTSTRAP, FailureExitStatus.NEEDS_MODEL_REVISION),
            (FailureStage.ECONOMIC_EVALUATION, FailureExitStatus.NEEDS_MODEL_REVISION),
            (FailureStage.ECONOMIC_BOOTSTRAP, FailureExitStatus.NEEDS_MODEL_REVISION),
            (FailureStage.ARTIFACT_VALIDATION, FailureExitStatus.NEEDS_PROTOCOL_REVISION),
            (FailureStage.MODEL_FIT, FailureExitStatus.TECHNICAL_FAILURE),
            (FailureStage.ARTIFACT_VALIDATION, FailureExitStatus.TECHNICAL_FAILURE),
        ],
    )
    def test_stage_and_exit_status_combinations(self, stage: FailureStage, status: FailureExitStatus) -> None:
        raw = build_failure_artifact(
            failed_stage=stage,
            error_type=f"Test_{stage.value}_{status.value}",
            development_exit_status=status,
        )
        parsed = validate_failure_artifact(raw)
        assert parsed["failed_stage"] == stage.value
        assert parsed["development_exit_status"] == status.value

    def test_svd_ambiguity_triplet(self) -> None:
        raw = build_failure_artifact(
            failed_stage=FailureStage.MODEL_INITIALIZATION,
            error_type="SVDLeadingSubspaceAmbiguity",
            development_exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
        parsed = validate_failure_artifact(raw)
        assert parsed["failed_stage"] == "MODEL_INITIALIZATION"
        assert parsed["error_type"] == "SVDLeadingSubspaceAmbiguity"
        assert parsed["development_exit_status"] == "NEEDS_MODEL_REVISION"

    def test_u005_fitting_failures(self) -> None:
        u005_cases = [
            (ConstraintViolation(), "ConstraintViolation", FailureExitStatus.NEEDS_MODEL_REVISION),
            (OptimizerNonConvergence(), "OptimizerNonConvergence", FailureExitStatus.NEEDS_MODEL_REVISION),
            (NonFiniteModelFit(), "NonFiniteModelFit", FailureExitStatus.NEEDS_MODEL_REVISION),
            (ForecastContractViolation(), "ForecastContractViolation", FailureExitStatus.NEEDS_MODEL_REVISION),
            (OptimizerExecutionError(), "OptimizerExecutionError", FailureExitStatus.TECHNICAL_FAILURE),
        ]
        for exc, expected_type, expected_status in u005_cases:
            raw = build_failure_artifact_from_exception(exc)
            parsed = validate_failure_artifact(raw)
            assert parsed["failed_stage"] == "MODEL_FIT"
            assert parsed["error_type"] == expected_type
            assert parsed["development_exit_status"] == expected_status.value

    def test_initialization_exceptions(self) -> None:
        init_cases = [
            (ZeroDigitMarginal(), "ZeroDigitMarginal"),
            (SVDLeadingSubspaceAmbiguity(), "SVDLeadingSubspaceAmbiguity"),
            (CenteredSingularVectorDegeneracy(), "CenteredSingularVectorDegeneracy"),
        ]
        for exc, expected_type in init_cases:
            raw = build_failure_artifact_from_exception(exc)
            parsed = validate_failure_artifact(raw)
            assert parsed["failed_stage"] == "MODEL_INITIALIZATION"
            assert parsed["error_type"] == expected_type
            assert parsed["development_exit_status"] == "NEEDS_MODEL_REVISION"

    def test_artifact_validation_exceptions(self) -> None:
        p_err = ArtifactProtocolInconsistencyError("Contract violated")
        raw_p = build_failure_artifact_from_exception(p_err)
        parsed_p = validate_failure_artifact(raw_p)
        assert parsed_p["failed_stage"] == "ARTIFACT_VALIDATION"
        assert parsed_p["error_type"] == "ArtifactProtocolInconsistencyError"
        assert parsed_p["development_exit_status"] == "NEEDS_PROTOCOL_REVISION"

        t_err = ArtifactTechnicalFailureError("Disk full")
        raw_t = build_failure_artifact_from_exception(t_err)
        parsed_t = validate_failure_artifact(raw_t)
        assert parsed_t["failed_stage"] == "ARTIFACT_VALIDATION"
        assert parsed_t["error_type"] == "ArtifactTechnicalFailureError"
        assert parsed_t["development_exit_status"] == "TECHNICAL_FAILURE"

    def test_from_protocol_failure_object(self) -> None:
        pf = ProtocolFailure(
            stage=FailureStage.FORECAST_BOOTSTRAP,
            error_type="BootstrapDegeneracy",
            exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
        raw = build_failure_artifact_from_failure(pf)
        parsed = validate_failure_artifact(raw)
        assert parsed["failed_stage"] == "FORECAST_BOOTSTRAP"
        assert parsed["error_type"] == "BootstrapDegeneracy"
        assert parsed["development_exit_status"] == "NEEDS_MODEL_REVISION"


class TestProtocolSnapshotSemantics:
    """Verify B-08 pre- vs post-AUTHORITY_COMPLETE semantics."""

    def test_pre_authority_must_be_literal_null(self) -> None:
        raw = build_failure_artifact(
            failed_stage=FailureStage.DATA_VALIDATION,
            error_type="RepositoryMismatch",
            development_exit_status=FailureExitStatus.TECHNICAL_FAILURE,
            protocol_snapshot=None,
        )
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["protocol_snapshot"] is None
        assert "null" in raw.decode("utf-8")

    def test_reject_sentinel_string_snapshot(self) -> None:
        payload = {
            "status": "FAILED",
            "failed_stage": "DATA_VALIDATION",
            "error_type": "RepositoryMismatch",
            "development_exit_status": "TECHNICAL_FAILURE",
            "protocol_snapshot": "0" * 64,  # Sentinel string is forbidden!
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with pytest.raises(FailureValidationError, match="protocol_snapshot must be object or null"):
            validate_failure_artifact(raw)

    def test_reject_partial_authority_snapshot(self) -> None:
        payload = {
            "status": "FAILED",
            "failed_stage": "DATA_VALIDATION",
            "error_type": "RepositoryMismatch",
            "development_exit_status": "TECHNICAL_FAILURE",
            "protocol_snapshot": {
                "authority": {"spec_version": "XPIS_V3_M3_DIGIT_FACTOR_DESIGN_V3"},  # Partial!
                "protocol_fingerprint_sha256": "a" * 64,
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with pytest.raises(FailureValidationError):
            validate_failure_artifact(raw)

    def test_reject_snapshot_with_corrupted_fingerprint(self) -> None:
        auth = _sample_authority()
        payload = {
            "status": "FAILED",
            "failed_stage": "MODEL_FIT",
            "error_type": "OptimizerNonConvergence",
            "development_exit_status": "NEEDS_MODEL_REVISION",
            "protocol_snapshot": {
                "authority": auth,
                "protocol_fingerprint_sha256": "0" * 64,  # Mismatched fingerprint!
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with pytest.raises(FailureValidationError, match="fingerprint"):
            validate_failure_artifact(raw)

    def test_accept_complete_valid_snapshot(self) -> None:
        auth = _sample_authority()
        fp = protocol_fingerprint(auth)
        raw = build_failure_artifact(
            failed_stage=FailureStage.FORECAST_GENERATION,
            error_type="NonFiniteForecast",
            development_exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
            protocol_snapshot={"authority": auth, "protocol_fingerprint_sha256": fp},
        )
        parsed = validate_failure_artifact(raw)
        assert parsed["protocol_snapshot"]["protocol_fingerprint_sha256"] == fp


class TestFailureSchemaEnforcement:
    """Verify strict fail-closed rejection of invalid failure artifacts."""

    def test_reject_extraneous_keys(self) -> None:
        extraneous = ["stack_trace", "debug_notes", "python_exception_repr", "timestamp", "hostname", "cwd"]
        for key in extraneous:
            payload = {
                "status": "FAILED",
                "failed_stage": "MODEL_FIT",
                "error_type": "ConstraintViolation",
                "development_exit_status": "NEEDS_MODEL_REVISION",
                "protocol_snapshot": None,
                key: "forbidden",
            }
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            with pytest.raises(FailureValidationError, match="Exact 5 top-level keys required"):
                validate_failure_artifact(raw)

    def test_reject_missing_keys(self) -> None:
        base = {
            "status": "FAILED",
            "failed_stage": "MODEL_FIT",
            "error_type": "ConstraintViolation",
            "development_exit_status": "NEEDS_MODEL_REVISION",
            "protocol_snapshot": None,
        }
        for key in list(base.keys()):
            incomplete = dict(base)
            del incomplete[key]
            raw = json.dumps(incomplete, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            with pytest.raises(FailureValidationError, match="Exact 5 top-level keys required"):
                validate_failure_artifact(raw)

    def test_reject_invalid_status(self) -> None:
        payload = {
            "status": "SUCCESS",  # Invalid!
            "failed_stage": "MODEL_FIT",
            "error_type": "ConstraintViolation",
            "development_exit_status": "NEEDS_MODEL_REVISION",
            "protocol_snapshot": None,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with pytest.raises(FailureValidationError, match="status must be 'FAILED'"):
            validate_failure_artifact(raw)

    def test_reject_invalid_failed_stage(self) -> None:
        payload = {
            "status": "FAILED",
            "failed_stage": "UNKNOWN_STAGE",  # Invalid!
            "error_type": "ConstraintViolation",
            "development_exit_status": "NEEDS_MODEL_REVISION",
            "protocol_snapshot": None,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with pytest.raises(FailureValidationError, match="Invalid failed_stage"):
            validate_failure_artifact(raw)

    def test_reject_invalid_exit_status(self) -> None:
        payload = {
            "status": "FAILED",
            "failed_stage": "MODEL_FIT",
            "error_type": "ConstraintViolation",
            "development_exit_status": "ECONOMIC_SIGNAL_FOUND",  # Success exit status, invalid in failure artifact!
            "protocol_snapshot": None,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with pytest.raises(FailureValidationError, match="Invalid development_exit_status"):
            validate_failure_artifact(raw)

    def test_reject_empty_error_type(self) -> None:
        payload = {
            "status": "FAILED",
            "failed_stage": "MODEL_FIT",
            "error_type": "",  # Empty!
            "development_exit_status": "NEEDS_MODEL_REVISION",
            "protocol_snapshot": None,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with pytest.raises(FailureValidationError, match="error_type must be a non-empty string"):
            validate_failure_artifact(raw)

    def test_reject_non_lf_termination(self) -> None:
        raw = b'{"development_exit_status":"TECHNICAL_FAILURE","error_type":"Err","failed_stage":"DATA_VALIDATION","protocol_snapshot":null,"status":"FAILED"}'
        with pytest.raises(FailureValidationError, match="LF"):
            validate_failure_artifact(raw)

    def test_reject_non_canonical_encoding(self) -> None:
        raw = b'{\n  "status": "FAILED"\n}\n'
        with pytest.raises(FailureValidationError):
            validate_failure_artifact(raw)


class TestArtifactBundleMutualExclusivity:
    """Verify strict mutual exclusivity between success and failure artifacts."""

    def test_valid_failure_set(self) -> None:
        failure_bundle = build_failure_artifact_set(
            failed_stage=FailureStage.MODEL_INITIALIZATION,
            error_type="ZeroDigitMarginal",
            development_exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
        assert validate_artifact_bundle_exclusivity(failure_bundle) == "FAILED"

    def test_valid_success_set(self) -> None:
        auth = _sample_authority()
        success_bundle = build_success_artifacts(
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
            ],
            forecast_metric_rows=[
                ["DEV", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
                ["DEV", "M3", "M3_W030", 0.69, 0.34, 0.44],
                ["DEV", "M3", "M3_W060", 0.68, 0.33, 0.43],
                ["DEV", "M3", "M3_W120", 0.69, 0.34, 0.44],
                ["DEV", "M3", "M3_W240", 0.70, 0.35, 0.45],
                ["DEV", "M3", "M3_W365", 0.71, 0.36, 0.46],
                ["VAL", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
                ["VAL", "M3", "M3_W060", 0.68, 0.33, 0.43],
            ],
        )
        assert validate_artifact_bundle_exclusivity(success_bundle) == "SUCCESS"

    def test_reject_coexistence_of_success_and_failure(self) -> None:
        auth = _sample_authority()
        bundle = build_success_artifacts(
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
            ],
            forecast_metric_rows=[
                ["DEV", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
                ["DEV", "M3", "M3_W030", 0.69, 0.34, 0.44],
                ["DEV", "M3", "M3_W060", 0.68, 0.33, 0.43],
                ["DEV", "M3", "M3_W120", 0.69, 0.34, 0.44],
                ["DEV", "M3", "M3_W240", 0.70, 0.35, 0.45],
                ["DEV", "M3", "M3_W365", 0.71, 0.36, 0.46],
                ["VAL", "B0", "B0_UNIFORM", 0.70, 0.35, 0.45],
                ["VAL", "M3", "M3_W060", 0.68, 0.33, 0.43],
            ],
        )
        bundle[FAILURE_ARTIFACT_NAME] = build_failure_artifact(
            failed_stage=FailureStage.MODEL_FIT,
            error_type="ConstraintViolation",
            development_exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
        )
        with pytest.raises(FailureValidationError, match="Mutual exclusivity violated"):
            validate_artifact_bundle_exclusivity(bundle)

    def test_directory_based_validation(self, tmp_path: Path) -> None:
        fail_set = build_failure_artifact_set(
            failed_stage=FailureStage.DATA_VALIDATION,
            error_type="InsufficientDevelopmentSample",
            development_exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
        for fname, content in fail_set.items():
            (tmp_path / fname).write_bytes(content)

        assert validate_artifact_bundle_exclusivity(tmp_path) == "FAILED"
        validate_failure_artifacts(tmp_path)

        # Adding an extraneous file must fail
        (tmp_path / "extra.txt").write_text("rogue")
        with pytest.raises(FailureValidationError, match="Unexpected extra"):
            validate_failure_artifacts(tmp_path)
