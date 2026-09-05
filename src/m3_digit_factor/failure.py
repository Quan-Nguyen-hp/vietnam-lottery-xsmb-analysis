"""Terminal failure lifecycle and development_run_FAILED.json for XPIS v3 M3 (Slice M3-12).

This module implements the exact terminal failure lifecycle for XPIS v3 M3:
1. Exact single failure artifact: 'development_run_FAILED.json'.
2. Closed 5-key schema: status, failed_stage, error_type, development_exit_status, protocol_snapshot.
3. Protocol snapshot semantics: null before AUTHORITY_COMPLETE, complete 72-key authority
   object + fingerprint after AUTHORITY_COMPLETE.
4. Canonical byte-exact JSON serialization (UTF-8, no BOM, compact delimiters, sort_keys=True, final LF).
5. Fail-closed schema enforcement and mutual exclusivity between success and failure artifacts.
6. Rejection of recursive failure generation (validation/construction errors raise directly).
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .artifacts import SUCCESS_ARTIFACT_NAMES
from .authority import (
    AuthoritySnapshot,
    AuthorityValidationError,
    protocol_fingerprint,
    validate_authority,
)
from .contracts import (
    FailureExitStatus,
    FailureStage,
    ProtocolFailure,
)
from .serialization import serialize_json
from .validation import (
    ArtifactValidationError,
    _load_raw_artifacts,
    validate_success_artifacts,
)


FAILURE_ARTIFACT_NAME: str = "development_run_FAILED.json"
FAILURE_ARTIFACT_COUNT: int = 1

FAILURE_ALLOWED_STAGES: tuple[str, ...] = tuple(s.value for s in FailureStage)
FAILURE_ALLOWED_EXIT_STATUSES: tuple[str, ...] = tuple(s.value for s in FailureExitStatus)

_FAILURE_SCHEMA_KEYS: frozenset[str] = frozenset({
    "status",
    "failed_stage",
    "error_type",
    "development_exit_status",
    "protocol_snapshot",
})


class FailureValidationError(ArtifactValidationError):
    """Raised when failure artifact schema, encoding, or bundle validation fails."""

    stage: FailureStage = FailureStage.ARTIFACT_VALIDATION
    exit_status: FailureExitStatus = FailureExitStatus.NEEDS_PROTOCOL_REVISION

    def __init__(
        self,
        message: str,
        error_type: str = "FailureValidationError",
        exit_status: FailureExitStatus = FailureExitStatus.NEEDS_PROTOCOL_REVISION,
    ) -> None:
        super().__init__(message, error_type=error_type, exit_status=exit_status)


def _resolve_protocol_snapshot(
    snapshot: AuthoritySnapshot | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve and validate the protocol snapshot representation.

    Under B-08:
    - Before AUTHORITY_COMPLETE: protocol_snapshot is null (None).
    - After AUTHORITY_COMPLETE: protocol_snapshot contains exactly the complete 72-key
      authority object and valid protocol_fingerprint_sha256.
    - Partial authority objects, placeholders, or sentinel hashes are strictly rejected.
    """
    if snapshot is None:
        return None

    if isinstance(snapshot, AuthoritySnapshot):
        if not snapshot.authority_complete:
            return None
        return snapshot.protocol_snapshot

    if isinstance(snapshot, Mapping):
        keys = set(snapshot.keys())
        expected_snap_keys = {"authority", "protocol_fingerprint_sha256"}
        if keys != expected_snap_keys:
            raise FailureValidationError(
                f"protocol_snapshot must have exactly keys {sorted(expected_snap_keys)}, got {sorted(keys)}",
                error_type="INVALID_PROTOCOL_SNAPSHOT_KEYS",
            )
        auth = snapshot["authority"]
        if not isinstance(auth, Mapping):
            raise FailureValidationError(
                "protocol_snapshot.authority must be a mapping",
                error_type="INVALID_AUTHORITY_TYPE",
            )
        materialized_auth = dict(auth)
        try:
            validate_authority(materialized_auth)
        except (AuthorityValidationError, ValueError) as err:
            raise FailureValidationError(
                f"Invalid authority in protocol_snapshot: {err}",
                error_type="INVALID_AUTHORITY_CONTENT",
            ) from err

        expected_fp = protocol_fingerprint(materialized_auth)
        provided_fp = snapshot["protocol_fingerprint_sha256"]
        if provided_fp != expected_fp:
            raise FailureValidationError(
                f"protocol_fingerprint_sha256 mismatch in protocol_snapshot: {provided_fp} vs {expected_fp}",
                error_type="PROTOCOL_FINGERPRINT_MISMATCH",
            )
        return {
            "authority": materialized_auth,
            "protocol_fingerprint_sha256": expected_fp,
        }

    raise FailureValidationError(
        f"protocol_snapshot must be object or null, got {type(snapshot).__name__}",
        error_type="INVALID_PROTOCOL_SNAPSHOT_TYPE",
    )


def build_failure_payload(
    failed_stage: FailureStage | str,
    error_type: str,
    development_exit_status: FailureExitStatus | str,
    protocol_snapshot: AuthoritySnapshot | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the exact 5-key terminal failure payload dictionary."""
    if isinstance(failed_stage, FailureStage):
        stage_str = failed_stage.value
    elif isinstance(failed_stage, str) and failed_stage in FAILURE_ALLOWED_STAGES:
        stage_str = failed_stage
    else:
        raise FailureValidationError(
            f"Invalid failed_stage: {failed_stage!r}. Allowed: {list(FAILURE_ALLOWED_STAGES)}",
            error_type="INVALID_FAILED_STAGE",
        )

    if not isinstance(error_type, str) or not error_type.strip():
        raise FailureValidationError(
            "error_type must be a non-empty string",
            error_type="INVALID_ERROR_TYPE",
        )
    error_type_str = error_type.strip()

    if isinstance(development_exit_status, FailureExitStatus):
        status_str = development_exit_status.value
    elif isinstance(development_exit_status, str) and development_exit_status in FAILURE_ALLOWED_EXIT_STATUSES:
        status_str = development_exit_status
    else:
        raise FailureValidationError(
            f"Invalid development_exit_status: {development_exit_status!r}. Allowed: {list(FAILURE_ALLOWED_EXIT_STATUSES)}",
            error_type="INVALID_DEVELOPMENT_EXIT_STATUS",
        )

    resolved_snapshot = _resolve_protocol_snapshot(protocol_snapshot)

    return {
        "development_exit_status": status_str,
        "error_type": error_type_str,
        "failed_stage": stage_str,
        "protocol_snapshot": resolved_snapshot,
        "status": "FAILED",
    }


def build_failure_artifact(
    failed_stage: FailureStage | str,
    error_type: str,
    development_exit_status: FailureExitStatus | str,
    protocol_snapshot: AuthoritySnapshot | Mapping[str, Any] | None = None,
) -> bytes:
    """Build canonical byte-exact serialized bytes for development_run_FAILED.json."""
    payload = build_failure_payload(
        failed_stage=failed_stage,
        error_type=error_type,
        development_exit_status=development_exit_status,
        protocol_snapshot=protocol_snapshot,
    )
    return serialize_json(payload)


def build_failure_artifact_set(
    failed_stage: FailureStage | str,
    error_type: str,
    development_exit_status: FailureExitStatus | str,
    protocol_snapshot: AuthoritySnapshot | Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Build the single-artifact failure set mapping."""
    return {
        FAILURE_ARTIFACT_NAME: build_failure_artifact(
            failed_stage=failed_stage,
            error_type=error_type,
            development_exit_status=development_exit_status,
            protocol_snapshot=protocol_snapshot,
        )
    }


def build_failure_artifact_from_failure(
    failure: ProtocolFailure,
    protocol_snapshot: AuthoritySnapshot | Mapping[str, Any] | None = None,
) -> bytes:
    """Build failure artifact bytes from an immutable ProtocolFailure contract instance."""
    return build_failure_artifact(
        failed_stage=failure.stage,
        error_type=failure.error_type,
        development_exit_status=failure.exit_status,
        protocol_snapshot=protocol_snapshot,
    )


def build_failure_artifact_from_exception(
    exc: Exception,
    protocol_snapshot: AuthoritySnapshot | Mapping[str, Any] | None = None,
) -> bytes:
    """Build failure artifact bytes from an M3 pipeline exception."""
    if hasattr(exc, "as_protocol_failure") and callable(exc.as_protocol_failure):
        pf: ProtocolFailure = exc.as_protocol_failure()
        return build_failure_artifact_from_failure(pf, protocol_snapshot=protocol_snapshot)

    stage = getattr(exc, "stage", None)
    error_type = getattr(exc, "error_type", type(exc).__name__)
    exit_status = getattr(exc, "exit_status", None)

    if stage is None or exit_status is None:
        if error_type in {
            "ZeroDigitMarginal",
            "SVDLeadingSubspaceAmbiguity",
            "CenteredSingularVectorDegeneracy",
        }:
            stage = stage or FailureStage.MODEL_INITIALIZATION
            exit_status = exit_status or FailureExitStatus.NEEDS_MODEL_REVISION
        elif error_type in {
            "ConstraintViolation",
            "OptimizerNonConvergence",
            "NonFiniteModelFit",
            "ForecastContractViolation",
        }:
            stage = stage or FailureStage.MODEL_FIT
            exit_status = exit_status or FailureExitStatus.NEEDS_MODEL_REVISION
        elif error_type == "OptimizerExecutionError":
            stage = stage or FailureStage.MODEL_FIT
            exit_status = exit_status or FailureExitStatus.TECHNICAL_FAILURE
        elif error_type == "InsufficientDevelopmentSample":
            stage = stage or FailureStage.DATA_VALIDATION
            exit_status = exit_status or FailureExitStatus.NEEDS_DATA_REVISION
        elif error_type == "ArtifactProtocolInconsistencyError":
            stage = stage or FailureStage.ARTIFACT_VALIDATION
            exit_status = exit_status or FailureExitStatus.NEEDS_PROTOCOL_REVISION
        elif error_type == "ArtifactTechnicalFailureError":
            stage = stage or FailureStage.ARTIFACT_VALIDATION
            exit_status = exit_status or FailureExitStatus.TECHNICAL_FAILURE
        else:
            stage = stage or FailureStage.DATA_VALIDATION
            exit_status = exit_status or FailureExitStatus.TECHNICAL_FAILURE

    return build_failure_artifact(
        failed_stage=stage,
        error_type=error_type,
        development_exit_status=exit_status,
        protocol_snapshot=protocol_snapshot,
    )


def validate_failure_artifact(raw_bytes: bytes) -> dict[str, Any]:
    """Validate development_run_FAILED.json byte-exactness and schema fail-closed."""
    if not raw_bytes.endswith(b"\n"):
        raise FailureValidationError(
            "Failure artifact does not terminate with LF",
            error_type="CANONICAL_ENCODING_MISMATCH",
        )
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except Exception as err:
        raise FailureValidationError(
            f"Failed to parse failure artifact JSON: {err}",
            error_type="JSON_DECODE_ERROR",
        ) from err

    if not isinstance(parsed, dict):
        raise FailureValidationError(
            f"Failure artifact JSON root must be object, got {type(parsed).__name__}",
            error_type="FAILURE_SCHEMA_ERROR",
        )

    re_serialized = serialize_json(parsed)
    if re_serialized != raw_bytes:
        raise FailureValidationError(
            "Canonical JSON byte-exactness mismatch for failure artifact",
            error_type="CANONICAL_ENCODING_MISMATCH",
        )

    keys = set(parsed.keys())
    if keys != _FAILURE_SCHEMA_KEYS:
        raise FailureValidationError(
            f"Exact 5 top-level keys required, got {sorted(keys)}",
            error_type="FAILURE_SCHEMA_ERROR",
        )

    if parsed["status"] != "FAILED":
        raise FailureValidationError(
            f"Failure artifact status must be 'FAILED', got {parsed['status']!r}",
            error_type="INVALID_FAILURE_STATUS",
        )

    if parsed["failed_stage"] not in FAILURE_ALLOWED_STAGES:
        raise FailureValidationError(
            f"Invalid failed_stage: {parsed['failed_stage']!r}. Allowed: {list(FAILURE_ALLOWED_STAGES)}",
            error_type="INVALID_FAILED_STAGE",
        )

    if not isinstance(parsed["error_type"], str) or not parsed["error_type"].strip():
        raise FailureValidationError(
            "error_type must be a non-empty string",
            error_type="INVALID_ERROR_TYPE",
        )

    if parsed["development_exit_status"] not in FAILURE_ALLOWED_EXIT_STATUSES:
        raise FailureValidationError(
            f"Invalid development_exit_status: {parsed['development_exit_status']!r}. Allowed: {list(FAILURE_ALLOWED_EXIT_STATUSES)}",
            error_type="INVALID_DEVELOPMENT_EXIT_STATUS",
        )

    # Validate protocol_snapshot semantics under B-08
    snapshot = parsed["protocol_snapshot"]
    if snapshot is not None:
        if not isinstance(snapshot, dict):
            raise FailureValidationError(
                f"protocol_snapshot must be object or null, got {type(snapshot).__name__}",
                error_type="INVALID_PROTOCOL_SNAPSHOT_TYPE",
            )
        _resolve_protocol_snapshot(snapshot)

    return parsed


def validate_failure_artifacts(artifacts: Mapping[str, bytes] | str | Path) -> dict[str, Any]:
    """Verify that artifact bundle contains exactly development_run_FAILED.json and passes validation."""
    raw_bundle = _load_raw_artifacts(artifacts)
    present_keys = set(raw_bundle.keys())

    if FAILURE_ARTIFACT_NAME not in present_keys:
        raise FailureValidationError(
            f"Missing required failure artifact: {FAILURE_ARTIFACT_NAME}",
            error_type="MISSING_FAILURE_ARTIFACT",
        )

    extra_keys = present_keys - {FAILURE_ARTIFACT_NAME}
    if extra_keys:
        raise FailureValidationError(
            f"Unexpected extra artifact(s) in failure bundle: {sorted(extra_keys)}",
            error_type="UNEXPECTED_ARTIFACT",
        )

    if len(raw_bundle) != FAILURE_ARTIFACT_COUNT:
        raise FailureValidationError(
            f"Expected exactly {FAILURE_ARTIFACT_COUNT} artifact in failure bundle, got {len(raw_bundle)}",
            error_type="ARTIFACT_COUNT_MISMATCH",
        )

    return validate_failure_artifact(raw_bundle[FAILURE_ARTIFACT_NAME])


def validate_artifact_bundle_exclusivity(artifacts: Mapping[str, bytes] | str | Path) -> str:
    """Verify mutual exclusivity between success and failure artifacts.

    Returns:
        'SUCCESS' if exactly the 9 success artifacts are present and valid.
        'FAILED' if exactly the 1 failure artifact is present and valid.

    Raises:
        FailureValidationError: if success and failure artifacts coexist or inventory is invalid.
    """
    raw_bundle = _load_raw_artifacts(artifacts)
    present_keys = set(raw_bundle.keys())

    has_failure = FAILURE_ARTIFACT_NAME in present_keys
    success_keys = present_keys.intersection(SUCCESS_ARTIFACT_NAMES)

    if has_failure and success_keys:
        raise FailureValidationError(
            f"Mutual exclusivity violated: failure artifact {FAILURE_ARTIFACT_NAME} cannot "
            f"coexist with success artifacts {sorted(success_keys)}",
            error_type="MUTUAL_EXCLUSIVITY_VIOLATION",
        )

    if has_failure:
        validate_failure_artifacts(raw_bundle)
        return "FAILED"

    if success_keys:
        validate_success_artifacts(raw_bundle)
        return "SUCCESS"

    raise FailureValidationError(
        "Artifact bundle is empty or contains no recognized protocol artifacts",
        error_type="EMPTY_OR_UNRECOGNIZED_BUNDLE",
    )
