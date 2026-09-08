"""Closed protocol vocabulary for XPIS v3 M3 digit-factor development."""

from dataclasses import dataclass
from enum import Enum


class FailureStage(str, Enum):
    """Stages that may be reported by a failed development run."""

    DATA_VALIDATION = 'DATA_VALIDATION'
    MODEL_INITIALIZATION = 'MODEL_INITIALIZATION'
    MODEL_FIT = 'MODEL_FIT'
    FORECAST_GENERATION = 'FORECAST_GENERATION'
    METRIC_EVALUATION = 'METRIC_EVALUATION'
    FORECAST_BOOTSTRAP = 'FORECAST_BOOTSTRAP'
    ECONOMIC_EVALUATION = 'ECONOMIC_EVALUATION'
    ECONOMIC_BOOTSTRAP = 'ECONOMIC_BOOTSTRAP'
    ARTIFACT_VALIDATION = 'ARTIFACT_VALIDATION'


class FailureExitStatus(str, Enum):
    """Closed exit statuses valid for a failed development run."""

    NEEDS_DATA_REVISION = 'NEEDS_DATA_REVISION'
    NEEDS_MODEL_REVISION = 'NEEDS_MODEL_REVISION'
    NEEDS_PROTOCOL_REVISION = 'NEEDS_PROTOCOL_REVISION'
    TECHNICAL_FAILURE = 'TECHNICAL_FAILURE'


class DevelopmentExitStatus(str, Enum):
    """All closed terminal development exit statuses."""

    NEEDS_DATA_REVISION = 'NEEDS_DATA_REVISION'
    NEEDS_MODEL_REVISION = 'NEEDS_MODEL_REVISION'
    NEEDS_PROTOCOL_REVISION = 'NEEDS_PROTOCOL_REVISION'
    TECHNICAL_FAILURE = 'TECHNICAL_FAILURE'
    FORECAST_GATE_FAILED = 'FORECAST_GATE_FAILED'
    ECONOMIC_GATE_FAILED = 'ECONOMIC_GATE_FAILED'
    ECONOMIC_SIGNAL_FOUND = 'ECONOMIC_SIGNAL_FOUND'


class ModelID(str, Enum):
    """Closed model identifiers emitted by M3 artifacts."""

    B0 = 'B0'
    M3 = 'M3'


class CandidateID(str, Enum):
    """Closed candidate identifiers emitted by M3 artifacts."""

    B0_UNIFORM = 'B0_UNIFORM'
    M3_W030 = 'M3_W030'
    M3_W060 = 'M3_W060'
    M3_W120 = 'M3_W120'
    M3_W240 = 'M3_W240'
    M3_W365 = 'M3_W365'


class CandidateStatus(str, Enum):
    """Closed qualification statuses for DEV candidates under P3 protocol."""

    COMPLETE_VALID = 'COMPLETE_VALID'
    DISQUALIFIED_MODEL_INITIALIZATION = 'DISQUALIFIED_MODEL_INITIALIZATION'
    DISQUALIFIED_MODEL_FIT = 'DISQUALIFIED_MODEL_FIT'
    DISQUALIFIED_FORECAST_CONTRACT = 'DISQUALIFIED_FORECAST_CONTRACT'


class DevelopmentStage(str, Enum):
    """Closed chronological stages used in development artifacts."""

    DEV = 'DEV'
    VAL = 'VAL'
    STABILITY = 'STABILITY'


class ArtifactStatus(str, Enum):
    """Closed lifecycle and evaluation statuses used by M3 artifacts."""

    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    EVALUATED = 'EVALUATED'
    NOT_EVALUATED_FORECAST_GATE_FAILED = 'NOT_EVALUATED_FORECAST_GATE_FAILED'


@dataclass(frozen=True)
class ProtocolFailure:
    """Immutable representation of the generic failed-run envelope."""

    stage: FailureStage
    error_type: str
    exit_status: FailureExitStatus

    def __post_init__(self) -> None:
        try:
            stage = FailureStage(self.stage)
        except (TypeError, ValueError) as error:
            raise ValueError(f'Unknown M3 failure stage: {self.stage!r}') from error
        try:
            exit_status = FailureExitStatus(self.exit_status)
        except (TypeError, ValueError) as error:
            raise ValueError(f'Invalid failure exit status: {self.exit_status!r}') from error
        if not isinstance(self.error_type, str) or not self.error_type.strip():
            raise ValueError('ProtocolFailure.error_type must be a non-empty string')

        object.__setattr__(self, 'stage', stage)
        object.__setattr__(self, 'exit_status', exit_status)
