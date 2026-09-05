"""XPIS v3 M3 digit-factor protocol contracts.

This package deliberately contains only the closed protocol vocabulary needed
by later implementation slices.
"""

from .contracts import (
    ArtifactStatus,
    CandidateID,
    DevelopmentExitStatus,
    DevelopmentStage,
    FailureExitStatus,
    FailureStage,
    ModelID,
    ProtocolFailure,
)
from .runner import (
    DevelopmentRunResult,
    HistoricalExecutionNotAuthorizedError,
    run_development,
)

__all__ = [
    'ArtifactStatus',
    'CandidateID',
    'DevelopmentExitStatus',
    'DevelopmentStage',
    'DevelopmentRunResult',
    'FailureExitStatus',
    'FailureStage',
    'HistoricalExecutionNotAuthorizedError',
    'ModelID',
    'ProtocolFailure',
    'run_development',
]

