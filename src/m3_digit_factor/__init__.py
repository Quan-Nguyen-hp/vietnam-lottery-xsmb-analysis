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

__all__ = [
    'ArtifactStatus',
    'CandidateID',
    'DevelopmentExitStatus',
    'DevelopmentStage',
    'FailureExitStatus',
    'FailureStage',
    'ModelID',
    'ProtocolFailure',
]
