from dataclasses import FrozenInstanceError, fields
from enum import Enum
import subprocess
import sys

import pytest

from src.m3_digit_factor.contracts import (
    ArtifactStatus,
    CandidateID,
    DevelopmentExitStatus,
    DevelopmentStage,
    FailureExitStatus,
    FailureStage,
    ModelID,
    ProtocolFailure,
)


def test_package_contracts_import_cleanly():
    assert FailureStage.DATA_VALIDATION.value == 'DATA_VALIDATION'


def test_closed_contract_values_match_the_frozen_protocol():
    assert {stage.value for stage in FailureStage} == {
        'DATA_VALIDATION',
        'MODEL_INITIALIZATION',
        'MODEL_FIT',
        'FORECAST_GENERATION',
        'METRIC_EVALUATION',
        'FORECAST_BOOTSTRAP',
        'ECONOMIC_EVALUATION',
        'ECONOMIC_BOOTSTRAP',
        'ARTIFACT_VALIDATION',
    }
    assert {status.value for status in DevelopmentExitStatus} == {
        'NEEDS_DATA_REVISION',
        'NEEDS_MODEL_REVISION',
        'NEEDS_PROTOCOL_REVISION',
        'TECHNICAL_FAILURE',
        'FORECAST_GATE_FAILED',
        'ECONOMIC_GATE_FAILED',
        'ECONOMIC_SIGNAL_FOUND',
    }
    assert {status.value for status in FailureExitStatus} == {
        'NEEDS_DATA_REVISION',
        'NEEDS_MODEL_REVISION',
        'NEEDS_PROTOCOL_REVISION',
        'TECHNICAL_FAILURE',
    }
    assert {candidate.value for candidate in CandidateID} == {
        'B0_UNIFORM',
        'M3_W030',
        'M3_W060',
        'M3_W120',
        'M3_W240',
        'M3_W365',
    }
    assert {model.value for model in ModelID} == {'B0', 'M3'}
    assert {stage.value for stage in DevelopmentStage} == {'DEV', 'VAL', 'STABILITY'}
    assert {status.value for status in ArtifactStatus} == {
        'COMPLETED',
        'FAILED',
        'EVALUATED',
        'NOT_EVALUATED_FORECAST_GATE_FAILED',
    }


@pytest.mark.parametrize('enum_type', [
    FailureStage,
    FailureExitStatus,
    DevelopmentExitStatus,
    CandidateID,
    ModelID,
    DevelopmentStage,
    ArtifactStatus,
])
def test_unknown_closed_values_are_rejected(enum_type: type[Enum]):
    with pytest.raises(ValueError):
        enum_type('UNAPPROVED_VALUE')


def test_protocol_failure_is_immutable_and_accepts_a_valid_failure_envelope():
    failure = ProtocolFailure(
        stage=FailureStage.MODEL_FIT,
        error_type='ConstraintViolation',
        exit_status=FailureExitStatus.NEEDS_MODEL_REVISION,
    )

    assert failure.stage is FailureStage.MODEL_FIT
    assert failure.exit_status is FailureExitStatus.NEEDS_MODEL_REVISION
    with pytest.raises(FrozenInstanceError):
        failure.error_type = 'OtherError'  # type: ignore[misc]


@pytest.mark.parametrize('exit_status', [
    DevelopmentExitStatus.FORECAST_GATE_FAILED,
    DevelopmentExitStatus.ECONOMIC_GATE_FAILED,
    DevelopmentExitStatus.ECONOMIC_SIGNAL_FOUND,
])
def test_protocol_failure_rejects_successful_or_gate_exit_statuses(exit_status: DevelopmentExitStatus):
    with pytest.raises(ValueError):
        ProtocolFailure(
            stage=FailureStage.DATA_VALIDATION,
            error_type='InsufficientDevelopmentSample',
            exit_status=exit_status,
        )


def test_protocol_failure_rejects_unknown_stage_and_blank_error_type():
    with pytest.raises(ValueError):
        ProtocolFailure(
            stage='UNKNOWN_STAGE',  # type: ignore[arg-type]
            error_type='Problem',
            exit_status=FailureExitStatus.TECHNICAL_FAILURE,
        )
    with pytest.raises(ValueError):
        ProtocolFailure(
            stage=FailureStage.DATA_VALIDATION,
            error_type=' ',
            exit_status=FailureExitStatus.NEEDS_DATA_REVISION,
        )
    with pytest.raises(ValueError):
        ProtocolFailure(
            stage=FailureStage.DATA_VALIDATION,
            error_type='Problem',
            exit_status='UNKNOWN_FAILURE_EXIT',  # type: ignore[arg-type]
        )


def test_contracts_expose_no_mutable_scientific_configuration_surface():
    assert [field.name for field in fields(ProtocolFailure)] == ['stage', 'error_type', 'exit_status']
    assert not hasattr(ProtocolFailure, 'configuration')
    assert not hasattr(ProtocolFailure, 'override_protocol')


def test_package_does_not_import_legacy_scientific_packages():
    result = subprocess.run(
        [sys.executable, '-c', 'import src.m3_digit_factor; import sys; print(sorted(sys.modules))'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'src.count_v2' not in result.stdout
