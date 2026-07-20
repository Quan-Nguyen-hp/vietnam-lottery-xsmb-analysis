import numpy as np

from src.meta.component_calibration import ComponentCalibrationManager


def test_component_calibration_fits_each_model_independently():
    rng = np.random.default_rng(42)
    labels_fit = rng.binomial(1, 0.25, size=(45, 100))
    labels_selection = rng.binomial(1, 0.25, size=(45, 100))
    raw_fit = {
        "overconfident": np.clip(0.05 + 0.9 * labels_fit + rng.normal(0, 0.1, labels_fit.shape), 0, 1),
        "flat": np.full((45, 100), 0.5),
    }
    raw_selection = {
        "overconfident": np.clip(
            0.05 + 0.9 * labels_selection + rng.normal(0, 0.1, labels_selection.shape), 0, 1
        ),
        "flat": np.full((45, 100), 0.5),
    }
    manager = ComponentCalibrationManager().fit(
        raw_fit,
        labels_fit,
        raw_selection,
        labels_selection,
    )
    calibrated = manager.calibrate_dict(raw_selection)
    assert set(calibrated) == {"overconfident", "flat"}
    assert set(manager.methods) == {"overconfident", "flat"}
    assert all(values.shape == (45, 100) for values in calibrated.values())
