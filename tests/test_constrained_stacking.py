import numpy as np

from src.meta.constrained_stacking import ConstrainedStacking


def test_weights_are_simplex_and_fuse_shape_is_preserved():
    rng = np.random.default_rng(42)
    labels = rng.binomial(1, 0.25, size=(45, 100))
    predictions = {
        "good": np.clip(labels * 0.4 + 0.1 + rng.normal(0, 0.02, labels.shape), 0.001, 0.999),
        "flat": np.full((45, 100), 0.25),
        "bad": np.full((45, 100), 0.7),
    }
    stack = ConstrainedStacking(regularization=0.10).fit(predictions, labels)
    assert np.isclose(sum(stack.weights.values()), 1.0)
    assert all(0.0 <= weight <= 1.0 for weight in stack.weights.values())
    assert stack.fuse(predictions).shape == labels.shape
