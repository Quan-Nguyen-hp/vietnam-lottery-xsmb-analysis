import numpy as np
import pandas as pd

from src.probability.count_poisson import CountEWMAPoissonPredictor


def test_count_ewma_poisson_returns_binary_any_hit_probability():
    rows = []
    for day in range(100):
        row = {"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day)}
        for prize in range(27):
            row[f"prize_{prize}"] = 7 if prize == 0 else (prize + day) % 100
        rows.append(row)
    history = pd.DataFrame(rows)
    model = CountEWMAPoissonPredictor()
    probabilities = model.predict_proba(pd.DataFrame({"number": np.arange(100)}), history)
    assert probabilities.shape == (100,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[7] > probabilities[99]
