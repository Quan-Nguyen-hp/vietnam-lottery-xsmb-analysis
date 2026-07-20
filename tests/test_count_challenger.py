import numpy as np

from src.probability.count_expectation import EWMAHitCountChallenger
from daily_update import evaluate_count_challenger


def test_count_challenger_uses_count_target_and_returns_auditable_arrays():
    counts = np.zeros((540, 100), dtype=np.int8)
    counts[:, 7] = 1
    challenger = EWMAHitCountChallenger(top_k=4)
    forecast = challenger.predict_from_counts(counts)
    assert forecast.expected_counts.shape == (100,)
    assert forecast.lower_bounds.shape == (100,)
    assert forecast.picks == [7]
    assert forecast.lower_bounds[7] > challenger.break_even_expected_count


def test_recent_observations_have_more_weight_than_old_observations():
    old_hits = np.zeros((540, 100), dtype=np.int8)
    old_hits[:30, 12] = 1
    recent_hits = np.zeros((540, 100), dtype=np.int8)
    recent_hits[-30:, 12] = 1
    challenger = EWMAHitCountChallenger()
    assert (
        challenger.predict_from_counts(recent_hits).expected_counts[12]
        > challenger.predict_from_counts(old_hits).expected_counts[12]
    )


def test_shadow_evaluation_counts_multiple_hits_and_remains_paper_trade():
    entry = {
        "count_challenger": {
            "picks": [
                {"number": 7, "decision": "PAPER_TRADE"},
                {"number": 12, "decision": "PAPER_TRADE"},
            ]
        }
    }
    evaluation = evaluate_count_challenger(entry, {7: 2, 12: 0})
    assert evaluation == {
        "actual_hits": 2,
        "cost_k": 54.0,
        "revenue_k": 198.0,
        "pnl_k": 144.0,
        "paper_trade": True,
    }
    assert entry["count_challenger"]["picks"][0]["actual_hits"] == 2
