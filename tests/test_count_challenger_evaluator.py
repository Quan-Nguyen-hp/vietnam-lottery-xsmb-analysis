from backtests.evaluate_count_challenger import evaluate


def _entry(date: str, picks: list[int], actual: list[int]) -> dict:
    return {
        "pipeline_metadata": {"date": date},
        "count_challenger": {"picks": [{"number": number} for number in picks]},
        "actual_results": actual,
    }


def test_evaluator_stays_pending_before_minimum_days_and_counts_multiple_hits():
    entries = [
        _entry("2026-07-20", [7, 12], [7, 7, 3]),
        _entry("2026-07-21", [], [12]),
    ]
    summary = evaluate(entries, "2026-07-20", minimum_days=180)
    assert summary["status"] == "PENDING"
    assert summary["n_days"] == 2
    assert summary["total_bets"] == 2
    assert summary["total_hits"] == 2


def test_evaluator_excludes_pre_start_and_incomplete_entries():
    entries = [
        _entry("2026-07-19", [7], [7]),
        {"pipeline_metadata": {"date": "2026-07-20"}, "count_challenger": {"picks": []}},
    ]
    summary = evaluate(entries, "2026-07-20", minimum_days=180)
    assert summary["n_days"] == 0
    assert summary["total_bets"] == 0
