"""Prospective evaluator cho EWMA count challenger ở shadow paper-trade."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
LOG_PATH = ROOT_DIR / "predictions" / "prediction_log.json"
POLICY_PATH = ROOT_DIR / "predictions" / "evaluation_policy.json"
DEFAULT_REPORT = ROOT_DIR / "backtests" / "results" / "count_challenger_prospective.md"

COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0
BOOTSTRAP_RESAMPLES = 20_000
SEED = 42


def extract_records(entries: list[dict], start_date: str) -> list[dict]:
    records = []
    for entry in entries:
        date_str = entry.get("pipeline_metadata", {}).get("date", entry.get("date", ""))
        challenger = entry.get("count_challenger")
        actual = entry.get("actual_results")
        if not date_str or date_str < start_date or not challenger or actual is None:
            continue
        actual_counts: dict[int, int] = {}
        for number in actual:
            actual_counts[int(number)] = actual_counts.get(int(number), 0) + 1
        picks = [int(pick["number"]) for pick in challenger.get("picks", [])]
        hits = sum(actual_counts.get(number, 0) for number in picks)
        records.append({
            "date": date_str,
            "n_bets": len(picks),
            "n_hits": hits,
            "daily_cost": len(picks) * COST_PER_BET,
            "daily_pnl": hits * PAYOUT_PER_HIT - len(picks) * COST_PER_BET,
        })
    return sorted(records, key=lambda row: row["date"])


def bootstrap_roi(records: list[dict]) -> tuple[float, float, float, float]:
    if not records:
        return 0.0, 0.0, 0.0, 0.0
    cost = np.asarray([row["daily_cost"] for row in records], dtype=float)
    pnl = np.asarray([row["daily_pnl"] for row in records], dtype=float)
    if cost.sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0
    observed = float(pnl.sum() / cost.sum())
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(records), size=(BOOTSTRAP_RESAMPLES, len(records)))
    sampled_cost = cost[indices].sum(axis=1)
    sampled_roi = np.divide(
        pnl[indices].sum(axis=1),
        sampled_cost,
        out=np.full(BOOTSTRAP_RESAMPLES, np.nan),
        where=sampled_cost > 0,
    )
    sampled_roi = sampled_roi[np.isfinite(sampled_roi)]
    lower, upper = np.quantile(sampled_roi, [0.025, 0.975])
    return observed, float(lower), float(upper), float(np.mean(sampled_roi > 0.0))


def evaluate(entries: list[dict], start_date: str, minimum_days: int) -> dict:
    records = extract_records(entries, start_date)
    roi, lower, upper, probability_positive = bootstrap_roi(records)
    n_days = len(records)
    total_bets = sum(row["n_bets"] for row in records)
    total_hits = sum(row["n_hits"] for row in records)
    if n_days < minimum_days:
        status = "PENDING"
        reason = f"Chỉ có {n_days}/{minimum_days} ngày prospective."
    elif total_bets == 0:
        status = "FAIL"
        reason = "Không có exposure để đánh giá."
    elif lower > 0.0:
        status = "PASS"
        reason = "Cận dưới bootstrap ROI 95% > 0."
    else:
        status = "FAIL"
        reason = "Cận dưới bootstrap ROI 95% không dương."
    return {
        "status": status,
        "reason": reason,
        "start_date": start_date,
        "minimum_days": minimum_days,
        "n_days": n_days,
        "total_bets": total_bets,
        "total_hits": total_hits,
        "roi": roi,
        "roi_lower_95": lower,
        "roi_upper_95": upper,
        "probability_roi_positive": probability_positive,
        "first_date": records[0]["date"] if records else None,
        "last_date": records[-1]["date"] if records else None,
    }


def write_report(summary: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join([
            "# Prospective evaluation — Count challenger",
            "",
            f"- **Status**: {summary['status']}",
            f"- **Lý do**: {summary['reason']}",
            f"- **Cửa sổ khóa**: từ {summary['start_date']}, tối thiểu {summary['minimum_days']} ngày",
            f"- **Dữ liệu hiện có**: {summary['first_date'] or '—'} đến {summary['last_date'] or '—'}",
            "",
            "| Metric | Giá trị |",
            "|---|---:|",
            f"| Ngày prospective | {summary['n_days']} |",
            f"| Bets | {summary['total_bets']} |",
            f"| Hits | {summary['total_hits']} |",
            f"| ROI | {summary['roi']:+.2%} |",
            f"| Bootstrap CI95 | [{summary['roi_lower_95']:+.2%}, {summary['roi_upper_95']:+.2%}] |",
            f"| P(ROI>0) | {summary['probability_roi_positive']:.1%} |",
            "",
            "> Challenger chỉ chạy shadow paper-trade và không tự động thay đổi production policy.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá prospective count challenger")
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    entries = json.loads(args.log.read_text(encoding="utf-8")) if args.log.exists() else []
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    challenger_policy = policy.get("count_model_challenger", {})
    start_date = challenger_policy.get("prospective_start_date", policy["locked_holdout"]["start_date"])
    minimum_days = int(policy["locked_holdout"]["minimum_days"])
    summary = evaluate(entries, start_date, minimum_days)
    write_report(summary, args.report)
    print(
        f"Count challenger: {summary['status']} | "
        f"days={summary['n_days']}/{minimum_days} | bets={summary['total_bets']} | "
        f"ROI={summary['roi']:+.2%}"
    )


if __name__ == "__main__":
    main()
