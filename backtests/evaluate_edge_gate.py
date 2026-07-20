"""
backtests/evaluate_edge_gate.py — Đánh giá và cập nhật Edge Gate state.

Chạy định kỳ (sau khi đủ dữ liệu holdout hoặc backtest mới) để:
  1. Tính bootstrap ROI CI trên kết quả backtest gần nhất
  2. Cập nhật Edge Gate state trong evaluation_policy.json
  3. In báo cáo rõ ràng

Chạy:
  python backtests/evaluate_edge_gate.py
  python backtests/evaluate_edge_gate.py --from-log     # Đọc từ prediction_log.json (production tracking)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from src.decision.edge_gate import EdgeGate

COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0
BOOTSTRAP_RESAMPLES = 20_000


def _bootstrap_roi_interval(daily_pnl: np.ndarray, daily_cost: np.ndarray) -> tuple[float, float, float]:
    if len(daily_pnl) == 0 or daily_cost.sum() <= 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(daily_pnl), size=(BOOTSTRAP_RESAMPLES, len(daily_pnl)))
    sampled_cost = daily_cost[indices].sum(axis=1)
    sampled_roi = np.divide(
        daily_pnl[indices].sum(axis=1),
        sampled_cost,
        out=np.zeros(BOOTSTRAP_RESAMPLES),
        where=sampled_cost > 0,
    )
    lower, upper = np.quantile(sampled_roi, [0.025, 0.975])
    prob_positive = float(np.mean(sampled_roi > 0.0))
    return float(lower), float(upper), prob_positive


def evaluate_from_backtest_report(report_path: Path) -> None:
    """Đọc kết quả từ báo cáo backtest Markdown đã có và cập nhật Edge Gate."""
    gate = EdgeGate()

    # Đọc báo cáo — extract ROI và CI từ file
    content = report_path.read_text(encoding="utf-8")

    # Parse báo cáo ablation mới
    roi_lower, roi_upper = None, None
    total_bets, total_hits, n_days = 0, 0, 0

    for line in content.splitlines():
        line = line.strip()
        if "ensemble_11" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            # Tìm ROI và CI
            for p in parts:
                if "%" in p and "+" in p or "-" in p:
                    try:
                        # Parse ROI
                        if "ROI" not in p and ":" not in p and not p.startswith("CI"):
                            p_clean = p.replace("%", "").strip()
                            roi_val = float(p_clean)
                    except ValueError:
                        pass

    print("⚠️ Không thể parse tự động từ Markdown. Dùng --from-log hoặc manual update.")


def evaluate_from_prediction_log() -> None:
    """Đọc prediction_log.json (có actual_results) và tính Edge Gate."""
    log_path = root_dir / "predictions" / "prediction_log.json"
    if not log_path.exists():
        print("❌ Không tìm thấy prediction_log.json")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    # Lọc các entry đã có kết quả
    completed = [e for e in log if e.get("actual_results") is not None]

    if len(completed) < 30:
        print(f"⚠️ Chỉ có {len(completed)} ngày có kết quả (cần ≥ 30). Gate giữ nguyên.")
        return

    daily_pnl = []
    daily_cost = []
    total_bets = 0
    total_hits = 0

    for e in completed:
        # Số bets: đọc từ decision_summary hoặc len(bets)
        is_v12 = "pipeline_metadata" in e
        if is_v12:
            bets = e.get("bets", [])
            n_bets = len(bets)
            # Lọc chỉ BET (không PAPER_TRADE)
            real_bets = [b for b in bets if b.get("decision") == "BET" or b.get("original_decision") == "BET"]
            n_bets = len(real_bets)
        else:
            n_bets = e.get("top_k", 0)

        hits = e.get("ensemble_hits", 0)
        pnl = e.get("pnl_k", hits * PAYOUT_PER_HIT - n_bets * COST_PER_BET)

        daily_pnl.append(float(pnl))
        daily_cost.append(float(n_bets * COST_PER_BET))
        total_bets += n_bets
        total_hits += hits

    daily_pnl = np.array(daily_pnl)
    daily_cost = np.array(daily_cost)

    total_cost = daily_cost.sum()
    total_pnl = daily_pnl.sum()
    roi = (total_pnl / total_cost) if total_cost > 0 else 0.0

    roi_lower, roi_upper, prob_positive = _bootstrap_roi_interval(daily_pnl, daily_cost)

    first_date = completed[0].get("pipeline_metadata", {}).get("date", completed[0].get("date", "?"))
    last_date = completed[-1].get("pipeline_metadata", {}).get("date", completed[-1].get("date", "?"))

    print("=" * 55)
    print("  📊 EDGE GATE EVALUATION")
    print("=" * 55)
    print(f"  Kỳ đánh giá: {first_date} → {last_date} ({len(completed)} ngày)")
    print(f"  Tổng cược  : {total_bets} số")
    print(f"  Tổng nháy  : {total_hits}")
    print(f"  Tổng chi phí: {total_cost:,.0f}k đ")
    print(f"  Tổng PnL   : {total_pnl:,.0f}k đ")
    print(f"  ROI        : {roi:+.2%}")
    print(f"  Bootstrap 95% CI: [{roi_lower:+.2%}, {roi_upper:+.2%}]")
    print(f"  P(ROI>0)   : {prob_positive:.1%}")
    print()

    # Cập nhật Edge Gate
    gate = EdgeGate()
    eval_period = f"{first_date} to {last_date}"
    result = gate.update(
        roi_lower_95=roi_lower,
        evaluation_period=eval_period,
        n_days=len(completed),
        total_bets=total_bets,
        total_hits=total_hits,
        roi=roi,
    )

    if result["pass"]:
        print("  ✅ EDGE GATE: PASS — Chuyển sang LIVE mode")
        print("     Hệ thống đã chứng minh lợi thế thống kê.")
    else:
        print(f"  ⛔ EDGE GATE: {result['status']} — Giữ PAPER_TRADE")
        print(f"     Lý do: cận dưới bootstrap 95% = {roi_lower:+.2%} ≤ 0")
        print("     Cần CI hoàn toàn dương để chuyển sang LIVE.")

    print("=" * 55)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá và cập nhật Edge Gate")
    parser.add_argument("--from-log", action="store_true", help="Đọc từ prediction_log.json")
    parser.add_argument("--report", type=str, default=None, help="Đọc từ báo cáo Markdown")
    args = parser.parse_args()

    if args.from_log:
        evaluate_from_prediction_log()
    elif args.report:
        evaluate_from_backtest_report(Path(args.report))
    else:
        # Mặc định: thử đọc từ prediction_log.json
        log_path = root_dir / "predictions" / "prediction_log.json"
        if log_path.exists():
            evaluate_from_prediction_log()
        else:
            print("❌ Không tìm thấy prediction_log.json. Dùng --from-log hoặc --report.")
