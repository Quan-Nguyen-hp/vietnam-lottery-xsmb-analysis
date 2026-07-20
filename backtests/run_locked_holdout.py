"""Chạy exact-mode duy nhất trên holdout đã khóa trong evaluation_policy.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from src.data.loader import DataLoader
from backtests.xpis_backtest import run_exact_production_backtest


POLICY_PATH = root_dir / "predictions" / "evaluation_policy.json"


def main() -> int:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    holdout = policy["locked_holdout"]
    start_date = pd.Timestamp(holdout["start_date"])
    minimum_days = int(holdout["minimum_days"])
    top_k = int(policy["decision_policy"]["top_k"])

    loader = DataLoader().load()
    available = loader.df[loader.df["date"] >= start_date]
    if len(available) < minimum_days:
        print(
            f"HOLDOUT CHƯA SẴN SÀNG: cần {minimum_days} ngày từ {start_date.date()}, "
            f"hiện có {len(available)} ngày. Giữ chế độ {policy['mode']}."
        )
        # Đây là trạng thái chờ dữ liệu hợp lệ, không phải lỗi pipeline/CI.
        return 0

    test_dates = available.iloc[:minimum_days]["date"]
    expected_start = pd.Timestamp(test_dates.iloc[0]).date()
    expected_end = pd.Timestamp(test_dates.iloc[-1]).date()
    dataset_tail = loader.df.iloc[-minimum_days:]["date"]
    if pd.Timestamp(dataset_tail.iloc[0]).date() != expected_start or pd.Timestamp(dataset_tail.iloc[-1]).date() != expected_end:
        raise RuntimeError("Holdout phải là cửa sổ dữ liệu mới nhất; không chạy trên cửa sổ đã bị chèn dữ liệu sau đó.")

    report_name = f"locked_holdout_{expected_start}_{expected_end}.md"
    results = run_exact_production_backtest(
        n_test_days=minimum_days,
        top_k=top_k,
        report_name=report_name,
    )
    print(f"Đã chạy locked holdout: {len(results)} ngày -> backtests/results/{report_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
