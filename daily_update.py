"""
daily_update.py — Cập nhật kết quả thực tế và đồng bộ dữ liệu XPIS v1.2 APPROVED.
Hỗ trợ tương thích ngược với log cũ v1.0 và cấu trúc v1.2 mới.

Cách dùng:
  python daily_update.py                         # Tự fetch kết quả hôm nay
  python daily_update.py --date 2026-07-13       # Cập nhật ngày cụ thể
"""
import sys
import json
import argparse
import subprocess
import shutil
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir / "src"))

import numpy as np
import pandas as pd

DATA_CSV = root_dir / "data" / "xsmb-2-digits.csv"
PRED_LOG = root_dir / "predictions" / "prediction_log.json"
RULES_FILE = root_dir / "predictions" / "matrix_rules.json"

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0
OPTIMIZE_EVERY = 7


def load_log() -> list:
    if PRED_LOG.exists():
        with open(PRED_LOG, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def save_log(log: list) -> None:
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def evaluate_count_challenger(entry: dict, actual_counts: dict[int, int]) -> dict | None:
    """Chấm shadow challenger theo số nháy; không thay đổi kết quả production."""
    challenger = entry.get("count_challenger")
    if not challenger:
        return None
    challenger_numbers = [pick["number"] for pick in challenger.get("picks", [])]
    challenger_hits = sum(actual_counts.get(number, 0) for number in challenger_numbers)
    challenger_cost = len(challenger_numbers) * COST_PER_BET
    challenger_revenue = challenger_hits * PAYOUT_PER_HIT
    evaluation = {
        "actual_hits": challenger_hits,
        "cost_k": challenger_cost,
        "revenue_k": challenger_revenue,
        "pnl_k": challenger_revenue - challenger_cost,
        "paper_trade": True,
    }
    challenger["evaluation"] = evaluation
    for pick in challenger.get("picks", []):
        pick["actual_hits"] = actual_counts.get(pick["number"], 0)
    return evaluation


def fetch_results_for_date(target_date: date) -> list[int] | None:
    """Fetch kết quả 2 chữ số cuối từ web cho target_date."""
    try:
        from lottery import Lottery
        print(f"🌐 Đang fetch kết quả ngày {target_date} từ web…")
        lottery = Lottery()
        lottery.load()
        lottery.fetch(target_date)
        lottery.generate_dataframes()
        lottery.dump()

        # Đọc lại CSV vừa cập nhật
        df = pd.read_csv(DATA_CSV)
        df["date"] = pd.to_datetime(df["date"])
        row = df[df["date"].dt.date == target_date]
        if row.empty:
            return None
        prize_cols = [c for c in df.columns if c != "date"]
        vals = row.iloc[0][prize_cols].values.astype(int).tolist()
        two_digits = [v % 100 for v in vals]
        return two_digits
    except Exception as e:
        print(f"⚠️  Không thể fetch: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Cập nhật kết quả XSMB XPIS v1.2 APPROVED")
    parser.add_argument("--date", type=str, default=None, help="Ngày cần cập nhật (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        now = datetime.now(TZ)
        target_date = now.date()
        if now.time() < dtime(18, 35):
            target_date -= timedelta(days=1)

    print(f"\n🔄 Cập nhật kết quả ngày: {target_date}")

    log = load_log()
    entry_idx = None
    for i, e in enumerate(log):
        # Đọc ngày tương thích v1.0 và v1.2
        d_val = e["pipeline_metadata"]["date"] if "pipeline_metadata" in e else e.get("date")
        if d_val == str(target_date):
            entry_idx = i
            break

    if entry_idx is None:
        print(f"⚠️  Không tìm thấy dự đoán cho ngày {target_date} trong log.")
        return

    entry = log[entry_idx]
    if entry.get("actual_results") is not None:
        print(f"✅ Ngày {target_date} đã có kết quả: {entry['actual_results']}")
    else:
        # Fetch kết quả
        actual = fetch_results_for_date(target_date)
        if actual is None:
            print(f"❌ Không thể lấy kết quả ngày {target_date}.")
            return

        # Đọc danh mục cược tương thích v1.0 và v1.2
        is_v12 = "pipeline_metadata" in entry
        if is_v12:
            picks = [b["number"] for b in entry.get("bets", [])]
            top_k = len(entry.get("bets", []))
        else:
            picks = entry.get("ensemble_picks", [])
            top_k = entry.get("top_k", 10)

        # Tính hits và PnL
        actual_counts = {}
        for v in actual:
            actual_counts[v] = actual_counts.get(v, 0) + 1
            
        ensemble_hits = sum(actual_counts.get(n, 0) for n in picks)
        revenue = ensemble_hits * PAYOUT_PER_HIT
        cost = top_k * COST_PER_BET
        pnl = revenue - cost

        # Cập nhật kết quả vào bản ghi log
        entry["actual_results"] = actual
        entry["ensemble_hits"] = ensemble_hits
        entry["revenue_k"] = revenue
        entry["pnl_k"] = pnl

        # Chấm shadow count challenger độc lập; đây là PnL giả lập, không phải tiền thật.
        evaluate_count_challenger(entry, actual_counts)
        
        # Nếu là v1.2, ghi nhận kết quả và PnL trực tiếp vào danh sách bets để Dashboard hiển thị
        if is_v12:
            for b in entry.get("bets", []):
                b_num = b["number"]
                b["actual_hits"] = actual_counts.get(b_num, 0)
                b["pnl"] = b["actual_hits"] * PAYOUT_PER_HIT - (COST_PER_BET if b["decision"] == "BET" else 0)

        log[entry_idx] = entry
        save_log(log)

        result_icon = "🎉" if ensemble_hits > 0 else "😔"
        print(f"\n{result_icon} KẾT QUẢ NGÀY {target_date}:")
        print(f"  Đã chọn  : {', '.join(f'{n:02d}' for n in picks) if picks else 'SKIP (Không cược)'}")
        print(f"  Kết quả  : {', '.join(f'{n:02d}' for n in sorted(set(actual)))}")
        print(f"  Trúng    : {ensemble_hits} nháy")
        print(f"  Chi phí  : {cost:,}k đ")
        print(f"  Thu về   : {revenue:,}k đ")
        print(f"  Lãi/Lỗ  : {pnl:+,}k đ")
        challenger = entry.get("count_challenger")
        if challenger:
            shadow_numbers = [pick["number"] for pick in challenger.get("picks", [])]
            shadow_eval = challenger.get("evaluation", {})
            print(
                "  Shadow count challenger: "
                f"{', '.join(f'{number:02d}' for number in shadow_numbers) if shadow_numbers else 'SKIP'} | "
                f"hits={shadow_eval.get('actual_hits', 0)} | "
                f"PnL giả lập={shadow_eval.get('pnl_k', 0):+,}k đ"
            )

    # Thống kê tổng hợp trượt
    completed_entries = []
    for e in log:
        if e.get("actual_results") is not None:
            completed_entries.append(e)
            
    if completed_entries:
        total_cost = sum(e.get("cost_k", 0) if "cost_k" in e else (len(e.get("bets", [])) if "bets" in e else e.get("top_k", 10)) * COST_PER_BET for e in completed_entries)
        total_rev = sum(e.get("revenue_k", 0) for e in completed_entries)
        total_pnl = total_rev - total_cost
        roi = total_pnl / total_cost * 100 if total_cost else 0
        win_days = sum(1 for e in completed_entries if (e.get("ensemble_hits") or 0) > 0)
        
        print(f"\n📊 Thống kê tổng hợp ({len(completed_entries)} ngày):")
        print(f"  Win rate : {win_days}/{len(completed_entries)} ngày ({win_days/len(completed_entries)*100:.0f}%)")
        print(f"  Tổng PnL : {total_pnl:+,}k đ")
        print(f"  ROI      : {roi:+.2f}%")


if __name__ == "__main__":
    main()
