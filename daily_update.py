"""
daily_update.py — Cập nhật kết quả thực tế + tối ưu bộ lọc ma trận quyết định (mỗi 7 ngày)

Cách dùng:
  python daily_update.py                         # tự fetch kết quả hôm nay
  python daily_update.py --date 2026-07-13       # cập nhật ngày cụ thể
  python daily_update.py --force-reweight        # bắt buộc tối ưu hóa lại bộ lọc ngay
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
sys.path.append(str(root_dir / "src"))

import numpy as np
import pandas as pd

# ── Đường dẫn ────────────────────────────────────────────────────────────────
DATA_CSV   = root_dir / "data" / "xsmb-2-digits.csv"
PRED_LOG   = root_dir / "predictions" / "prediction_log.json"
RULES_FILE = root_dir / "predictions" / "matrix_rules.json"

TZ             = ZoneInfo("Asia/Ho_Chi_Minh")
COST_PER_NUM   = 27
PAYOUT_PER_HIT = 99
OPTIMIZE_EVERY = 7       # số ngày giữa các lần chạy tối ưu hóa

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_log() -> list:
    if PRED_LOG.exists():
        with open(PRED_LOG, encoding="utf-8") as f:
            return json.load(f)
    return []

def save_log(log: list) -> None:
    with open(PRED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def load_rules_data() -> dict:
    if RULES_FILE.exists():
        with open(RULES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"updated_at": None, "rules": {}}

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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cập nhật kết quả XSMB và tối ưu bộ lọc")
    parser.add_argument("--date",          type=str, default=None, help="Ngày cần cập nhật (YYYY-MM-DD)")
    parser.add_argument("--force-reweight", action="store_true",   help="Bắt buộc chạy tối ưu lại bộ lọc ngay")
    args = parser.parse_args()

    # Xác định ngày cần cập nhật
    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        now = datetime.now(TZ)
        target_date = now.date()
        # Nếu chưa đến 18:35 thì cập nhật hôm qua
        if now.time() < dtime(18, 35):
            target_date -= timedelta(days=1)

    print(f"\n🔄 Cập nhật kết quả ngày: {target_date}")

    # Load log
    log = load_log()
    entry_idx = None
    for i, e in enumerate(log):
        if e["date"] == str(target_date):
            entry_idx = i
            break

    if entry_idx is None:
        print(f"⚠️  Không tìm thấy dự đoán cho ngày {target_date} trong log.")
        print("   Hãy chạy daily_predict.py trước.")
        return

    entry = log[entry_idx]
    if entry.get("actual_results") is not None:
        print(f"✅ Ngày {target_date} đã có kết quả: {entry['actual_results']}")
    else:
        # Fetch kết quả từ web
        actual = fetch_results_for_date(target_date)
        if actual is None:
            print(f"❌ Không thể lấy kết quả ngày {target_date}.")
            return

        top_k  = entry.get("top_k", 10)
        picks  = entry.get("ensemble_picks", [])

        # Tính hits và PnL
        actual_counts = {}
        for v in actual:
            actual_counts[v] = actual_counts.get(v, 0) + 1
        ensemble_hits = sum(actual_counts.get(n, 0) for n in picks)
        revenue       = ensemble_hits * PAYOUT_PER_HIT
        cost          = top_k * COST_PER_NUM
        pnl           = revenue - cost

        per_method_hits = {}
        for method, preds in entry.get("per_method", {}).items():
            method_picks = preds[:top_k]
            per_method_hits[method] = sum(actual_counts.get(n, 0) for n in method_picks)

        # Cập nhật entry
        entry["actual_results"]  = actual
        entry["ensemble_hits"]   = ensemble_hits
        entry["per_method_hits"] = per_method_hits
        entry["revenue_k"]       = revenue
        entry["pnl_k"]           = pnl
        log[entry_idx] = entry
        save_log(log)

        # In kết quả ngày
        result_icon = "🎉" if ensemble_hits > 0 else "😔"
        print(f"\n{result_icon} KẾT QUẢ NGÀY {target_date}:")
        print(f"  Đã chọn  : {', '.join(f'{n:02d}' for n in picks)}")
        print(f"  Kết quả  : {', '.join(f'{n:02d}' for n in sorted(set(actual)))}")
        print(f"  Trúng    : {ensemble_hits} nháy")
        print(f"  Chi phí  : {cost:,}k đ")
        print(f"  Thu về   : {revenue:,}k đ")
        print(f"  Lãi/Lỗ  : {pnl:+,}k đ")
        print()
        print("  Từng phương pháp:")
        for method, hits in per_method_hits.items():
            icon = "✅" if hits > 0 else "  "
            print(f"    {icon} {method:<22}: {hits} nháy")

    # ── Tối ưu hoá bộ lọc ma trận (mỗi 7 ngày) ───────────────────────────
    rules_data = load_rules_data()
    last_update = rules_data.get("updated_at")
    should_optimize = args.force_reweight

    if not should_optimize and last_update:
        last_dt = datetime.fromisoformat(last_update)
        days_since = (datetime.now(TZ) - last_dt).days
        if days_since >= OPTIMIZE_EVERY:
            should_optimize = True
            print(f"\n📅 Đã {days_since} ngày kể từ lần tối ưu hoá gần nhất → chạy lại bộ tối ưu…")

    if not should_optimize and not last_update:
        completed_count = sum(1 for e in log if e.get("actual_results") is not None)
        if completed_count >= OPTIMIZE_EVERY:
            should_optimize = True

    if should_optimize:
        print("\n⚖️  Đang chạy tối ưu hoá lại bộ lọc ma trận quyết định trên dữ liệu mới nhất...")
        
        # Chạy matrix_optimizer.py
        uv_bin = shutil.which("uv")
        if uv_bin:
            cmd = [uv_bin, "run", "backtests/matrix_optimizer.py"]
        else:
            cmd = [sys.executable, "backtests/matrix_optimizer.py"]
            
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root_dir))
        if result.returncode == 0:
            print("✅ Tối ưu hoá hoàn thành thành công!")
            if result.stdout:
                # Trích xuất 10 dòng cuối cùng của log tối ưu
                lines = result.stdout.strip().splitlines()
                for line in lines[-8:]:
                    print(f"  {line}")
        else:
            print(f"❌ Lỗi khi chạy tối ưu hoá bộ lọc: {result.stderr}")
    else:
        completed = sum(1 for e in log if e.get("actual_results") is not None)
        print(f"\nℹ️  Bộ lọc chưa được tối ưu lại ({completed} ngày có kết quả / cần {OPTIMIZE_EVERY})")

    # ── Thống kê tổng hợp ─────────────────────────────────────────────────
    completed_entries = [e for e in log if e.get("actual_results") is not None]
    if completed_entries:
        total_cost = sum(e.get("cost_k", 0) for e in completed_entries)
        total_rev  = sum(e.get("revenue_k", 0) for e in completed_entries)
        total_pnl  = total_rev - total_cost
        roi        = total_pnl / total_cost * 100 if total_cost else 0
        win_days   = sum(1 for e in completed_entries if (e.get("ensemble_hits") or 0) > 0)
        print(f"\n📊 Thống kê tổng hợp ({len(completed_entries)} ngày):")
        print(f"  Win rate : {win_days}/{len(completed_entries)} ngày ({win_days/len(completed_entries)*100:.0f}%)")
        print(f"  Tổng PnL : {total_pnl:+,}k đ")
        print(f"  ROI      : {roi:+.2f}%")

if __name__ == "__main__":
    main()
