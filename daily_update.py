"""
daily_update.py — Cập nhật kết quả thực tế + điều chỉnh trọng số (mỗi 7 ngày)

Cách dùng:
  uv run daily_update.py                         # tự fetch kết quả hôm nay
  uv run daily_update.py --date 2026-07-13       # cập nhật ngày cụ thể
  uv run daily_update.py --force-reweight        # bắt buộc tính lại trọng số ngay
"""
import sys
import json
import argparse
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

root_dir = Path(__file__).parent
sys.path.append(str(root_dir / "src"))

import numpy as np
import pandas as pd

# ── Đường dẫn ────────────────────────────────────────────────────────────────
DATA_CSV     = root_dir / "data" / "xsmb-2-digits.csv"
PRED_LOG     = root_dir / "predictions" / "prediction_log.json"
WEIGHTS_FILE = root_dir / "predictions" / "adaptive_weights.json"

TZ             = ZoneInfo("Asia/Ho_Chi_Minh")
COST_PER_NUM   = 27
PAYOUT_PER_HIT = 99
ALPHA          = 0.15    # tốc độ học trọng số
WEIGHT_MIN     = 0.1
WEIGHT_MAX     = 10.0
REWEIGHT_EVERY = 7       # số ngày giữa các lần cập nhật trọng số

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_log() -> list:
    if PRED_LOG.exists():
        with open(PRED_LOG, encoding="utf-8") as f:
            return json.load(f)
    return []

def save_log(log: list) -> None:
    with open(PRED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def load_weights_data() -> dict:
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"weights": {}, "updated_at": None, "window_days": 7, "rolling_stats": {}}

def save_weights_data(data: dict) -> None:
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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

def compute_hits(picks: list[int], actual: list[int]) -> dict:
    """Tính số nháy trúng và revenue."""
    actual_counts = {}
    for v in actual:
        actual_counts[v] = actual_counts.get(v, 0) + 1
    hits = sum(actual_counts.get(n, 0) for n in picks)
    return hits

def recompute_weights(log: list, current_weights: dict) -> tuple[dict, dict]:
    """
    Tính lại trọng số dựa trên 7 ngày gần nhất có kết quả.
    Trả về (new_weights, rolling_stats)
    """
    # Lọc entries có actual_results
    completed = [e for e in log if e.get("actual_results") is not None]
    if len(completed) < 3:
        print(f"  Chưa đủ dữ liệu ({len(completed)} ngày có kết quả, cần ≥ 3)")
        return current_weights, {}

    window = completed[-REWEIGHT_EVERY:]  # 7 ngày gần nhất

    # Tính ROI của từng phương pháp trong window
    method_names = list(window[0].get("per_method", {}).keys()) if window else []
    method_stats = {}

    for method in method_names:
        hits_list = []
        roi_list  = []
        for entry in window:
            picks   = entry.get("per_method", {}).get(method, [])[:entry.get("top_k", 4)]
            actual  = entry.get("actual_results", [])
            top_k   = entry.get("top_k", 4)
            if not picks or not actual:
                continue
            hits = compute_hits(picks, actual)
            cost = top_k * COST_PER_NUM
            rev  = hits * PAYOUT_PER_HIT
            roi  = (rev - cost) / cost * 100
            hits_list.append(hits)
            roi_list.append(roi)
        if not roi_list:
            continue
        method_stats[method] = {
            "avg_roi"   : float(np.mean(roi_list)),
            "hit_rate"  : float(np.mean([h > 0 for h in hits_list])),
            "n_days"    : len(roi_list),
        }

    if not method_stats:
        return current_weights, {}

    # Điều chỉnh trọng số theo ROI tương đối
    avg_roi = np.mean([s["avg_roi"] for s in method_stats.values()])
    new_weights = dict(current_weights)

    for method, stats in method_stats.items():
        # Tìm tên đầy đủ trong current_weights (partial match)
        matched_key = None
        for k in current_weights:
            # Match theo từ đầu tiên của tên phương pháp
            method_short = method.split()[0].lower()
            k_short      = k.split()[0].lower()
            if method_short in k.lower() or method in k or k in method:
                matched_key = k
                break
        if matched_key is None:
            continue

        old_w    = current_weights.get(matched_key, 1.0)
        delta    = stats["avg_roi"] - avg_roi
        adj      = ALPHA * delta / 100  # normalize
        new_w    = old_w * (1 + adj)
        new_w    = max(WEIGHT_MIN, min(WEIGHT_MAX, new_w))
        new_weights[matched_key] = round(new_w, 3)

        direction = "⬆" if new_w > old_w else ("⬇" if new_w < old_w else "─")
        print(f"  {direction} {matched_key:<45}: {old_w:.3f} → {new_w:.3f}  (ROI {stats['avg_roi']:+.1f}%)")

    return new_weights, method_stats

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cập nhật kết quả XSMB và trọng số")
    parser.add_argument("--date",          type=str, default=None, help="Ngày cần cập nhật (YYYY-MM-DD)")
    parser.add_argument("--force-reweight", action="store_true",   help="Bắt buộc tính lại trọng số ngay")
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

        top_k  = entry.get("top_k", 4)
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

    # ── Cập nhật trọng số (mỗi 7 ngày) ───────────────────────────────────
    weights_data = load_weights_data()
    last_update  = weights_data.get("updated_at")
    should_reweight = args.force_reweight

    if not should_reweight and last_update:
        last_dt = datetime.fromisoformat(last_update)
        days_since = (datetime.now(TZ) - last_dt).days
        if days_since >= REWEIGHT_EVERY:
            should_reweight = True
            print(f"\n📅 Đã {days_since} ngày kể từ lần cập nhật cuối → cập nhật trọng số…")

    if not should_reweight and not last_update:
        completed_count = sum(1 for e in log if e.get("actual_results") is not None)
        if completed_count >= REWEIGHT_EVERY:
            should_reweight = True

    if should_reweight:
        print("\n⚖️  Tính lại trọng số Ensemble dựa trên hiệu suất gần đây:")
        current_weights = weights_data.get("weights", {})
        new_weights, rolling_stats = recompute_weights(log, current_weights)

        weights_data["weights"]       = new_weights
        weights_data["updated_at"]    = datetime.now(TZ).isoformat()
        weights_data["rolling_stats"] = rolling_stats
        save_weights_data(weights_data)
        print(f"\n✅ Đã lưu trọng số mới vào {WEIGHTS_FILE}")
    else:
        completed = sum(1 for e in log if e.get("actual_results") is not None)
        print(f"\nℹ️  Trọng số chưa được cập nhật ({completed} ngày có kết quả / cần {REWEIGHT_EVERY})")

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
