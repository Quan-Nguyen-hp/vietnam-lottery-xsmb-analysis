"""
prediction_dashboard.py — Xem thống kê hiệu suất dự đoán (bản nâng cấp Matrix Decision)

Cách dùng:
  python prediction_dashboard.py           # toàn bộ lịch sử
  python prediction_dashboard.py --days 30 # 30 ngày gần nhất
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root_dir = Path(__file__).parent

PRED_LOG   = root_dir / "predictions" / "prediction_log.json"
RULES_FILE = root_dir / "predictions" / "matrix_rules.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

COST_PER_NUM   = 27
PAYOUT_PER_HIT = 99

def bar_ascii(value, max_val=100, width=20):
    if value >= 0:
        n = min(int(value / max_val * width), width)
        return ("█" * n).ljust(width)
    else:
        n = min(int(abs(value) / max_val * width), width)
        return ("░" * n).ljust(width)

def main():
    parser = argparse.ArgumentParser(description="Dashboard dự đoán XSMB")
    parser.add_argument("--days", type=int, default=None, help="Số ngày gần nhất (mặc định: toàn bộ)")
    args = parser.parse_args()

    if not PRED_LOG.exists():
        print("❌ Chưa có dữ liệu log. Hãy chạy daily_predict.py trước.")
        return

    with open(PRED_LOG, encoding="utf-8") as f:
        log = json.load(f)

    completed = [e for e in log if e.get("actual_results") is not None]
    pending   = [e for e in log if e.get("actual_results") is None]

    if args.days:
        completed = completed[-args.days:]

    SEP = "═" * 62
    print(f"\n{'╔' + SEP + '╗'}")
    title = f"  📊  DASHBOARD DỰ ĐOÁN XSMB (Matrix Decision)"
    if args.days:
        title += f" — {args.days} NGÀY GẦN NHẤT"
    print(f"║{title:<62}║")
    print(f"{'╠' + SEP + '╣'}")

    if not completed:
        print(f"║  Chưa có kết quả nào được ghi nhận.{' '*26}║")
        print(f"{'╚' + SEP + '╝'}")
        return

    # ── Thống kê tổng ─────────────────────────────────────────────────────
    total_cost = sum(e.get("cost_k", 0) for e in completed)
    total_rev  = sum(e.get("revenue_k", 0) for e in completed)
    total_pnl  = total_rev - total_cost
    roi        = total_pnl / total_cost * 100 if total_cost else 0
    win_days   = [e for e in completed if (e.get("ensemble_hits") or 0) > 0]
    win_rate   = len(win_days) / len(completed) * 100

    # Chuỗi thắng/thua hiện tại
    streak = 0
    streak_type = None
    for e in reversed(completed):
        hit = (e.get("ensemble_hits") or 0) > 0
        if streak_type is None:
            streak_type = "W" if hit else "L"
            streak = 1
        elif (streak_type == "W" and hit) or (streak_type == "L" and not hit):
            streak += 1
        else:
            break

    print(f"║  Kỳ theo dõi  : {completed[0]['date']} → {completed[-1]['date']}{' '*16}║")
    print(f"║  Số ngày theo dõi : {len(completed):<41}║")
    print(f"║  Chưa có kết quả  : {len(pending):<41}║")
    print(f"{'╠' + SEP + '╣'}")
    print(f"║  💰 Tổng chi phí  : {total_cost:>10,}k đ{' '*27}║")
    print(f"║  💵 Tổng thu về   : {total_rev:>10,}k đ{' '*27}║")
    pnl_icon = "🟢" if total_pnl >= 0 else "🔴"
    print(f"║  {pnl_icon} Lãi/Lỗ ròng  : {total_pnl:>+10,}k đ{' '*27}║")
    print(f"║  📈 ROI           : {roi:>+10.2f}%{' '*28}║")
    print(f"║  🎯 Win rate      : {win_rate:>10.1f}%  ({len(win_days)}/{len(completed)} ngày){' '*10}║")
    streak_icon = "🔥" if streak_type == "W" else "❄️"
    streak_label = f"Chuỗi {'thắng' if streak_type=='W' else 'thua'} hiện tại"
    print(f"║  {streak_icon} {streak_label:<18}: {streak} ngày{' '*27}║")
    print(f"{'╠' + SEP + '╣'}")

    # ── Phân phối PnL theo tháng ──────────────────────────────────────────
    from collections import defaultdict
    monthly = defaultdict(lambda: {"pnl": 0, "cost": 0, "win": 0, "total": 0})
    for e in completed:
        ym = e["date"][:7]
        monthly[ym]["pnl"]   += e.get("pnl_k", 0) or 0
        monthly[ym]["cost"]  += e.get("cost_k", 0) or 0
        monthly[ym]["total"] += 1
        if (e.get("ensemble_hits") or 0) > 0:
            monthly[ym]["win"] += 1

    print(f"║  📅  PHÂN PHỐI THEO THÁNG{' '*36}║")
    print(f"║  {'Tháng':<9} {'Ngày':>4} {'Thắng':>5} {'Lãi/Lỗ':>12} {'ROI':>7}  Bar{' '*14}║")
    print(f"║  {'-'*9} {'-'*4} {'-'*5} {'-'*12} {'-'*7}  {'-'*14}║")
    for ym in sorted(monthly.keys()):
        m  = monthly[ym]
        roi_m = m["pnl"] / m["cost"] * 100 if m["cost"] else 0
        icon  = "🟢" if m["pnl"] >= 0 else "🔴"
        b     = bar_ascii(roi_m, max_val=80, width=14)
        pnl_str = f"{m['pnl']:>+,}k"
        print(f"║  {ym:<9} {m['total']:>4} {m['win']:>5} {pnl_str:>12} {roi_m:>+6.1f}%  {icon}{b}║")

    print(f"{'╠' + SEP + '╣'}")

    # ── Quy tắc lọc ma trận hiện tại ──────────────────────────────────────
    if RULES_FILE.exists():
        with open(RULES_FILE, encoding="utf-8") as f:
            rdata = json.load(f)
        rules = rdata.get("rules", {})
        updated_at = rdata.get("updated_at", "Chưa cập nhật")
        print(f"║  ⚖️  BỘ LỌC MA TRẬN TỐI ƯU (Cập nhật: {updated_at[:10] if updated_at else 'N/A':<14})║")
        for name, val in rules.items():
            bar_w = "▓" * min(int(val * 15), 14)
            print(f"║    {name:<22}: {val:>5.2f}  {bar_w:<14}{' '*14}║")

    print(f"║{' '*62}║")
    # ── Lịch sử 10 ngày gần nhất ──────────────────────────────────────────
    print(f"║  🗂️  10 NGÀY GẦN NHẤT{' '*41}║")
    print(f"║  {'Ngày':<12} {'Số chọn':>14} {'Trúng':>6} {'PnL':>10}  {'':>12}║")
    print(f"║  {'-'*12} {'-'*14} {'-'*6} {'-'*10}  {'-'*12}║")
    for e in completed[-10:]:
        picks_list = e.get("ensemble_picks") or []
        picks_str = ",".join(f"{n:02d}" for n in picks_list[:5])
        if len(picks_list) > 5:
            picks_str += ".."
        hits  = e.get("ensemble_hits") or 0
        pnl   = e.get("pnl_k") or 0
        icon  = "✅" if hits > 0 else "  "
        pnl_s = f"{pnl:>+,}kđ"
        print(f"║  {e['date']:<12} {picks_str:>14} {icon}{hits:>4} {pnl_s:>10}  {'':>12}║")

    print(f"{'╚' + SEP + '╝'}")

    # Pending
    if pending:
        print(f"\n⏳ Đang chờ kết quả ({len(pending)} ngày):")
        for e in pending[-5:]:
            picks_list = e.get("ensemble_picks") or []
            picks_str = ",".join(f"{n:02d}" for n in picks_list)
            print(f"   {e['date']}: chọn [{picks_str}]")

if __name__ == "__main__":
    main()
